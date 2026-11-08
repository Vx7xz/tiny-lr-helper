"""The final test: a helper meta-trained only on 3-layer MLPs, used as is on a small CNN (CIFAR-10)
and a small character-level transformer (TinyShakespeare), against AdamW + cosine tuned by the usual
13-point grid search (picked by validation loss, the tuning runs measured and counted).

Per method and paired seed: test metric (CNN: accuracy; transformer: loss in nats per character,
lower is better), steps until validation reaches a target set by the rule below, and what the runs
cost the machine (resources.Meter). Target: 98% of tuned AdamW's mean final validation accuracy
(CNN), or 102% of its mean final validation loss (transformer).

    python final_test.py --task cnn --nets controller_cos_v18.pt
    python final_test.py --task transformer --nets controller_cos_v18.pt
    python final_test.py --task cnn --fake --steps 60 --seeds 1   # random data: checks the code only
"""
import argparse
import math
import os
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F

import common
from common import RESTED, load_helper
from general import CNN, CharTransformer, augment, train_model
from resources import Meter

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["cnn", "transformer"], required=True)
ap.add_argument("--nets", nargs="+", default=[])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay of the baseline")
ap.add_argument("--data", default="", choices=["", "tinyimagenet", "tinystories"],
                help="another dataset instead of CIFAR-10 (cnn) / TinyShakespeare (transformer)")
ap.add_argument("--limits", action="store_true", help="each method's best: AdamW over lr x weight decay x steps, "
                                                       "the --pg helper over a global lr scale x steps (no retraining)")
ap.add_argument("--target", type=float, help="skip the grid search and AdamW (their numbers from an earlier run "
                                              "of the same test) and run only the --pg helpers against this target")
ap.add_argument("--fake", action="store_true", help="random data of the right shapes, to check the code")
ap.add_argument("--augment", action="store_true", help="CNN: random crop + flip of every training batch")
ap.add_argument("--cnn-width", type=int, default=32, help="CNN: channels of the first stage (general.CNN's c)")
ap.add_argument("--average", action="store_true",
                help="helpers hand over a moving average of their weights (as budget_test --average); adds tuned "
                     "AdamW with the same averaging")
ap.add_argument("--avg-warm", type=float, default=0.0, help="with --average: common.train's ema_warm")
ap.add_argument("--pg", nargs="+", default=[], help="per-group helpers (pg.py, or pg2 with 14 numbers), run by "
                                                    "pg_fast, always averaged")
args = ap.parse_args()
T, SEEDS, BATCH, CONTEXT = args.steps, range(500, 500 + args.seeds), 128, 64  # seeds not used anywhere else

# ---------------------------------------------------------------- data, kept on the GPU as bytes
IMG = 32
if args.data == "tinyimagenet":  # 10 fixed classes (lr_tasks seed 0): 4500 train / 500 val from its train split,
    import lr_tasks  # its 500 val images as test
    t = lr_tasks.tinyimagenet_task(0)
    perm = torch.randperm(len(t["xtr"]), generator=torch.Generator().manual_seed(0)).cuda()
    X = {"train": t["xtr"][perm[500:]], "val": t["xtr"][perm[:500]], "test": t["xva"]}
    Y = {"train": t["ytr"][perm[500:]], "val": t["ytr"][perm[:500]], "test": t["yva"]}
    mean, std, IMG = t["mean"], t["std"], 64
    prep = lambda x: (x.float() / 255 - mean) / std
elif args.data == "tinystories":  # first 3M bytes of its validation file, byte-level
    raw = open(os.path.join("data", "tinystories", "TinyStories-valid.txt"), "rb").read(3_000_000)
    chars = sorted(set(raw))
    lut = torch.zeros(256, dtype=torch.uint8)
    lut[chars] = torch.arange(len(chars), dtype=torch.uint8)
    ids = lut[torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()].cuda()
    n = len(ids)
    D = {"train": ids[:int(0.9 * n)], "val": ids[int(0.9 * n):int(0.95 * n)], "test": ids[int(0.95 * n):]}
elif args.task == "cnn":
    if args.fake:
        xs, ys = torch.randint(0, 256, (6000, 3, 32, 32), dtype=torch.uint8), torch.randint(0, 10, (6000,))
        xt, yt = xs[:1000], ys[:1000]
    else:
        from torchvision import datasets
        tr, te = (datasets.CIFAR10("data", train=s, download=True) for s in (True, False))
        xs, ys = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(tr.targets)
        xt, yt = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(te.targets)
    perm = torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))
    va, tr_i = perm[:5000], perm[5000:]
    X = {"train": xs[tr_i].cuda(), "val": xs[va].cuda(), "test": xt.cuda()}
    Y = {"train": ys[tr_i].cuda(), "val": ys[va].cuda(), "test": yt.cuda()}
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device="cuda")[:, None, None]
    std = torch.tensor([0.2470, 0.2435, 0.2616], device="cuda")[:, None, None]
    prep = lambda x: (x.float() / 255 - mean) / std
else:
    path = os.path.join("data", "tinyshakespeare.txt")
    if args.fake:
        text = "".join(chr(97 + i % 26) for i in torch.randint(0, 26, (200000,)).tolist())
    else:
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
        text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text], dtype=torch.uint8).cuda()
    n = len(ids)
    D = {"train": ids[:int(0.9 * n)], "val": ids[int(0.9 * n):int(0.95 * n)], "test": ids[int(0.95 * n):]}


def new_model(seed):
    torch.manual_seed(seed)
    return (CNN(args.cnn_width, size=IMG) if args.task == "cnn" else CharTransformer(len(chars), context=CONTEXT)).cuda()


@torch.no_grad()
def evaluate(model, split, limit=None):
    """(mean loss, metric): CNN accuracy %, transformer next-character accuracy %."""
    loss, correct, count = 0.0, 0, 0
    if args.task == "cnn":
        x, y = X[split][:limit], Y[split][:limit]
        for s in range(0, len(x), 250):  # small chunks: evaluation, not training, would set the GPU peak
            out = model(prep(x[s:s + 250]))
            loss += F.cross_entropy(out, y[s:s + 250], reduction="sum").item()
            correct += (out.argmax(1) == y[s:s + 250]).sum().item()
        count = len(x)
    else:
        d = D[split][:limit]
        w = d[:(len(d) - 1) // CONTEXT * CONTEXT + 1].long()
        xb, yb = w[:-1].view(-1, CONTEXT), w[1:].view(-1, CONTEXT)
        for s in range(0, len(xb), 64):
            out = model(xb[s:s + 64])
            loss += F.cross_entropy(out.flatten(0, 1), yb[s:s + 64].flatten(), reduction="sum").item()
            correct += (out.argmax(-1) == yb[s:s + 64]).sum().item()
        count = yb.numel()
    return loss / count, 100 * correct / count


def run(seed, lr_fn=None, helper=None, wd=0.0, eval_every=0, avg=False, pg=None, lr_scale=1.0):
    model = new_model(seed)
    g = torch.Generator(device="cuda").manual_seed(seed)

    def batch_loss(t):
        if args.task == "cnn":
            i = torch.randint(0, len(X["train"]), (BATCH,), device="cuda", generator=g)
            x = prep(X["train"][i])
            return F.cross_entropy(model(augment(x, g) if args.augment else x), Y["train"][i])
        i = torch.randint(0, len(D["train"]) - CONTEXT - 1, (BATCH // 2,), device="cuda", generator=g)
        w = D["train"][i[:, None] + torch.arange(CONTEXT + 1, device="cuda")].long()
        return F.cross_entropy(model(w[:, :-1]).flatten(0, 1), w[:, 1:].flatten())

    model.train()
    eval_fn = lambda: evaluate(model, "val", 2000 if args.task == "cnn" else 50000)
    if pg is not None:
        from pg_fast import train_pg_fast
        hist = train_pg_fast(model, batch_loss, T, pg, avg=True, avg_warm=args.avg_warm or 8.0, eval_fn=eval_fn,
                             eval_every=eval_every, lr_scale=lr_scale)
    else:
        hist = train_model(model.groups(), batch_loss, T, model.width, lr_fn=lr_fn, helper=helper, wd=wd,
                           eval_fn=eval_fn, eval_every=eval_every, avg=avg, avg_warm=args.avg_warm)
    model.eval()
    return model, hist


def cosine(loglr):
    return lambda t: loglr + math.log10(0.5 * (1 + math.cos(math.pi * t / T)) + 1e-8)


def measured(runs):
    """Run each (seed, kwargs) with a Meter around all of them; GPU seconds leave the GD_MAX_BUSY rest out."""
    t0, r0, out = time.time(), RESTED[0], []
    with Meter() as m:
        for seed, kw in runs:
            out.append(run(seed, **kw))
    return out, (time.time() - t0 - (RESTED[0] - r0)) / len(runs), m


if args.limits:  # each method's best: a wide search on seed 0 (picked by validation), then seeds 500-502
    better = (lambda a, b: a[1] > b[1]) if args.task == "cnn" else (lambda a, b: a[0] < b[0])  # val acc / val loss
    steps_list = (1500, 3000, 6000) if args.task == "cnn" else (3000, 6000)
    lrs = [x / 4 for x in (range(-18, -11) if args.task == "cnn" else range(-12, -5))]
    helper = torch.load(args.pg[0])
    search = {"AdamW (lr x wd x steps), avg": [(S, dict(lr_fn=lr, wd=wd, avg=True)) for S in steps_list for lr in lrs
                                               for wd in (0.0, 0.01, 0.1, 0.5)],
              f"{args.pg[0]} (lr scale x steps)": [(S, dict(pg=helper, lr_scale=10 ** (k / 8))) for S in steps_list
                                                   for k in range(-4, 5)]}
    print(f"=== {args.task} limits: search on seed 0, then seeds {list(SEEDS)} ===", flush=True)
    for label, cands in search.items():
        best_v, best_c, t0 = None, None, time.time()
        for S, kw in cands:
            T = S  # run() and cosine() read the global T
            desc = f"T={S} " + ", ".join(f"{k}={x:.3g}" for k, x in kw.items() if isinstance(x, float))
            kw = dict(kw, lr_fn=cosine(kw["lr_fn"])) if "lr_fn" in kw else kw
            mdl, _ = run(0, **kw)
            v = evaluate(mdl, "val")
            del mdl
            print(f"  {label[:5]} {desc:45s} val loss {v[0]:.4f}  val metric {v[1]:.2f}%", flush=True)
            if best_v is None or better(v, best_v):
                best_v, best_c, best_d = v, (S, kw), desc
        search_s = time.time() - t0
        T = best_c[0]
        out, sec, m = measured([(s, dict(best_c[1], eval_every=50)) for s in SEEDS])
        test = torch.tensor([evaluate(mdl, "test") for mdl, _ in out])
        del out
        print(f"BEST {label}: {best_d} | test loss {test[:, 0].mean():.4f}  test metric {test[:, 1].mean():.2f}% (seeds "
              f"{' '.join(f'{x:.2f}' for x in test[:, 1].tolist())}) | search {len(cands)} runs {search_s:.0f} s, "
              f"final run {sec:.1f} s, {m.energy_j / len(SEEDS):.0f} J, GPU {m.gpu_mb:.0f} MB", flush=True)
    sys.exit()

if args.target is not None:  # helpers only; AdamW's numbers come from an earlier run of the same test
    print(f"=== {args.task}: {T} steps, batch {BATCH}, seeds {list(SEEDS)}, helpers only ===")
    print(f"target: validation {'accuracy' if args.task == 'cnn' else 'loss'} {args.target:.3f} (given)")
    print(f"{'method':28s} {'test loss':>9s} {'test metric':>13s} {'steps to target':>16s} {'s/run':>6s} {'J/run':>6s} "
          f"{'GPU MB':>7s} {'RAM MB':>6s}")
    for p in args.pg:
        out, sec, m = measured([(s, dict(pg=torch.load(p), eval_every=50)) for s in SEEDS])
        test = torch.tensor([evaluate(mdl, "test") for mdl, _ in out])
        ok = (lambda v: v >= args.target) if args.task == "cnn" else (lambda v: v <= args.target)
        key = "val_metric" if args.task == "cnn" else "val_loss"
        steps = [next((s for s, v in zip(h["step"], h[key]) if ok(v)), T) for _, h in out]
        print(f"{p + ' (pg)':28s} {test[:, 0].mean():9.4f} {test[:, 1].mean():12.2f}% {sum(steps) / len(steps):9.0f} "
              f"{steps}  {sec:6.2f} {m.energy_j / len(SEEDS):6.0f} {m.gpu_mb:7.0f} {m.ram_mb:6.0f}", flush=True)
        del out
    sys.exit()

# ---------------------------------------------------------------- tuned AdamW: the grid search
grid = [x / 4 for x in range(-18, -5)]  # 1e-4.5 ... 1e-1.5, as in common.cosine_grid_search
res, grid_s, gm = measured([(0, dict(lr_fn=cosine(g), wd=args.wd)) for g in grid])
vloss = [evaluate(mdl, "val")[0] for mdl, _ in res]
del res
best = grid[min(range(len(grid)), key=lambda i: vloss[i] if math.isfinite(vloss[i]) else 1e9)]
print(f"=== {args.task}{' (fake data)' if args.fake else ''}: {T} steps, batch {BATCH}, seeds {list(SEEDS)} ===")
print(f"grid search (13 runs, one after another): best lr 1e{best:.2f}, {grid_s * len(grid):.0f} GPU s, "
      f"{gm.energy_j:.0f} J, GPU {gm.gpu_mb:.0f} MB, RAM {gm.ram_mb:.0f} MB", flush=True)

methods = [("AdamW + cosine, tuned", dict(lr_fn=cosine(best), wd=args.wd))]
if args.average:  # the same averaging for AdamW, so its own share of any gain shows
    methods.append(("AdamW + cosine, tuned, avg", dict(lr_fn=cosine(best), wd=args.wd, avg=True)))
methods += [(p + (" avg" if args.average else ""), dict(helper=load_helper(p), avg=args.average)) for p in args.nets]
methods += [(p + " (pg)", dict(pg=torch.load(p))) for p in args.pg]
results = {}
for label, kw in methods:
    out, sec, m = measured([(s, dict(kw, eval_every=50)) for s in SEEDS])
    test = torch.tensor([evaluate(mdl, "test") for mdl, _ in out])  # (seeds, 2): loss, metric
    results[label] = (test, [{k: x.cpu() if torch.is_tensor(x) else x for k, x in h.items()} for _, h in out], sec, m)
    del out  # otherwise this method's 5 models stay on the GPU while the next one is measured (vram_check.py)

ref_hists = results[methods[0][0]][1]
if args.task == "cnn":
    target = 0.98 * sum(h["val_metric"][-1] for h in ref_hists) / len(ref_hists)
    reached = lambda h: next((s for s, v in zip(h["step"], h["val_metric"]) if v >= target), T)
else:
    target = 1.02 * sum(h["val_loss"][-1] for h in ref_hists) / len(ref_hists)
    reached = lambda h: next((s for s, v in zip(h["step"], h["val_loss"]) if v <= target), T)
tuned_j = gm.energy_j + results[methods[0][0]][3].energy_j / len(SEEDS)
unit = "acc" if args.task == "cnn" else "char acc"
print(f"target: validation {'accuracy' if args.task == 'cnn' else 'loss'} {target:.3f} "
      f"({'98% of' if args.task == 'cnn' else '102% of'} tuned AdamW's final)")
print(f"{'method':28s} {'test loss':>9s} {'test ' + unit:>13s} {'vs tuned':>9s} {'steps to target':>16s} {'s/run':>6s} "
      f"{'J/run':>6s} {'GPU %':>5s} {'GPU MB':>7s} {'CPU s':>6s} {'RAM MB':>6s} {'J vs tuned':>10s}")
ref = results[methods[0][0]][0]
for label, (test, hists, sec, m) in results.items():
    d = (test[:, 1] - ref[:, 1]) if args.task == "cnn" else (ref[:, 0] - test[:, 0])  # > 0: better
    steps = [reached(h) for h in hists]
    print(f"{label:28s} {test[:, 0].mean():9.4f} {test[:, 1].mean():12.2f}% {d.mean():+9.3f} "
          f"{sum(steps) / len(steps):9.0f} ({sum(s < r for s, r in zip(steps, [reached(h) for h in ref_hists]))}/"
          f"{len(steps)}) {sec:6.2f} {m.energy_j / len(SEEDS):6.0f} {m.gpu_util:5.0f} {m.gpu_mb:7.0f} "
          f"{m.cpu_s / len(SEEDS):6.2f} {m.ram_mb:6.0f} {m.energy_j / len(SEEDS) / tuned_j:10.0%}", flush=True)
