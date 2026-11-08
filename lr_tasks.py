"""Datasets and task runners for the lr-choice work, with a hard train/test split.

  TRAIN (calibration / meta-training only): MNIST, EMNIST-balanced (10 of its 47 classes per seed),
         CIFAR-100 (10 of 20 superclasses per seed), stdlib Python text, enwik8 (Wikipedia XML, bytes)
  TEST (measuring only, never fitted on):   Fashion-MNIST, KMNIST, CIFAR-10, SVHN, TinyShakespeare,
         WikiText-2 (clean Wikipedia articles, bytes), Tiny ImageNet (10 of 200 classes per seed, 64x64),
         TinyStories (simple children's stories, bytes)

Every image set: uint8 (n, C, H, W) on the GPU, a 5000-image validation split (from the training split, as
elsewhere in the project; CIFAR-100 uses its test split), per-channel mean/std. Every text set: uint8 ids,
a training part and a 50k-character validation part. A run is f(seed, s) -> final validation loss, training
with the helper's lrs x 10**s.
"""
import glob
import io
import math
import os
import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATASETS, _idx
from general import CNN, CharTransformer, augment
from pg_fast import train_pg_fast

DEV = "cuda"
TRAIN_SETS = ("mnist", "emnist", "cifar100", "stdlib", "enwik8")
TEST_SETS = ("fashion", "kmnist", "cifar10", "svhn", "shakespeare", "wikitext2",
             "tinyimagenet", "tinystories")  # the last two: the day's final test (lr_final_test.py)
assert not set(TRAIN_SETS) & set(TEST_SETS)
_cache = {}


def _split(x, y, mean=None, std=None):
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0)).to(x.device)
    va, tr = perm[:5000], perm[5000:]
    f = x[tr[:5000]].float() / 255
    mean = f.mean((0, 2, 3)) if mean is None else torch.tensor([mean], device=x.device)
    std = f.std((0, 2, 3)) if std is None else torch.tensor([std], device=x.device)
    return dict(xtr=x[tr], ytr=y[tr], xva=x[va], yva=y[va], mean=mean[:, None, None], std=std[:, None, None])


def image_set(name):
    if name in _cache:
        return _cache[name]
    if name in ("mnist", "fashion", "kmnist"):
        folder, mean, std = DATASETS[name]
        raw = os.path.join("data", folder, "raw")
        x = _idx(os.path.join(raw, "train-images-idx3-ubyte")).to(DEV)[:, None]
        d = _split(x, _idx(os.path.join(raw, "train-labels-idx1-ubyte")).to(DEV).long(), mean, std)
    elif name == "emnist":
        raw = os.path.join("data", "emnist", "EMNIST", "raw")
        x = _idx(os.path.join(raw, "emnist-balanced-train-images-idx3-ubyte")).to(DEV)[:, None]
        d = _split(x, _idx(os.path.join(raw, "emnist-balanced-train-labels-idx1-ubyte")).to(DEV).long())
        d["classes"] = 47
    elif name == "cifar10":
        import pickle
        raw = []
        for k in range(1, 6):
            with open(os.path.join("data", "cifar-10-batches-py", f"data_batch_{k}"), "rb") as f:
                raw.append(pickle.load(f, encoding="bytes"))
        x = torch.from_numpy(np.concatenate([r[b"data"] for r in raw]).reshape(-1, 3, 32, 32)).to(DEV)
        d = _split(x, torch.tensor(sum((list(r[b"labels"]) for r in raw), []), device=DEV))
    elif name == "svhn":
        from scipy.io import loadmat
        m = loadmat(os.path.join("data", "svhn", "train_32x32.mat"))
        x = torch.from_numpy(np.ascontiguousarray(m["X"].transpose(3, 2, 0, 1))).to(DEV)
        d = _split(x, torch.from_numpy(m["y"].astype(np.int64).ravel() % 10).to(DEV))
    elif name == "cifar100":
        from cifar100 import load_cifar100
        d = load_cifar100(DEV)
        d["classes"] = 20
    elif name == "tinyimagenet":
        d = dict(classes=200)  # decoded per task (10 classes at a time): tinyimagenet_task
    else:
        raise KeyError(name)
    _cache[name] = d
    return d


def tinyimagenet_task(seed):
    """10 of Tiny ImageNet's 200 classes picked by seed: 64x64 photos, 500 train and 50 validation each
    (its own val split; the 10k test images have no labels)."""
    from PIL import Image
    root = os.path.join("data", "tinyimagenet")
    zf = zipfile.ZipFile(os.path.join(root, "tiny-imagenet-200.zip"))
    wnids = sorted(zf.read("tiny-imagenet-200/wnids.txt").decode().split())
    classes = [wnids[k] for k in torch.randperm(200, generator=torch.Generator().manual_seed(seed))[:10].tolist()]
    load = lambda n: np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
    xs, ys = [], []
    for k, wn in enumerate(classes):
        names = [n for n in zf.namelist() if n.startswith(f"tiny-imagenet-200/train/{wn}/images/") and n.endswith(".JPEG")]
        xs += [load(n) for n in names]
        ys += [k] * len(names)
    xv, yv = [], []
    for line in zf.read("tiny-imagenet-200/val/val_annotations.txt").decode().splitlines():
        fn, wn = line.split("\t")[:2]
        if wn in classes:
            xv.append(load(f"tiny-imagenet-200/val/images/{fn}"))
            yv.append(classes.index(wn))
    t = lambda a: torch.from_numpy(np.stack(a).transpose(0, 3, 1, 2).copy()).to(DEV)
    task = dict(xtr=t(xs), ytr=torch.tensor(ys, device=DEV), xva=t(xv), yva=torch.tensor(yv, device=DEV))
    f = task["xtr"][:5000].float() / 255
    task["mean"], task["std"] = f.mean((0, 2, 3))[:, None, None], f.std((0, 2, 3))[:, None, None]
    return task


def image_task(name, seed):
    """10-class task: the set itself, or 10 classes picked by seed (EMNIST, CIFAR-100 superclasses)."""
    if name == "tinyimagenet":
        if ("tin", seed) not in _cache:
            _cache[("tin", seed)] = tinyimagenet_task(seed)
        return _cache[("tin", seed)]
    d = image_set(name)
    if name == "cifar100":
        from cifar100 import cifar100_task
        return cifar100_task(d, seed)
    if "classes" not in d:
        return d
    classes = torch.randperm(d["classes"], generator=torch.Generator().manual_seed(seed))[:10].tolist()
    lut = torch.full((d["classes"],), -1, dtype=torch.long, device=DEV)
    lut[classes] = torch.arange(10, device=DEV)
    t = dict(mean=d["mean"], std=d["std"])
    for s in ("tr", "va"):
        y = lut[d["y" + s]]
        t["x" + s], t["y" + s] = d["x" + s][y >= 0], y[y >= 0]
    return t


def text_set(name):
    if name in _cache:
        return _cache[name]
    if name == "stdlib":  # meta_train_pg.py's text
        lib, ok, parts = os.path.join(sys.base_prefix, "Lib"), set(chr(c) for c in range(32, 127)) | {"\n"}, []
        for path in sorted(glob.glob(os.path.join(lib, "*.py"))):
            parts.append("".join(ch for ch in open(path, encoding="utf-8", errors="ignore").read() if ch in ok))
            if sum(map(len, parts)) > 1_200_000:
                break
        raw = "".join(parts)[:1_200_000].encode()
        n = len(raw)
        tr, va = raw[:int(0.9 * n)], raw[int(0.9 * n):][:50_000]
    elif name == "shakespeare":  # lr_choice_test.py's split
        raw = open(os.path.join("data", "tinyshakespeare.txt"), encoding="utf-8").read().encode()
        n = len(raw)
        tr, va = raw[:int(0.9 * n)], raw[int(0.9 * n):int(0.95 * n)][:50_001]
    elif name == "enwik8":
        with zipfile.ZipFile(os.path.join("data", "enwik8", "enwik8.zip")) as z:
            raw = z.open("enwik8").read(3_000_000)
        tr, va = raw[:2_700_000], raw[2_700_000:2_750_000]
    elif name == "tinystories":  # its validation file (~19 MB): first 2.7M bytes train, next 50k validation
        raw = open(os.path.join("data", "tinystories", "TinyStories-valid.txt"), "rb").read()
        tr, va = raw[:2_700_000], raw[2_700_000:2_750_000]
    elif name == "wikitext2":
        import pyarrow.parquet as pq
        get = lambda s: "".join(pq.read_table(os.path.join("data", "wikitext2", f"{s}.parquet"))["text"].to_pylist()).encode()
        tr, va = get("train"), get("validation")[:50_000]
    else:
        raise KeyError(name)
    used = sorted(set(tr) | set(va))
    lut = np.zeros(256, dtype=np.uint8)
    lut[used] = np.arange(len(used))
    ids = lambda b: torch.from_numpy(lut[np.frombuffer(b, dtype=np.uint8)]).to(DEV)
    _cache[name] = d = dict(tr=ids(tr), va=ids(va), vocab=len(used))
    return d


def _train(w, model, bl, T, s, evals=0, eval_fn=None):
    """evals > 0: also the validation loss at T/evals, 2T/evals, ... (hist["val_loss"], meta_train_v5.py)."""
    kw = dict(eval_fn=lambda: (eval_fn(), 0.0), eval_every=T // evals) if evals else {}
    return train_pg_fast(model, bl, T, w, **kw, **({"lr_scale": 10 ** s} if s else {}))


def mlp_run(w, name, H, T, he=True, evals=0):
    def run(seed, s):
        t = image_task(name, seed)
        prep = lambda x: ((x.float() / 255 - t["mean"]) / t["std"]).flatten(1)
        torch.manual_seed(seed)
        n_in = t["xtr"][0].numel()
        net = nn.Sequential(nn.Linear(n_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 10)).to(DEV)
        if he:
            for mod in net:
                if isinstance(mod, nn.Linear):
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                    nn.init.zeros_(mod.bias)
        g = torch.Generator(device=DEV).manual_seed(seed)
        bl = lambda _: F.cross_entropy(
            net(prep(t["xtr"][i := torch.randint(0, len(t["xtr"]), (128,), device=DEV, generator=g)])), t["ytr"][i])
        h = _train(w, net, bl, T, s, evals, lambda: _eval_images(net, t, prep))
        return _eval_images(net, t, prep), net, h
    return run


def cnn_run(w, name, c, T, aug=False, evals=0):
    def run(seed, s):
        t = image_task(name, seed)
        prep = lambda x: (x.float() / 255 - t["mean"]) / t["std"]
        torch.manual_seed(seed)
        model = CNN(c, in_ch=t["xtr"].shape[1], size=t["xtr"].shape[2]).to(DEV)
        g = torch.Generator(device=DEV).manual_seed(seed)

        def bl(_):
            i = torch.randint(0, len(t["xtr"]), (128,), device=DEV, generator=g)
            x = prep(t["xtr"][i])
            return F.cross_entropy(model(augment(x, g) if aug else x), t["ytr"][i])

        h = _train(w, model, bl, T, s, evals, lambda: _eval_images(model, t, prep))
        return _eval_images(model, t, prep), model, h
    return run


def _eval_images(model, t, prep):
    with torch.no_grad():
        x, y = t["xva"], t["yva"]
        return sum(F.cross_entropy(model(prep(x[k:k + 500])), y[k:k + 500], reduction="sum").item()
                   for k in range(0, len(x), 500)) / len(x)


def lm_run(w, name, d, layers, T, evals=0):
    def run(seed, s):
        t = text_set(name)
        torch.manual_seed(seed)
        model = CharTransformer(t["vocab"], d=d, layers=layers, context=64).to(DEV)
        g = torch.Generator(device=DEV).manual_seed(seed)
        tr, va = t["tr"], t["va"]

        def bl(_):
            i = torch.randint(0, len(tr) - 65, (64,), device=DEV, generator=g)
            seq = tr[i[:, None] + torch.arange(65, device=DEV)].long()
            return F.cross_entropy(model(seq[:, :-1]).flatten(0, 1), seq[:, 1:].flatten())

        @torch.no_grad()
        def val():
            seq = va[:(len(va) - 1) // 64 * 64 + 1].long()
            xb, yb = seq[:-1].view(-1, 64), seq[1:].view(-1, 64)
            return sum(F.cross_entropy(model(xb[k:k + 64]).flatten(0, 1), yb[k:k + 64].flatten(), reduction="sum").item()
                       for k in range(0, len(xb), 64)) / yb.numel()

        h = _train(w, model, bl, T, s, evals, val)
        return val(), model, h
    return run
