"""lr_choice_test.py on more datasets and fresh seeds (1500+, unused anywhere else), test datasets only
(lr_tasks.TEST_SETS: Fashion, KMNIST, CIFAR-10, SVHN, TinyShakespeare, WikiText-2). Fixed before any results.
Same scoring: 5 scales, correct = best within +-0.25 decade of the helper's own choice, regret.

    python lr_choice_test2.py pg2_v2.pt [--final]
--final: the day's final test instead, on two datasets new to the project (Tiny ImageNet CNNs, TinyStories
transformers), seeds 1600+.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

import lr_tasks as L

w = torch.load(sys.argv[1])
SCALES = (-0.5, -0.25, 0.0, 0.25, 0.5)
SEEDS = (1500, 1501)

tasks = [(f"mlp {n} H={H} T={T}", L.mlp_run(w, n, H, T)) for n in ("fashion", "kmnist")
         for H in (64, 256, 1024) for T in (500, 2000, 4000)]
tasks += [(f"mlp svhn H={H} T={T}", L.mlp_run(w, "svhn", H, T)) for H in (256, 1024) for T in (1000, 4000)]
tasks += [(f"cnn cifar10 c={c} aug={int(a)} T=2000", L.cnn_run(w, "cifar10", c, 2000, a)) for c in (16, 32) for a in (False, True)]
tasks += [(f"cnn svhn c={c} T={T}", L.cnn_run(w, "svhn", c, T)) for c in (16, 32) for T in (1000, 3000)]
tasks += [(f"lm shakespeare d={d} L={l} T={T}", L.lm_run(w, "shakespeare", d, l, T))
          for d, l, T in ((64, 2, 1000), (128, 4, 1000), (128, 2, 2000))]
tasks += [(f"lm wikitext2 d={d} L={l} T={T}", L.lm_run(w, "wikitext2", d, l, T))
          for d, l, T in ((64, 2, 1000), (128, 2, 1000), (64, 4, 1000), (128, 4, 1000), (128, 4, 2000))]
FINAL = "--final" in sys.argv  # the day's final test: two new datasets, seeds 1600+
if FINAL:
    SEEDS = (1600, 1601)
    tasks = [(f"cnn tinyimagenet c={c} aug={int(a)} T=2000", L.cnn_run(w, "tinyimagenet", c, 2000, a))
             for c in (16, 32) for a in (False, True)]
    tasks += [(f"lm tinystories d={d} L={l} T={T}", L.lm_run(w, "tinystories", d, l, T))
              for d, l, T in ((64, 2, 1000), (128, 2, 1000), (64, 4, 1000), (128, 4, 2000))]
for name, _ in tasks:  # test datasets only
    assert name.split()[1] in L.TEST_SETS, name

results = []
for name, run in tasks:
    for seed in SEEDS:
        losses = [run(seed, s)[0] for s in SCALES]
        best = int(np.nanargmin(losses))
        r = dict(task=f"{name} seed={seed}", losses=losses, best=SCALES[best], correct=abs(SCALES[best]) <= 0.25,
                 exact=SCALES[best] == 0.0, regret=(losses[2] - losses[best]) / losses[best])
        results.append(r)
        print(f"{r['task']:44s} " + " ".join(f"{x:.4f}" for x in losses) + f"  best x10^{SCALES[best]:+.2f}"
              f"{'' if r['correct'] else '  MISS'}", flush=True)

out = os.path.splitext(os.path.basename(sys.argv[1]))[0]
with open(f"lr_{'final' if FINAL else 'choice2'}_{out}.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"\n=== {sys.argv[1]}: lr-choice accuracy on {len(results)} tasks (fresh seeds, test datasets) ===")
groups = ["all", "mlp", "cnn", "lm"] + list(L.TEST_SETS)
for gname in groups:
    rs = [r for r in results if gname == "all" or r["task"].split()[0] == gname or r["task"].split()[1] == gname]
    if not rs:
        continue
    print(f"{gname:12s} ({len(rs):3d}): correct {100 * np.mean([r['correct'] for r in rs]):5.1f}%  exactly right "
          f"{100 * np.mean([r['exact'] for r in rs]):5.1f}%  mean regret {100 * np.mean([r['regret'] for r in rs]):.2f}%  "
          f"misses: lr too high {sum(r['best'] < -0.25 for r in rs)}, too low {sum(r['best'] > 0.25 for r in rs)}")
grp = defaultdict(list)
for r in results:
    grp[re.sub(r" seed=\d+", "", r["task"])].append(r["losses"])
ok = sum(abs(SCALES[int(np.mean(v, 0).argmin())]) <= 0.25 for v in grp.values())
print(f"seed-averaged per config: {ok}/{len(grp)} correct")
