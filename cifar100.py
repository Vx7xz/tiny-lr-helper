"""CIFAR-100 (binary version, data/cifar-100-binary) as a source of natural-image tasks for
meta-training, so CIFAR-10 can stay unseen. Each task: 10 of CIFAR-100's 20 superclasses picked by
a seed, labels 0-9 by superclass; train images from CIFAR-100's training split (2500 per
superclass, 25k per task), validation images from its test split (500 per superclass). The images
are other photos than CIFAR-10's, in other classes. Kept on the GPU as bytes, with per-channel
mean/std of the task's training images for normalizing.
"""
import os

import numpy as np
import torch

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cifar-100-binary")


def load_cifar100(device="cuda"):
    """{"xtr", "ctr", "xte", "cte"}: uint8 images (n, 3, 32, 32) and superclass labels (0-19)."""
    out = {}
    for split, name in (("tr", "train.bin"), ("te", "test.bin")):
        raw = np.fromfile(os.path.join(DIR, name), dtype=np.uint8).reshape(-1, 3074)
        out["x" + split] = torch.from_numpy(raw[:, 2:].reshape(-1, 3, 32, 32).copy()).to(device)
        out["c" + split] = torch.from_numpy(raw[:, 0].astype(np.int64)).to(device)
    return out


def cifar100_task(d, seed):
    """{"xtr", "ytr", "xva", "yva", "mean", "std", "classes"} for 10 superclasses picked by seed."""
    g = torch.Generator().manual_seed(seed)
    classes = torch.randperm(20, generator=g)[:10].tolist()
    lut = torch.full((20,), -1, dtype=torch.long, device=d["ctr"].device)
    lut[classes] = torch.arange(10, device=lut.device)
    task = dict(classes=classes)
    for split, src in (("tr", "tr"), ("va", "te")):
        y = lut[d["c" + src]]
        keep = y >= 0
        task["x" + split], task["y" + split] = d["x" + src][keep], y[keep]
    f = task["xtr"][:5000].float() / 255
    task["mean"], task["std"] = f.mean((0, 2, 3))[:, None, None], f.std((0, 2, 3))[:, None, None]
    return task
