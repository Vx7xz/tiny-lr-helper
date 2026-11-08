"""Training any PyTorch model (for the final test: a small CNN and a small transformer) with AdamW +
cosine or with a saved cosine helper, through the same fused kernels as common.train.

The helper was meta-trained on 3-layer MLPs and sets one learning rate (and rule strength) per layer.
Here the model's parameters are split into three groups that play those parts: the input layer
(first convolution / embeddings), everything in the middle, and the output layer (classifier /
language-model head)."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from common import N_LAYERS, _Launch, _strides


def flatten_params(groups):
    """Put all parameters in one flat (1, n) buffer, group by group, and their gradients in another:
    each parameter (and its .grad, which backward adds into in place) becomes a view into them.
    Returns flat, grad and where groups 1 and 2 begin."""
    params = [p for g in groups for p in g]
    flat = torch.cat([p.detach().flatten() for p in params])[None]
    grad = torch.zeros_like(flat)
    off = 0
    for p in params:
        k = p.numel()
        p.data = flat[0, off:off + k].view_as(p)
        p.grad = grad[0, off:off + k].view_as(p)
        off += k
    s1 = sum(p.numel() for p in groups[0])
    return flat, grad, s1, s1 + sum(p.numel() for p in groups[1])


def train_model(groups, batch_loss, T, width, lr_fn=None, helper=None, wd=0.0, eval_fn=None, eval_every=0,
                avg=False, avg_warm=0.0, trace=None, gn_peak=False):
    """T steps of Adam on the parameters in `groups` (3 lists). batch_loss(t) -> scalar training loss.
    Either lr_fn(t) -> log10 lr (a fixed schedule, e.g. AdamW + cosine) or helper = load_helper(path).
    width: the model's hidden size, for the helper's width input. eval_fn() -> (val loss, val metric),
    called every eval_every steps. avg: keep a moving average of the weights (horizon T/10, as
    common.train's ema_eval), evaluate it and leave it in the model at the end. avg_warm: common.train's
    ema_warm (horizon t/k early on). trace: a list that gets (step, the helper's state row) every 25
    steps, to see its inputs (diagnostics only). gn_peak (probe): the helper's gradient-norm input is
    taken against the peak so far instead of the first step's norm. On the MLPs it was trained on the
    norm falls from the start, so this changes little there; on a CNN with PyTorch's default init it
    rises ~30x, which the helper reads as trouble.
    Returns {"step", "val_loss", "val_metric", "loglr"}."""
    flat, grad, start1, start2 = flatten_params(groups)
    n = flat.shape[1]
    m, v = torch.zeros_like(flat), torch.zeros_like(flat)
    ema = flat.clone() if avg else None
    hist = dict(step=[], val_loss=[], val_metric=[])
    if helper is not None:  # lr and up are filled in by helper_step every step
        make, agree = helper
        fn = make()
        theta = fn.theta.contiguous()
        ab = agree.contiguous() if agree is not None else theta
        n_out = 2 * N_LAYERS if fn.rule else N_LAYERS
        nb = min(64, (n + 255) // 256)
        prev_g = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
        part = torch.empty(1, nb, 2, device="cuda")
        st = torch.zeros(1, 7 + n_out, device="cuda")
        up, lr = torch.empty(1, N_LAYERS, device="cuda"), torch.empty(1, N_LAYERS, device="cuda")
        hist_lr = torch.empty(T, 1, N_LAYERS, device="cuda")
        stats = _Launch(common._cuda_stats, (nb, 1, 1), (256, 1, 1), [grad, prev_g, part, n, 1])
        helper_k = _Launch(common._cuda_helper, (1, 1, 1), (128, 1, 1), [
            part, nb, flat, theta, theta.stride(0), st, ab, ab.stride(0), ab.shape[1] // 2, up, lr, hist_lr, 0, T,
            float((math.log2(width) - 7) / 2), float(math.log10(T) - 3), n_out, int(agree is not None), 1])
    else:
        hist_lr = torch.empty(T)
    adam = None
    rest = 0.0
    for t in range(T):
        t_step = time.perf_counter()
        grad.zero_()
        loss = batch_loss(t)
        loss.backward()
        with torch.no_grad():
            if helper is not None:
                stats((4, int(t == 0)))
                helper_k((2, loss), (12, t))
                rule = up if agree is not None else lr
                if gn_peak:  # st columns: cos, fast, slow, gn0, gn_ema, ...
                    torch.maximum(st[:, 3], st[:, 4], out=st[:, 3])
                if trace is not None and t % 25 == 0:
                    trace.append((t, st[0].cpu()))
            else:
                loglr = lr_fn(t)
                if torch.is_tensor(loglr):  # (1, 3): one log10 lr per group (group_lr_headroom.py)
                    hist_lr[t] = loglr.mean()
                    lr = (10 ** loglr).float()
                else:
                    hist_lr[t] = loglr
                    lr = torch.full((1,), 10 ** loglr, device="cuda")
                rule = lr
            if adam is None:
                adam = _Launch(common._cuda_adam, ((n + 255) // 256, 1, 1), (256, 1, 1), [
                    flat, grad, m, v, lr, rule, n, start1, start2, *_strides(lr), *_strides(rule), 0.0, 0.0,
                    float(wd), int(helper is not None and agree is not None)])
            adam((4, lr), (5, rule), (13, 1 - 0.9 ** (t + 1)), (14, 1 - 0.999 ** (t + 1)))
            if avg:
                ema.lerp_(flat, max(10 / T, min(1.0, avg_warm / (t + 1))))
        if eval_every and ((t + 1) % eval_every == 0 or t == T - 1):
            if avg:  # evaluate the averaged weights: swap them into the model for the call
                raw = flat.clone()
                with torch.no_grad():
                    flat.copy_(ema)
            vl, vm = eval_fn()
            if avg:
                with torch.no_grad():
                    flat.copy_(raw)
                del raw
            hist["step"].append(t + 1)
            hist["val_loss"].append(vl)
            hist["val_metric"].append(vm)
        if common.MAX_BUSY < 1:  # same rest rule as common.train
            torch.cuda.synchronize()
            rest += (time.perf_counter() - t_step) * (1 / common.MAX_BUSY - 1)
            if rest > 0.02:
                t_rest = time.perf_counter()
                time.sleep(rest)
                common.RESTED[0] += time.perf_counter() - t_rest
                rest = 0.0
    if avg:
        with torch.no_grad():
            flat.copy_(ema)  # the model the method hands over is the averaged one
    hist["loglr"] = hist_lr
    return hist


def augment(x, g):
    """The standard CIFAR augmentation on a batch (n, C, H, W), on the GPU: a random crop of the image
    padded by 4 pixels (zeros, i.e. the mean after normalizing), and a random horizontal flip."""
    n, c, h, w = x.shape
    xp = F.pad(x, (4, 4, 4, 4))
    oy, ox = torch.randint(0, 9, (2, n), device=x.device, generator=g)
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    out = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
             rows, cols]
    flip = torch.rand(n, device=x.device, generator=g) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


# ---------------------------------------------------------------- the two models

class CNN(nn.Module):
    """A small CIFAR-10 network (~0.29M parameters): three conv stages and a linear classifier."""

    def __init__(self, c=32, in_ch=3, size=32):  # in_ch=1, size=28 for MNIST-like images (meta-training)
        super().__init__()
        self.first = nn.Conv2d(in_ch, c, 3, padding=1)
        self.middle = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, 2 * c, 3, padding=1),
                                     nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.Conv2d(2 * c, 4 * c, 3, padding=1)])
        self.last = nn.Linear(4 * c * (size // 8) ** 2, 10)
        self.width = 2 * c

    def forward(self, x):
        x = F.relu(self.first(x))
        for i, conv in enumerate(self.middle):
            x = F.relu(conv(x))
            if i in (0, 2, 3):
                x = F.max_pool2d(x, 2)
        return self.last(x.flatten(1))

    def groups(self):
        return [list(self.first.parameters()), list(self.middle.parameters()), list(self.last.parameters())]


class CharTransformer(nn.Module):
    """A small character-level language model (~0.8M parameters): pre-norm transformer decoder blocks."""

    def __init__(self, vocab, d=128, layers=4, heads=4, context=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(context, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                                                norm_first=True) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(context), persistent=False)
        self.width = d

    def forward(self, x):
        h = self.tok(x) + self.pos.weight[:x.shape[1]]
        for b in self.blocks:
            h = b(h, src_mask=self.mask[:x.shape[1], :x.shape[1]], is_causal=True)
        return self.head(self.norm(h))

    def groups(self):
        return [list(self.tok.parameters()) + list(self.pos.parameters()),
                list(self.blocks.parameters()) + list(self.norm.parameters()), list(self.head.parameters())]
