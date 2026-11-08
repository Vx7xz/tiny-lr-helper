"""A per-group learning-rate helper ("pg"), for any PyTorch model.

The MLP helpers (v13 and on) see 8 global inputs and give one learning rate per *layer role* (input /
middle / output of a 3-layer MLP). On a CNN and a transformer they set the middle group's lr far off
(README, "The final test"). group_lr_headroom.py showed what these nets need: the middle group near
the tuned lr, the input and output groups higher. So pg looks at every parameter tensor on its own
and gives each its own lr, from that tensor's own facts:

  per tensor: (log2 fan-in - 7) / 2 (embeddings and vectors count as fan-in 1), is it a vector
              (bias / norm), position in the network (-1 first ... +1 last), cosine between its
              successive gradients (smoothed), ln(its gradient norm / the peak so far), its last
              log10 lr + 3
  shared:     t/T, log10(T) - 3, log10 loss (smoothed), loss trend 10*ln(fast/slow) in [-3, 3]

  lr of tensor g = 1e-3 * cosine(t) * 10**c_g,  c_g = 1.5 * tanh(w . features_g + b), smoothed

w and b are the helper: 11 numbers, the same for every tensor of every model. With w = b = 0 it is
plain Adam + cosine at 1e-3. Training: Adam (no weight decay) with a moving average of the weights
handed over at the end (horizon t/k early, T/10 later), as the tests' "averaged" methods.
"""
import math
import time

import torch
import torch.nn as nn

import common

N_PG = 10  # inputs per tensor; the helper is N_PG weights + 1 bias


def group_info(model):
    """The model's trainable tensors and their fixed inputs (G, 3): fan-in term, is-vector, position."""
    owner = {id(p): mod for mod in model.modules() for p in mod.parameters(recurse=False)}
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for k, p in enumerate(params):
        vec = p.dim() < 2
        fan = 1 if vec or isinstance(owner[id(p)], nn.Embedding) else p[0].numel()
        rows.append(((math.log2(fan) - 7) / 2, float(vec), 2 * k / max(1, len(params) - 1) - 1))
    return params, torch.tensor(rows)


def train_pg(model, batch_loss, T, w, avg=True, avg_warm=8.0, eval_fn=None, eval_every=0, trace=None):
    """T steps on model with the pg helper w (N_PG + 1 numbers). batch_loss(t) -> scalar loss.
    Returns {"step", "val_loss", "val_metric", "loglr" (T, G)}; leaves the averaged weights in the model."""
    dev = next(model.parameters()).device
    params, info = group_info(model)
    G, sizes = len(params), [p.numel() for p in params]
    flat = torch.cat([p.detach().flatten() for p in params])
    grad = torch.zeros_like(flat)
    off = 0
    for p, k in zip(params, sizes):  # every tensor (and its .grad) becomes a view into the flat buffers
        p.data = flat[off:off + k].view_as(p)
        p.grad = grad[off:off + k].view_as(p)
        off += k
    gid = torch.repeat_interleave(torch.arange(G, device=dev), torch.tensor(sizes, device=dev))
    m, v, prev = torch.zeros_like(flat), torch.zeros_like(flat), torch.zeros_like(flat)
    ema = flat.clone() if avg else None
    static = info.to(dev)
    w = w.to(dev).float()
    W, b = w[:N_PG], w[N_PG]
    length = math.log10(T) - 3
    fast = slow = None
    cos_g, gn_ema, peak, last, c = (torch.zeros(G, device=dev) for _ in range(5))
    prev_gn = torch.ones(G, device=dev)
    hist = dict(step=[], val_loss=[], val_metric=[])
    hist_lr = torch.empty(T, G, device=dev)
    rest = 0.0
    for t in range(T):
        t_step = time.perf_counter()
        grad.zero_()
        loss = batch_loss(t)
        loss.backward()
        with torch.no_grad():
            L = loss.detach().float()
            if t == 0:
                fast, slow = L.clone(), L.clone()
            else:
                fast.lerp_(L, 0.1)
                slow.lerp_(L, 0.01)
            gn = torch.zeros(G, device=dev).index_add_(0, gid, grad * grad).sqrt_()
            if t == 0:
                gn_ema.copy_(gn)
            else:
                dot = torch.zeros(G, device=dev).index_add_(0, gid, grad * prev)
                cos_g.lerp_(dot / (gn * prev_gn + 1e-12), 0.1)
                gn_ema.lerp_(gn, 0.1)
            torch.maximum(peak, gn_ema, out=peak)
            prev.copy_(grad)
            prev_gn.copy_(gn)
            shared = torch.stack([torch.tensor(t / T, device=dev), torch.tensor(length, device=dev), fast.log10(),
                                  (10 * (fast / slow).log()).clamp(-3, 3)])
            feats = torch.cat([shared.expand(G, 4), cos_g[:, None], (gn_ema / (peak + 1e-12)).log().clamp(-5, 0)[:, None],
                               last[:, None], static], 1)
            raw = 1.5 * torch.tanh(feats @ W + b)
            if t == 0:
                c.copy_(raw)
            else:
                c.lerp_(raw, 0.1)
            loglr = -3 + c + math.log10(0.5 * (1 + math.cos(math.pi * t / T)) + 1e-8)
            hist_lr[t] = loglr
            last.copy_(loglr + 3)
            m.mul_(0.9).add_(grad, alpha=0.1)
            v.mul_(0.999).addcmul_(grad, grad, value=0.001)
            step = (10 ** loglr)[gid] / (1 - 0.9 ** (t + 1))
            flat.addcdiv_(m * step, (v / (1 - 0.999 ** (t + 1))).sqrt_().add_(1e-8), value=-1)
            if avg:
                ema.lerp_(flat, max(10 / T, min(1.0, avg_warm / (t + 1))))
        if trace is not None and t % 25 == 0:
            trace.append((t, feats.cpu(), loglr.cpu()))
        if eval_every and ((t + 1) % eval_every == 0 or t == T - 1):
            if avg:
                raw_w = flat.clone()
                with torch.no_grad():
                    flat.copy_(ema)
            vl, vm = eval_fn()
            if avg:
                with torch.no_grad():
                    flat.copy_(raw_w)
                del raw_w
            hist["step"].append(t + 1)
            hist["val_loss"].append(vl)
            hist["val_metric"].append(vm)
        if common.MAX_BUSY < 1:
            torch.cuda.synchronize()
            rest += (time.perf_counter() - t_step) * (1 / common.MAX_BUSY - 1)
            if rest > 0.02:
                t_rest = time.perf_counter()
                time.sleep(rest)
                common.RESTED[0] += time.perf_counter() - t_rest
                rest = 0.0
    if avg:
        with torch.no_grad():
            flat.copy_(ema)
    hist["loglr"] = hist_lr
    return hist
