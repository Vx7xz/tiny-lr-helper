"""pg2: the pg helper (pg.py, pg_fast.py) with less GPU memory than AdamW and three more inputs.

Memory. pg keeps, per weight: the weight, its gradient, Adam's m and v, the weight average and the
previous gradient (24 bytes; AdamW: 16). pg2 keeps 14:
  - no previous gradient: the cosine input compares the gradient with Adam's momentum m (before this
    step's update) instead of with the previous gradient;
  - m, v and the weight average are stored in bfloat16 with stochastic rounding (the average as its
    difference from the current weight, which is small, so 16 bits keep it well); the arithmetic is
    float32;
  - the per-tensor lr history is kept every 25 steps only.

Inputs. Per tensor, besides pg's: is it a convolution kernel (3+ dims), an embedding table, a
normalization layer's weight or bias. The CNN on photos and the transformer pulled pg's 11 numbers
in opposite directions (README, CIFAR-100); these let the helper tell those tensors apart.

  lr of tensor g = 1e-3 * cosine(t) * 10**c_g,  c_g = 1.5 * tanh(w . features_g + b), smoothed

w and b are N_PG2 + 1 = 14 numbers. A pg helper (11 numbers) is expanded with zeros for the new inputs.
"""
import math
import time

import torch
import torch.nn as nn

import common
from common import _Launch

N_PG2 = 13
N_STATIC = 6  # fan-in term, is-vector, position, is-conv, is-embedding, is-norm
HIST_EVERY = 25
CHUNK = 4096

_SRC = r"""
__device__ __forceinline__ unsigned int mix(unsigned int a, unsigned int b) {
    unsigned int h = a * 0x9E3779B1u ^ (b * 0x85EBCA77u + 0x7F4A7C15u);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
__device__ __forceinline__ float bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ __forceinline__ unsigned short to_bf(float x, unsigned int r) {  // stochastic rounding
    return (unsigned short)((__float_as_uint(x) + (r & 0xFFFFu)) >> 16);
}

extern "C" __global__ void pg2_stats(const float* g, const unsigned short* m, const int* bstart, const int* bend,
                                     float* part) {
    __shared__ float s_sq[256], s_dot[256], s_m[256];
    int b = blockIdx.x, s = bstart[b], e = bend[b];
    float sq = 0.f, dot = 0.f, mm = 0.f;
    for (int i = s + threadIdx.x; i < e; i += 256) {
        float gi = g[i], mi = bf(m[i]);
        sq += gi * gi;
        dot += gi * mi;
        mm += mi * mi;
    }
    s_sq[threadIdx.x] = sq; s_dot[threadIdx.x] = dot; s_m[threadIdx.x] = mm;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            s_sq[threadIdx.x] += s_sq[threadIdx.x + k];
            s_dot[threadIdx.x] += s_dot[threadIdx.x + k];
            s_m[threadIdx.x] += s_m[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { part[3 * b] = s_sq[0]; part[3 * b + 1] = s_dot[0]; part[3 * b + 2] = s_m[0]; }
}

extern "C" __global__ void pg2_helper(const float* part, const int* gfirst, const int* glast, const float* loss,
                                      const float* w, const float* stat, float* st, float* sh, float* lr, float* hist,
                                      int t, int T, double length, int G, int every) {
    // st per tensor: cos, gn_ema, peak, last, c; sh: fast, slow loss averages
    __shared__ float fast, slow;
    int k = threadIdx.x;
    if (k == 0) {
        float L = loss[0];
        if (t == 0) { sh[0] = L; sh[1] = L; }
        else { sh[0] += 0.1f * (L - sh[0]); sh[1] += 0.01f * (L - sh[1]); }
        fast = sh[0]; slow = sh[1];
    }
    __syncthreads();
    if (k >= G) return;
    float sq = 0.f, dot = 0.f, mm = 0.f;
    for (int b = gfirst[k]; b < glast[k]; b++) { sq += part[3 * b]; dot += part[3 * b + 1]; mm += part[3 * b + 2]; }
    float gn = sqrtf(sq);
    float* s = st + 5 * k;
    if (t == 0) s[1] = gn;
    else {
        s[0] += 0.1f * (dot / (gn * sqrtf(mm) + 1e-12f) - s[0]);
        s[1] += 0.1f * (gn - s[1]);
    }
    s[2] = fmaxf(s[2], s[1]);
    const float* q = stat + 6 * k;
    float f[13] = {(float)t / (float)T, (float)length, log10f(fast), fminf(fmaxf(10.f * logf(fast / slow), -3.f), 3.f),
                   s[0], fminf(fmaxf(logf(s[1] / (s[2] + 1e-12f)), -5.f), 0.f), s[3],
                   q[0], q[1], q[2], q[3], q[4], q[5]};
    float a = w[13];
    for (int j = 0; j < 13; j++) a += w[j] * f[j];
    float raw = 1.5f * tanhf(a);
    s[4] = t == 0 ? raw : s[4] + 0.1f * (raw - s[4]);
    float l = -3.f + s[4] + log10f(0.5f * (1.f + cosf(3.14159265358979f * (float)t / (float)T)) + 1e-8f);
    if (t % every == 0) hist[(long long)(t / every) * G + k] = l;
    s[3] = l + 3.f;
    lr[k] = powf(10.f, l);
}

extern "C" __global__ void pg2_adam(float* p, const float* g, unsigned short* m, unsigned short* v, unsigned short* d,
                                    const float* lr, const int* bstart, const int* bend, const int* bgroup,
                                    double bc1, double bc2, double ema_w, int avg, int seed) {
    // Adam in float32 on bfloat16 m, v; d = weight average - weight, also bfloat16
    int b = blockIdx.x, s = bstart[b], e = bend[b];
    float l = lr[bgroup[b]] / (float)bc1, c2 = (float)bc2, ew = (float)ema_w;
    unsigned int sd = (unsigned int)seed * 3u;
    for (int i = s + threadIdx.x; i < e; i += 256) {
        float gi = g[i];
        float mi = bf(m[i]) * 0.9f + 0.1f * gi;
        float vi = bf(v[i]) * 0.999f + 0.001f * gi * gi;
        m[i] = to_bf(mi, mix((unsigned int)i, sd));
        v[i] = to_bf(vi, mix((unsigned int)i, sd + 1u));
        float po = p[i];
        float pn = po - (mi * l) / (sqrtf(vi / c2) + 1e-8f);
        p[i] = pn;
        if (avg) {
            float a = po + bf(d[i]);
            a += ew * (pn - a);
            d[i] = to_bf(a - pn, mix((unsigned int)i, sd + 2u));
        }
    }
}
"""
_k = {name: torch.cuda._compile_kernel(_SRC, name) for name in ("pg2_stats", "pg2_helper", "pg2_adam")}

_NORMS = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d)


def group_info2(model):
    """The model's trainable tensors and their fixed inputs (G, 6): pg's three, is-conv, is-embedding, is-norm."""
    owner = {id(p): mod for mod in model.modules() for p in mod.parameters(recurse=False)}
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for k, p in enumerate(params):
        vec, emb = p.dim() < 2, isinstance(owner[id(p)], nn.Embedding)
        fan = 1 if vec or emb else p[0].numel()
        rows.append(((math.log2(fan) - 7) / 2, float(vec), 2 * k / max(1, len(params) - 1) - 1,
                     float(p.dim() >= 3), float(emb), float(isinstance(owner[id(p)], _NORMS))))
    return params, torch.tensor(rows)


def expand_pg(w):
    """A pg helper (11 numbers) as a pg2 helper: zeros for the three new inputs."""
    w = w.float().flatten()
    return w if w.numel() == N_PG2 + 1 else torch.cat([w[:10], torch.zeros(3), w[10:]])


def train_pg2(model, batch_loss, T, w, avg=True, avg_warm=8.0, eval_fn=None, eval_every=0, lr_scale=1.0):
    """As pg_fast.train_pg_fast, with pg2's memory and inputs. Returns {"step", "val_loss", "val_metric",
    "loglr" ((T + 24) // 25, G), every 25th step}; leaves the averaged weights in the model.
    lr_scale != 1: every lr the helper picks is multiplied by it (to test the helper's choice,
    lr_choice_test.py); its own 'last lr' input stays the unscaled one."""
    dev = next(model.parameters()).device
    params, info = group_info2(model)
    G, sizes = len(params), [p.numel() for p in params]
    flat = torch.cat([p.detach().flatten() for p in params])
    grad = torch.zeros_like(flat)
    off, bs, be, bg, gfirst, glast = 0, [], [], [], [], []
    for gi, (p, k) in enumerate(zip(params, sizes)):
        p.data = flat[off:off + k].view_as(p)
        p.grad = grad[off:off + k].view_as(p)
        gfirst.append(len(bs))
        for s in range(off, off + k, CHUNK):
            bs.append(s)
            be.append(min(s + CHUNK, off + k))
            bg.append(gi)
        glast.append(len(bs))
        off += k
    ints = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    bstart, bend, bgroup, gf, gl = ints(bs), ints(be), ints(bg), ints(gfirst), ints(glast)
    nb = len(bs)
    m, v = (torch.zeros(flat.numel(), dtype=torch.bfloat16, device=dev) for _ in range(2))
    d = torch.zeros(flat.numel() if avg else 1, dtype=torch.bfloat16, device=dev)
    part = torch.zeros(nb, 3, device=dev)
    st = torch.zeros(G, 5, device=dev)
    sh = torch.zeros(2, device=dev)
    lr = torch.empty(G, device=dev)
    hist_lr = torch.empty((T + HIST_EVERY - 1) // HIST_EVERY, G, device=dev)
    wt = expand_pg(w).to(dev).contiguous()
    stat = info.to(dev).float().contiguous()
    stats = _Launch(_k["pg2_stats"], (nb, 1, 1), (256, 1, 1), [grad, m, bstart, bend, part])
    helper = _Launch(_k["pg2_helper"], (1, 1, 1), (max(32, (G + 31) // 32 * 32), 1, 1),
                     [part, gf, gl, flat, wt, stat, st, sh, lr, hist_lr, 0, T, float(math.log10(T) - 3), G,
                      HIST_EVERY])
    adam = _Launch(_k["pg2_adam"], (nb, 1, 1), (256, 1, 1),
                   [flat, grad, m, v, d, lr, bstart, bend, bgroup, 1.0, 1.0, 0.0, int(avg), 0])
    hist = dict(step=[], val_loss=[], val_metric=[])
    rest = 0.0
    for t in range(T):
        t_step = time.perf_counter()
        grad.zero_()
        loss = batch_loss(t)
        loss.backward()
        with torch.no_grad():
            L = loss.detach().float()
            stats()
            helper((3, L), (10, t))
            if torch.is_tensor(lr_scale) or lr_scale != 1.0:  # a tensor: one factor per tensor
                lr.mul_(lr_scale)
            adam((9, 1 - 0.9 ** (t + 1)), (10, 1 - 0.999 ** (t + 1)),
                 (11, max(10 / T, min(1.0, avg_warm / (t + 1)))), (13, t))
        if eval_every and ((t + 1) % eval_every == 0 or t == T - 1):
            if avg:  # the raw weights wait in RAM, not on the GPU, while the average is evaluated
                raw_w = flat.cpu()
                with torch.no_grad():
                    flat.add_(d)
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
            flat.add_(d)
    hist["loglr"] = hist_lr
    return hist
