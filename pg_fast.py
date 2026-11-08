"""pg.train_pg as three NVRTC kernels per step instead of ~40 PyTorch ops: the same math (float32;
the cosine and logs in single precision, so results match train_pg closely, not bit for bit).

  pg_stats:  per block of one tensor, sum of g*g and of g*previous g; keeps g as the previous one
  pg_helper: one thread per tensor: its gradient norm and cosine, the features, the 11-number helper,
             its lr (one block; the shared loss averages in thread 0)
  pg_adam:   Adam with each block's tensor lr, plus the weight average

Blocks never straddle two tensors, so each knows its tensor without a per-weight index.
"""
import math
import time

import torch

import common
from common import _Launch
from pg import N_PG, group_info

_SRC = r"""
extern "C" __global__ void pg_stats(const float* g, float* prev, const int* bstart, const int* bend, float* part,
                                    int first) {
    __shared__ float s_sq[256], s_dot[256];
    int b = blockIdx.x, s = bstart[b], e = bend[b];
    float sq = 0.f, dot = 0.f;
    for (int i = s + threadIdx.x; i < e; i += 256) {
        float gi = g[i];
        if (!first) dot += gi * prev[i];
        sq += gi * gi;
        prev[i] = gi;
    }
    s_sq[threadIdx.x] = sq;
    s_dot[threadIdx.x] = dot;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {  // fixed order: the same numbers every run
        if (threadIdx.x < k) {
            s_sq[threadIdx.x] += s_sq[threadIdx.x + k];
            s_dot[threadIdx.x] += s_dot[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part[2 * b] = s_sq[0];
        part[2 * b + 1] = s_dot[0];
    }
}

extern "C" __global__ void pg_helper(const float* part, const int* gfirst, const int* glast, const float* loss,
                                     const float* w, const float* stat, float* st, float* sh, float* lr, float* hist,
                                     int t, int T, double length, int G) {
    // st per tensor: cos, gn_ema, peak, last, c, prev_gn; sh: fast, slow loss averages
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
    float sq = 0.f, dot = 0.f;
    for (int b = gfirst[k]; b < glast[k]; b++) { sq += part[2 * b]; dot += part[2 * b + 1]; }
    float gn = sqrtf(sq);
    float* s = st + 6 * k;
    if (t == 0) s[1] = gn;
    else {
        s[0] += 0.1f * (dot / (gn * s[5] + 1e-12f) - s[0]);
        s[1] += 0.1f * (gn - s[1]);
    }
    s[2] = fmaxf(s[2], s[1]);
    s[5] = gn;
    float f[10] = {(float)t / (float)T, (float)length, log10f(fast), fminf(fmaxf(10.f * logf(fast / slow), -3.f), 3.f),
                   s[0], fminf(fmaxf(logf(s[1] / (s[2] + 1e-12f)), -5.f), 0.f), s[3],
                   stat[3 * k], stat[3 * k + 1], stat[3 * k + 2]};
    float a = w[10];
    for (int j = 0; j < 10; j++) a += w[j] * f[j];
    float raw = 1.5f * tanhf(a);
    s[4] = t == 0 ? raw : s[4] + 0.1f * (raw - s[4]);
    float l = -3.f + s[4] + log10f(0.5f * (1.f + cosf(3.14159265358979f * (float)t / (float)T)) + 1e-8f);
    hist[(long long)t * G + k] = l;
    s[3] = l + 3.f;
    lr[k] = powf(10.f, l);
}

extern "C" __global__ void pg_adam(float* p, const float* g, float* m, float* v, float* ema, const float* lr,
                                   const int* bstart, const int* bend, const int* bgroup, double bc1, double bc2,
                                   double ema_w, int avg) {
    int b = blockIdx.x, s = bstart[b], e = bend[b];
    float l = lr[bgroup[b]] / (float)bc1, c2 = (float)bc2, ew = (float)ema_w;
    for (int i = s + threadIdx.x; i < e; i += 256) {
        float gi = g[i];
        float mi = m[i] * 0.9f + 0.1f * gi;
        float vi = v[i] * 0.999f + 0.001f * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i] - (mi * l) / (sqrtf(vi / c2) + 1e-8f);
        p[i] = pi;
        if (avg) ema[i] += ew * (pi - ema[i]);
    }
}

extern "C" __global__ void pg_adamw(float* p, const float* g, float* m, float* v, float* ema, const float* lr,
                                    const int* bstart, const int* bend, const int* bgroup, double bc1, double bc2,
                                    double ema_w, int avg, double wd) {
    // pg_adam plus AdamW's decoupled weight decay (each weight also shrinks by lr * wd); a separate kernel
    // so that pg_adam, which the earlier results used, stays exactly as it was
    int b = blockIdx.x, s = bstart[b], e = bend[b];
    float lrb = lr[bgroup[b]], l = lrb / (float)bc1, c2 = (float)bc2, ew = (float)ema_w, d = lrb * (float)wd;
    for (int i = s + threadIdx.x; i < e; i += 256) {
        float gi = g[i];
        float mi = m[i] * 0.9f + 0.1f * gi;
        float vi = v[i] * 0.999f + 0.001f * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i];
        pi = pi - (mi * l) / (sqrtf(vi / c2) + 1e-8f) - d * pi;
        p[i] = pi;
        if (avg) ema[i] += ew * (pi - ema[i]);
    }
}
"""
_k = {name: torch.cuda._compile_kernel(_SRC, name) for name in ("pg_stats", "pg_helper", "pg_adam", "pg_adamw")}
CHUNK = 4096  # weights per block


def train_pg_fast(model, batch_loss, T, w, avg=True, avg_warm=8.0, eval_fn=None, eval_every=0, wd=0.0,
                  lr_scale=1.0):
    """Same as pg.train_pg (minus trace). wd > 0: AdamW's decoupled weight decay (the pg_adamw kernel).
    lr_scale != 1: every lr the helper picks is multiplied by it before the update (the helper's own
    'last lr' input stays the unscaled one). A pg2 helper (14 numbers) runs pg2_fast.train_pg2 instead;
    a calibrated pg2 helper (15 numbers, fit_calib.py) is pg2 plus a global lr offset of 10**w[14]; one with
    22 numbers (fit_calib2.py) is pg2 with every lr times 10**(z . w[14:]), z the parameter-weighted mean of
    the tensors' fixed inputs (one factor for the whole model, as the calibration measured)."""
    if w.numel() == 22:
        from pg2_fast import group_info2
        params, q = group_info2(model)
        x = torch.cat([q.float(), torch.full((len(q), 1), math.log10(T) - 3), torch.ones(len(q), 1)], 1)
        size = torch.tensor([float(p.numel()) for p in params])
        scale = 10 ** ((size @ x / size.sum()) @ w[14:].float().cpu()).item() * lr_scale
        return train_pg_fast(model, batch_loss, T, w[:14], avg=avg, avg_warm=avg_warm, eval_fn=eval_fn,
                             eval_every=eval_every, wd=wd, lr_scale=scale)
    if w.numel() == 15:
        assert not torch.is_tensor(lr_scale)
        return train_pg_fast(model, batch_loss, T, w[:14], avg=avg, avg_warm=avg_warm, eval_fn=eval_fn,
                             eval_every=eval_every, wd=wd, lr_scale=lr_scale * 10 ** w[14].item())
    if w.numel() == 14:
        assert wd == 0.0
        from pg2_fast import train_pg2
        return train_pg2(model, batch_loss, T, w, avg=avg, avg_warm=avg_warm, eval_fn=eval_fn, eval_every=eval_every,
                         lr_scale=lr_scale)
    dev = next(model.parameters()).device
    params, info = group_info(model)
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
    m, v, prev = torch.zeros_like(flat), torch.zeros_like(flat), torch.zeros_like(flat)
    ema = flat.clone() if avg else flat  # flat: any pointer when not averaging
    part = torch.zeros(nb, 2, device=dev)
    st = torch.zeros(G, 6, device=dev)
    st[:, 5] = 1.0
    sh = torch.zeros(2, device=dev)
    lr = torch.empty(G, device=dev)
    hist_lr = torch.empty(T, G, device=dev)
    wt = w.to(dev).float().contiguous()
    stat = info.to(dev).float().contiguous()
    stats = _Launch(_k["pg_stats"], (nb, 1, 1), (256, 1, 1), [grad, prev, bstart, bend, part, 1])
    helper = _Launch(_k["pg_helper"], (1, 1, 1), (max(32, (G + 31) // 32 * 32), 1, 1),
                     [part, gf, gl, flat, wt, stat, st, sh, lr, hist_lr, 0, T, float(math.log10(T) - 3), G])
    adam = _Launch(_k["pg_adamw" if wd else "pg_adam"], (nb, 1, 1), (256, 1, 1),
                   [flat, grad, m, v, ema, lr, bstart, bend, bgroup, 1.0, 1.0, 0.0, int(avg)] + ([float(wd)] if wd else []))
    hist = dict(step=[], val_loss=[], val_metric=[])
    rest = 0.0
    for t in range(T):
        t_step = time.perf_counter()
        grad.zero_()
        loss = batch_loss(t)
        loss.backward()
        with torch.no_grad():
            L = loss.detach().float()
            stats((5, int(t == 0)))
            helper((3, L), (10, t))
            if torch.is_tensor(lr_scale) or lr_scale != 1.0:  # a tensor: one factor per tensor (group_info order)
                lr.mul_(lr_scale)
            adam((9, 1 - 0.9 ** (t + 1)), (10, 1 - 0.999 ** (t + 1)),
                 (11, max(10 / T, min(1.0, avg_warm / (t + 1)))))
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
