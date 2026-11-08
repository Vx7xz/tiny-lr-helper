"""Shared pieces: MNIST on the GPU, a batch of P small MLPs trained in parallel,
hand-written Adam/SGD, and the tiny neural network that predicts the learning rate.

Every tensor has a leading "population" dimension P, so P independent networks
(different learning rates, or different controller weights) train in one pass.
"""
import ctypes
import math
import os
import time

import struct

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # no GPU compiler (e.g. a CPU machine): train uses plain PyTorch ops
    triton = None

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# On a shared machine, GD_MAX_BUSY=0.75 keeps training busy at most 75% of the time: after
# each step it rests for a third of the time the step took.
MAX_BUSY = float(os.environ.get("GD_MAX_BUSY", "1"))
RESTED = [0.0]  # total seconds slept for MAX_BUSY, so timings can leave the rest out


DATASETS = {  # all 28x28 grayscale, 10 classes: name -> (torchvision class / folder, pixel mean, pixel std)
    "mnist": ("MNIST", 0.1307, 0.3081),         # handwritten digits
    "fashion": ("FashionMNIST", 0.2860, 0.3530),  # clothing photos, harder
    "kmnist": ("KMNIST", 0.1918, 0.3483),       # handwritten Japanese characters
}


class Images:
    """A dataset's images kept on the GPU as bytes (4x smaller than floats) and turned into the
    normalized floats when indexed: lut[byte] is exactly the (byte / 255 - mean) / std a float
    copy would hold, so training sees the same numbers. Supports len, shape and indexing."""

    def __init__(self, u8, mean, std):
        self.u8, self.shape = u8, u8.shape
        # on the CPU, as before: the GPU rounds this chain differently in the last bit for 175 of 256 values
        self.lut = torch.arange(256, dtype=torch.uint8).float().div(255).sub(mean).div(std).to(u8.device)

    def __len__(self):
        return len(self.u8)

    def __getitem__(self, i):
        return self.lut[self.u8[i].int()]


def _idx(path):
    """An idx file (MNIST's format) read straight into a uint8 tensor: magic number, sizes, bytes."""
    t = torch.from_file(path, size=os.path.getsize(path), dtype=torch.uint8)
    ndim = int(t[3])
    shape = struct.unpack(f">{ndim}I", bytes(t[4:4 + 4 * ndim].tolist()))
    return t[4 + 4 * ndim:].view(shape)


def load_mnist(root="data", name="mnist"):
    folder, mean, std = DATASETS[name]
    raw = os.path.join(root, folder, "raw")
    if not os.path.exists(os.path.join(raw, "t10k-labels-idx1-ubyte")):  # first use: torchvision downloads it
        from torchvision import datasets
        for train in (True, False):
            getattr(datasets, folder)(root, train=train, download=True)

    def prep(split):  # read without torchvision (~90 MB less RAM); pixels stay bytes, on the GPU
        x = _idx(os.path.join(raw, f"{split}-images-idx3-ubyte")).to(DEV)
        return x.view(len(x), -1), _idx(os.path.join(raw, f"{split}-labels-idx1-ubyte")).to(DEV).long()

    xtr, ytr = prep("train")
    xte, yte = prep("t10k")
    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0)).to(DEV)
    tr, va = perm[5000:], perm[:5000]  # 55k train, 5k validation (used for tuning)
    return dict(xtr=Images(xtr[tr], mean, std), ytr=ytr[tr], xva=Images(xtr[va], mean, std), yva=ytr[va],
                xte=Images(xte, mean, std), yte=yte)


_cache = {}


def cheap_task(name, seed):
    """Small, cheap tasks for meta-training, none of them MNIST or Fashion-MNIST:
      kmnist14: KMNIST shrunk to 14x14 (196 inputs, 4x less compute in the first layer)
      teacher:  64 random inputs, labels from a random tanh network (new network per seed)
      blobs:    32 inputs, 10 overlapping Gaussian clusters (new clusters per seed)
    """
    if name == "kmnist14":
        if name not in _cache:
            d = load_mnist(name="kmnist")
            small = {k: F.avg_pool2d(v[:].view(-1, 1, 28, 28), 2).flatten(1) if k[0] == "x" else v for k, v in d.items()}
            mu, sd = small["xtr"].mean(), small["xtr"].std()
            _cache[name] = {k: (v - mu) / sd if k[0] == "x" else v for k, v in small.items()}
        return _cache[name]
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = 25000
    if name == "teacher":
        x = torch.randn(n, 64, device=DEV, generator=g)
        W1 = torch.randn(64, 64, device=DEV, generator=g) / 8
        W2 = torch.randn(64, 10, device=DEV, generator=g)
        y = (torch.tanh(x @ W1) @ W2 + 0.3 * torch.randn(n, 10, device=DEV, generator=g)).argmax(1)
    else:  # blobs
        y = torch.randint(0, 10, (n,), device=DEV, generator=g)
        x = 0.3 * torch.randn(10, 32, device=DEV, generator=g)[y] + torch.randn(n, 32, device=DEV, generator=g)
    return dict(xtr=x[5000:], ytr=y[5000:], xva=x[:5000], yva=y[:5000])


# ---------------------------------------------------------------- the network being trained

def init_mlp(P, sizes, seed):
    """He init. All P copies start from the same weights, so they differ only by optimizer."""
    g = torch.Generator().manual_seed(seed)
    params = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        W = torch.randn(1, i, o, generator=g) * math.sqrt(2 / i)
        params += [W.expand(P, -1, -1).clone().to(DEV), torch.zeros(P, 1, o, device=DEV)]
    return params


def forward(params, x):
    h = x  # (B, in) @ (P, in, out) broadcasts to (P, B, out)
    for k in range(0, len(params), 2):
        h = h @ params[k] + params[k + 1]
        if k < len(params) - 2:
            h = F.relu(h)
    return h


def loss_per_net(params, x, y):
    logits = forward(params, x)
    P = logits.shape[0]
    return F.cross_entropy(logits.flatten(0, 1), y.repeat(P), reduction="none").view(P, -1).mean(1)


@torch.no_grad()
def evaluate(params, x, y, chunk=5000):
    loss, correct = 0.0, 0.0
    for s in range(0, len(x), chunk):
        logits = forward(params, x[s:s + chunk])
        P = logits.shape[0]
        yy = y[s:s + chunk]
        loss = loss + F.cross_entropy(logits.flatten(0, 1), yy.repeat(P), reduction="none").view(P, -1).sum(1)
        correct = correct + (logits.argmax(-1) == yy).float().sum(1)
    return loss / len(x), correct / len(x)


# ---------------------------------------------------------------- training loop

N_FEAT = 8


MUP_BASE = 128  # with mup=True, a 128-wide network trains exactly as without it


if triton is not None:
    @triton.jit
    def _adam_kernel(p_ptr, g_ptr, m_ptr, v_ptr, lr_ptr, up_ptr, n, start1, start2,
                     lr_row, lr_layer, up_row, up_layer, bc1, bc2, wd, RULE: tl.constexpr, BLOCK: tl.constexpr):
        """Adam, the agreement rule and the weight step in one pass over memory. As ~12 separate
        PyTorch ops the same update reads and writes the weights' worth of memory ~35 times,
        which is what a wide network's step waits on."""
        row = tl.program_id(1)  # which of the P networks
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = col < n
        i = row * n + col
        g = tl.load(g_ptr + i, mask=mask, other=0.0)
        m = tl.load(m_ptr + i, mask=mask, other=0.0) * 0.9 + 0.1 * g
        v = tl.load(v_ptr + i, mask=mask, other=0.0) * 0.999 + 0.001 * g * g
        tl.store(m_ptr + i, m, mask=mask)
        tl.store(v_ptr + i, v, mask=mask)
        u = (m / bc1) / (tl.sqrt(v / bc2) + 1e-8)
        layer = (col >= start1).to(tl.int32) + (col >= start2).to(tl.int32)
        if RULE:  # up**s with s = sign(g * m): steady weights get up, oscillating ones 1 / up
            up = tl.load(up_ptr + row * up_row + layer * up_layer, mask=mask, other=1.0)
            gm = g * m
            u = tl.where(gm > 0, u * up, tl.where(gm < 0, u / up, u))
        lr = tl.load(lr_ptr + row * lr_row + layer * lr_layer, mask=mask, other=0.0)
        p = tl.load(p_ptr + i, mask=mask, other=0.0)
        tl.store(p_ptr + i, p - lr * (u + wd * p), mask=mask)  # wd: AdamW's decoupled weight decay


_ADAM_CUDA = r"""
extern "C" __global__ void adam_step(float* p, const float* g, float* m, float* v, const float* lr, const float* up,
                                     int n, int start1, int start2, int lr_row, int lr_layer, int up_row, int up_layer,
                                     double bc1, double bc2, double wd, int rule) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    int row = blockIdx.y;
    long long i = (long long)row * n + col;
    float gi = g[i];
    float mi = m[i] * 0.9f + 0.1f * gi;
    float vi = v[i] * 0.999f + 0.001f * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float u = (mi / (float)bc1) / (sqrtf(vi / (float)bc2) + 1e-8f);
    int layer = (col >= start1) + (col >= start2);
    if (rule) {
        float f = up[row * up_row + layer * up_layer];
        float gm = gi * mi;
        u = gm > 0 ? u * f : (gm < 0 ? u / f : u);
    }
    float pi = p[i];
    p[i] = pi - lr[row * lr_row + layer * lr_layer] * (u + (float)wd * pi);
}
"""

# The cosine helper's whole per-step bookkeeping (features, controller, learning rates) in two kernels.
# As ~40 PyTorch ops it needed a copy of the gradients, a CUDA graph, an 8 MB cuBLAS workspace and
# ~90 MB of host RAM for the code of all those ops; here it reads the gradients once.
_HELPER_CUDA = r"""
extern "C" __global__ void grad_stats(const float* g, unsigned short* prev, float* part, int n, int first) {
    // per block: sum of g*g and of g * previous g over a strided slice of one row; prev = g as bfloat16
    __shared__ float s_sq[256], s_dot[256];
    int row = blockIdx.y;
    const float* gr = g + (long long)row * n;
    unsigned short* pr = prev + (long long)row * n;
    float sq = 0.f, dot = 0.f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float gi = gr[i];
        if (!first) dot += gi * __uint_as_float((unsigned int)pr[i] << 16);
        sq += gi * gi;
        unsigned int u = __float_as_uint(gi);
        pr[i] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // round to nearest even
    }
    s_sq[threadIdx.x] = sq;
    s_dot[threadIdx.x] = dot;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {  // fixed order: the same numbers every run
        if (threadIdx.x < s) {
            s_sq[threadIdx.x] += s_sq[threadIdx.x + s];
            s_dot[threadIdx.x] += s_dot[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* o = part + 2 * ((long long)row * gridDim.x + blockIdx.x);
        o[0] = s_sq[0];
        o[1] = s_dot[0];
    }
}

extern "C" __global__ void helper_step(const float* part, int nb, const float* loss, const float* theta, int th_row,
                                       float* st, const float* agree, int ag_row, int ag_layers, float* up, float* lr,
                                       float* hist, int t, int T, double width, double length, int n_out,
                                       int has_agree, int P) {
    // one thread per network: the same math as train's PyTorch helper and ctrl_cos_lr_fn
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float sq = 0.f, dot = 0.f;
    for (int b = 0; b < nb; b++) {
        sq += part[2 * (p * nb + b)];
        dot += part[2 * (p * nb + b) + 1];
    }
    float gn = sqrtf(sq), L = loss[p];
    float* s = st + p * (7 + n_out);  // cos, fast, slow, gn0, gn_ema, prev_gn, last, then the smoothed outputs
    if (t == 0) {
        s[1] = L; s[2] = L; s[3] = gn; s[4] = gn;
    } else {
        s[0] += 0.1f * (dot / (gn * s[5] + 1e-12f) - s[0]);
        s[1] += 0.1f * (L - s[1]);
        s[2] += 0.01f * (L - s[2]);
        s[4] += 0.1f * (gn - s[4]);
    }
    s[5] = gn;
    float f[8] = {(float)t / (float)T, log10f(s[1]), fminf(fmaxf(10.f * logf(s[1] / s[2]), -3.f), 3.f), s[0],
                  fminf(fmaxf(logf(s[4] / s[3]), -5.f), 5.f), s[6], (float)width, (float)length};
    const float* W1 = theta + p * th_row;  // (8, 16), then b1 (16), W2 (16, n_out), b2 (n_out)
    const float* b1 = W1 + 8 * 16;
    const float* W2 = b1 + 16;
    const float* b2 = W2 + 16 * n_out;
    float h[16];
    for (int j = 0; j < 16; j++) {
        float a = b1[j];
        for (int k = 0; k < 8; k++) a += f[k] * W1[k * 16 + j];
        h[j] = tanhf(a);
    }
    float* c = s + 7;
    for (int o = 0; o < n_out; o++) {
        float a = b2[o];
        for (int j = 0; j < 16; j++) a += h[j] * W2[j * n_out + o];
        float raw = tanhf(a);
        c[o] = t == 0 ? raw : c[o] + 0.1f * (raw - c[o]);
    }
    float cosine = log10f(0.5f * (1.f + cosf(3.14159265358979f * f[0])) + 1e-8f);
    float mean = 0.f;
    for (int i = 0; i < 3; i++) {
        float l = -3.f + cosine + c[i];
        mean += l;
        hist[((long long)t * P + p) * 3 + i] = l;
        float scale = 1.f;
        if (has_agree) {  // the per-weight rule: up = 10**(a + d/2) (d only from a cosdyn helper), scale = 10**b
            const float* ab = agree + p * ag_row + (ag_layers == 1 ? 0 : 2 * i);
            float a = 0.5f * tanhf(ab[0]);
            up[p * 3 + i] = powf(10.f, n_out > 3 ? a + 0.5f * c[3 + i] : a);
            scale = powf(10.f, 0.5f * tanhf(ab[1]));
        }
        lr[p * 3 + i] = powf(10.f, l) * scale;
    }
    s[6] = mean / 3.f + 3.f;
}
"""


class _Launch:
    """An NVRTC kernel launch whose argument array is built once. torch's launcher converts every
    argument on every call (35-45 us for these kernels, a few % of a step); a step here only
    rewrites the arguments that changed (~7 us)."""

    def __init__(self, kernel, grid, block, args):
        self.func, self.grid, self.block = kernel.func, grid, block
        self.lib = torch.cuda._utils._get_gpu_runtime_library()
        self.vals = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else
                     ctypes.c_int(a) if isinstance(a, int) else ctypes.c_double(a) for a in args]
        self.args = (ctypes.c_void_p * len(args))(*map(ctypes.addressof, self.vals))
        self.stream = torch.cuda.current_stream()._as_parameter_

    def __call__(self, *changes):  # (argument index, new value) pairs
        for i, x in changes:
            self.vals[i].value = x.data_ptr() if isinstance(x, torch.Tensor) else x
        err = self.lib.cuLaunchKernel(self.func, *self.grid, *self.block, 0, self.stream, self.args, None)
        if err:
            raise RuntimeError(f"cuLaunchKernel failed with CUDA error {err}")


_cuda_adam = _cuda_stats = _cuda_helper = None
if DEV == "cuda" and hasattr(torch.cuda, "_compile_kernel"):
    import torch.utils.cpp_extension as _cpp
    if _cpp.CUDA_HOME is None:  # torch asks for a CUDA install to add its headers; these kernels need none,
        _cpp.CUDA_HOME = os.path.dirname(torch.__file__)  # and NVRTC itself ships with torch
    if triton is None:  # without Triton (e.g. on Windows) the same one-pass Adam kernel, compiled by NVRTC
        _cuda_adam = torch.cuda._compile_kernel(_ADAM_CUDA, "adam_step")
    _cuda_stats = torch.cuda._compile_kernel(_HELPER_CUDA, "grad_stats")
    _cuda_helper = torch.cuda._compile_kernel(_HELPER_CUDA, "helper_step")


_side = []


def _capture_stream():
    """The one side stream every run captures its CUDA graph on. cuBLAS keeps an 8 MB workspace per
    stream for good, so a fresh stream per run left one behind each time, and it pinned the cache
    block around it: GPU memory grew run after run (24 MB a run here, gigabytes in meta-training)."""
    if not _side:
        _side.append(torch.cuda.Stream())
    return _side[0]


def _strides(x):
    """Row and layer strides of a (P,) or (P, N_LAYERS) tensor, 0 where it broadcasts."""
    return x.stride(0) if x.shape[0] > 1 else 0, x.stride(1) if x.dim() == 2 and x.shape[1] > 1 else 0


def train(params, lr_fn, T, data, opt="adam", batch=128, seed=0, eval_every=0, upd_fn=None, mup=False,
          ema_eval=False, agree=None, features=True, wd=0.0, eval_at=(), ema_out=False, ema_warm=0.0):
    """Train P networks for T steps. lr_fn(features (P, N_FEAT), t) -> log10 learning rate (P,).
    With opt="learned", upd_fn (see upd_net_fn) decides every weight's update, and the
    learning rate only sets the overall scale.

    mup=True applies the muP rule for Adam: weight matrices fed by a hidden layer of
    width n get learning rate * MUP_BASE / n, so the best learning rate stops moving
    when the network gets wider.

    ema_eval=True scores the eval_every checkpoints on a moving average of the weights
    (horizon T/10). The average already shows what annealing would give, so cutting the
    learning rate just before a checkpoint no longer looks like progress. ema_warm=k > 0 shortens
    the horizon early on to t/k (step weight max(10/T, k/(t+1))): a T/10 average still holds the
    first, poor weights when a target is reached in the first tenth of a run.

    eval_at: extra checkpoint steps (counted from 1), e.g. drawn at random so a helper cannot
    know when it will be scored.

    agree (P, 2) adds a cheap learned per-weight rule on top of Adam: each weight's step
    is scaled by 10**(a * s + b), where s = +1 if its gradient agrees in sign with its
    momentum (steady progress) and -1 if not (oscillating). a, b = 0.5 * tanh(agree).
    agree (P, 2 * N_LAYERS) gives every layer its own (a, b). With an lr_fn made by
    ctrl_cos_lr_fn(..., rule=True), the helper also shifts each layer's a every step.

    features=False skips the features (lr_fn gets None): a fixed schedule does not need
    them, so plain Adam is timed without the helper's bookkeeping.

    wd > 0 turns Adam into AdamW: every step also shrinks each weight by lr * wd (decoupled
    weight decay, applied with the step's learning rate but not the rule's sign factor).

    Features describe how training is going:
      t/T, log10 loss, loss trend, cosine between successive gradients,
      gradient norm relative to the start, last log10 lr, network width, and run length
      (longer runs need a smaller peak learning rate).
    """
    P = params[0].shape[0]
    width = (math.log2(params[0].shape[-1]) - 7) / 2  # 0 at 128 units, +1 at 512
    length = math.log10(T) - 3  # 0 at 1000 steps
    g = torch.Generator(device=DEV).manual_seed(seed)
    # All weights live in one flat (P, n) tensor, so every per-weight op below runs once instead of
    # once per weight matrix and bias: the step is launch-bound, and that cuts its launches by ~60.
    shapes = [p.shape[1:] for p in params]
    sizes = [s.numel() for s in shapes]

    def unflat(x):
        return [c.view(P, *s) for c, s in zip(x.split(sizes, 1), shapes)]

    def per_weight(lr):  # (P,) or one per layer (P, N_LAYERS) -> broadcasts against (P, n)
        return lr[:, None] if lr.dim() == 1 else lr[:, layer]

    flat = torch.cat([p.detach().flatten(1) for p in params], 1).requires_grad_()
    n_in = params[0].shape[1]
    del params  # flat holds the weights now: without this the caller's copy stays alive for the whole run
    mult = torch.cat([torch.full((n,), MUP_BASE / s[0] if s[0] > 1 and s[0] != n_in else 1.0, device=DEV)
                      for n, s in zip(sizes, shapes)]) if mup else None
    m = torch.zeros_like(flat)
    v = torch.zeros_like(flat)
    ms = torch.zeros_like(flat) if opt == "learned" else None  # slow momentum
    hist = dict(loglr=[], step=[], val_loss=[], val_acc=[])
    ema = flat.detach().clone() if ema_eval else None
    ema_decay = 1 - 10 / T
    net_rule = agree is not None and agree.shape[1] > 2 * N_LAYERS  # (a, b) + the tiny per-weight network
    fused = (DEV == "cuda" and (triton is not None or _cuda_adam is not None) and opt == "adam" and mult is None
             and not net_rule)
    # each weight's layer (int64: twice the weights' memory); the fused kernels work it out from start1 and start2
    layer = None if fused else torch.cat([torch.full((n,), k // 2, device=DEV) for k, n in enumerate(sizes)])
    # a cosine helper (ctrl_cos_lr_fn) runs as the two kernels of _HELPER_CUDA, which also apply the rule's numbers
    lean = features and fused and _cuda_helper is not None and hasattr(lr_fn, "theta")
    if agree is not None and not lean:  # the rule's numbers are fixed during a run: turn them into factors once
        n_ab = 2 * N_LAYERS if agree.shape[1] == 2 * N_LAYERS else 2  # an (a, b) per layer, or one for all
        a, b = (0.5 * torch.tanh(agree[:, :n_ab])).view(P, -1, 2).unbind(2)  # (P, 1) or (P, N_LAYERS)
        up, scale = (10 ** a).contiguous(), 10 ** b  # 10**(a*s + b) = up**s * scale
    rule_out = getattr(lr_fn, "rule", False)  # the helper's network also sets the rule's strength...
    if rule_out and agree is not None or lean:  # ...per layer and every step: up = 10**(a + d/2), filled in below
        up = torch.empty(P, N_LAYERS, device=DEV)
    if fused:
        n, start1, start2 = flat.shape[1], sum(sizes[:2]), sum(sizes[:4])  # where layers 1 and 2 begin
        grid = (triton.cdiv(n, 1024), P) if triton is not None else ((n + 255) // 256, P, 1)

    if lean:  # no gradient copy and no CUDA graph; the previous gradient is kept in bfloat16 (half the memory)
        theta = lr_fn.theta.contiguous()
        ab = agree.contiguous() if agree is not None else theta  # any pointer when there is no rule
        n_out = 2 * N_LAYERS if rule_out else N_LAYERS
        nb = min(64, (n + 255) // 256)  # blocks per network in grad_stats
        prev_g = torch.empty(P, n, dtype=torch.bfloat16, device=DEV)
        part = torch.empty(P, nb, 2, device=DEV)
        st = torch.zeros(P, 7 + n_out, device=DEV)
        lr = torch.empty(P, N_LAYERS, device=DEV)
        hist_lr = torch.empty(T, P, N_LAYERS, device=DEV)
        stats = _Launch(_cuda_stats, (nb, P, 1), (256, 1, 1), [flat, prev_g, part, n, 1])  # flat: a placeholder
        helper_k = _Launch(_cuda_helper, ((P + 127) // 128, 1, 1), (128, 1, 1), [
            part, nb, flat, theta, theta.stride(0), st, ab, ab.stride(0), ab.shape[1] // 2, up, lr, hist_lr, 0, T,
            float(width), float(length), n_out, int(agree is not None), P])
    elif features:
        # The helper's bookkeeping is ~40 tiny GPU ops per step, and here each op costs its launch,
        # not its math. So it works in place on fixed buffers, and from step 2 on the whole of it
        # (features, controller, learning rate) replays as one CUDA graph: one launch per step.
        buf = torch.empty(P, 1 + flat.shape[1], device=DEV)  # this step's loss and gradients
        lossb, gflat = buf[:, 0], buf[:, 1:]
        prev_g = torch.empty_like(gflat)
        cos, fast, slow, gn0, gn_ema, prev_gn, last = torch.zeros(7, P, device=DEV)  # last = last log10 lr + 3
        consts = torch.tensor([width, length], device=DEV)[:, None].expand(2, P)
        step = torch.zeros(1, dtype=torch.long, device=DEV)
        hist_lr = graph = None
        use_graph = DEV == "cuda" and getattr(lr_fn, "graphable", False)

        def helper(first, t):
            nonlocal hist_lr
            gn = gflat.norm(dim=1)
            if first:
                for x, v in ((fast, lossb), (slow, lossb), (gn0, gn), (gn_ema, gn)):
                    x.copy_(v)
            else:
                dot = torch.bmm(gflat[:, None], prev_g[:, :, None])[:, 0, 0]  # no weights-sized temporary
                cos.lerp_(dot / (gn * prev_gn + 1e-12), 0.1)
                fast.lerp_(lossb, 0.1)
                slow.lerp_(lossb, 0.01)
                gn_ema.lerp_(gn, 0.1)
            prev_g.copy_(gflat)
            prev_gn.copy_(gn)
            feats = torch.stack([(step / T).expand(P), fast.log10(), (10 * (fast / slow).log()).clamp(-3, 3), cos,
                                 (gn_ema / gn0).log().clamp(-5, 5), last, *consts], 1)
            loglr = lr_fn(feats, t)
            if rule_out:
                if agree is not None:
                    up.copy_(10 ** (a + 0.5 * loglr[:, N_LAYERS:]))
                loglr = loglr[:, :N_LAYERS]
            last.copy_(loglr if loglr.dim() == 1 else loglr.mean(1)).add_(3)
            if hist_lr is None:
                hist_lr = loglr.new_empty((T,) + loglr.shape)
            hist_lr.index_copy_(0, step, loglr[None])
            step.add_(1)
            lr = 10 ** loglr  # (P,) or one per layer (P, N_LAYERS)
            return lr * scale if agree is not None and not net_rule else lr  # the rule's constant part

    rest = 0.0  # seconds of rest owed under MAX_BUSY
    adam = None  # the NVRTC Adam launch, built at the first step
    for t in range(T):
        t_step = time.perf_counter()
        idx = torch.randint(0, len(data["xtr"]), (batch,), device=DEV, generator=g)
        loss = loss_per_net(unflat(flat), data["xtr"][idx], data["ytr"][idx])
        grad, = torch.autograd.grad(loss.sum(), flat)  # backprop, all P nets at once

        with torch.no_grad():
            if not features:
                loglr = lr_fn(None, t)
                lr = 10 ** loglr
                hist["loglr"].append(loglr)
                if agree is not None and not net_rule:
                    lr = lr * scale  # the rule's constant part, applied once instead of per weight
            elif lean:
                stats((0, grad), (4, int(t == 0)))
                helper_k((2, loss), (12, t))
            else:
                torch.cat([loss.detach()[:, None], grad], 1, out=buf)
                if graph is not None:
                    graph.replay()
                    lr = lr_graph
                elif use_graph and t == 1:  # run step 1 on a side stream (the warm-up a capture needs), then capture it
                    side = _capture_stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        lr = helper(False, t)
                        graph = torch.cuda.CUDAGraph()
                        graph.capture_begin()  # records, does not run: the state is not advanced twice
                        lr_graph = helper(False, t)
                        graph.capture_end()
                    torch.cuda.current_stream().wait_stream(side)
                else:
                    lr = helper(t == 0, t)

            if fused:
                rule = up if agree is not None else lr  # without the rule this pointer is never read
                if triton is not None:
                    _adam_kernel[grid](flat, grad, m, v, lr, rule, n, start1, start2, *_strides(lr), *_strides(rule),
                                       1 - 0.9 ** (t + 1), 1 - 0.999 ** (t + 1), wd, RULE=agree is not None, BLOCK=1024)
                else:
                    if adam is None:  # lr's shape (so its strides) is known from the first step on
                        adam = _Launch(_cuda_adam, grid, (256, 1, 1), [
                            flat, grad, m, v, lr, rule, n, start1, start2, *_strides(lr), *_strides(rule),
                            0.0, 0.0, float(wd), int(agree is not None)])
                    adam((1, grad), (4, lr), (5, rule), (13, 1 - 0.9 ** (t + 1)), (14, 1 - 0.999 ** (t + 1)))
                u = None
            elif opt in ("adam", "learned"):
                m.mul_(0.9).add_(grad, alpha=0.1)
                v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                denom = (v / (1 - 0.999 ** (t + 1))).sqrt() + 1e-8
                u = (m / (1 - 0.9 ** (t + 1))) / denom
                if agree is not None:
                    s = torch.sign(grad * m)
                    if net_rule:  # the tiny per-weight network adds its own correction
                        u = u * 10 ** (a * s + b + agree_net(agree[:, 2:], s, u, t / T))
                    else:
                        u = u * (up if up.shape[1] == 1 else up[:, layer]).pow(s)
                if opt == "learned":
                    ms.mul_(0.99).add_(grad, alpha=0.01)
                    u = upd_fn(u, (grad / denom).clamp(-10, 10),
                               (ms / (1 - 0.99 ** (t + 1)) / denom).clamp(-10, 10), t / T, width, length)
            else:  # plain gradient descent
                u = grad
            if u is not None:
                step_lr = per_weight(lr) if mult is None else per_weight(lr) * mult
                if wd:
                    flat.mul_(1 - step_lr * wd)
                flat.addcmul_(step_lr, u, value=-1)
            if ema_eval:
                ema.lerp_(flat, max(1 - ema_decay, min(1.0, ema_warm / (t + 1))))

        if eval_every and ((t + 1) % eval_every == 0 or t == T - 1) or t + 1 in eval_at:
            vl, va = evaluate(unflat(ema if ema_eval else flat), data["xva"], data["yva"])
            hist["step"].append(t + 1)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)

        if MAX_BUSY < 1:
            if DEV == "cuda":
                torch.cuda.synchronize()  # count the GPU's time too, not just the launches
            rest += (time.perf_counter() - t_step) * (1 / MAX_BUSY - 1)
            if rest > 0.02:  # rest in chunks: short sleeps are not precise
                t_rest = time.perf_counter()
                time.sleep(rest)
                RESTED[0] += time.perf_counter() - t_rest
                rest = 0.0

    hist["loglr"] = hist_lr if features else torch.stack(hist["loglr"])  # (T, P) or (T, P, N_LAYERS)
    return unflat((ema if ema_out else flat).detach()), hist  # ema_out (with ema_eval): the averaged weights


def cosine_grid_search(data, H, T, wd=0.0):
    """The traditional tuning, as evaluate.py did it for tuned_lr.json: Adam (AdamW with wd > 0)
    + cosine at 13 learning rates on seed 0, all at once; returns the log10 lr with the lowest
    final validation loss."""
    grid = torch.arange(-4.5, -1.49, 0.25, device=DEV)
    sched = lambda f, t: grid + math.log10(0.5 * (1 + math.cos(math.pi * t / T)) + 1e-8)
    params, _ = train(init_mlp(len(grid), [data["xtr"].shape[1], H, H, 10], seed=0), sched, T, data, seed=0,
                      features=False, wd=wd)
    return grid[torch.nan_to_num(evaluate(params, data["xva"], data["yva"])[0], nan=1e9).argmin()].item()


# ---------------------------------------------------------------- the learning-rate network

HID = 16
N_LAYERS = 3  # every network here is 784-H-H-10


def ctrl_size(n_out=1):
    return N_FEAT * HID + HID + HID * n_out + n_out  # 161 weights for one output


def ctrl_init(seed=0, n_out=1):
    g = torch.Generator().manual_seed(seed)
    th = torch.zeros(ctrl_size(n_out))
    th[:N_FEAT * HID] = torch.randn(N_FEAT * HID, generator=g) / math.sqrt(N_FEAT)
    th[N_FEAT * HID + HID:-n_out] = torch.randn(HID * n_out, generator=g) * 0.1 / math.sqrt(HID)
    return th.to(DEV)  # outputs start near 0


def _ctrl_net(theta, n_out):
    """theta (P, ctrl_size(n_out)): one controller per network. Returns feats -> (P, n_out) in [-1, 1]."""
    P = theta.shape[0]
    W1 = theta[:, :N_FEAT * HID].view(P, N_FEAT, HID)
    b1 = theta[:, N_FEAT * HID:N_FEAT * HID + HID]
    W2 = theta[:, N_FEAT * HID + HID:-n_out].view(P, HID, n_out)
    b2 = theta[:, -n_out:]

    def net(feats):
        h = torch.tanh(torch.baddbmm(b1[:, None], feats[:, None], W1))  # (P, 1, HID), one kernel per layer
        return torch.tanh(torch.baddbmm(b2[:, None], h, W2))[:, 0]

    return net


def ctrl_lr_fn(theta):
    """Predict the learning rate directly: log10 lr in [-5, -1]."""
    net = _ctrl_net(theta, 1)
    return lambda feats, t: -3 + 2 * net(feats)[:, 0]


def ctrl_cos_lr_fn(theta, rule=False):
    """Correct a cosine schedule, separately for each layer:
    lr of layer i = 1e-3 * cosine(t) * 10**c_i with c_i in [-1, 1], smoothed so it cannot jitter.
    rule=True: the same network also outputs, per layer, a change d_i in [-1, 1] to the per-weight
    rule's strength (train uses a + d_i / 2), so the rule can follow how training is going.
    Returns (P, N_LAYERS) log10 lr, or (P, 2 * N_LAYERS) with the d_i appended."""
    net = _ctrl_net(theta, 2 * N_LAYERS if rule else N_LAYERS)
    c = None

    def fn(feats, t):
        nonlocal c
        raw = net(feats)
        if c is None:
            c = raw.clone()
        else:
            c.lerp_(raw, 0.1)  # in place: a CUDA graph replay must find it at the same address
        loglr = -3 + torch.log10(0.5 * (1 + torch.cos(math.pi * feats[:, :1])) + 1e-8) + c[:, :N_LAYERS]
        return torch.cat([loglr, c[:, N_LAYERS:]], 1) if rule else loglr

    fn.graphable = True  # uses only the features (not t) and keeps its state in place: train may replay it
    fn.rule = rule
    fn.theta = theta  # train can run this whole function as one kernel (helper_step)
    return fn


# ---------------------------------------------------------------- the weight-update network
# A tiny MLP run separately on EVERY weight (shared weights, like a 1x1 convolution).
# Per weight it sees: Adam's direction, the normalized raw gradient, a slow momentum,
# plus t/T, network width and run length. It outputs a direction and a size for that one weight:
#   update = 1e-3 * (adam_dir + d) * 10**q
# d and q start at 0, so before training it is exactly Adam with lr 1e-3.

U_IN, U_HID = 6, 8
N_UPD = U_IN * U_HID + U_HID + U_HID * 2 + 2  # 74 weights


def upd_init(seed=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.zeros(N_UPD)
    th[:U_IN * U_HID] = torch.randn(U_IN * U_HID, generator=g) / math.sqrt(U_IN)
    return th.to(DEV)  # output layer zero: starts as plain Adam


def upd_net_fn(theta):
    """theta (P, N_UPD): one update network per trained network."""
    P = theta.shape[0]
    W1 = theta[:, :U_IN * U_HID].view(P, U_IN, U_HID)
    b1 = theta[:, U_IN * U_HID:U_IN * U_HID + U_HID]
    W2 = theta[:, U_IN * U_HID + U_HID:-2].view(P, U_HID, 2)
    b2 = theta[:, -2:]

    def fn(adam_dir, g, mslow, frac, width, length):
        shape = adam_dir.shape
        x = torch.stack([adam_dir, g, mslow], -1).view(P, -1, 3)
        x = torch.cat([x, x.new_tensor([frac, width, length]).expand(P, x.shape[1], 3)], -1)
        h = torch.tanh(torch.einsum("pnf,pfh->pnh", x, W1) + b1[:, None])
        out = torch.einsum("pnh,pho->pno", h, W2) + b2[:, None]
        d, q = out[..., 0], out[..., 1].clamp(-3, 1)
        return ((x[..., 0] + d) * 10 ** q).view(shape)

    return fn


# ---------------------------------------------------------------- the per-weight agreement network
# A 21-weight MLP (3 -> 4 -> 1) run on every weight, on top of the two-number agreement rule.
# Per weight it sees: does the gradient agree with the momentum (+1/-1), log10 of Adam's
# step size, and t/T. It outputs a correction in [-0.5, 0.5] to that weight's log10 step.
# The output layer starts at zero, so before training it is exactly the two-number rule.

A_IN, A_HID = 3, 4
N_AGREE_NET = A_IN * A_HID + A_HID + A_HID + 1  # 21 weights


def agree_net_init(seed=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.zeros(N_AGREE_NET)
    th[:A_IN * A_HID] = torch.randn(A_IN * A_HID, generator=g) / math.sqrt(A_IN)
    return th.to(DEV)


def agree_net(theta, s, u, frac):
    """theta (P, N_AGREE_NET); s, u (P, in, out) -> log10 correction per weight."""
    P = theta.shape[0]
    W1 = theta[:, :A_IN * A_HID].view(P, A_IN, A_HID)
    b1 = theta[:, A_IN * A_HID:A_IN * A_HID + A_HID]
    W2 = theta[:, A_IN * A_HID + A_HID:-1]
    b2 = theta[:, -1:]
    x = torch.stack([s, u.abs().add(1e-8).log10().clamp(-3, 1), torch.full_like(s, frac)], -1).view(P, -1, A_IN)
    h = torch.tanh(torch.einsum("pnf,pfh->pnh", x, W1) + b1[:, None])
    return (0.5 * torch.tanh(torch.einsum("pnh,ph->pn", h, W2) + b2)).view(s.shape)


def load_helper(path):
    """A saved cosine helper: (maker of a fresh lr_fn, per-weight rule (1, 2) or None)."""
    th = torch.load(path).to(DEV)[None]
    n2 = ctrl_size(2 * N_LAYERS)
    rule = th.shape[1] in (n2 + 2, n2 + 2 * N_LAYERS)  # its network also sets the rule's strength (cosdyn)
    n = n2 if rule else ctrl_size(N_LAYERS)
    return (lambda: ctrl_cos_lr_fn(th[:, :n], rule)), (th[:, n:] if th.shape[1] > n else None)
