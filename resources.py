"""What a piece of work costs the machine, measured while it runs:
GPU energy (the card's own energy counter, in joules), average GPU busy %, peak GPU memory this
process allocated, CPU time this process used, and its peak RAM.

    with Meter() as m:
        ...work...
    m.energy_j, m.gpu_util, m.gpu_mb, m.cpu_s, m.ram_mb

Energy and GPU busy % are for the whole card, so only one job should use it at a time (the tests
run one method after another). With GD_MAX_BUSY < 1 the rests are inside the measured window too;
every method rests the same third of its busy time, so comparisons between methods stay fair.
"""
import threading
import time

import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # no NVIDIA driver (e.g. a CPU machine): GPU numbers stay nan
    _gpu = None
_proc = psutil.Process()


class Meter:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._samples, self._ram, self._stop = [], _proc.memory_info().rss, threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(_gpu) if _gpu else None  # millijoules
        self._cpu0 = sum(_proc.cpu_times()[:2])
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _sample(self):  # every 50 ms: how busy the GPU is, and this process's RAM
        while not self._stop.wait(0.05):
            if _gpu:
                self._samples.append(pynvml.nvmlDeviceGetUtilizationRates(_gpu).gpu)
            self._ram = max(self._ram, _proc.memory_info().rss)

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        self.cpu_s = sum(_proc.cpu_times()[:2]) - self._cpu0
        self.energy_j = (pynvml.nvmlDeviceGetTotalEnergyConsumption(_gpu) - self._e0) / 1000 if _gpu else float("nan")
        self.gpu_util = sum(self._samples) / len(self._samples) if self._samples else float("nan")
        self.gpu_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float("nan")
        self.ram_mb = self._ram / 2 ** 20
        return False
