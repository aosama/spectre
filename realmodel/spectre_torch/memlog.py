"""RAM/GPU memory diagnostics for training runs.

Usage:
    from spectre_torch.memlog import MemLogger
    mem = MemLogger("train")
    mem.log("after model load")

Prints process RSS (from ps), MPS allocated memory, and swap usage, so
memory blowups are visible per phase instead of discovered via system swap.
"""
import os
import subprocess

import torch


def _rss_gb() -> float:
    pid = os.getpid()
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        )
        return int(out.stdout.strip()) / 1048576
    except Exception:
        return -1.0


def _swap_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        # "total = 4096.00M  used = 2556.19M  free = 1539.81M  (encrypted)"
        for part in out.stdout.split():
            if part.startswith("used"):
                continue
        used = out.stdout.split("used = ")[1].split("M")[0]
        return float(used) / 1024
    except Exception:
        return -1.0


class MemLogger:
    def __init__(self, label: str, log_every_s: float = 30.0):
        self.label = label
        self.log_every_s = log_every_s
        self._last = 0.0

    def log(self, phase: str, force: bool = False) -> None:
        import time

        now = time.time()
        if not force and now - self._last < self.log_every_s:
            return
        self._last = now
        mps_alloc = mps_reserved = -1.0
        if torch.backends.mps.is_available():
            mps_alloc = torch.mps.current_allocated_memory() / 2**30
            mps_reserved = torch.mps.driver_allocated_memory() / 2**30
        print(
            f"[mem:{self.label}] {phase}: rss {_rss_gb():.2f}GB, "
            f"mps alloc {mps_alloc:.2f}GB / driver {mps_reserved:.2f}GB, "
            f"swap used {_swap_gb():.2f}GB"
        )