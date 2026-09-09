"""Timings at the reads' real shapes, on whatever device this process
has: what one call costs, and what a panel at cube scale costs. Run
locally with `python -m glosskernels.measure`; Modal's entrypoint
calls `run` on the GPU (modal_app.py)."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from .kernels import Kernels

ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]


def run(panel_rows: int = 5000, use_amp: bool = False, misfit_workers: int | None = None) -> dict:
    t0 = time.perf_counter()
    k = Kernels(use_amp=use_amp)
    out = {
        "device": k.device,
        "gpu": torch.cuda.get_device_name(0) if k.device == "cuda" else None,
        "amp": use_amp,
        "misfit_workers": misfit_workers or __import__("glosskernels.kernels", fromlist=["misfit_workers"]).misfit_workers(k.device),
        "torch": torch.__version__,
        "load_s": round(time.perf_counter() - t0, 3),
    }
    rng = np.random.default_rng(0)
    cuda = k.device == "cuda"

    def timed(name, fn, n=1, warm=True):
        if warm:
            fn()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        if cuda:
            torch.cuda.synchronize()
        out[name] = {"s_per_call": round((time.perf_counter() - t) / n, 4)}
        if cuda:
            out[name]["peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # A walk point: two years of months, the five walk features.
    tx, ty, qx = rng.normal(size=(24, 5)), rng.normal(size=24), rng.normal(size=5)
    timed("band_point_24x5", lambda: k.band_point(tx, ty, qx, ALPHAS, 0.1), n=6)
    # A replay grid: nine worlds by twelve months, four levers and the month index.
    gx, gy, gq = rng.normal(size=(108, 5)), rng.normal(size=108), rng.normal(size=(12, 5))
    timed("band_grid_108x5_to_12", lambda: k.band_grid(gx, gy, gq, ALPHAS), n=2)
    # A misfit frame at the door's cap.
    mx = rng.normal(size=(2000, 16))
    timed("misfit_2000x16", lambda: k.misfit(mx, workers=misfit_workers), n=1, warm=False)
    # A panel at cube scale: the ensemble read over it, a hundred queries.
    px, py, pq = rng.normal(size=(panel_rows, 20)), rng.normal(size=panel_rows), rng.normal(size=(100, 20))
    timed(f"band_grid_panel_{panel_rows}x20_to_100", lambda: k.band_grid(px, py, pq, ALPHAS), n=1, warm=False)
    return out


if __name__ == "__main__":
    import os

    workers = os.environ.get("GLOSSKERNELS_MISFIT_WORKERS")
    print(json.dumps(run(use_amp=os.environ.get("GLOSSKERNELS_AMP") == "1", misfit_workers=int(workers) if workers else None), indent=1))
