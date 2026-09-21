"""Parity against the port repo's pinned oracle fixtures, as numbers
rather than assertions — for a device the tests do not run on (a GPU,
AMP on). What the tests assert, this reports: the largest relative
deviation on the bands and the coverage flips over the pinned walk,
the largest deviation over the ensemble grids and the density read."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .kernels import Kernels

ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]


def _rel(q: np.ndarray, expect: np.ndarray) -> float:
    return float(np.max(np.abs(q - expect) / np.maximum(1.0, np.abs(expect))))


def _spread(devs: list[float]) -> dict:
    """Per-fit deviations: the worst says whether anything broke, the
    median and the 99th say whether it is one fit or all of them."""
    d = np.asarray(devs)
    return {"max_rel_dev": float(d.max()), "p99_rel_dev": float(np.quantile(d, 0.99)), "median_rel_dev": float(np.median(d))}


def run(fixtures: Path, use_amp: bool = False, bf16: bool = False) -> dict:
    if bf16:  # autocast's CUDA default is float16; bfloat16 keeps float32's range
        torch.set_autocast_dtype("cuda", torch.bfloat16)
    k = Kernels(use_amp=use_amp)
    out = {
        "device": k.device,
        "gpu": torch.cuda.get_device_name(0) if k.device == "cuda" else None,
        "amp": use_amp,
        "autocast": ("bfloat16" if bf16 else "float16") if use_amp else None,
    }

    walk = np.load(fixtures / "bands_walk.npz")
    pinned = np.load(fixtures / "bands_pinned.npz")["bands"]
    devs, flips80, flips90 = [], 0, 0
    t = time.perf_counter()
    for i, (_g, _s, _m, _t, off, size, _id) in enumerate(walk["index"]):
        q, _pit = k.band_point(
            walk["train_x_all"][off : off + size],
            walk["train_y_all"][off : off + size],
            walk["test_x"][i],
            ALPHAS,
            float(walk["actual"][i]),
        )
        q = np.asarray(q)
        devs.append(_rel(q, pinned[i]))
        a = float(walk["actual"][i])
        for lo_i, hi_i, name in ((1, 3, "80"), (0, 4, "90")):
            lo, hi = sorted((q[lo_i], q[hi_i]))
            plo, phi = sorted((pinned[i][lo_i], pinned[i][hi_i]))
            if (lo <= a <= hi) != (plo <= a <= phi):
                if name == "80":
                    flips80 += 1
                else:
                    flips90 += 1
    out["walk"] = {
        "fits": int(len(walk["index"])),
        **_spread(devs),
        "flips80": flips80,
        "flips90": flips90,
        "s": round(time.perf_counter() - t, 1),
    }

    d = np.load(fixtures / "e4_walk.npz")
    oracle = np.load(fixtures / "e4_ensemble.npz")["grid_bands"]
    devs = []
    t = time.perf_counter()
    for i in range(oracle.shape[0]):
        off, size = (int(v) for v in d["grid_offsets"][i])
        q = k.band_grid(
            d["grid_train_x"][off : off + size], d["grid_train_y"][off : off + size], d["grid_test_x"][i], ALPHAS
        )
        devs.append(_rel(q, oracle[i].T))
    out["grids"] = {"fits": int(oracle.shape[0]), **_spread(devs), "s": round(time.perf_counter() - t, 1)}

    dens = np.load(fixtures / "density_scores.npz")
    logs = k.misfit(
        dens["x_test"].astype(np.float64), perms=list(dens["perms"]), random_state=42, train=dens["x_train"].astype(np.float64)
    )
    out["density"] = {"max_rel_dev": _rel(np.exp(logs), dens["scores"]), "finite": bool(np.isfinite(logs).all())}
    return out


if __name__ == "__main__":
    import os
    import sys

    fixtures = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "fixtures"
    print(json.dumps(run(fixtures, use_amp=os.environ.get("GLOSSKERNELS_AMP") == "1"), indent=1))
