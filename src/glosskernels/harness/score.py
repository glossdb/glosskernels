"""The walk over a panel and what it scores.

Every voice is asked for a dense quantile grid (the percentiles), so one
answer gives the door's five bands, their coverage, and a PIT read the
way `band_point` reads it — the share of the grid at or under the
actual. Calibrated bands leave the PITs uniform; `pit_ks` is their
largest distance from that."""

from __future__ import annotations

import time

import numpy as np

from .panels import Panel

BANDS = [0.05, 0.10, 0.50, 0.90, 0.95]  # the door's alphas
GRID = [round(a, 2) for a in np.arange(1, 100) / 100.0]
_AT = [GRID.index(a) for a in BANDS]


def walk(panel: Panel, voices: list, months: int = 6) -> dict:
    """The last `months` of the panel, one step ahead each, every voice
    at every origin. Scores are over the points every voice called, so
    the voices are compared on the same months of the same series."""
    total = panel.y.shape[1]
    actual = panel.y[:, total - months :]  # (series, months)
    answers = {v.name: np.full((*actual.shape, len(GRID)), np.nan) for v in voices}
    seconds = {v.name: 0.0 for v in voices}
    for w in range(months):
        t = total - months + w
        for v in voices:
            start = time.perf_counter()
            answers[v.name][:, w] = v.step(panel.y[:, :t], panel.moy[: t + 1], GRID)
            seconds[v.name] += time.perf_counter() - start

    scored = ~np.isnan(actual)
    for q in answers.values():
        scored &= ~np.isnan(q).any(axis=-1)
    out = {"panel": panel.name, "series": int(panel.y.shape[0]), "months": months, "points": int(scored.sum()), "voices": {}}
    for name, q in answers.items():
        out["voices"][name] = {**score(q[scored], actual[scored]), "s": round(seconds[name], 1)}
    floor = out["voices"].get("seasonal_naive", {}).get("wql")
    if floor:
        for scores in out["voices"].values():
            scores["wql_vs_naive"] = round(scores["wql"] / floor, 3)
    return out


def score(q: np.ndarray, actual: np.ndarray) -> dict:
    """`q` is (points, GRID) and `actual` (points,)."""
    if actual.size == 0:
        return {}
    q = np.sort(q, axis=1)  # a voice's grid may cross; the door's PIT reads a monotone one
    bands = q[:, _AT]
    gap = actual[:, None] - bands
    pinball = np.maximum(np.asarray(BANDS) * gap, (np.asarray(BANDS) - 1.0) * gap)
    pit = np.sort((q <= actual[:, None]).sum(axis=1) / (len(GRID) + 1))
    n = pit.shape[0]
    ks = max(np.max(np.arange(1, n + 1) / n - pit), np.max(pit - np.arange(n) / n))
    scale = max(float(np.abs(actual).sum()), 1e-12)
    return {
        # Weighted quantile loss over the door's bands: lower is sharper and better placed.
        "wql": round(float(2.0 * pinball.sum() / (scale * len(BANDS))), 4),
        "coverage80": round(float(np.mean((bands[:, 1] <= actual) & (actual <= bands[:, 3]))), 3),
        "coverage90": round(float(np.mean((bands[:, 0] <= actual) & (actual <= bands[:, 4]))), 3),
        "width80": round(float((bands[:, 3] - bands[:, 1]).sum() / scale), 4),
        "pit_ks": round(float(ks), 3),
    }
