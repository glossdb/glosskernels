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


def walk(panel: Panel, voices: list, months: int = 6, burn: int = 0) -> dict:
    """The last `months` of the panel, one step ahead each, every voice
    at every origin. `burn` walks that many months first, unscored — the
    record a calibrated voice reads itself through. Scores are over the
    points every voice called, so the voices are compared on the same
    months of the same series."""
    total = panel.y.shape[1]
    actual = panel.y[:, total - months :]  # (series, months)
    answers = {v.name: np.full((*actual.shape, len(GRID)), np.nan) for v in voices}
    seconds = {v.name: 0.0 for v in voices}
    for w in range(-burn, months):
        t = total - months + w
        for v in voices:
            start = time.perf_counter()
            q = v.step(panel.y[:, :t], panel.moy[: t + 1], GRID)
            seconds[v.name] += time.perf_counter() - start
            if w >= 0:
                answers[v.name][:, w] = q

    scored = _called(answers, actual)
    out = {
        "panel": panel.name,
        "series": int(panel.y.shape[0]),
        "months": months,
        "burn": burn,
        "points": int(scored.sum()),
        "voices": {name: {**score(q[scored], actual[scored]), "s": round(seconds[name], 1)} for name, q in answers.items()},
    }
    _against_floor(out["voices"])
    return out


def project(panel: Panel, voices: list, origins: int = 2, horizon: int = 12) -> dict:
    """Projections: from each origin (a year apart, the last one a year
    before the panel ends) every month out to `horizon`, and the total
    over those months two ways. `total_summed` adds the monthly bands up
    — what a plan built from monthly bands does, and right only if every
    month misses the same way at once. `total_direct` calls the total as
    its own series: the trailing `horizon`-month sum, read `horizon`
    months out — the cube's own cell for the year."""
    total = panel.y.shape[1]
    starts = [total - horizon * (o + 1) for o in reversed(range(origins))]
    rolling = trailing_sum(panel.y, horizon)

    shape = (panel.y.shape[0], origins)
    monthly = {v.name: np.full((*shape, horizon, len(GRID)), np.nan) for v in voices}
    direct = {v.name: np.full((*shape, len(GRID)), np.nan) for v in voices}
    seconds = {v.name: 0.0 for v in voices}
    actual = np.stack([panel.y[:, t : t + horizon] for t in starts], axis=1)  # (series, origins, horizon)
    for o, t in enumerate(starts):
        for v in voices:
            start = time.perf_counter()
            for h in range(1, horizon + 1):
                monthly[v.name][:, o, h - 1] = v.step(panel.y[:, :t], panel.moy[: t + h], GRID, h)
            direct[v.name][:, o] = v.step(rolling[:, :t], panel.moy[: t + horizon], GRID, horizon)
            seconds[v.name] += time.perf_counter() - start

    out = {"panel": panel.name, "series": int(shape[0]), "origins": origins, "horizon": horizon, "voices": {}}
    reads = {f"h{h}": None for h in sorted({1, 3, 6, horizon}) if h <= horizon}
    for name in monthly:
        out["voices"][name] = {"s": round(seconds[name], 1)}
    for key in reads:
        h = int(key[1:])
        answers = {name: q[:, :, h - 1] for name, q in monthly.items()}
        scored = _called(answers, actual[:, :, h - 1])
        for name, q in answers.items():
            out["voices"][name][key] = score(q[scored], actual[:, :, h - 1][scored])
    summed = {name: np.sort(q, axis=-1).sum(axis=2) for name, q in monthly.items()}
    for key, answers in (("total_summed", summed), ("total_direct", direct)):
        scored = _called({**summed, **{f"d:{n}": q for n, q in direct.items()}}, actual.sum(axis=2))
        for name, q in answers.items():
            out["voices"][name][key] = score(q[scored], actual.sum(axis=2)[scored])
    return out


def trailing_sum(y: np.ndarray, months: int) -> np.ndarray:
    """Column m holds the sum of months m-months+1..m; NaN until that
    many months exist, and wherever one of them is absent."""
    out = np.full_like(y, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(y, months, axis=1)
    out[:, months - 1 :] = windows.sum(axis=-1)
    return out


def _called(answers: dict, actual: np.ndarray) -> np.ndarray:
    """The points with an actual that every voice called."""
    scored = ~np.isnan(actual)
    for q in answers.values():
        scored &= ~np.isnan(q).any(axis=-1)
    return scored


def _against_floor(voices: dict) -> None:
    floor = voices.get("seasonal_naive", {}).get("wql")
    if floor:
        for scores in voices.values():
            scores["wql_vs_naive"] = round(scores["wql"] / floor, 3)


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
