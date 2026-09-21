"""A voice read through its record: a pure function of the raw answer
and the PITs the caller has kept.

The kernel holds nothing between calls. The record keeps each walked
point's PIT — data, like the training rows — and hands them back with a
request as a histogram: how many past PITs fell in each hundredth, a
hundred counts whatever the history's length. What is done with them is
statistics, and lives here, versioned and graded with the reads. Nothing
is fitted: a band at alpha is the raw quantile at the level the past
PITs put alpha at. A voice whose actuals fell under its 10% line 16% of
the time is read lower there.

A tenant's counts are added to a default record shipped with the kernel
(`calibration_default.json`, built from public panels by
`glosskernels.harness.defaults`, counted as `DEFAULT_WEIGHT`
observations): a new tenant is read honestly on the first day, and its
own record takes over as it grows. No tenant's PITs reach another's.

PITs are always taken against the raw answer, never the recalibrated
one — a record of corrected answers would chase its own correction."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np

BINS = 100
# What the default record counts for beside a tenant's own PITs.
DEFAULT_WEIGHT = 200.0
# Below this many PITs in all, a history says too little and the voice
# speaks raw. With a default record for the voice this is always met.
MIN_HISTORY = 100


def pit(grid: np.ndarray, actual: np.ndarray, salt: np.ndarray | None = None) -> np.ndarray:
    """The PIT of each actual against its row of a quantile grid: the
    share of the grid under the actual. `grid` is (rows, levels),
    `actual` (rows,).

    An actual tied with part of the grid — a zero month under a voice
    whose lower quantiles are all zero — has no one PIT: it lies
    anywhere in the tie, and counting the whole tie as under it (or none
    of it) piles a mostly-zero metric's PITs at one end and makes an
    honest voice look wrong. It is placed uniformly within the tie, the
    draw taken from the row's own bytes and its `salt`, so a replay
    repeats it. The salt tells apart points whose answers are identical
    (every quiet month of a dead metric): any integer that names the
    point — the caller's hash of metric and month — and, left out, the
    row's position in the request."""
    grid = np.sort(np.asarray(grid, dtype=np.float64), axis=1)
    actual = np.asarray(actual, dtype=np.float64)
    under = (grid < actual[:, None]).sum(axis=1).astype(np.float64)
    tied = (grid == actual[:, None]).sum(axis=1)
    salt = np.arange(grid.shape[0]) if salt is None else np.asarray(salt)
    for row in np.flatnonzero(tied):
        named = grid[row].tobytes() + actual[row].tobytes() + int(salt[row]).to_bytes(8, "big", signed=True)
        digest = hashlib.blake2b(named, digest_size=8).digest()
        under[row] += tied[row] * (int.from_bytes(digest, "big") / 2.0**64)
    return under / (grid.shape[1] + 1)


def histogram(pits: np.ndarray) -> np.ndarray:
    """PITs as the wire carries them: counts per hundredth."""
    pits = np.asarray(pits, dtype=np.float64)
    return np.histogram(pits[~np.isnan(pits)], bins=BINS, range=(0.0, 1.0))[0].astype(np.float64)


@cache
def _defaults() -> dict:
    path = Path(__file__).with_name("calibration_default.json")
    return json.loads(path.read_text())["records"] if path.exists() else {}


def default_record(voice: str, horizon: int = 1) -> np.ndarray | None:
    """The shipped record for a voice at a horizon, or None."""
    counts = _defaults().get(voice, {}).get(str(horizon))
    return None if counts is None else np.asarray(counts, dtype=np.float64)


def recalibrate(
    quantiles: np.ndarray,
    levels: list[float],
    history: np.ndarray | None = None,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """`quantiles` (rows, levels) at increasing `levels`, re-read through
    a record: the caller's `history` counts, and a `default` record
    weighed in at `DEFAULT_WEIGHT`. Rows with an absent quantile stay
    absent. What lies past the grid's ends cannot be reached: a voice too
    narrow beyond its first level stays so."""
    counts = np.zeros(BINS) if history is None else np.asarray(history, dtype=np.float64).copy()
    if counts.shape != (BINS,):
        raise ValueError(f"a PIT history is {BINS} counts, got {counts.shape}")
    if default is not None and default.sum() > 0:
        counts += default * (DEFAULT_WEIGHT / default.sum())
    if counts.sum() < MIN_HISTORY:
        return quantiles
    # The record's own quantile function: the level under which a share
    # `alpha` of past PITs fell. A sliver per bin keeps it increasing.
    counts = counts + 1e-9
    share = np.concatenate([[0.0], np.cumsum(counts) / counts.sum()])
    read_at = np.interp(levels, share, np.linspace(0.0, 1.0, BINS + 1))
    out = np.full_like(quantiles, np.nan)
    for row in np.flatnonzero(~np.isnan(quantiles).any(axis=1)):
        out[row] = np.interp(read_at, levels, np.sort(quantiles[row]))
    return out
