"""A voice read through its record: a pure function of the raw answer
and the PITs the caller has kept.

The kernel holds nothing between calls. The record keeps each walked
point's PIT — data, like the training rows — and hands the relevant
ones back with the next request; what is done with them is statistics,
and lives here, versioned and graded with the reads. Nothing is fitted:
a band at alpha is the raw quantile at the level the past PITs put
alpha at. A voice whose actuals fell under its 10% line 16% of the time
is read lower there; one with no history is read as it spoke.

PITs are always taken against the raw answer, never the recalibrated
one — a record of corrected answers would chase its own correction."""

from __future__ import annotations

import numpy as np

# Below this many landed PITs a history says too little, and the voice
# speaks raw. Graded in the harness; the one constant the read carries.
MIN_HISTORY = 100


def pit(grid: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """The PIT of each actual against its row of a monotone quantile
    grid, the way `band_point` reads it: the share of the grid at or
    under the actual. `grid` is (rows, levels), `actual` (rows,)."""
    grid = np.sort(grid, axis=1)
    return (grid <= actual[:, None]).sum(axis=1) / (grid.shape[1] + 1)


def recalibrate(
    quantiles: np.ndarray, levels: list[float], pits: np.ndarray, min_history: int = MIN_HISTORY
) -> np.ndarray:
    """`quantiles` (rows, levels) at increasing `levels`, re-read through
    `pits`. Rows with an absent quantile stay absent. What lies past the
    grid's ends cannot be reached: a voice too narrow beyond its first
    level stays so, and a denser grid is the remedy."""
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[~np.isnan(pits)]
    if pits.shape[0] < min_history:
        return quantiles
    read_at = np.quantile(pits, levels)
    out = np.full_like(quantiles, np.nan)
    for row in np.flatnonzero(~np.isnan(quantiles).any(axis=1)):
        out[row] = np.interp(read_at, levels, np.sort(quantiles[row]))
    return out
