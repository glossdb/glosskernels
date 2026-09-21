"""A read made ready for the model, off the GPU's critical path.

What the package's `fit` and `predict` do to a table before the model
sees it — the target scaled, the features encoded, the members' views
generated — with the package's own classes, so the numbers are its. It
is pure host work, a few milliseconds a read of mostly input checking,
and with the forward passes batched it is what a walk waits on (4.2 s of
5.1 s for 374 points beside an L4). Reads are independent, so a pool of
processes prepares them side by side; the model lives only in the parent.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Fewer reads than this are prepared in line: a pool's round trip is not free.
POOL_FROM = 64


def prepare(read: tuple[np.ndarray, np.ndarray, np.ndarray], members: int):
    """`(tables, y_scaler)` — tables a list of (Xs, ys), member-major, as
    `EnsembleGenerator.transform(mode="both")` yields them — or the text
    of the package's refusal."""
    from sklearn.preprocessing import StandardScaler
    from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical

    train_x, train_y, test_x = read
    try:
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(np.asarray(train_y, dtype=np.float32).reshape(-1, 1)).flatten()
        encoder = TransformToNumerical(verbose=False)
        x = encoder.fit_transform(train_x)
        generator = EnsembleGenerator(
            classification=False,
            n_estimators=members,
            # One member is the pinned one: no normalization, no feature shuffle.
            norm_methods="none" if members == 1 else ["none", "power"],
            feat_shuffle_method="none" if members == 1 else "latin",
            outlier_threshold=4.0,
            random_state=0,
        )
        generator.fit(x, y_scaled)
        tables = list(generator.transform(encoder.transform(test_x), mode="both").values())
    except Exception as e:  # the package's own refusals, by their text
        return str(e)
    return tables, y_scaler


def _prepare_chunk(chunk, members):
    return [prepare(read, members) for read in chunk]


_POOL: ProcessPoolExecutor | None = None
_LOCK = threading.Lock()


def workers() -> int:
    """`GLOSSKERNELS_PREPARE_WORKERS`, else the cores but one, at most eight; 0 prepares in line."""
    named = os.environ.get("GLOSSKERNELS_PREPARE_WORKERS", "").strip()
    if named.isdigit():
        return int(named)
    return max(0, min(8, (os.cpu_count() or 1) - 1))


def prepare_many(reads: list, members: int) -> list:
    global _POOL
    n = workers()
    if n < 2 or len(reads) < POOL_FROM:
        return [prepare(read, members) for read in reads]
    with _LOCK:
        if _POOL is None:
            import multiprocessing

            # Spawned, not forked: the parent holds a CUDA context.
            _POOL = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
    size = max(1, len(reads) // (n * 4))
    chunks = [reads[i : i + size] for i in range(0, len(reads), size)]
    return [done for chunk in _POOL.map(_prepare_chunk, chunks, [members] * len(chunks)) for done in chunk]


def warm() -> None:
    """Start the pool now — a worker's first import takes seconds, and the
    first walk should not be the one to pay them."""
    n = workers()
    if n >= 2:
        dummy = (np.arange(12.0).reshape(6, 2), np.arange(6.0), np.ones((1, 2)))
        prepare_many([dummy] * max(POOL_FROM, n * 4), 1)
