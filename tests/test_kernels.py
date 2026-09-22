"""Parity with the pinned oracle fixtures (fixtures/, copied from the
port repo — see fixtures/SOURCE): the same package produced them, so the
reads here reproduce them up to device drift (they were generated on
CPU)."""

import os
from pathlib import Path

import numpy as np
import pytest

from glosskernels.kernels import KernelError, Kernels

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]

pytestmark = pytest.mark.skipif(not FIXTURES.is_dir(), reason="no fixtures/ in the repo")


@pytest.fixture(scope="module")
def k():
    return Kernels()


def _tol(k):
    # CPU reproduces the CPU oracle to float noise; an accelerator drifts
    # in the fourth decimal on standardized values.
    return (1e-5, 1e-6) if k.device == "cpu" else (2e-3, 1e-3)


def test_band_point_reproduces_the_pinned_walk(k):
    walk = np.load(FIXTURES / "bands_walk.npz")
    pinned = np.load(FIXTURES / "bands_pinned.npz")["bands"]
    n = int(os.environ.get("GLOSSKERNELS_TEST_FITS", "40"))
    rtol, atol = _tol(k)
    flips = 0
    for i, (_grain, _seed, _months, _t, off, size, _sid) in enumerate(walk["index"][:n]):
        train_x = walk["train_x_all"][off : off + size]
        train_y = walk["train_y_all"][off : off + size]
        actual = float(walk["actual"][i])
        q, pit = k.band_point(train_x, train_y, walk["test_x"][i], ALPHAS, actual)
        q = np.asarray(q)
        scale = max(1.0, float(np.abs(pinned[i]).max()))
        assert np.allclose(q, pinned[i], rtol=rtol, atol=atol * scale), f"fit {i}: {q} vs {pinned[i]}"
        assert 0.0 <= pit <= 1.0
        # The coverage read the harness grades: inside the 80 band or not.
        lo, hi = sorted((q[1], q[3]))
        plo, phi = sorted((pinned[i][1], pinned[i][3]))
        flips += int((lo <= actual <= hi) != (plo <= actual <= phi))
    assert flips <= (0 if k.device == "cpu" else 1)


def test_the_walk_in_one_call_matches_the_pinned_bands(k):
    """Points grouped by shape and answered a group a forward pass —
    batching must not move a band."""
    walk = np.load(FIXTURES / "bands_walk.npz")
    pinned = np.load(FIXTURES / "bands_pinned.npz")["bands"]
    n = int(os.environ.get("GLOSSKERNELS_TEST_FITS", "40"))
    rtol, atol = _tol(k)
    reads = [
        (walk["train_x_all"][off : off + size], walk["train_y_all"][off : off + size], walk["test_x"][i][None, :])
        for i, (_g, _s, _m, _t, off, size, _id) in enumerate(walk["index"][:n])
    ]
    for i, (q, grid) in enumerate(k.bands_many(reads, ALPHAS, members=1)):
        scale = max(1.0, float(np.abs(pinned[i]).max()))
        assert np.allclose(q[0], pinned[i], rtol=rtol, atol=atol * scale), f"fit {i}: {q[0]} vs {pinned[i]}"
        assert grid.shape[0] == 1 and np.all(np.diff(grid[0]) >= 0)


def test_band_grid_reproduces_the_ensemble_oracle(k):
    d = np.load(FIXTURES / "e4_walk.npz")
    oracle = np.load(FIXTURES / "e4_ensemble.npz")["grid_bands"]  # (fits, alphas, rows)
    rtol, atol = _tol(k)
    for i in range(int(os.environ.get("GLOSSKERNELS_TEST_GRIDS", "3"))):
        off, size = (int(v) for v in d["grid_offsets"][i])
        q = k.band_grid(
            d["grid_train_x"][off : off + size], d["grid_train_y"][off : off + size], d["grid_test_x"][i], ALPHAS
        )
        expect = oracle[i].T  # rows × alphas
        scale = max(1.0, float(np.abs(expect).max()))
        assert q.shape == expect.shape
        assert np.allclose(q, expect, rtol=rtol, atol=atol * scale), f"grid fit {i}"


def test_misfit_reproduces_the_density_oracle(k):
    d = np.load(FIXTURES / "density_scores.npz")
    # The fixture recorded the package's own orderings; the read takes them
    # as given and defaults to identity and reverse otherwise.
    logs = k.misfit(
        d["x_test"].astype(np.float64), perms=list(d["perms"]), random_state=42, train=d["x_train"].astype(np.float64)
    )
    # A score is the product of 2 x cols conditional densities, so the
    # estimators' float noise compounds: a CPU with another BLAS than the
    # oracle's (x86 oneDNN against arm64 Accelerate) lands 1e-3 off in
    # relative terms; an accelerator further.
    rtol, atol = _tol(k)
    rtol = max(rtol * 10, 5e-3)
    assert np.allclose(np.exp(logs), d["scores"], rtol=rtol, atol=atol), f"{np.exp(logs)} vs {d['scores']}"
    # The read the door consumes: one finite log density per row.
    assert logs.shape == (d["x_test"].shape[0],) and np.isfinite(logs).all()


def test_self_fit_misfit_ranks_an_outlier_last(k):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 4))
    x[:, 1] = 0.8 * x[:, 0] + 0.2 * x[:, 1]
    x[7] = [6.0, -6.0, 6.0, -6.0]
    logs = k.misfit(x)
    assert int(np.argmin(logs)) == 7


def test_misfit_columns_sum_to_the_score_and_name_the_cell(k):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 4))
    x[:, 1] = 0.8 * x[:, 0] + 0.2 * x[:, 1]
    x[7, 2] = 9.0  # one cell off; the rest of the row ordinary
    logs, shares = k.misfit(x, columns=True)
    assert shares.shape == x.shape
    assert np.array_equal(logs, k.misfit(x))  # asking per column changes no score
    assert np.allclose(shares.sum(axis=1), logs, rtol=1e-6, atol=1e-6)
    assert int(np.argmin(shares[7])) == 2
    # Scored from a context it is not in, the cell is the frame's least likely.
    held, held_shares = k.misfit(x, columns=True, folds=2)
    assert np.allclose(held_shares.sum(axis=1), held, rtol=1e-6, atol=1e-6)
    assert int(np.argmin(held)) == 7
    assert np.unravel_index(np.argmin(held_shares), held_shares.shape) == (7, 2)
    with pytest.raises(KernelError, match="folds"):
        k.misfit(x[:3], folds=2)
