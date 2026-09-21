"""The calibration arithmetic: the PIT on ties, the histogram as the
wire's form of a history, and how a default record gives way to a
tenant's own."""

import numpy as np
from scipy.stats import norm

from glosskernels import calibration

LEVELS = [round(a, 2) for a in np.arange(1, 100) / 100.0]


def test_pit_without_ties_is_the_share_of_the_grid_under_the_actual():
    grid = np.tile(np.arange(1.0, 100.0), (3, 1))
    got = calibration.pit(grid, np.array([0.5, 50.5, 99.5]))
    assert got.tolist() == [0.0, 0.5, 0.99]


def test_pit_on_a_mostly_zero_metric_reads_an_honest_voice_as_honest():
    # Demand is zero 70% of the time, else lognormal; the voice knows it exactly.
    rng = np.random.default_rng(0)
    n = 4000
    actual = np.where(rng.random(n) < 0.7, 0.0, rng.lognormal(1.0, 0.5, n))
    levels = np.asarray(LEVELS)
    row = np.where(levels <= 0.7, 0.0, np.exp(1.0 + 0.5 * norm.ppf(np.clip((levels - 0.7) / 0.3, 1e-9, 1 - 1e-9))))
    pits = calibration.pit(np.tile(row, (n, 1)), actual)
    counts = calibration.histogram(pits)
    # Uniform within sampling noise — counting the whole tie as under the
    # actual would put 70% of the PITs at 0.70.
    assert counts[:70].sum() / n == 0.7 or abs(counts[:70].sum() / n - 0.7) < 0.03
    assert counts.max() < 3 * n / calibration.BINS
    # And the same request gives the same PIT.
    assert np.array_equal(pits, calibration.pit(np.tile(row, (n, 1)), actual))


def test_recalibrate_widens_a_narrow_voice_from_its_histogram():
    rng = np.random.default_rng(1)
    narrow = np.tile(0.7 * norm.ppf(LEVELS), (3000, 1))
    history = calibration.histogram(calibration.pit(narrow, rng.normal(size=3000)))
    fresh = rng.normal(size=3000)
    raw_cover = np.mean((narrow[:, 9] <= fresh) & (fresh <= narrow[:, 89]))
    read = calibration.recalibrate(narrow, LEVELS, history)
    cover = np.mean((read[:, 9] <= fresh) & (fresh <= read[:, 89]))
    assert raw_cover < 0.7 and abs(cover - 0.8) < 0.03


def test_no_history_speaks_raw_and_a_default_gives_way_to_the_tenants_own():
    q = np.tile(norm.ppf(LEVELS), (2, 1))
    assert calibration.recalibrate(q, LEVELS) is q
    assert calibration.recalibrate(q, LEVELS, np.full(calibration.BINS, 0.5)) is q  # 50 PITs: too few

    flat = np.ones(calibration.BINS)  # a record of an honest voice
    ends = np.zeros(calibration.BINS)
    ends[[0, -1]] = 1.0  # a record of a voice far too narrow
    by_default = calibration.recalibrate(q, LEVELS, None, default=ends)
    assert by_default[0, 89] - by_default[0, 9] > 1.5 * (q[0, 89] - q[0, 9])
    # 200 default observations against 20,000 of the tenant's own: the tenant's say.
    own = calibration.recalibrate(q, LEVELS, flat * 200.0, default=ends)
    assert np.allclose(own[0, 9:90], q[0, 9:90], atol=0.05)
