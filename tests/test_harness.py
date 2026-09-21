"""The harness's own arithmetic: the recipe against the door's, the
scores against cases worked by hand, and a walk end to end on the
synthetic panel with the voices that need no model."""

import numpy as np

from glosskernels.harness import panels, score, voices


def test_recipe_is_the_doors():
    v = np.arange(1.0, 15.0)  # months 0..13, the 14th to call
    moy = np.arange(15) % 12 + 1
    feats, labels = voices.recipe(v, moy)
    assert feats.shape == (14, 5) and labels.shape == (13,)
    # Month 1: index, month of year, lag 1, the mean of what came before, no lag 12 yet.
    assert feats[0, :4].tolist() == [1.0, 2.0, 1.0, 1.0] and np.isnan(feats[0, 4])
    # Month 12: the trailing three-month mean (not the t-3 value), and the first lag 12.
    assert feats[11].tolist() == [12.0, 1.0, 12.0, 11.0, 1.0]
    # The last row is the month to call, and carries no label.
    assert feats[-1].tolist() == [14.0, 3.0, 14.0, 13.0, 3.0]
    assert labels.tolist() == v[1:].tolist()


def test_fill_is_the_training_median():
    train = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 8.0], [np.nan, np.nan]])
    test = np.array([[np.nan, np.nan]])
    filled_train, filled_test = voices._filled(train, test)
    assert filled_test.tolist() == [[2.0, 6.0]]
    assert filled_train[0, 1] == 6.0 and filled_train[3].tolist() == [2.0, 6.0]
    # Absent throughout: 0.0.
    assert voices._filled(np.full((3, 1), np.nan), np.full((1, 1), np.nan))[1].tolist() == [[0.0]]


def test_score_reads_a_calibrated_voice_as_calibrated():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=4000)
    from scipy.stats import norm

    q = np.tile(norm.ppf(score.GRID), (actual.shape[0], 1))
    s = score.score(q, actual)
    assert abs(s["coverage80"] - 0.80) < 0.03 and abs(s["coverage90"] - 0.90) < 0.03
    assert s["pit_ks"] < 0.03
    # Bands half as wide cover less and pile the PITs at the ends.
    narrow = score.score(q / 2, actual)
    assert narrow["coverage80"] < 0.55 and narrow["pit_ks"] > 0.1


def test_walk_scores_voices_on_the_points_all_called():
    class Silent:
        name = "silent_on_first"

        def step(self, y, moy, alphas):
            out = voices.SeasonalNaive().step(y, moy, alphas)
            out[0] = np.nan
            return out

    panel = panels.synthetic(series=5, months=40)
    out = score.walk(panel, [voices.SeasonalNaive(), Silent()], months=4)
    assert out["points"] == 4 * 4
    assert out["voices"]["silent_on_first"]["wql"] == out["voices"]["seasonal_naive"]["wql"]
    assert out["voices"]["seasonal_naive"]["wql_vs_naive"] == 1.0


def test_panel_axis():
    panel = panels.synthetic(series=3, months=14)
    assert panel.moy.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
    assert panel.head(2).y.shape == (2, 14) and panel.head(9) is panel
