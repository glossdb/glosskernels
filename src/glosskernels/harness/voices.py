"""The voices: each answers next month's quantiles for every series in
the panel from what was known at the origin.

A voice's `step(y, moy, alphas)` gets the panel cut at the origin — `y`
is (series, t), the month to call is column t, `moy` is (t + 1,) and
ends on that month — and returns (series, alphas), NaN for a series it
will not call.

The tabular voices share the walk's graded recipe (glossql's
`metric_band_walk`: index, month of year, the 1- and 12-month lags and
the trailing three-month mean, absent features filled by the median of
the training rows) and differ in what forms the context — one series'
own months, or the panel's — and in the model behind it."""

from __future__ import annotations

from typing import Callable

import numpy as np

MIN_TRAIN = 5  # the walk's floor on training rows

# (train_x, train_y, test_x, alphas) -> (test rows, alphas)
Backend = Callable[[np.ndarray, np.ndarray, np.ndarray, list[float]], np.ndarray]


def _trim(v: np.ndarray) -> np.ndarray:
    """A series from its first value on."""
    present = np.flatnonzero(~np.isnan(v))
    return v[present[0] :] if present.size else v[:0]


def recipe(v: np.ndarray, moy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The walk's feature rows for months 1..n of `v`, the last one the
    month to call (`moy` runs one past `v`): features (n, 5), labels
    (n - 1,). Absent stays NaN — the fill is the caller's, from its own
    training rows."""
    n = v.shape[0]
    feats = np.full((n, 5), np.nan)
    for i in range(1, n + 1):
        recent = v[max(0, i - 3) : i]
        recent = recent[~np.isnan(recent)]
        feats[i - 1] = [
            i,
            moy[i],
            v[i - 1],
            recent.mean() if recent.size else np.nan,
            v[i - 12] if i >= 12 else np.nan,
        ]
    return feats, v[1:]


def _filled(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absent features take the training rows' median; one absent
    throughout takes 0.0."""
    with np.errstate(all="ignore"):
        fills = np.nan_to_num(np.nanmedian(train_x, axis=0) if train_x.size else np.zeros(train_x.shape[1]))
    return np.where(np.isnan(train_x), fills, train_x), np.where(np.isnan(test_x), fills, test_x)


# -- the models behind the tabular voices ------------------------------------


def tabicl(ensemble: bool) -> Backend:
    """The kernel's own reads: the pinned member (`band_point`'s) or the
    package's default ensemble (`band_grid`'s)."""
    from .. import kernels

    def backend(train_x, train_y, test_x, alphas):
        k = kernels.get()
        est = k._regressor(random_state=0) if ensemble else k._regressor(
            n_estimators=1, norm_methods="none", feat_shuffle_method="none", random_state=0
        )
        est.fit(train_x, train_y)
        q = est.predict(test_x, output_type="quantiles", alphas=alphas)
        return np.asarray(q, dtype=np.float64).reshape(test_x.shape[0], len(alphas))

    return backend


def nori(model: str = "nori-6m") -> Backend:
    from synthefy_nori import NoriRegressor

    def backend(train_x, train_y, test_x, alphas):
        reg = NoriRegressor(model=model)
        reg.fit(train_x, train_y)
        q = reg.predict(test_x, output_type="quantiles", quantiles=alphas)
        return np.asarray(q, dtype=np.float64).reshape(len(alphas), test_x.shape[0]).T

    return backend


# -- the voices --------------------------------------------------------------


class SeasonalNaive:
    """The floor: the same month last year, banded by the spread of the
    series' own year-over-year changes."""

    name = "seasonal_naive"

    def step(self, y, moy, alphas):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        for s in range(y.shape[0]):
            v = _trim(y[s])
            lag = 12 if v.shape[0] > 12 + MIN_TRAIN else 1
            if v.shape[0] <= lag + MIN_TRAIN or np.isnan(v[-lag]):
                continue
            moves = v[lag:] - v[:-lag]
            out[s] = v[-lag] + np.quantile(moves[~np.isnan(moves)], alphas)
        return out


class Walk:
    """The graded protocol: each series alone, its own months the context."""

    def __init__(self, name: str, backend: Backend):
        self.name, self.backend = name, backend

    def step(self, y, moy, alphas):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        for s in range(y.shape[0]):
            v = _trim(y[s])
            feats, labels = recipe(v, moy[moy.shape[0] - v.shape[0] - 1 :])
            known = ~np.isnan(labels)
            if known.sum() < MIN_TRAIN:
                continue
            # The fill reads every training row, labelled or not, as the door's does.
            train_x, test_x = _filled(feats[:-1], feats[-1:])
            out[s] = self.backend(train_x[known], labels[known], test_x, alphas)[0]
        return out


class Pooled:
    """The panel as the context: every series' recent rows in one table,
    each series in units of its own trailing year so levels compare, and
    one fit answering every series at the origin. The recipe's index
    column is left out: it counts from each series' own first month, so
    across series it orders nothing, and the month to call always sits
    past its end (on tourism_monthly it cost a factor of three in
    quantile loss)."""

    def __init__(self, name: str, backend: Backend, recent: int = 36, max_rows: int = 3000, seed: int = 0):
        self.name, self.backend = name, backend
        self.recent, self.max_rows, self.seed = recent, max_rows, seed

    def step(self, y, moy, alphas):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        train_x, train_y, test_x, called, scales = [], [], [], [], []
        for s in range(y.shape[0]):
            v = _trim(y[s])
            if v.shape[0] < MIN_TRAIN + 1:
                continue
            with np.errstate(all="ignore"):
                scale = np.nanmean(np.abs(v[-12:]))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            feats, labels = recipe(v / scale, moy[moy.shape[0] - v.shape[0] - 1 :])
            known = ~np.isnan(labels)
            known[: -self.recent] = False
            train_x.append(feats[:-1][known])
            train_y.append(labels[known])
            test_x.append(feats[-1])
            called.append(s)
            scales.append(scale)
        if not called:
            return out
        train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
        if train_x.shape[0] > self.max_rows:
            keep = np.random.default_rng(self.seed).choice(train_x.shape[0], size=self.max_rows, replace=False)
            train_x, train_y = train_x[keep], train_y[keep]
        train_x, test_x = _filled(train_x[:, 1:], np.asarray(test_x)[:, 1:])
        out[called] = self.backend(train_x, train_y, test_x, alphas) * np.asarray(scales)[:, None]
        return out


class Chronos:
    """A time-series model reading each series alone, all of them in one batch."""

    def __init__(self, name: str = "chronos2", model: str = "amazon/chronos-2", context: int = 512):
        from chronos import BaseChronosPipeline

        from ..kernels import pick_device

        self.name, self.context = name, context
        device = pick_device()
        self.pipeline = BaseChronosPipeline.from_pretrained(model, device_map="cpu" if device == "mps" else device)

    def step(self, y, moy, alphas):
        import torch

        out = np.full((y.shape[0], len(alphas)), np.nan)
        called, inputs = [], []
        for s in range(y.shape[0]):
            v = _trim(y[s])[-self.context :]
            if v.shape[0] < MIN_TRAIN + 1:
                continue
            called.append(s)
            inputs.append(torch.tensor(v, dtype=torch.float32))
        if called:
            quantiles, _mean = self.pipeline.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(alphas))
            out[called] = np.stack([np.asarray(q, dtype=np.float64).reshape(-1, len(alphas))[0] for q in quantiles])
        return out


def build(names: list[str]) -> list:
    """Voices by name; one whose package is not installed says so."""
    makers = {
        "seasonal_naive": SeasonalNaive,
        "walk:tabicl": lambda: Walk("walk:tabicl", tabicl(ensemble=False)),
        "pooled:tabicl": lambda: Pooled("pooled:tabicl", tabicl(ensemble=True)),
        "walk:nori": lambda: Walk("walk:nori", nori()),
        "pooled:nori": lambda: Pooled("pooled:nori", nori()),
        "chronos2": Chronos,
    }
    voices = []
    for name in names:
        if name not in makers:
            raise ValueError(f"no voice {name!r}; the harness knows {sorted(makers)}")
        try:
            voices.append(makers[name]())
        except ImportError as e:
            raise SystemExit(f"{name}: {e} — `uv sync --group harness`") from e
    return voices
