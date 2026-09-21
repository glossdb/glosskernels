"""The voices: each answers a coming month's quantiles for every series
in the panel from what was known at the origin.

A voice's `step(y, moy, alphas, h)` gets the panel cut at the origin —
`y` is (series, t), the month to call is column t + h - 1, `moy` is
(t + h,) and ends on that month — and returns (series, alphas), NaN for
a series it will not call. Past the first month the tabular voices call
directly: one fit per horizon, its rows built from what was known `h`
months before each label, never a forecast fed back in.

The tabular voices share the walk's graded recipe (glossql's
`metric_band_walk`: index, month of year, the 1- and 12-month lags and
the trailing three-month mean, absent features filled by the median of
the training rows) and differ in what forms the context — one series'
own months, or the panel's — and in the model behind it."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .. import calibration
from .. import voices as service

MIN_TRAIN = 5  # the walk's floor on training rows

# (train_x, train_y, test_x, alphas) -> (test rows, alphas)
Backend = Callable[[np.ndarray, np.ndarray, np.ndarray, list[float]], np.ndarray]


def _trim(v: np.ndarray) -> np.ndarray:
    """A series from its first value on."""
    present = np.flatnonzero(~np.isnan(v))
    return v[present[0] :] if present.size else v[:0]


def recipe(v: np.ndarray, moy: np.ndarray, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The walk's feature rows for months h..n+h-1 of `v`, the last one
    the month to call (`moy` runs `h` past `v`): features (n, 5), labels
    (n - h,). At h = 1 this is the door's recipe exactly; further out
    the lag and the trailing mean step back to the last month known `h`
    before the label, and the 12-month lag stays while it is still known.
    Absent stays NaN — the fill is the caller's, from its own training
    rows."""
    n = v.shape[0]
    feats = np.full((n, 5), np.nan)
    for i in range(h, n + h):
        known = i - h + 1  # months 0..i-h were known when month i was called
        recent = v[max(0, known - 3) : known]
        recent = recent[~np.isnan(recent)]
        feats[i - h] = [
            i,
            moy[i],
            v[known - 1],
            recent.mean() if recent.size else np.nan,
            v[i - 12] if h <= 12 <= i else np.nan,
        ]
    return feats, v[h:]


def seasonal_recipe(v: np.ndarray, moy: np.ndarray, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Rows for a call further out, where the last known month is the
    wrong anchor — it is another season. The month is placed against the
    same month in the last year or two known, and against where the
    level has gone since: the mean of the last twelve known months, and
    of the twelve before them. Same shapes as `recipe`; nothing in a row
    is later than `h` months before its label."""
    n = v.shape[0]
    feats = np.full((n, 7), np.nan)
    back = 12 * int(np.ceil(h / 12))  # the same month, the last time it was known
    with np.errstate(all="ignore"):
        for i in range(h, n + h):
            known = i - h + 1
            year, before = v[max(0, known - 12) : known], v[max(0, known - 24) : max(0, known - 12)]
            feats[i - h] = [
                i,
                moy[i],
                v[i - back] if i >= back else np.nan,
                v[i - back - 12] if i >= back + 12 else np.nan,
                np.nanmean(year) if np.isfinite(year).any() else np.nan,
                np.nanmean(before) if np.isfinite(before).any() else np.nan,
                v[known - 1],
            ]
    return feats, v[h:]


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
    """The floor — the service's own `voices.seasonal_naive`, a panel at a time."""

    name = "seasonal_naive"

    def step(self, y, moy, alphas, h=1):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        for s in range(y.shape[0]):
            try:
                out[s] = service.seasonal_naive(y[s], np.array([h]), list(alphas))[0]
            except service.VoiceError:
                continue
        return out


class Walk:
    """The graded protocol: each series alone, its own months the context."""

    def __init__(self, name: str, backend: Backend, rows=recipe, record: str | None = None):
        self.name, self.backend, self.rows, self.record = name, backend, rows, record

    def step(self, y, moy, alphas, h=1):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        for s in range(y.shape[0]):
            v = _trim(y[s])
            if v.shape[0] < h + MIN_TRAIN:
                continue
            feats, labels = self.rows(v, moy[moy.shape[0] - v.shape[0] - h :], h)
            known = ~np.isnan(labels)
            if known.sum() < MIN_TRAIN:
                continue
            # The fill reads every training row, labelled or not, as the door's does.
            train_x, test_x = _filled(feats[: labels.shape[0]], feats[-1:])
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

    def step(self, y, moy, alphas, h=1):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        train_x, train_y, test_x, called, scales = [], [], [], [], []
        for s in range(y.shape[0]):
            v = _trim(y[s])
            if v.shape[0] < MIN_TRAIN + h:
                continue
            with np.errstate(all="ignore"):
                scale = np.nanmean(np.abs(v[-12:]))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            feats, labels = recipe(v / scale, moy[moy.shape[0] - v.shape[0] - h :], h)
            known = ~np.isnan(labels)
            known[: -self.recent] = False
            train_x.append(feats[: labels.shape[0]][known])
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
    """A time-series model reading each series alone, all of them in one
    batch — the service's own `voices.Chronos2`, a panel at a time."""

    def __init__(self, name: str = "chronos2"):
        from ..kernels import pick_device

        self.name, self.record = name, "chronos2"
        self.voice = service.Chronos2(pick_device())

    def step(self, y, moy, alphas, h=1):
        out = np.full((y.shape[0], len(alphas)), np.nan)
        called = [s for s in range(y.shape[0]) if service._known(y[s]).shape[0] >= service.MIN_HISTORY]
        if called:
            got = self.voice.quantiles([y[s] for s in called], [np.array([h])] * len(called), list(alphas))
            out[called] = np.stack([q[0] for q in got])
        return out


class _Once:
    """A voice asked once per origin and horizon, however many listen —
    the raw voice and its calibrated reading share the fits."""

    def __init__(self, inner):
        self.inner, self.name, self._answers = inner, inner.name, {}
        self.record = getattr(inner, "record", None)

    def step(self, y, moy, alphas, h=1):
        # The last known column tells one panel from another cut at the
        # same origin — the months, and their trailing sums.
        key = (y.shape[1], h, len(alphas), y[:, -1].tobytes())
        if key not in self._answers:
            self._answers[key] = self.inner.step(y, moy, alphas, h)
        return self._answers[key]


class Calibrated:
    """A voice read through its own record — the harness playing the
    record's part (keeping the PITs, handing them back) around the
    kernel's pure `calibration.recalibrate`. Each answer is kept; once
    the month it called has landed, the PIT of the actual against it
    joins the voice's history — the panel's, every series together, as
    a metric's own six walked months say too little alone — kept apart
    per horizon, since a voice honest a month out may be narrow a year
    out. A band at alpha is then the raw quantile at the level the past
    PITs put alpha at: a voice whose 90s held four times in five is read
    further out.

    Point in time by construction: an origin's PITs come only from
    months already in the `y` it was handed. Until `min_history` PITs
    have landed at a horizon the voice speaks raw there. What lies
    outside the raw grid's ends cannot be reached — a voice too narrow
    past its first percentile stays so."""

    def __init__(self, inner, shipped: bool = False):
        # `cal:` reads the voice through its own record alone — the grade of
        # the method. `dcal:` adds the kernel's shipped default record, as a
        # deployment does; grade it only on panels the default was not built from.
        self.inner, self.shipped = inner, shipped
        self.name = f"{'dcal' if shipped else 'cal'}:{inner.name}"
        self._pending: dict[tuple, np.ndarray] = {}  # (panel, h, origin) -> the raw answer
        self._pits: dict[tuple, list[np.ndarray]] = {}  # (panel, h) -> landed PITs

    def step(self, y, moy, alphas, h=1):
        raw = self.inner.step(y, moy, alphas, h)
        t = y.shape[1]
        # A panel is known by its first two years — the same under every
        # origin, and what tells the months from their trailing sums.
        panel = y[:, :24].tobytes()
        for key in sorted(k for k in self._pending if k[0] == panel and k[2] + k[1] - 1 < t):
            answer, actual = self._pending.pop(key), y[:, key[2] + key[1] - 1]
            landed = ~np.isnan(actual) & ~np.isnan(answer).any(axis=1)
            # A point is a series at a month: what its tie draw is salted with.
            salt = np.flatnonzero(landed) * 100_003 + key[2]
            self._pits.setdefault(key[:2], []).append(calibration.pit(answer[landed], actual[landed], salt))
        self._pending[(panel, h, t)] = raw
        landed = self._pits.get((panel, h))
        history = calibration.histogram(np.concatenate(landed)) if landed else None
        record = getattr(self.inner, "record", None)
        default = calibration.default_record(record, h) if self.shipped and record else None
        return calibration.recalibrate(raw, alphas, history, default)


class Blend:
    """Several voices as one: the mean of their quantiles, level by
    level — each voice's band ends averaged, which keeps a band a band
    (averaging the distributions instead would only ever widen it)."""

    def __init__(self, voices: list):
        self.voices, self.name = voices, "blend:" + "+".join(v.name for v in voices)

    def step(self, y, moy, alphas, h=1):
        return np.mean([np.sort(v.step(y, moy, alphas, h), axis=1) for v in self.voices], axis=0)


def build(names: list[str]) -> list:
    """Voices by name: `cal:<voice>` for one read through its own record,
    `blend:<voice>+<voice>` for several as one, and `cal:blend:…` for
    both. One whose package is not installed says so."""
    makers = {
        "seasonal_naive": SeasonalNaive,
        "walk:tabicl": lambda: Walk("walk:tabicl", tabicl(ensemble=False), record="tabicl"),
        "seasonal:tabicl": lambda: Walk("seasonal:tabicl", tabicl(ensemble=False), rows=seasonal_recipe),
        "pooled:tabicl": lambda: Pooled("pooled:tabicl", tabicl(ensemble=True)),
        "walk:nori": lambda: Walk("walk:nori", nori()),
        "pooled:nori": lambda: Pooled("pooled:nori", nori()),
        "chronos2": Chronos,
    }
    raw: dict[str, _Once] = {}

    def once(name: str) -> _Once:
        if name not in makers:
            raise ValueError(f"no voice {name!r}; the harness knows {sorted(makers)}, each also as cal:<voice>")
        if name not in raw:
            try:
                raw[name] = _Once(makers[name]())
            except ImportError as e:
                raise SystemExit(f"{name}: {e} — `uv sync --group harness`") from e
        return raw[name]

    def named(name: str):
        if name.startswith("cal:"):
            return Calibrated(named(name[4:]))
        if name.startswith("dcal:"):
            return Calibrated(named(name[5:]), shipped=True)
        if name.startswith("blend:"):
            if name not in raw:
                raw[name] = _Once(Blend([named(part) for part in name[6:].split("+")]))
            return raw[name]
        return once(name)

    return [named(n) for n in names]
