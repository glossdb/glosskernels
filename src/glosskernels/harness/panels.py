"""Monthly panels on one calendar axis.

A panel is a matrix, series by month, NaN before a series starts. The
public ones come from the Monash archive as `autogluon/chronos_datasets`
holds it on the hub (one parquet per panel: id, timestamps, targets).
Series that do not end on the panel's common last month are dropped:
the walk's origins are calendar months, and a pooled context may only
hold what every series had seen by then."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HUB_REPO = "autogluon/chronos_datasets"
# Enterprise-shaped monthlies: demand by region, patient counts by
# service, and intermittent spare-part sales (mostly zeros).
PANELS = {
    "tourism_monthly": "monash_tourism_monthly",
    "hospital": "monash_hospital",
    "car_parts": "monash_car_parts",
    # Held out of the default calibration record: macro indicators, and a
    # mixed bag of short real and synthetic monthlies.
    "fred_md": "monash_fred_md",
    "cif_2016": "monash_cif_2016",
}


@dataclass
class Panel:
    name: str
    months: np.ndarray  # (T,) datetime64[M]
    y: np.ndarray  # (series, T) float64, NaN where a series has no value

    @property
    def moy(self) -> np.ndarray:
        """Month of year per column, 1–12."""
        return self.months.astype(int) % 12 + 1

    def head(self, series: int, seed: int = 0) -> "Panel":
        """A seeded sample of `series` rows, the axis unchanged."""
        if series >= self.y.shape[0]:
            return self
        keep = np.sort(np.random.default_rng(seed).choice(self.y.shape[0], size=series, replace=False))
        return Panel(self.name, self.months, self.y[keep])


def load(name: str) -> Panel:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if name not in PANELS:
        raise ValueError(f"no panel {name!r}; the harness knows {sorted(PANELS)}")
    path = hf_hub_download(HUB_REPO, f"{PANELS[name]}/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    spans = [
        (np.asarray(ts, dtype="datetime64[M]"), np.asarray(target, dtype=np.float64))
        for ts, target in zip(df["timestamp"], df["target"])
    ]
    ends, counts = np.unique([m[-1] for m, _ in spans], return_counts=True)
    end = ends[np.argmax(counts)]
    spans = [(m, v) for m, v in spans if m[-1] == end]
    start = min(m[0] for m, _ in spans)
    months = np.arange(start, end + np.timedelta64(1, "M"), dtype="datetime64[M]")
    y = np.full((len(spans), months.shape[0]), np.nan)
    for row, (m, v) in enumerate(spans):
        y[row, (m - start).astype(int)] = v
    return Panel(name, months, y)


def synthetic(series: int = 12, months: int = 60, seed: int = 0) -> Panel:
    """A panel with a shared season and per-series level and trend — for
    the tests, and for a run with no network."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    level = rng.uniform(50, 5000, size=(series, 1))
    trend = rng.normal(0.0, 0.004, size=(series, 1))
    season = 0.2 * np.sin(2 * np.pi * (t % 12) / 12.0)
    y = level * (1 + trend * t + season + rng.normal(0, 0.05, size=(series, months)))
    start = np.datetime64("2019-01", "M")
    return Panel("synthetic", np.arange(start, start + np.timedelta64(months, "M"), dtype="datetime64[M]"), y)
