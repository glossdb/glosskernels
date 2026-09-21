"""Build the default calibration record the kernel ships: each voice
walked over the public panels, its PITs counted per hundredth.

    uv run python -m glosskernels.harness.defaults

The panels named here are the record's; grade `dcal:<voice>` on others."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import calibration
from . import panels, score, voices

BUILT_FROM = ("tourism_monthly", "hospital")
VOICES = ("walk:tabicl", "chronos2")


def build(series: int = 60, months: int = 18) -> dict:
    records: dict[str, dict[str, list[float]]] = {}
    for name in VOICES:
        (voice,) = voices.build([name])
        pits = []
        for panel_name in BUILT_FROM:
            panel = panels.load(panel_name).head(series)
            total = panel.y.shape[1]
            for t in range(total - months, total):
                q, actual = voice.step(panel.y[:, :t], panel.moy[: t + 1], score.GRID), panel.y[:, t]
                landed = ~np.isnan(actual) & ~np.isnan(q).any(axis=1)
                pits.append(calibration.pit(q[landed], actual[landed], np.flatnonzero(landed) * 100_003 + t))
        records[voice.record] = {"1": calibration.histogram(np.concatenate(pits)).tolist()}
    return {
        "built_from": list(BUILT_FROM),
        "series_per_panel": series,
        "months_walked": months,
        "bins": calibration.BINS,
        "records": records,
    }


if __name__ == "__main__":
    path = Path(calibration.__file__).with_name("calibration_default.json")
    path.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {path}")
