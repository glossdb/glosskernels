"""Build the default calibration records the kernel ships: each voice
walked over the public panels, its PITs counted per hundredth — a month
ahead on the months, and a year ahead on their trailing annual totals.

    uv run python -m glosskernels.harness.defaults

The panels named here are the record's; grade `dcal:<voice>` on others."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import calibration
from . import panels, score, voices

BUILT_FROM = ("tourism_monthly", "hospital")
VOICES = ("walk:tabicl", "chronos2", "seasonal_naive")
WINDOWS = {1: 1, calibration.TOTAL_WINDOW: calibration.TOTAL_WINDOW}  # window -> the horizon it is walked at


def build(series: int = 60, months: int = 18) -> dict:
    records: dict[str, dict[str, list[float]]] = {}
    for name in VOICES:
        (voice,) = voices.build([name])
        records[voice.record] = {}
        for window, h in WINDOWS.items():
            pits = []
            for panel_name in BUILT_FROM:
                panel = panels.load(panel_name).head(series)
                y = panel.y if window == 1 else score.trailing_sum(panel.y, window)
                total = y.shape[1]
                for t in range(total - months - h + 1, total - h + 1):
                    q, actual = voice.step(y[:, :t], panel.moy[: t + h], score.GRID, h), y[:, t + h - 1]
                    landed = ~np.isnan(actual) & ~np.isnan(q).any(axis=1)
                    pits.append(calibration.pit(q[landed], actual[landed], np.flatnonzero(landed) * 100_003 + t))
            records[voice.record][str(window)] = calibration.histogram(np.concatenate(pits)).tolist()
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
