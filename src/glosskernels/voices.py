"""The voices beside TabICL, and how voices are blended.

TabICL reads a month through feature rows and pools what the rows hold;
a voice here reads the series itself. They answer the same question —
quantiles for a month some steps past the history — and each is
returned on its own, to land as its own voice, with their blend.

- `chronos2`: Chronos-2, a time-series model (Apache-2.0), every series
  of a cycle in one batch.
- `seasonal_naive`: the same month last season, banded by the series'
  own season-over-season moves. No model; the floor, and as a blend
  member what keeps a projection from losing to it (in the harness the
  three-voice blend is the only reading at or under this floor at every
  horizon on both panels).

The harness grades these same functions."""

from __future__ import annotations

import threading

import numpy as np

SERIES_VOICES = ("chronos2", "seasonal_naive")
VOICES = ("tabicl", *SERIES_VOICES)
MIN_HISTORY = 6  # months a series voice needs behind it
CHRONOS = "amazon/chronos-2"
CHRONOS_CONTEXT = 512


class VoiceError(Exception):
    """A series a voice will not call, with the reason."""


def _known(history: np.ndarray) -> np.ndarray:
    """A series from its first value on."""
    history = np.asarray(history, dtype=np.float64)
    present = np.flatnonzero(~np.isnan(history))
    return history[present[0] :] if present.size else history[:0]


def seasonal_naive(history: np.ndarray, horizons: np.ndarray, levels: list[float], season: int = 12) -> np.ndarray:
    """(horizons, levels): for each month `h` past the history, the value
    a season before it — while that is known; else the last month — plus
    the quantiles of the series' own moves over that lag."""
    v = _known(history)
    out = np.full((len(horizons), len(levels)), np.nan)
    for row, h in enumerate(int(h) for h in horizons):
        lag = season if v.shape[0] > season + MIN_HISTORY - 1 and h <= season else h
        if v.shape[0] <= lag + MIN_HISTORY - 1 or np.isnan(v[h - 1 - lag]):
            raise VoiceError(f"seasonal_naive: {v.shape[0]} months of history, {h} out — too short to band on")
        moves = v[lag:] - v[:-lag]
        out[row] = v[h - 1 - lag] + np.quantile(moves[~np.isnan(moves)], levels)
    return out


class Chronos2:
    """Loaded on first use, kept; one batch answers every series it is given."""

    def __init__(self, device: str):
        self.device = "cpu" if device == "mps" else device  # its kernels are not all on Metal
        self._pipeline = None
        self._lock = threading.Lock()

    def pipeline(self):
        with self._lock:
            if self._pipeline is None:
                try:
                    from chronos import BaseChronosPipeline
                except ImportError as e:
                    raise VoiceError(f"chronos2: not installed in this service ({e})") from e
                self._pipeline = BaseChronosPipeline.from_pretrained(CHRONOS, device_map=self.device)
            return self._pipeline

    def quantiles(self, histories: list[np.ndarray], horizons: list[np.ndarray], levels: list[float]) -> list[np.ndarray]:
        """Per series, (its horizons, levels)."""
        import torch

        series = []
        for history in histories:
            v = _known(history)[-CHRONOS_CONTEXT:]
            if v.shape[0] < MIN_HISTORY:
                raise VoiceError(f"chronos2: {v.shape[0]} months of history — too short to read")
            series.append(torch.tensor(v, dtype=torch.float32))
        furthest = int(max(int(np.max(h)) for h in horizons))
        answered, _mean = self.pipeline().predict_quantiles(series, prediction_length=furthest, quantile_levels=list(levels))
        out = []
        for got, mine in zip(answered, horizons):
            steps = np.asarray(got, dtype=np.float64).reshape(-1, len(levels))  # (furthest, levels)
            out.append(steps[np.asarray(mine, dtype=int) - 1])
        return out


def blend(answers: list[np.ndarray]) -> np.ndarray:
    """Several voices as one: the mean of their quantiles, level by level —
    each voice's band ends averaged, which keeps a band a band (averaging
    the distributions instead would only ever widen it)."""
    return np.mean([np.sort(a, axis=-1) for a in answers], axis=0)


def fetch_checkpoints() -> None:
    """Pull Chronos-2 into the hub cache — the image bakes it."""
    from huggingface_hub import snapshot_download

    snapshot_download(CHRONOS)
