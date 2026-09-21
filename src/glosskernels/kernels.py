"""The reads over the reference package, one loaded checkpoint.

`Kernels` holds the regressor checkpoint on one device and answers two
reads:

- `bands`: quantiles for query rows given context rows. One member is
  the pinned one (one estimator, no normalization, no feature shuffle) —
  the metric-bands walk; more is the package's ensemble (the `none` and
  `power` pipelines with latin feature shuffles) — the what-if replay,
  where sparse grids are the regime the ensemble was ruled in for.
  `band_point` and `band_grid` are those two as the oracle fixtures pin
  them, kept for the parity tests.
- `misfit`: the chain-rule density over one frame, fit and scored on
  the same rows, log space, mean over permutations — higher fits the
  frame better; the server negates.

The checkpoint loads once; the package reloads it on every `fit`
(`TabICLRegressor.fit` → `_load_model`), so the estimators built here
carry the shared module the way `TabICLUnsupervised` shares its own.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabicl import TabICLRegressor, TabICLUnsupervised

HUB_REPO = "jingang/TabICL"
REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


class KernelError(Exception):
    """A read the kernel refuses, with the reason the door reports."""


def fetch_checkpoints() -> None:
    """Pull the regressor into the hub cache — the image bakes it."""
    hf_hub_download(repo_id=HUB_REPO, filename=REGRESSOR)


def pick_device() -> str:
    """`GLOSSKERNELS_DEVICE`, else cuda, else mps, else cpu."""
    named = os.environ.get("GLOSSKERNELS_DEVICE", "").strip()
    if named:
        return named
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Regressor(TabICLRegressor):
    """A regressor that keeps a model it was handed instead of reloading."""

    def _load_model(self) -> None:
        if not hasattr(self, "model_"):
            super()._load_model()


class _Unsupervised(TabICLUnsupervised):
    """The density read over the shared regressor; numeric columns only."""

    def __init__(self, shared: torch.nn.Module, **kwargs: Any):
        super().__init__(**kwargs)
        self._shared = shared

    def _load_shared_model(self, estimator_cls):
        if estimator_cls is not TabICLRegressor:
            raise KernelError("misfit: the read serves numeric columns only")
        return self._shared


class Kernels:
    def __init__(self, device: str | None = None, use_amp: bool = False):
        self.device = device or pick_device()
        self.use_amp = use_amp
        offline = os.environ.get("HF_HUB_OFFLINE", "") == "1"
        loader = TabICLRegressor(device=self.device, allow_auto_download=not offline)
        loader._resolve_device()
        loader._load_model()
        loader.model_.to(loader.device_)
        self.model = loader.model_

    def _regressor(self, **kwargs: Any) -> _Regressor:
        est = _Regressor(device=self.device, use_amp=self.use_amp, **kwargs)
        est.model_ = self.model
        return est

    # -- the reads --------------------------------------------------------

    def bands(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        alphas: list[float],
        members: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantiles (test rows x alphas) and the raw quantile grid (test
        rows x grid) for `test_x` given the context rows. One member is
        the pinned one (no normalization, no feature shuffle); more is
        the package's ensemble of that size."""
        rows, cols = train_x.shape
        if rows < 2 or train_y.shape[0] != rows or test_x.ndim != 2 or test_x.shape[1] != cols:
            raise KernelError(
                f"bands: {rows} train rows x {cols} features against {train_y.shape[0]} train values "
                f"and test rows of shape {test_x.shape}"
            )
        # The package's preprocessor drops constant columns (mean-imputed
        # first) and its member generator fails on an empty set with a
        # message about sequences; say what happened instead.
        imputed = np.where(np.isnan(train_x), np.nanmean(train_x, axis=0), train_x)
        if members > 1 and not np.any(imputed != imputed[0], axis=0).any():
            raise KernelError(
                "bands: every feature column is constant over the training rows — nothing varies to band on"
            )
        if members == 1:
            est = self._regressor(n_estimators=1, norm_methods="none", feat_shuffle_method="none", random_state=0)
        else:
            est = self._regressor(n_estimators=members, random_state=0)
        try:
            est.fit(train_x, train_y)
            out = est.predict(test_x, output_type=["quantiles", "raw_quantiles"], alphas=alphas)
        except Exception as e:  # the package's own refusals, by their text
            raise KernelError(f"bands: {e}") from e
        n = test_x.shape[0]
        return (
            np.asarray(out["quantiles"], dtype=np.float64).reshape(n, len(alphas)),
            np.asarray(out["raw_quantiles"], dtype=np.float64).reshape(n, -1),
        )

    def band_point(self, train_x, train_y, test_x, alphas, actual) -> tuple[list[float], float]:
        """The walk point as the fixtures pin it: the pinned member, and
        the PIT as the share of the raw grid at or under the actual."""
        quantiles, grid = self.bands(train_x, train_y, test_x[None, :], alphas, members=1)
        pit = float(np.count_nonzero(grid[0] <= actual)) / (grid.shape[1] + 1)
        return quantiles[0].tolist(), pit

    def band_grid(self, train_x, train_y, test_x, alphas) -> np.ndarray:
        """The replay grid as the fixtures pin it: the default ensemble."""
        return self.bands(train_x, train_y, test_x, alphas, members=8)[0]

    def misfit(
        self,
        x: np.ndarray,
        *,
        perms: list[np.ndarray] | None = None,
        random_state: int = 0,
        train: np.ndarray | None = None,
        workers: int | None = None,
    ) -> np.ndarray:
        """Mean log density per row over the orderings in `perms`, fit on
        `train` (default: the frame itself). Log space end to end — the
        package's `score_samples` exponentiates the same mean.

        The orderings default to the port's protocol: identity and
        reverse, so every column conditions both early and late and the
        read costs two conditionals per column. Not the package's own
        `score_samples`, whose latin shuffler ignores the count it is
        asked for and returns one ordering per column — sixteen on a
        sixteen-column frame, eight times the work.

        The chain rule is one conditional per column per ordering, each
        independent given the frame: the same fits the package runs one
        after another (`_compute_log_density`), dispatched here over
        `workers` threads so the accelerator is not idle while the next
        conditional is prepared. Every conditional is computed exactly as
        in the sequential loop, and the sums are taken in the loop's
        order, so the numbers do not depend on the schedule. The one
        stateful thing in that loop is the dummy column for the first
        conditional of each ordering, drawn from one random stream in
        loop order; it is drawn here in that order before dispatch.
        """
        rows, cols = x.shape
        if rows < 2 or cols < 2:
            raise KernelError(f"misfit: {rows} rows x {cols} features — the read needs two of each")
        u = _Unsupervised(
            self.model,
            n_estimators=1,
            categorical_features=[],
            random_state=random_state,
            device=self.device,
            estimator_params={
                "norm_methods": "none",
                "feat_shuffle_method": "none",
                "use_amp": self.use_amp,
            },
        )
        try:
            u.fit(x if train is None else train)
        except Exception as e:
            raise KernelError(f"misfit: {e}") from e
        xf = x.astype(np.float32)
        rng = np.random.default_rng(random_state)
        if perms is None:
            perms = [np.arange(cols), np.arange(cols)[::-1]]
        perms = [np.asarray(p, dtype=np.int64) for p in perms]
        n_permutations = len(perms)

        # (ordering, column, conditioning columns, train mask, noise) —
        # in the package's loop order, the noise drawn where it draws it.
        tasks = []
        for pi, perm in enumerate(perms):
            for i, col in enumerate(perm):
                col = int(col)
                train_mask = ~np.isnan(u.X_[:, col])
                if train_mask.sum() < u._MIN_SAMPLES_PER_CONDITIONAL:
                    continue
                cond = [int(c) for c in perm[:i]]
                noise = None
                if not cond:
                    noise = (
                        rng.standard_normal((int(train_mask.sum()), 1)).astype(np.float32),
                        rng.standard_normal((rows, 1)).astype(np.float32),
                    )
                tasks.append((pi, col, cond, train_mask, noise))

        def conditional(task):
            pi, col, cond, train_mask, noise = task
            if noise is None:
                x_train, x_test = u.X_[train_mask][:, cond], xf[:, cond]
            else:
                x_train, x_test = noise
            y_train = u.X_[train_mask, col]
            est, _categorical = u._fit_conditional_estimator(col, x_train, y_train)
            return pi, u._log_prob_numerical(est, x_test, xf[:, col])

        workers = workers or misfit_workers(self.device)
        try:
            if workers <= 1:
                results = [conditional(t) for t in tasks]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(conditional, tasks))
        except Exception as e:
            raise KernelError(f"misfit: {e}") from e
        sums = np.zeros((n_permutations, rows))
        for pi, lp in results:  # task order — the loop's summation order
            sums[pi] += lp
        return np.mean(sums, axis=0)


def misfit_workers(device: str) -> int:
    """`GLOSSKERNELS_MISFIT_WORKERS`, else four on CUDA and one elsewhere
    (Metal runs one queue, and the CPU is the bottleneck on the CPU)."""
    named = os.environ.get("GLOSSKERNELS_MISFIT_WORKERS", "").strip()
    if named.isdigit() and int(named) > 0:
        return int(named)
    return 4 if device == "cuda" else 1


# -- the process's one instance ---------------------------------------------

_LOCK = threading.Lock()
_KERNELS: Kernels | None = None


def get() -> Kernels:
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = Kernels()
        return _KERNELS


def peek() -> Kernels | None:
    return _KERNELS
