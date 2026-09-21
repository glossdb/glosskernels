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
import time
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabicl import TabICLRegressor, TabICLUnsupervised

from .prepare import prepare_many, warm

HUB_REPO = "jingang/TabICL"
REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


# Tables of one shape a forward pass carries: an L4 is saturated at ~256
# walk-sized tables, and a pass is kept under a cell budget as tables grow.
_TABLES_PER_PASS = 256
_CELLS_PER_PASS = 4_000_000


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


def _free_without_flushing(manager) -> float:
    """What the package's probe answers, without what it does to get it.

    Before every forward pass (three times a pass: columns, rows, the
    ICL stack) the package sizes its batches from free device memory, and
    to read that it synchronizes the device and empties PyTorch's
    allocator cache — so every pass waits for the device, then buys its
    memory back from the driver. For a walk-sized table that is most of
    the pass. The same number is there without the flush: what the
    driver has free, plus what the allocator holds and is not using."""
    device = manager.exe_device
    free, _total = torch.cuda.mem_get_info(device)
    idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return (free + idle) / (1024 * 1024)


def _probe_memory_cheaply() -> None:
    from tabicl._model.inference import InferenceManager

    flushing = InferenceManager.get_available_gpu_memory

    def available(self) -> float:
        if getattr(self.exe_device, "type", None) == "cuda":
            return _free_without_flushing(self)
        return flushing(self)

    InferenceManager.get_available_gpu_memory = available


if os.environ.get("GLOSSKERNELS_FLUSHING_PROBE", "") != "1":  # the package's own, for a comparison
    _probe_memory_cheaply()


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

    def bands(self, train_x, train_y, test_x, alphas, members: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """One read: quantiles (test rows x alphas) and the raw quantile
        grid (test rows x grid) for `test_x` given the context rows."""
        return self.bands_many([(train_x, train_y, test_x)], alphas, members)[0]

    def bands_many(
        self,
        reads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        alphas: list[float],
        members: int = 1,
        timings: dict | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Many reads, the device kept busy. One member is the pinned one
        (no normalization, no feature shuffle); more is the package's
        ensemble of that size.

        A small read's forward pass is a long chain of tiny kernels the
        host launches one by one — 35 ms on an L4 that is 10% busy, and
        no faster on a bigger card. The model takes a batch of tables of
        one shape, so the reads' tables (a read has one per member) are
        prepared as the package prepares them, grouped by shape, and each
        group rides one chain: 814 tables a second on the same L4, the
        answers within 2e-5 of one at a time. What comes back is rescaled
        and averaged per read exactly as the package's `predict` does.
        `timings`, when given, is filled with where the seconds went."""
        clock = [time.perf_counter()]

        def lap(name: str) -> None:
            if timings is not None:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                timings[name] = round(timings.get(name, 0.0) + now - clock[0], 3)
                clock[0] = now

        for index, (train_x, train_y, test_x) in enumerate(reads):
            rows, cols = train_x.shape
            if rows < 2 or train_y.shape[0] != rows or test_x.ndim != 2 or test_x.shape[1] != cols:
                raise KernelError(
                    f"bands: read {index}: {rows} train rows x {cols} features against "
                    f"{train_y.shape[0]} train values and test rows of shape {test_x.shape}"
                )
            # The package's preprocessor drops constant columns (mean-imputed
            # first) and its member generator fails on an empty set with a
            # message about sequences; say what happened instead.
            imputed = np.where(np.isnan(train_x), np.nanmean(train_x, axis=0), train_x)
            if members > 1 and not np.any(imputed != imputed[0], axis=0).any():
                raise KernelError(
                    f"bands: read {index}: every feature column is constant over the training rows — "
                    "nothing varies to band on"
                )
        scalers, tables = [], []  # tables: (read, Xs member-major, ys)
        for index, made in enumerate(prepare_many(reads, members)):
            if isinstance(made, str):
                raise KernelError(f"bands: read {index}: {made}")
            scalers.append(made[1])
            tables.extend((index, xs, ys) for xs, ys in made[0])
        lap("prepare_s")
        groups: dict[tuple, list[int]] = {}
        for at, (_index, xs, ys) in enumerate(tables):
            groups.setdefault((xs.shape[1:], ys.shape[1:]), []).append(at)
        answered: list[dict[str, np.ndarray] | None] = [None] * len(tables)
        kinds = ["quantiles", "raw_quantiles"]
        for (x_shape, _), members_of in groups.items():
            # As many tables a pass as saturate the device, fewer as they grow.
            per_pass = max(1, min(_TABLES_PER_PASS, _CELLS_PER_PASS // int(np.prod(x_shape))))
            device, config = self._configured(x_shape[0])
            spans = np.cumsum([0] + [tables[at][1].shape[0] for at in members_of])
            xs = np.concatenate([tables[at][1] for at in members_of])
            ys = np.concatenate([tables[at][2] for at in members_of])
            outs = {kind: [] for kind in kinds}
            try:
                with torch.no_grad():
                    for lo in range(0, xs.shape[0], per_pass):
                        out = self.model.predict_stats(
                            torch.from_numpy(xs[lo : lo + per_pass]).float().to(device),
                            torch.from_numpy(ys[lo : lo + per_pass]).float().to(device),
                            output_type=kinds,
                            alphas=alphas,
                            inference_config=config,
                        )
                        for kind in kinds:
                            outs[kind].append(out[kind].float().cpu().numpy())
            except Exception as e:
                raise KernelError(f"bands: {e}") from e
            whole = {kind: np.concatenate(parts) for kind, parts in outs.items()}
            for slot, at in enumerate(members_of):
                answered[at] = {kind: whole[kind][spans[slot] : spans[slot + 1]] for kind in kinds}

        lap("forward_s")
        if timings is not None:
            timings["tables"], timings["shapes"] = len(tables), len(groups)
        results = []
        mine_of: dict[int, list] = {}
        for at, table in enumerate(tables):
            mine_of.setdefault(table[0], []).append(answered[at])
        for index, y_scaler in enumerate(scalers):
            mine = mine_of[index]
            pair = []
            for kind in kinds:
                arr = np.concatenate([m[kind] for m in mine])  # (members, test rows, quantiles)
                scaled = y_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
                pair.append(np.mean(scaled, axis=0).astype(np.float64))
            results.append((pair[0], pair[1]))
        lap("finish_s")
        return results

    def _configured(self, rows: int):
        """The device and inference configuration the package would build
        for a table of `rows` — its own code, without the fit around it."""
        est = self._regressor()
        est._resolve_device()
        est.n_samples_in_ = rows
        est._build_inference_config()
        return est.device_, est.inference_config_

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
            threading.Thread(target=warm, daemon=True).start()
        return _KERNELS


def peek() -> Kernels | None:
    return _KERNELS
