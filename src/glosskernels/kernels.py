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
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabicl import TabICLRegressor, TabICLUnsupervised

from . import voices
from .contexts import Contexts, ContextUnknown, Held, budget_bytes, context_id
from .prepare import prepare_many, warm

HUB_REPO = "jingang/TabICL"
REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


# Tables of one shape a forward pass carries: an L4 is saturated at ~256
# walk-sized tables, and a pass is kept under a cell budget as tables grow.
_TABLES_PER_PASS = 256
_CELLS_PER_PASS = 4_000_000
# Precision is the kernel's business. Small reads are float32: exact, and
# faster (autocast's casts outweigh the arithmetic on a small table, and
# there float16 is percent-level wrong one fit in a hundred). A kept
# context of this many rows or more is built in float16 on CUDA, where it
# is half the memory and a third of the build: measured on an L4 at 5k
# and 20k rows x 8 members, within 0.0024 of the float32 read in units of
# the target's spread at worst, 0.0006 at the median.
_HALF_FROM_ROWS = 5000


class KernelError(Exception):
    """A read the kernel refuses, with the reason the door reports."""


@dataclass
class Read:
    """One band read: query rows against context rows — the rows
    themselves, or the id of a context this caller had kept."""

    test_x: np.ndarray
    train_x: np.ndarray | None = None
    train_y: np.ndarray | None = None
    context: str | None = None  # in place of the rows
    cache: bool = False  # keep this read's context for later reads
    caller: str = "open"
    # For the voices that read a series: the series up to the origin, how
    # many months past it each query row lies, and its season's length.
    voices: tuple[str, ...] = ("tabicl",)
    history: np.ndarray | None = None
    horizon: np.ndarray | None = None
    season: int = 12


@dataclass
class Answered:
    """A read's answer: per voice (quantiles at the levels asked, the grid
    its PIT is read from), and the id of its context where one is kept."""

    voices: dict[str, tuple[np.ndarray, np.ndarray]]
    context: str | None = None


def fetch_checkpoints() -> None:
    """Pull the regressor and Chronos-2 into the hub cache — the image bakes them."""
    hf_hub_download(repo_id=HUB_REPO, filename=REGRESSOR)
    voices.fetch_checkpoints()


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
        self.contexts = Contexts(budget_bytes(self.device))
        self.chronos = voices.Chronos2(self.device)  # loads on first use

    def _regressor(self, **kwargs: Any) -> _Regressor:
        kwargs.setdefault("use_amp", self.use_amp)
        est = _Regressor(device=self.device, **kwargs)
        est.model_ = self.model
        return est

    # -- the reads --------------------------------------------------------

    def answer(self, reads: list[Read], alphas: list[list[float]], members: int = 1) -> list[Answered | Exception]:
        """Every read of a cycle, each `Answered` or with its refusal in place. Reads that
        bring their rows and keep nothing ride the forward passes
        together; a read against a kept context costs only its query
        rows; a read asking to be kept builds its context once (the
        same rows again find it) and is answered from it."""
        out: list = [None] * len(reads)
        plain = [i for i, r in enumerate(reads) if r.context is None and not r.cache]
        if plain:
            served = self.bands_many(
                [(reads[i].train_x, reads[i].train_y, reads[i].test_x) for i in plain],
                [alphas[i] for i in plain],
                members,
                refusals=True,
            )
            for i, got in zip(plain, served):
                out[i] = got if isinstance(got, Exception) else Answered({"tabicl": (got[0], got[1])})
        for i, read in enumerate(reads):
            if out[i] is not None:
                continue
            try:
                id_ = read.context or self._keep(read, members)
                held = self.contexts.get(read.caller, id_)
                if read.test_x.ndim != 2 or read.test_x.shape[1] != held.cols:
                    raise KernelError(f"bands: test rows of shape {read.test_x.shape} against a context of {held.cols} features")
                got = held.fitted.predict(read.test_x, output_type=["quantiles", "raw_quantiles"], alphas=list(alphas[i]))
                n = read.test_x.shape[0]
                out[i] = Answered(
                    {
                        "tabicl": (
                            np.asarray(got["quantiles"], dtype=np.float64).reshape(n, len(alphas[i])),
                            np.asarray(got["raw_quantiles"], dtype=np.float64).reshape(n, -1),
                        )
                    },
                    id_,
                )
            except (KernelError, ContextUnknown) as e:
                out[i] = e
            except Exception as e:  # the package's own refusals, by their text
                out[i] = KernelError(f"bands: {e}")
        self._series_voices(reads, alphas, out)
        return out

    def _series_voices(self, reads: list[Read], alphas: list[list[float]], out: list) -> None:
        """The voices that read the series, added to the reads that asked
        for them and were answered. Chronos-2 takes every such series of
        the cycle asking the same levels in one batch. A voice's grid is
        its answer at the levels asked (the door asks the percentiles)."""
        asking = [i for i, r in enumerate(reads) if isinstance(out[i], Answered) and len(r.voices) > 1]
        for i in asking:
            read = reads[i]
            if "seasonal_naive" in read.voices:
                try:
                    q = voices.seasonal_naive(read.history, read.horizon, list(alphas[i]), read.season)
                    out[i].voices["seasonal_naive"] = (q, np.sort(q, axis=1))
                except voices.VoiceError as e:
                    out[i] = KernelError(f"bands: {e}")
        by_levels: dict[tuple, list[int]] = {}
        for i in asking:
            if isinstance(out[i], Answered) and "chronos2" in reads[i].voices:
                by_levels.setdefault(tuple(alphas[i]), []).append(i)
        for levels, mine in by_levels.items():
            try:
                got = self.chronos.quantiles([reads[i].history for i in mine], [reads[i].horizon for i in mine], list(levels))
            except Exception:
                # One series a voice will not read must not fail the others: find whose.
                got = []
                for i in mine:
                    try:
                        got.append(self.chronos.quantiles([reads[i].history], [reads[i].horizon], list(levels))[0])
                    except Exception as e:
                        got.append(KernelError(f"bands: {e}"))
            for i, q in zip(mine, got):
                if isinstance(q, Exception):
                    out[i] = q
                else:
                    out[i].voices["chronos2"] = (q, np.sort(q, axis=1))

    def _keep(self, read: Read, members: int) -> str:
        """The read's context, built and kept unless this caller already has it."""
        rows, cols = read.train_x.shape
        if rows < 2 or read.train_y.shape[0] != rows:
            raise KernelError(f"bands: {rows} train rows x {cols} features against {read.train_y.shape[0]} train values")
        id_ = context_id(read.train_x, read.train_y, members, REGRESSOR)
        if self.contexts.has(read.caller, id_):
            return id_
        half = self.use_amp or (self.device == "cuda" and rows >= _HALF_FROM_ROWS)
        # What the package keeps: keys and values of every training row, at
        # each of the ICL stack's 12 layers, 512 wide, per member.
        estimate = rows * members * 12 * 2 * 512 * (2 if half else 4)
        if not self.contexts.fits(estimate):
            raise KernelError(
                f"bands: a context of {rows} rows x {members} members keeps ~{estimate >> 20} MB and a caller's "
                f"share of the cache is {int(self.contexts.budget * self.contexts.caller_share) >> 20} MB — "
                "fewer members, or send it without `cache`"
            )
        if members == 1:
            est = self._regressor(
                n_estimators=1, norm_methods="none", feat_shuffle_method="none", random_state=0, kv_cache="kv", use_amp=half
            )
        else:
            est = self._regressor(n_estimators=members, random_state=0, kv_cache="kv", use_amp=half)
        try:
            est.fit(read.train_x, read.train_y)
        except torch.OutOfMemoryError as e:
            del est
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise KernelError(f"bands: the device ran out of memory building a {rows}-row context: {e}") from e
        nbytes = sum(c.cache_size_mb() + 1 for c in est.model_kv_cache_.values()) << 20
        self.contexts.put(read.caller, id_, Held(est, nbytes, rows, cols))
        return id_

    def bands(self, train_x, train_y, test_x, alphas, members: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """One read: quantiles (test rows x alphas) and the raw quantile
        grid (test rows x grid) for `test_x` given the context rows."""
        return self.bands_many([(train_x, train_y, test_x)], alphas, members)[0]

    def bands_many(
        self,
        reads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        alphas: list[float] | list[list[float]],
        members: int = 1,
        timings: dict | None = None,
        refusals: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray] | KernelError]:
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

        `alphas` is one list for every read, or a list per read: the
        forward pass does not depend on them, so reads asking for
        different levels (each caller's record puts them elsewhere) still
        ride together. With `refusals` a read the kernel refuses comes
        back as its `KernelError` in place, the others answered — reads
        from different callers must not fail each other; without, the
        first refusal is raised. `timings`, when given, is filled with
        where the seconds went."""
        per_read = bool(alphas) and isinstance(alphas[0], (list, tuple, np.ndarray))
        levels = [tuple(float(a) for a in (alphas[i] if per_read else alphas)) for i in range(len(reads))]
        refused: dict[int, KernelError] = {}

        def refuse(index: int, why: str) -> None:
            refused[index] = KernelError(f"bands: read {index}: {why}")
            if not refusals:
                raise refused[index]

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
                refuse(
                    index,
                    f"{rows} train rows x {cols} features against {train_y.shape[0]} train values "
                    f"and test rows of shape {test_x.shape}",
                )
                continue
            # The package's preprocessor drops constant columns (mean-imputed
            # first) and its member generator fails on an empty set with a
            # message about sequences; say what happened instead.
            imputed = np.where(np.isnan(train_x), np.nanmean(train_x, axis=0), train_x)
            if members > 1 and not np.any(imputed != imputed[0], axis=0).any():
                refuse(index, "every feature column is constant over the training rows — nothing varies to band on")
        sound = [index for index in range(len(reads)) if index not in refused]
        scalers, tables = {}, []  # tables: (read, Xs member-major, ys)
        for index, made in zip(sound, prepare_many([reads[i] for i in sound], members)):
            if isinstance(made, str):
                refuse(index, made)
                continue
            scalers[index] = made[1]
            tables.extend((index, xs, ys) for xs, ys in made[0])
        lap("prepare_s")
        groups: dict[tuple, list[int]] = {}
        for at, (_index, xs, ys) in enumerate(tables):
            groups.setdefault((xs.shape[1:], ys.shape[1:]), []).append(at)
        answered: list[list | None] = [None] * len(tables)
        answered_rows: dict[tuple, tuple] = {}
        for group_key, members_of in groups.items():
            x_shape = group_key[0]
            # As many tables a pass as saturate the device, fewer as they grow.
            per_pass = max(1, min(_TABLES_PER_PASS, _CELLS_PER_PASS // int(np.prod(x_shape))))
            device, config = self._configured(x_shape[0])
            spans = np.cumsum([0] + [tables[at][1].shape[0] for at in members_of])
            xs = np.concatenate([tables[at][1] for at in members_of])
            ys = np.concatenate([tables[at][2] for at in members_of])
            owner = np.repeat([tables[at][0] for at in members_of], np.diff(spans))  # the read of each stacked table
            try:
                with torch.no_grad():
                    for lo in range(0, xs.shape[0], per_pass):
                        raw = self.model._inference_forward(
                            torch.from_numpy(xs[lo : lo + per_pass]).float().to(device),
                            torch.from_numpy(ys[lo : lo + per_pass]).float().to(device),
                            inference_config=config,
                        )
                        # As `predict_stats` reads them: the monotone grid, and
                        # the quantiles at each caller's own levels.
                        by_levels: dict[tuple, list[int]] = {}
                        for row, read in enumerate(owner[lo : lo + per_pass]):
                            by_levels.setdefault(levels[read], []).append(row)
                        for asked, rows_ in by_levels.items():
                            dist = self.model.quantile_dist(raw[rows_])
                            q = dist.icdf(alpha=torch.tensor(asked, device=raw.device, dtype=raw.dtype))
                            grid, q = dist.quantiles.float().cpu().numpy(), q.float().cpu().numpy()
                            for slot, row in enumerate(rows_):
                                answered_rows[(group_key, lo + row)] = (q[slot], grid[slot])
            except Exception as e:
                for read in set(owner.tolist()):
                    refused[read] = KernelError(f"bands: {e}")
                if not refusals:
                    raise KernelError(f"bands: {e}") from e
                continue
            for slot, at in enumerate(members_of):
                answered[at] = [answered_rows.pop((group_key, row)) for row in range(spans[slot], spans[slot + 1])]

        lap("forward_s")
        if timings is not None:
            timings["tables"], timings["shapes"] = len(tables), len(groups)
        mine_of: dict[int, list] = {}
        for at, table in enumerate(tables):
            if answered[at] is not None:
                mine_of.setdefault(table[0], []).extend(answered[at])
        results = []
        for index in range(len(reads)):
            if index in refused:
                results.append(refused[index])
                continue
            pair = []
            for kind in (0, 1):  # quantiles at the levels, the raw grid
                arr = np.stack([member[kind] for member in mine_of[index]])  # (members, test rows, quantiles)
                scaled = scalers[index].inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
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
