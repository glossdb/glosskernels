"""Timings at the reads' real shapes, on whatever device this process
has: what one call costs, and what a panel at cube scale costs. Run
locally with `python -m glosskernels.measure`; Modal's entrypoint
calls `run` on the GPU (modal_app.py).

With `context_rows` the run adds the two questions a shared instance
turns on: what a cached context costs to build, hold and query against
refitting it per call (`contexts`), and what the reads do under
concurrent callers with and without the service's one lock
(`concurrency`)."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .kernels import Kernels, misfit_workers as default_misfit_workers

ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]


def _sync(k: Kernels) -> None:
    if k.device == "cuda":
        torch.cuda.synchronize()


class _Utilization:
    """The GPU's busy fraction while the block runs, sampled from NVML
    (`torch.cuda.utilization`); None off CUDA or without NVML."""

    def __init__(self, k: Kernels, every_s: float = 0.05):
        self.on = k.device == "cuda"
        self.every_s = every_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                self.samples.append(torch.cuda.utilization())
            except Exception:  # no NVML on this host
                return

    def __enter__(self) -> "_Utilization":
        if self.on:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def mean(self) -> float | None:
        return round(float(np.mean(self.samples)), 1) if self.samples else None


def contexts(k: Kernels, rows_list: list[int], cols: int = 20, queries: int = 100, members: int = 8) -> list[dict]:
    """A panel as a context held across calls: for each size, the fit
    (with the cache build, where there is one), what the cache holds,
    and a hundred-row and a one-row query — uncached, `repr` and `kv`.
    A failure is an answer here (the size this device cannot hold), so
    it is recorded and the run goes on."""
    rng = np.random.default_rng(1)
    cuda = k.device == "cuda"
    out = []
    for rows in rows_list:
        x, y, q = rng.normal(size=(rows, cols)), rng.normal(size=rows), rng.normal(size=(queries, cols))
        for mode in (False, "repr", "kv"):
            entry: dict = {"rows": rows, "cols": cols, "members": members, "cache": mode or "none"}
            est = None
            try:
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                est = k._regressor(n_estimators=members, kv_cache=mode, random_state=0)
                t = time.perf_counter()
                est.fit(x, y)
                _sync(k)
                entry["fit_s"] = round(time.perf_counter() - t, 3)
                if mode:
                    entry["cache_mb"] = sum(c.cache_size_mb() for c in est.model_kv_cache_.values())
                for name, rows_q in ((f"query_{queries}_s", q), ("query_1_s", q[:1])):
                    t = time.perf_counter()
                    est.predict(rows_q, output_type="quantiles", alphas=ALPHAS)
                    _sync(k)
                    entry[name] = round(time.perf_counter() - t, 3)
                if cuda:
                    entry["peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                del est
            out.append(entry)
    return out


def concurrency(k: Kernels, clients: tuple[int, ...] = (1, 4, 16), calls: int = 48) -> list[dict]:
    """Mixed short reads (three walk points to one replay grid) from
    `clients` threads, under one lock as the service runs them today
    and without it. Reads per second, the GPU's busy fraction, and the
    largest deviation from the same reads run one after another — the
    lock can only go if that stays zero."""
    rng = np.random.default_rng(2)
    tx, ty, qx = rng.normal(size=(24, 5)), rng.normal(size=24), rng.normal(size=5)
    gx, gy, gq = rng.normal(size=(108, 5)), rng.normal(size=108), rng.normal(size=(12, 5))

    def point() -> np.ndarray:
        return np.asarray(k.band_point(tx, ty, qx, ALPHAS, 0.1)[0])

    def grid() -> np.ndarray:
        return k.band_grid(gx, gy, gq, ALPHAS).reshape(-1)

    alone = {"point": point(), "grid": grid()}
    lock = threading.Lock()
    out = []
    for n in clients:
        for locked in (True, False):
            # One client needs no lock; and Metal does not survive two
            # threads on one command queue — the process aborts.
            if not locked and (n == 1 or k.device == "mps"):
                continue

            def job(i: int) -> float:
                name, read = ("grid", grid) if i % 4 == 3 else ("point", point)
                if locked:
                    with lock:
                        got = read()
                else:
                    got = read()
                return float(np.max(np.abs(got - alone[name])))

            entry: dict = {"clients": n, "locked": locked, "calls": calls}
            try:
                with _Utilization(k) as util:
                    t = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=n) as pool:
                        devs = list(pool.map(job, range(calls)))
                    _sync(k)
                    entry["reads_per_s"] = round(calls / (time.perf_counter() - t), 2)
                entry["max_dev_from_alone"] = max(devs)
                entry["gpu_util_pct"] = util.mean
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.append(entry)
    return out


def batching(k: Kernels, sizes: tuple[int, ...] = (1, 8, 64, 256, 1024), rows: int = 24, cols: int = 5) -> list[dict]:
    """Why a bigger GPU does not speed up a small read, and what does: a
    walk point's forward pass is a long chain of tiny kernels the host
    launches one by one, the device idle between them. The model takes a
    batch of same-shaped tables, so many reads can ride one chain: the
    forward alone at each batch size, the tables it answers per second,
    how busy the device is, and how far a table's answer moves when it
    is batched with others (it should not)."""
    rng = np.random.default_rng(3)
    est = k._regressor(n_estimators=1, norm_methods="none", feat_shuffle_method="none", random_state=0)
    est.fit(rng.normal(size=(rows, cols)), rng.normal(size=rows))
    config, device = est.inference_config_, est.device_
    x = torch.from_numpy(rng.normal(size=(max(sizes), rows + 1, cols))).float().to(device)
    y = torch.from_numpy(rng.normal(size=(max(sizes), rows))).float().to(device)
    out = []
    with torch.no_grad():
        alone = torch.cat([k.model(x[i : i + 1], y[i : i + 1], inference_config=config) for i in range(8)])
        for b in sizes:
            k.model(x[:b], y[:b], inference_config=config)  # warm
            _sync(k)
            repeats = max(3, 64 // b)
            with _Utilization(k) as util:
                t = time.perf_counter()
                for _ in range(repeats):
                    got = k.model(x[:b], y[:b], inference_config=config)
                _sync(k)
                per = (time.perf_counter() - t) / repeats
            entry = {"tables": b, "forward_ms": round(per * 1000, 2), "tables_per_s": round(b / per, 1), "gpu_util_pct": util.mean}
            if b >= 8:
                entry["max_dev_from_alone"] = float((got[:8] - alone).abs().max())
            out.append(entry)
    return out


def callers(k: Kernels, clients: int = 16, requests: int = 8, points: int = 6) -> dict:
    """Many callers, small walks: `clients` threads each send `requests`
    walks of `points` reads (a metric's six months, the context growing a
    row a month). One after another, as a service answering a request at
    a time would; then through the owner's queue, where what is waiting
    rides together."""
    from .kernels import Read
    from .owner import Owner

    rng = np.random.default_rng(4)

    def walk():
        return [(rng.normal(size=(18 + m, 5)), rng.normal(size=18 + m), rng.normal(size=(1, 5))) for m in range(points)]

    work = [[walk() for _ in range(requests)] for _ in range(clients)]
    total = clients * requests * points
    from .prepare import warm

    warm()  # the pool's start is a service's start, not a request's
    k.bands_many([read for reads in work[0] for read in reads] * 2, ALPHAS)  # and the shapes, batched
    out: dict = {"clients": clients, "requests_each": requests, "reads": total}

    with _Utilization(k) as util:
        t = time.perf_counter()
        for mine in work:
            for reads in mine:
                k.bands_many(reads, ALPHAS)
        _sync(k)
        took = time.perf_counter() - t
    out["one_request_at_a_time"] = {"s": round(took, 2), "reads_per_s": round(total / took, 1), "gpu_util_pct": util.mean}

    own = Owner(lambda: k)

    def client(i: int) -> float:
        worst = 0.0
        for reads in work[i]:
            t0 = time.perf_counter()
            own.bands(f"caller-{i}", [Read(q, x, y) for x, y, q in reads], ALPHAS, 1).result()
            worst = max(worst, time.perf_counter() - t0)
        return worst

    with _Utilization(k) as util:
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            slowest = max(pool.map(client, range(clients)))
        _sync(k)
        took = time.perf_counter() - t
    out["through_the_owner"] = {
        "s": round(took, 2),
        "reads_per_s": round(total / took, 1),
        "gpu_util_pct": util.mean,
        "slowest_request_s": round(slowest, 3),
        "cycles": own.stats["cycles"],
        "largest_cycle_jobs": own.stats["largest_cycle_jobs"],
    }
    return out


def spoken(k: Kernels, series: int = 600, months: int = 36) -> dict:
    """A cycle with every voice: `series` one-step reads (100 metrics x 6
    walked months), TabICL's tables riding together and Chronos-2 taking
    the series in one batch, against TabICL alone."""
    from .kernels import Read
    from .prepare import warm

    rng = np.random.default_rng(6)
    levels = ALPHAS + [round(a, 2) for a in np.arange(1, 100) / 100.0]
    reads = []
    for i in range(series):
        v = 100.0 + np.cumsum(rng.normal(size=months)) + 5.0 * np.sin(np.arange(months) * np.pi / 6)
        rows = 18 + i % 6
        reads.append((rng.normal(size=(rows, 5)), rng.normal(size=rows), rng.normal(size=(1, 5)), v))
    warm()
    out: dict = {"reads": series}
    for name, voices_ in (("tabicl", ("tabicl",)), ("all_three", ("tabicl", "chronos2", "seasonal_naive"))):
        asking = [Read(q, x, y, voices=voices_, history=v, horizon=np.ones(1)) for x, y, q, v in reads]
        k.answer(asking[:12], [levels] * 12, 1)  # warm: the shapes, and Chronos-2's load
        _sync(k)
        t = time.perf_counter()
        got = k.answer(asking, [levels] * series, 1)
        _sync(k)
        refused = [g for g in got if isinstance(g, Exception)]
        out[name] = {"s": round(time.perf_counter() - t, 2), "refused": len(refused), **({"first": str(refused[0])[:160]} if refused else {})}
    return out


def kept(k: Kernels, rows_list: tuple[int, ...] = (5000, 20000), members: int = 8, cols: int = 20, queries: int = 100) -> list[dict]:
    """The context cache as the service runs it (`Kernels.answer`): the
    build, a query by id, and how far the kept read stands from the fp32
    read of the rows — in float32, and in float16 where the cache is
    half the size. Deviations are in units of the target's spread."""
    from .kernels import Read

    rng = np.random.default_rng(5)
    k.contexts.budget = 1 << 50  # measure the device, not the budget
    out = []
    for rows in rows_list:
        x, q = rng.normal(size=(rows, cols)), rng.normal(size=(queries, cols))
        y = x[:, 0] - 0.5 * x[:, 1] * x[:, 2] + rng.normal(scale=0.5, size=rows)
        entry: dict = {"rows": rows, "members": members}
        k.use_amp = False
        (reference,) = k.answer([Read(q, x, y)], [ALPHAS], members)
        for name, amp in (("fp32", False), ("fp16", True)):
            k.use_amp = amp
            k.contexts.drop("measure")
            try:
                if k.device == "cuda":
                    torch.cuda.empty_cache()
                t = time.perf_counter()
                (built,) = k.answer([Read(q, x, y, cache=True, caller="measure")], [ALPHAS], members)
                _sync(k)
                if isinstance(built, Exception):
                    raise built
                build_s = time.perf_counter() - t
                t = time.perf_counter()
                for _ in range(5):
                    (got,) = k.answer([Read(q, context=built.context, caller="measure")], [ALPHAS], members)
                _sync(k)
                dev = np.abs(got.voices["tabicl"][0] - reference.voices["tabicl"][0]).max(axis=1) / float(np.std(y))
                entry[name] = {
                    "build_s": round(build_s, 2),
                    "query_s": round((time.perf_counter() - t) / 5, 3),
                    "cache_mb": k.contexts.of("measure") >> 20,
                    "dev_median": float(np.median(dev)),
                    "dev_p99": float(np.quantile(dev, 0.99)),
                    "dev_max": float(dev.max()),
                }
            except Exception as e:
                entry[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        k.use_amp = False
        k.contexts.drop("measure")
        out.append(entry)
    return out


def run(
    panel_rows: int = 5000,
    use_amp: bool = False,
    misfit_workers: int | None = None,
    context_rows: list[int] | None = None,
) -> dict:
    t0 = time.perf_counter()
    k = Kernels(use_amp=use_amp)
    out = {
        "device": k.device,
        "gpu": torch.cuda.get_device_name(0) if k.device == "cuda" else None,
        "amp": use_amp,
        "misfit_workers": misfit_workers or default_misfit_workers(k.device),
        "torch": torch.__version__,
        "load_s": round(time.perf_counter() - t0, 3),
    }
    rng = np.random.default_rng(0)
    cuda = k.device == "cuda"

    def timed(name, fn, n=1, warm=True):
        if warm:
            fn()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        if cuda:
            torch.cuda.synchronize()
        out[name] = {"s_per_call": round((time.perf_counter() - t) / n, 4)}
        if cuda:
            out[name]["peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # A walk point: two years of months, the five walk features.
    tx, ty, qx = rng.normal(size=(24, 5)), rng.normal(size=24), rng.normal(size=5)
    timed("band_point_24x5", lambda: k.band_point(tx, ty, qx, ALPHAS, 0.1), n=6)
    # A replay grid: nine worlds by twelve months, four levers and the month index.
    gx, gy, gq = rng.normal(size=(108, 5)), rng.normal(size=108), rng.normal(size=(12, 5))
    timed("band_grid_108x5_to_12", lambda: k.band_grid(gx, gy, gq, ALPHAS), n=2)
    # A misfit frame at the door's cap.
    mx = rng.normal(size=(2000, 16))
    timed("misfit_2000x16", lambda: k.misfit(mx, workers=misfit_workers), n=1, warm=False)
    # A panel at cube scale: the ensemble read over it, a hundred queries.
    px, py, pq = rng.normal(size=(panel_rows, 20)), rng.normal(size=panel_rows), rng.normal(size=(100, 20))
    timed(f"band_grid_panel_{panel_rows}x20_to_100", lambda: k.band_grid(px, py, pq, ALPHAS), n=1, warm=False)

    if context_rows:
        out["concurrency"] = concurrency(k)
        out["contexts"] = contexts(k, context_rows)
    return out


def _rows(named: str | None) -> list[int] | None:
    return [int(r) for r in named.split(",") if r.strip()] if named else None


if __name__ == "__main__":
    import os

    workers = os.environ.get("GLOSSKERNELS_MISFIT_WORKERS")
    print(
        json.dumps(
            run(
                use_amp=os.environ.get("GLOSSKERNELS_AMP") == "1",
                misfit_workers=int(workers) if workers else None,
                context_rows=_rows(os.environ.get("GLOSSKERNELS_CONTEXT_ROWS")),
            ),
            indent=1,
        )
    )
