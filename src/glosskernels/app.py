"""The doors of the kernel service: reads over HTTP, JSON bodies.

One route per kind of read — `/bands` and `/misfit` — matrices as nested
lists, a null where the caller has NaN. Nothing here knows about datasets,
metrics or actors — a request is numbers and a key.

Authentication: when `GLOSSKERNELS_KEYS` names one or more keys
(comma-separated), a request must carry one as `Authorization: Bearer
<key>`; unset, the doors are open (a laptop, or a host that
authenticates in front — Modal's proxy auth). Same header either way.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import calibration, kernels

# One model on one device answers one call at a time: the reads are
# short, and the accelerators do not share a device across threads.
_LOCK = threading.Lock()


class Refusal(Exception):
    """A request the kernel will not serve, with the reason. 4xx."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _keys() -> set[str]:
    raw = os.environ.get("GLOSSKERNELS_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _authorized(request: Request) -> bool:
    keys = _keys()
    if not keys:
        return True
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and token.strip() in keys


def _matrix(body: dict[str, Any], name: str, *, ndim: int) -> np.ndarray:
    """A JSON list as a float64 array, nulls as NaN, shape checked."""
    value = body.get(name)
    if value is None:
        raise Refusal(f"`{name}` is required", 400)
    try:
        arr = np.array(value, dtype=object)
        arr = np.where(arr == None, np.nan, arr).astype(np.float64)  # noqa: E711
    except (TypeError, ValueError) as e:
        raise Refusal(f"`{name}` is not a numeric {'matrix' if ndim == 2 else 'vector'}: {e}", 400)
    if arr.ndim != ndim:
        raise Refusal(f"`{name}` must have {ndim} dimension(s), got {arr.ndim}", 400)
    if arr.size == 0:
        raise Refusal(f"`{name}` is empty", 400)
    return arr


def _scalar(body: dict[str, Any], name: str) -> float:
    value = body.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Refusal(f"`{name}` must be a number", 400)
    return float(value)


def _alphas(body: dict[str, Any]) -> list[float]:
    value = body.get("alphas")
    if not isinstance(value, list) or not value:
        raise Refusal("`alphas` must be a non-empty list of probabilities", 400)
    out = []
    for a in value:
        if not isinstance(a, (int, float)) or not 0.0 < float(a) < 1.0:
            raise Refusal("every alpha lies strictly between 0 and 1", 400)
        out.append(float(a))
    return out


def _json(value: Any) -> Any:
    """Arrays to lists, non-finite floats to null — JSON has no NaN."""
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class _Answer(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(_json(content), allow_nan=False).encode()


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise Refusal(f"the body is not JSON: {e}", 400)
    if not isinstance(body, dict):
        raise Refusal("the body is a JSON object", 400)
    return body


def _door(read):
    """Wrap a read: auth, body, the kernel under the lock, errors as JSON."""

    async def endpoint(request: Request) -> Response:
        if not _authorized(request):
            return JSONResponse({"error": "unauthorized: a bearer key this service issued"}, status_code=401)
        try:
            body = await _body(request)
            args = read.parse(body)

            def run():
                with _LOCK:
                    return read.serve(kernels.get(), **args)

            result = await run_in_threadpool(run)
            return _Answer(result)
        except Refusal as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)
        except kernels.KernelError as e:
            return JSONResponse({"error": str(e)}, status_code=422)

    return endpoint


class Bands:
    """Quantiles for query rows given context rows — every band read:
    a walk point is one read of one row with its actual, a walk is many
    reads, a what-if or a projection is a read of many rows.

    `members` is 1 for the pinned member and more for the ensemble. A
    read's `actual` (per test row, null where none) gets its PIT back,
    always against the raw answer; `salt` names the points for the draw
    that places a tied actual. With `pit_history` (a hundred counts of
    past PITs, zeros for a caller with none yet) the `quantiles` are
    read through that record and the kernel's default one, and `raw`
    carries the answer as the model spoke it."""

    @staticmethod
    def parse(body):
        reads = body.get("reads")
        if not isinstance(reads, list) or not reads:
            raise Refusal("`reads` must be a non-empty list", 400)
        members = body.get("members", 1)
        if not isinstance(members, int) or isinstance(members, bool) or not 1 <= members <= 32:
            raise Refusal("`members` is an integer from 1 to 32", 400)
        history = None
        if body.get("pit_history") is not None:
            history = _matrix(body, "pit_history", ndim=1)
            if history.shape != (calibration.BINS,) or np.isnan(history).any() or (history < 0).any():
                raise Refusal(f"`pit_history` is {calibration.BINS} non-negative counts", 400)
        parsed = []
        for i, read in enumerate(reads):
            if not isinstance(read, dict):
                raise Refusal(f"read {i} is a JSON object", 400)
            one = {name: _matrix(read, name, ndim=ndim) for name, ndim in (("train_x", 2), ("train_y", 1), ("test_x", 2))}
            for name in ("actual", "salt"):
                one[name] = _matrix(read, name, ndim=1) if read.get(name) is not None else None
                if one[name] is not None and one[name].shape[0] != one["test_x"].shape[0]:
                    raise Refusal(f"read {i}: `{name}` has one entry per test row", 400)
            parsed.append(one)
        return {"reads": parsed, "alphas": _alphas(body), "members": members, "history": history}

    @staticmethod
    def serve(k: kernels.Kernels, reads, alphas, members, history):
        asked = list(alphas)
        if history is not None:
            read_at = calibration.levels(alphas, history, calibration.default_record("tabicl"))
            asked += np.clip(read_at, 0.001, 0.999).tolist()
        answers = []
        served = k.bands_many([(r["train_x"], r["train_y"], r["test_x"]) for r in reads], asked, members)
        for read, (quantiles, grid) in zip(reads, served):
            answer = {"quantiles": quantiles[:, : len(alphas)]}
            if history is not None:
                answer = {"quantiles": np.sort(quantiles[:, len(alphas) :], axis=1), "raw": answer["quantiles"]}
            if read["actual"] is not None:
                landed = ~np.isnan(read["actual"])
                pit = np.full(landed.shape[0], np.nan)
                salt = None if read["salt"] is None else read["salt"][landed].astype(np.int64)
                pit[landed] = calibration.pit(grid[landed], read["actual"][landed], salt)
                answer["pit"] = pit
            answers.append(answer)
        return {"reads": answers}


class Misfit:
    """The chain-rule density read over one frame, log space."""

    @staticmethod
    def parse(body):
        return {"x": _matrix(body, "x", ndim=2)}

    @staticmethod
    def serve(k: kernels.Kernels, **a):
        return {"scores": k.misfit(**a)}


async def healthz(_: Request) -> Response:
    k = kernels.peek()
    return JSONResponse({"status": "ok", "device": k.device if k else None, "loaded": k is not None})


app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Route("/bands", _door(Bands), methods=["POST"]),
        Route("/misfit", _door(Misfit), methods=["POST"]),
    ]
)
