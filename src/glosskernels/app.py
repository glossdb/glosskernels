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

import asyncio
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
from .contexts import ContextUnknown
from .owner import Busy, Owner, TooLarge

# Handlers never touch the model: one owner per device takes what is
# waiting from every caller and answers it together (owner.py).
_OWNER: Owner | None = None
_OWNER_LOCK = threading.Lock()

# A body is parsed before the queue can refuse it, so it is bounded first.
MAX_BODY_MB = int(os.environ.get("GLOSSKERNELS_MAX_BODY_MB", "") or 256)


def owner() -> Owner:
    global _OWNER
    with _OWNER_LOCK:
        if _OWNER is None:
            _OWNER = Owner(lambda: kernels.get())
        return _OWNER


class Refusal(Exception):
    """A request the kernel will not serve, with the reason. 4xx."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _keys() -> set[str]:
    raw = os.environ.get("GLOSSKERNELS_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _caller(request: Request) -> str | None:
    """Who is asking — the bearer key, which is also whose share of the
    queue the request counts against; None when the key is not one of ours."""
    keys = _keys()
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not keys:
        return "open"
    return token.strip() if scheme.lower() == "bearer" and token.strip() in keys else None


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
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_MB * 1024 * 1024:
        raise Refusal(f"the body is over {MAX_BODY_MB} MB — send the reads in parts", 413)
    raw = await request.body()
    if len(raw) > MAX_BODY_MB * 1024 * 1024:
        raise Refusal(f"the body is over {MAX_BODY_MB} MB — send the reads in parts", 413)
    try:
        body = await run_in_threadpool(json.loads, raw)
    except json.JSONDecodeError as e:
        raise Refusal(f"the body is not JSON: {e}", 400)
    if not isinstance(body, dict):
        raise Refusal("the body is a JSON object", 400)
    return body


def _door(read):
    """Wrap a read: auth, body, the owner's queue, errors as JSON."""

    async def endpoint(request: Request) -> Response:
        caller = _caller(request)
        if caller is None:
            return JSONResponse({"error": "unauthorized: a bearer key this service issued"}, status_code=401)
        try:
            body = await _body(request)
            args = await run_in_threadpool(read.parse, body)
            return _Answer(await read.serve(owner(), caller, **args))
        except Refusal as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)
        except Busy as e:
            return JSONResponse({"error": str(e)}, status_code=429, headers={"Retry-After": str(e.retry_after)})
        except TooLarge as e:
            return JSONResponse({"error": str(e)}, status_code=413)
        except ContextUnknown as e:
            return JSONResponse({"error": str(e)}, status_code=404)
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
    that places a tied actual. A read with `cache` has its context
    kept and its id returned as `context`; a later read sends that id in
    place of the rows and pays only for its query rows — until the
    context ages out, when the answer is a 404 and the rows are sent
    again. With `pit_history` (a hundred counts of
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
            one = {"test_x": _matrix(read, "test_x", ndim=2), "context": read.get("context"), "cache": read.get("cache", False)}
            if one["context"] is not None:
                if not isinstance(one["context"], str) or "train_x" in read or "train_y" in read:
                    raise Refusal(f"read {i}: `context` is an id this service gave, in place of `train_x` and `train_y`", 400)
            else:
                one |= {"train_x": _matrix(read, "train_x", ndim=2), "train_y": _matrix(read, "train_y", ndim=1)}
            if not isinstance(one["cache"], bool):
                raise Refusal(f"read {i}: `cache` is true or false", 400)
            for name in ("actual", "salt"):
                one[name] = _matrix(read, name, ndim=1) if read.get(name) is not None else None
                if one[name] is not None and one[name].shape[0] != one["test_x"].shape[0]:
                    raise Refusal(f"read {i}: `{name}` has one entry per test row", 400)
            parsed.append(one)
        return {"reads": parsed, "alphas": _alphas(body), "members": members, "history": history}

    @staticmethod
    async def serve(own: Owner, caller: str, reads, alphas, members, history):
        asked = list(alphas)
        if history is not None:
            read_at = calibration.levels(alphas, history, calibration.default_record("tabicl"))
            asked += np.clip(read_at, 0.001, 0.999).tolist()
        answers = []
        asking = [
            kernels.Read(r["test_x"], r.get("train_x"), r.get("train_y"), context=r["context"], cache=r["cache"])
            for r in reads
        ]
        served = await asyncio.wrap_future(own.bands(caller, asking, asked, members))
        for read, (quantiles, grid, context) in zip(reads, served):
            answer = {"quantiles": quantiles[:, : len(alphas)]}
            if history is not None:
                answer = {"quantiles": np.sort(quantiles[:, len(alphas) :], axis=1), "raw": answer["quantiles"]}
            if read["actual"] is not None:
                landed = ~np.isnan(read["actual"])
                pit = np.full(landed.shape[0], np.nan)
                salt = None if read["salt"] is None else read["salt"][landed].astype(np.int64)
                pit[landed] = calibration.pit(grid[landed], read["actual"][landed], salt)
                answer["pit"] = pit
            if context is not None:
                answer["context"] = context
            answers.append(answer)
        return {"reads": answers}


class Misfit:
    """The chain-rule density read over one frame, log space."""

    @staticmethod
    def parse(body):
        return {"x": _matrix(body, "x", ndim=2)}

    @staticmethod
    async def serve(own: Owner, caller: str, x):
        # The chain rule fits a conditional per column per ordering: the frame, that many times.
        waiting = own.exclusive(caller, "misfit", 2 * x.shape[1] * x.size, lambda k: k.misfit(x))
        return {"scores": await asyncio.wrap_future(waiting)}


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
