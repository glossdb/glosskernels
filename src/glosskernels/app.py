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

from . import calibration, kernels, voices
from .voices import blend as voices_blend
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


# The levels a voice's PIT is read from when several voices answer: the percentiles.
GRID = [round(a, 2) for a in np.arange(1, 100) / 100.0]


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
    again. With `pit_history` (a hundred counts of past PITs, zeros for
    a caller with none yet) the `quantiles` are read through that record
    and the kernel's default one, and `raw` carries the answer as the
    model spoke it.

    With `voices` — `tabicl` and any of `chronos2`, `seasonal_naive` —
    each read also carries `history` (the series up to the origin) and
    may carry `horizon` (per test row, how many months past the history
    it lies; 1 where left out). Every voice answers on its own under
    `voices`, beside their `blend`; `pit_history` is then an object of
    histories by voice name (`blend` among them). The blend is of the
    voices as read — each through its record, where it has one.

    `window` says how many periods each value of the series sums (1, a
    month; 12, a trailing annual total asked for as its own series): it
    picks the default record a history is weighed with."""

    @staticmethod
    def parse(body):
        reads = body.get("reads")
        if not isinstance(reads, list) or not reads:
            raise Refusal("`reads` must be a non-empty list", 400)
        members = body.get("members", 1)
        if not isinstance(members, int) or isinstance(members, bool) or not 1 <= members <= 32:
            raise Refusal("`members` is an integer from 1 to 32", 400)
        asked = body.get("voices")
        if asked is not None and (
            not isinstance(asked, list) or "tabicl" not in asked or len(set(asked)) != len(asked) or set(asked) - set(voices.VOICES)
        ):
            raise Refusal(f"`voices` names `tabicl` and any of {list(voices.SERIES_VOICES)}, each once", 400)
        season = body.get("season", 12)
        if not isinstance(season, int) or isinstance(season, bool) or not 1 <= season <= 366:
            raise Refusal("`season` is the season's length in periods, an integer", 400)
        window = body.get("window", 1)
        if not isinstance(window, int) or isinstance(window, bool) or not 1 <= window <= 366:
            raise Refusal("`window` is how many periods each value sums, an integer", 400)
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
            rows = one["test_x"].shape[0]
            for name in ("actual", "salt", "horizon"):
                one[name] = _matrix(read, name, ndim=1) if read.get(name) is not None else None
                if one[name] is not None and one[name].shape[0] != rows:
                    raise Refusal(f"read {i}: `{name}` has one entry per test row", 400)
            one["history"] = None
            if asked and len(asked) > 1:
                one["history"] = _matrix(read, "history", ndim=1)
                if one["horizon"] is None:
                    one["horizon"] = np.ones(rows)
                h = one["horizon"]
                if np.isnan(h).any() or (h != np.round(h)).any() or (h < 1).any() or (h > 120).any():
                    raise Refusal(f"read {i}: `horizon` is whole periods past the history, 1 to 120", 400)
            parsed.append(one)
        return {
            "reads": parsed,
            "alphas": _alphas(body),
            "members": members,
            "voices": tuple(asked) if asked else None,
            "season": season,
            "window": window,
            "histories": Bands._histories(body.get("pit_history"), asked),
        }

    @staticmethod
    def _histories(value, asked) -> dict[str, np.ndarray] | None:
        """PIT histories by voice; a bare list is TabICL's."""
        if value is None:
            return None
        if asked and len(asked) > 1:
            names = {*asked, "blend"}
            if not isinstance(value, dict) or set(value) - names:
                raise Refusal(f"with `voices`, `pit_history` is an object of histories by voice, any of {sorted(names)}", 400)
        elif isinstance(value, dict):
            raise Refusal("`pit_history` is a list of counts; by voice only with `voices`", 400)
        else:
            value = {"tabicl": value}
        out = {}
        for name, counts in value.items():
            history = _matrix({"pit_history": counts}, "pit_history", ndim=1)
            if history.shape != (calibration.BINS,) or np.isnan(history).any() or (history < 0).any():
                raise Refusal(f"`pit_history` is {calibration.BINS} non-negative counts", 400)
            out[name] = history
        return out

    @staticmethod
    async def serve(own: Owner, caller: str, reads, alphas, members, voices, season, window, histories):
        several = voices is not None and len(voices) > 1
        n = len(alphas)
        levels = list(alphas)
        if several:
            levels += GRID  # every voice's PIT, and its record's reading, come from the percentiles
        elif histories is not None:
            read_at = calibration.levels(alphas, histories["tabicl"], calibration.default_record("tabicl", window))
            levels += np.clip(read_at, 0.001, 0.999).tolist()
        asking = [
            kernels.Read(
                r["test_x"], r.get("train_x"), r.get("train_y"), context=r["context"], cache=r["cache"],
                voices=voices or ("tabicl",), history=r["history"], horizon=r["horizon"], season=season,
            )
            for r in reads
        ]  # fmt: skip
        served = await asyncio.wrap_future(own.bands(caller, asking, levels, members))

        def pits(read, grid):
            landed = ~np.isnan(read["actual"])
            pit = np.full(landed.shape[0], np.nan)
            salt = None if read["salt"] is None else read["salt"][landed].astype(np.int64)
            pit[landed] = calibration.pit(grid[landed], read["actual"][landed], salt)
            return pit

        def through_its_record(name, read, bands, grid):
            """A voice among several: its bands at the alphas, read through
            its record off its percentiles — and those percentiles as read,
            which is what the blend is made of. The PIT is against the raw."""
            answer, as_read = {"quantiles": bands}, grid
            if histories is not None and name in histories:
                default = calibration.default_record(name, window)

                def re_read(levels):
                    read_at = calibration.levels(levels, histories[name], default)
                    return np.stack([np.interp(read_at, GRID, row) for row in grid])

                answer, as_read = {"quantiles": re_read(alphas), "raw": bands}, re_read(GRID)
            if read["actual"] is not None:
                answer["pit"] = pits(read, grid)
            return answer, as_read

        answers = []
        for read, got in zip(reads, served):
            if several:
                spoke = {
                    name: through_its_record(name, read, q[:, :n], np.sort(q[:, n:], axis=1))
                    for name, (q, _grid) in got.voices.items()
                }
                # The bands and the percentiles are two sets of levels: blended apart.
                bands = voices_blend([answer["quantiles"] for answer, _ in spoke.values()])
                percentiles = voices_blend([as_read for _, as_read in spoke.values()])
                answer = {
                    "voices": {name: answer for name, (answer, _) in spoke.items()},
                    "blend": through_its_record("blend", read, bands, percentiles)[0],
                }
            else:
                quantiles, grid = got.voices["tabicl"]
                answer = {"quantiles": quantiles[:, :n]}
                if histories is not None:
                    answer = {"quantiles": np.sort(quantiles[:, n:], axis=1), "raw": answer["quantiles"]}
                if read["actual"] is not None:
                    answer["pit"] = pits(read, grid)
            if got.context is not None:
                answer["context"] = got.context
            answers.append(answer)
        return {"reads": answers}


class Misfit:
    """The chain-rule density read over one frame, log space — per row,
    and with `columns` per cell: the same conditionals, not summed. With
    `folds`, each row is scored from a context it is not in."""

    @staticmethod
    def parse(body):
        columns = body.get("columns", False)
        if not isinstance(columns, bool):
            raise Refusal("`columns` is true or false", 400)
        folds = body.get("folds", 1)
        if not isinstance(folds, int) or isinstance(folds, bool) or not 1 <= folds <= 10:
            raise Refusal("`folds` is a whole number from 1 to 10", 400)
        return {"x": _matrix(body, "x", ndim=2), "columns": columns, "folds": folds}

    @staticmethod
    async def serve(own: Owner, caller: str, x, columns, folds):
        # The chain rule fits a conditional per column per ordering: the frame, that many times — per fold.
        work = 2 * x.shape[1] * x.size * folds
        waiting = own.exclusive(caller, "misfit", work, lambda k: k.misfit(x, columns=columns, folds=folds))
        got = await asyncio.wrap_future(waiting)
        return {"scores": got[0], "columns": got[1]} if columns else {"scores": got}


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
