"""The wire, without the model: nulls, shapes, keys, errors."""

import os

import httpx
import numpy as np
import pytest

from glosskernels import app as app_module
from glosskernels import kernels
from glosskernels.contexts import ContextUnknown


class Fake:
    device = "fake"

    def answer(self, reads, alphas, members):
        # Each quantile is its own level plus the member count; the grid is 1..9.
        out = []
        for read, levels in zip(reads, alphas):
            if read.context == "gone":
                out.append(ContextUnknown("context_unknown: gone is not held here — send the rows again"))
                continue
            n = read.test_x.shape[0]
            kept = read.context or ("kept-for-" + read.caller if read.cache else None)
            spoken = {"tabicl": (np.tile(np.asarray(levels) + members, (n, 1)), np.tile(np.arange(1.0, 10.0), (n, 1)))}
            for at, name in enumerate(read.voices[1:], start=1):  # each further voice 10 higher than the last
                q = np.tile(np.asarray(levels) + members + 10.0 * at, (n, 1))
                spoken[name] = (q, q)
            out.append(kernels.Answered(spoken, kept))
        return out

    def misfit(self, x, columns=False, folds=1):
        scores = np.array([0.1, float("nan")])
        return (scores, np.array([[0.3, -0.2], [float("nan"), 1.0]])) if columns else scores


class _Sync:
    """httpx's ASGI transport is async; the tests read better synchronous."""

    def __init__(self):
        self.transport = httpx.ASGITransport(app=app_module.app)

    def _run(self, method, path, **kw):
        import asyncio

        async def go():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://kernel") as c:
                return await c.request(method, path, **kw)

        return asyncio.run(go())

    def get(self, path, **kw):
        return self._run("GET", path, **kw)

    def post(self, path, **kw):
        return self._run("POST", path, **kw)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(kernels, "get", lambda: Fake())
    monkeypatch.setattr(kernels, "peek", lambda: Fake())
    monkeypatch.setattr(app_module, "_OWNER", None)
    return _Sync()


READ = {"train_x": [[1, 2], [3, None], [5, 6]], "train_y": [1, 2, 3], "test_x": [[1, 2], [3, 4]]}


def test_bands_answers_every_read_and_pits_the_actuals(client):
    body = {"alphas": [0.1, 0.9], "reads": [READ | {"actual": [4.5, None]}, READ], "members": 8}
    r = client.post("/bands", json=body)
    assert r.status_code == 200, r.text
    first, second = r.json()["reads"]
    assert first["quantiles"] == [[8.1, 8.9]] * 2
    assert first["pit"] == [0.4, None]  # four of the nine grid points under 4.5, over ten
    assert second == {"quantiles": [[8.1, 8.9]] * 2}


def test_bands_reads_through_a_record_and_keeps_the_raw_answer(client):
    # Every past actual fell in the top hundredth: the record reads every band far higher.
    history = [0] * 99 + [5000]
    r = client.post("/bands", json={"alphas": [0.1, 0.9], "reads": [READ], "pit_history": history})
    assert r.status_code == 200, r.text
    (read,) = r.json()["reads"]
    assert read["raw"] == [[1.1, 1.9]] * 2
    assert all(0.99 <= q - 1 <= 0.999 for q in read["quantiles"][0])
    r = client.post("/bands", json={"alphas": [0.5], "reads": [READ], "pit_history": [1, 2]})
    assert r.status_code == 400 and "100" in r.json()["error"]


def test_misfit_non_finite_becomes_null(client):
    r = client.post("/misfit", json={"x": [[1, 2], [3, 4]]})
    assert r.status_code == 200
    assert r.json() == {"scores": [0.1, None]}
    r = client.post("/misfit", json={"x": [[1, 2], [3, 4]], "columns": True})
    assert r.json() == {"scores": [0.1, None], "columns": [[0.3, -0.2], [None, 1.0]]}
    r = client.post("/misfit", json={"x": [[1, 2], [3, 4]], "columns": "yes"})
    assert r.status_code == 400 and "columns" in r.json()["error"]
    r = client.post("/misfit", json={"x": [[1, 2], [3, 4]], "folds": 0})
    assert r.status_code == 400 and "folds" in r.json()["error"]


def test_shape_and_type_refusals(client):
    r = client.post("/misfit", json={"x": [1, 2]})
    assert r.status_code == 400 and "dimension" in r.json()["error"]
    r = client.post("/bands", json={"reads": [READ], "alphas": [1.5]})
    assert r.status_code == 400 and "alpha" in r.json()["error"]
    r = client.post("/bands", json={"reads": [READ | {"actual": [1.0]}], "alphas": [0.5]})
    assert r.status_code == 400 and "per test row" in r.json()["error"]
    r = client.post("/bands", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_keys_gate_when_set(client, monkeypatch):
    monkeypatch.setenv("GLOSSKERNELS_KEYS", "k1, k2")
    body = {"x": [[1, 2], [3, 4]]}
    assert client.post("/misfit", json=body).status_code == 401
    assert client.post("/misfit", json=body, headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.post("/misfit", json=body, headers={"authorization": "Bearer k2"}).status_code == 200
    monkeypatch.delenv("GLOSSKERNELS_KEYS")
    assert client.post("/misfit", json=body).status_code == 200


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["device"] == "fake"


def test_a_full_queue_answers_429_with_retry_after(client, monkeypatch):
    from glosskernels.owner import Busy

    class Full:
        def bands(self, *_a, **_k):
            raise Busy("the kernel's queue is full — retry shortly", retry_after=7)

    monkeypatch.setattr(app_module, "_OWNER", Full())
    r = client.post("/bands", json={"alphas": [0.5], "reads": [READ]})
    assert r.status_code == 429 and r.headers["retry-after"] == "7" and "full" in r.json()["error"]


def test_a_body_over_the_cap_is_refused_before_it_is_parsed(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BODY_MB", 0)
    r = client.post("/bands", json={"alphas": [0.5], "reads": [READ]})
    assert r.status_code == 413 and "in parts" in r.json()["error"]


def test_a_kept_context_is_named_and_read_by_its_id(client):
    r = client.post("/bands", json={"alphas": [0.5], "reads": [READ | {"cache": True}]})
    assert r.status_code == 200 and r.json()["reads"][0]["context"] == "kept-for-open"
    r = client.post("/bands", json={"alphas": [0.5], "reads": [{"context": "kept-for-open", "test_x": [[1, 2]]}]})
    assert r.status_code == 200 and r.json()["reads"][0] == {"quantiles": [[1.5]], "context": "kept-for-open"}
    r = client.post("/bands", json={"alphas": [0.5], "reads": [{"context": "gone", "test_x": [[1, 2]]}]})
    assert r.status_code == 404 and "context_unknown" in r.json()["error"]
    r = client.post("/bands", json={"alphas": [0.5], "reads": [READ | {"context": "x"}]})
    assert r.status_code == 400 and "in place of" in r.json()["error"]


def test_voices_answer_on_their_own_beside_their_blend(client):
    body = {
        "alphas": [0.1, 0.9],
        "voices": ["tabicl", "chronos2", "seasonal_naive"],
        "reads": [READ | {"history": [1, 2, 3, 4, 5, 6, 7, 8], "actual": [1.505, None]}],
    }
    r = client.post("/bands", json=body)
    assert r.status_code == 200, r.text
    (read,) = r.json()["reads"]
    assert set(read) == {"voices", "blend"} and set(read["voices"]) == {"tabicl", "chronos2", "seasonal_naive"}
    assert read["voices"]["tabicl"]["quantiles"] == [[1.1, 1.9]] * 2
    assert read["voices"]["chronos2"]["quantiles"] == [[11.1, 11.9]] * 2
    assert read["blend"]["quantiles"] == [[11.1, 11.9]] * 2  # the mean of 1.x, 11.x and 21.x
    # Each voice's PIT is read off its own percentiles: 1.505 sits just over tabicl's median, under the others' floors.
    assert read["voices"]["tabicl"]["pit"] == [0.5, None] and read["voices"]["chronos2"]["pit"] == [0.0, None]


def test_voices_are_read_through_their_own_records(client):
    body = {
        "alphas": [0.5],
        "voices": ["tabicl", "chronos2"],
        "reads": [READ | {"history": [1, 2, 3, 4, 5, 6, 7, 8], "horizon": [3, 3]}],
        "pit_history": {"chronos2": [0] * 99 + [5000]},  # its actuals always landed at the very top
    }
    (read,) = client.post("/bands", json=body).json()["reads"]
    assert "raw" not in read["voices"]["tabicl"] and read["voices"]["chronos2"]["raw"] == [[11.5]] * 2
    assert read["voices"]["chronos2"]["quantiles"][0][0] > 11.98


def test_voices_refusals(client):
    base = {"alphas": [0.5], "reads": [READ | {"history": [1, 2, 3]}]}
    r = client.post("/bands", json=base | {"voices": ["chronos2"]})
    assert r.status_code == 400 and "tabicl" in r.json()["error"]
    r = client.post("/bands", json={"alphas": [0.5], "voices": ["tabicl", "chronos2"], "reads": [READ]})
    assert r.status_code == 400 and "history" in r.json()["error"]
    r = client.post("/bands", json=base | {"voices": ["tabicl", "chronos2"], "pit_history": [0] * 100})
    assert r.status_code == 400 and "by voice" in r.json()["error"]
    r = client.post("/bands", json={"alphas": [0.5], "voices": ["tabicl", "chronos2"], "reads": [READ | {"history": [1, 2, 3], "horizon": [0, 1]}]})
    assert r.status_code == 400 and "horizon" in r.json()["error"]
