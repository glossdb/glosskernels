"""The wire, without the model: nulls, shapes, keys, errors."""

import os

import httpx
import numpy as np
import pytest

from glosskernels import app as app_module
from glosskernels import kernels


class Fake:
    device = "fake"

    def bands_many(self, reads, alphas, members):
        # Each quantile is its own level plus the member count; the grid is 1..9.
        return [
            (np.tile(np.asarray(alphas) + members, (test_x.shape[0], 1)), np.tile(np.arange(1.0, 10.0), (test_x.shape[0], 1)))
            for _train_x, _train_y, test_x in reads
        ]

    def misfit(self, x):
        return np.array([0.1, float("nan")])


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
