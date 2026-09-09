"""The wire, without the model: nulls, shapes, keys, errors."""

import os

import httpx
import numpy as np
import pytest

from glosskernels import app as app_module
from glosskernels import kernels


class Fake:
    device = "fake"

    def band_point(self, train_x, train_y, test_x, alphas, actual):
        assert train_x.shape == (3, 2) and np.isnan(train_x[1, 1])
        return [float(a) for a in alphas], 0.5

    def band_grid(self, train_x, train_y, test_x, alphas):
        return np.full((test_x.shape[0], len(alphas)), 1.5)

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


def test_band_point_nulls_ride_as_nan(client):
    r = client.post(
        "/v1/band_point",
        json={
            "train_x": [[1, 2], [3, None], [5, 6]],
            "train_y": [1, 2, 3],
            "test_x": [1, 2],
            "alphas": [0.05, 0.5, 0.95],
            "actual": 2.0,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"quantiles": [0.05, 0.5, 0.95], "pit": 0.5}


def test_band_grid_is_rows_by_alphas(client):
    r = client.post(
        "/v1/band_grid",
        json={"train_x": [[1], [2]], "train_y": [1, 2], "test_x": [[1], [2], [3]], "alphas": [0.1, 0.9]},
    )
    assert r.status_code == 200
    assert r.json() == {"quantiles": [[1.5, 1.5]] * 3}


def test_misfit_non_finite_becomes_null(client):
    r = client.post("/v1/misfit", json={"x": [[1, 2], [3, 4]]})
    assert r.status_code == 200
    assert r.json() == {"scores": [0.1, None]}


def test_shape_and_type_refusals(client):
    r = client.post("/v1/misfit", json={"x": [1, 2]})
    assert r.status_code == 400 and "dimension" in r.json()["error"]
    r = client.post("/v1/band_point", json={"train_x": [[1]], "train_y": [1], "test_x": [1], "alphas": [1.5], "actual": 1})
    assert r.status_code == 400 and "alpha" in r.json()["error"]
    r = client.post("/v1/band_grid", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_keys_gate_when_set(client, monkeypatch):
    monkeypatch.setenv("GLOSSKERNELS_KEYS", "k1, k2")
    body = {"x": [[1, 2], [3, 4]]}
    assert client.post("/v1/misfit", json=body).status_code == 401
    assert client.post("/v1/misfit", json=body, headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/misfit", json=body, headers={"authorization": "Bearer k2"}).status_code == 200
    monkeypatch.delenv("GLOSSKERNELS_KEYS")
    assert client.post("/v1/misfit", json=body).status_code == 200


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["device"] == "fake"
