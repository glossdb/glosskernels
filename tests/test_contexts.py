"""Kept contexts, with the model: the same answer for less, a caller's
own, and gone when the budget says so."""

import numpy as np
import pytest

from glosskernels import kernels
from glosskernels.contexts import Contexts, ContextUnknown, Held
from glosskernels.kernels import Read

ALPHAS = [0.05, 0.5, 0.95]


@pytest.fixture(scope="module")
def k():
    return kernels.Kernels()


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 4))
    return x, x[:, 0] * 2.0 + rng.normal(scale=0.3, size=300), rng.normal(size=(5, 4))


@pytest.mark.parametrize("members", [1, 4])
def test_a_kept_context_answers_as_the_rows_do(k, panel, members):
    x, y, q = panel
    (rows,) = k.answer([Read(q, x, y)], [ALPHAS], members)
    (kept,) = k.answer([Read(q, x, y, cache=True, caller="a")], [ALPHAS], members)
    assert rows[2] is None and isinstance(kept[2], str)
    assert np.allclose(kept[0], rows[0], rtol=2e-3, atol=2e-3) and kept[1].shape == rows[1].shape
    # By its id, other query rows and other levels, without the rows.
    (again,) = k.answer([Read(q[:2], context=kept[2], caller="a")], [[0.5]], members)
    assert np.allclose(again[0][:, 0], rows[0][:2, 1], rtol=2e-3, atol=2e-3)
    # The same rows again find it rather than build it.
    builds = k.contexts.stats["builds"]
    k.answer([Read(q, x, y, cache=True, caller="a")], [ALPHAS], members)
    assert k.contexts.stats["builds"] == builds


def test_a_context_is_its_callers_alone(k, panel):
    x, y, q = panel
    (kept,) = k.answer([Read(q, x, y, cache=True, caller="a")], [ALPHAS], 1)
    (other,) = k.answer([Read(q, context=kept[2], caller="b")], [ALPHAS], 1)
    assert isinstance(other, ContextUnknown) and "send the rows again" in str(other)
    # And refusing one read leaves the cycle's others answered.
    good, bad = k.answer([Read(q, context=kept[2], caller="a"), Read(q[:, :2], context=kept[2], caller="a")], [ALPHAS] * 2, 1)
    assert not isinstance(good, Exception) and "features" in str(bad)


def test_the_budget_evicts_the_least_recently_read_a_callers_own_first():
    held = lambda mb: Held(object(), mb << 20, 10, 2)  # noqa: E731
    c = Contexts(budget=100 << 20, caller_share=0.5)
    c.put("a", "one", held(20))
    c.put("a", "two", held(20))
    c.put("b", "theirs", held(40))
    c.get("a", "one")  # read again: "two" is now a's oldest
    c.put("a", "three", held(20))  # a would hold 60 of its 50
    assert c.has("a", "one") and c.has("a", "three") and not c.has("a", "two") and c.has("b", "theirs")
    with pytest.raises(ContextUnknown):
        c.get("a", "two")
    assert not c.fits(51 << 20) and c.fits(50 << 20)
    assert c.drop("a") == 2 and c.nbytes == 40 << 20


def test_a_context_too_large_to_keep_is_refused_with_what_to_do(k, panel, monkeypatch):
    x, y, q = panel
    monkeypatch.setattr(k.contexts, "budget", 1 << 20)
    (refused,) = k.answer([Read(q, x, y, cache=True, caller="a")], [ALPHAS], 8)
    assert isinstance(refused, kernels.KernelError) and "fewer members" in str(refused)
