"""The owner's queue: callers answered together, in turn, and refused
before memory is."""

import threading
import time

import numpy as np
import pytest

from glosskernels.kernels import Answered, KernelError, Read
from glosskernels.owner import Busy, Owner, TooLarge


class Gated:
    """A kernel that holds each cycle until told, and records what rode together."""

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.cycles: list[list[float]] = []

    def answer(self, reads, alphas, members):
        self.entered.set()
        self.gate.wait(5)
        self.cycles.append([float(r.train_x[0, 0]) for r in reads])  # each read is tagged in its first cell
        return [
            KernelError(f"bands: read {i}: no")
            if r.train_x[0, 0] < 0
            else Answered({"tabicl": (np.full((1, len(a)), r.train_x[0, 0]), np.zeros((1, 3)))})
            for i, (r, a) in enumerate(zip(reads, alphas))
        ]


def read(tag: float, rows: int = 4):
    x = np.zeros((rows, 2))
    x[0, 0] = tag
    return Read(np.zeros((1, 2)), x, np.zeros(rows))


def held(kernel: Gated, **kw) -> Owner:
    """An owner already inside a cycle, so what is submitted next waits together."""
    own = Owner(lambda: kernel, **kw)
    own.bands("warm", [read(0.0)], [0.5], 1)
    assert kernel.entered.wait(5)
    return own


def test_waiting_callers_ride_one_cycle_and_get_their_own_answers():
    kernel = Gated()
    own = held(kernel)
    a = own.bands("a", [read(1.0), read(2.0)], [0.1, 0.9], 1)
    b = own.bands("b", [read(3.0)], [0.5], 1)
    kernel.gate.set()
    assert [got.voices["tabicl"][0][0].tolist() for got in a.result(5)] == [[1.0, 1.0], [2.0, 2.0]]
    assert [got.voices["tabicl"][0][0].tolist() for got in b.result(5)] == [[3.0]]
    assert kernel.cycles[1] == [1.0, 2.0, 3.0]
    assert own.stats["largest_cycle_jobs"] == 2


def test_a_refused_read_fails_its_own_caller_only_and_names_the_callers_index():
    kernel = Gated()
    own = held(kernel)
    good = own.bands("a", [read(1.0)], [0.5], 1)
    bad = own.bands("b", [read(2.0), read(-1.0)], [0.5], 1)
    kernel.gate.set()
    assert good.result(5)[0].voices["tabicl"][0].tolist() == [[1.0]]
    with pytest.raises(KernelError, match="read 1: no"):  # the cycle's index was 2
        bad.result(5)


def test_callers_are_taken_in_turn_within_a_cycles_budget():
    kernel = Gated()
    # A cycle of 1 MB holds 131072 cells; each job below is 40,000.
    own = held(kernel, cycle_mb=1)
    big = lambda tag: read(tag, rows=20_000)  # noqa: E731
    flood = [own.bands("flood", [big(10.0 + i)], [0.5], 1) for i in range(4)]
    late = own.bands("late", [big(99.0)], [0.5], 1)
    kernel.gate.set()
    for f in [*flood, late]:
        f.result(5)
    # Three jobs fit a cycle: the late caller is in the first one, not behind the flood.
    assert kernel.cycles[1] == [10.0, 99.0, 11.0]


def test_ensemble_sizes_do_not_mix_and_an_exclusive_job_runs_alone():
    kernel = Gated()
    own = held(kernel)
    one = own.bands("a", [read(1.0)], [0.5], 1)
    eight = own.bands("b", [read(8.0)], [0.5], 8)
    alone = own.exclusive("c", "misfit", 10, lambda k: "scored")
    kernel.gate.set()
    assert one.result(5) and eight.result(5) and alone.result(5) == "scored"
    assert kernel.cycles[1:] == [[1.0], [8.0]]


def test_the_queue_pushes_back_before_memory_does():
    kernel = Gated()
    own = held(kernel, queue_mb=1)  # 131072 cells in all, half of it a caller's share
    big = lambda: read(1.0, rows=15_000)  # 30,002 cells  # noqa: E731
    own.bands("a", [big()], [0.5], 1)
    own.bands("a", [big()], [0.5], 1)
    with pytest.raises(Busy) as busy:  # a third would pass a's share
        own.bands("a", [big()], [0.5], 1)
    assert busy.value.retry_after >= 1
    own.bands("b", [big()], [0.5], 1)  # another caller still gets in
    with pytest.raises(TooLarge, match="in parts"):
        own.bands("b", [read(1.0, rows=40_000)], [0.5], 1)
    assert own.stats["refused_busy"] == 1
    kernel.gate.set()


def test_a_failed_cycle_recovers_the_device_and_leaves_the_rate_alone():
    """A pass that dies fails its callers, has the kernel hand its memory
    back, and does not teach the rate estimator anything."""

    class Dying:
        def __init__(self):
            self.recovered = 0
            self.die = True

        def answer(self, reads, alphas, members):
            if self.die:
                raise RuntimeError("CUDA out of memory")
            return [Answered({"tabicl": (np.zeros((1, len(a))), np.zeros((1, 3)))}) for a in alphas]

        def recover(self):
            self.recovered += 1

    kernel = Dying()
    own = Owner(lambda: kernel)
    with pytest.raises(RuntimeError, match="out of memory"):
        own.bands("a", [read(1.0)], [0.5], 1).result(5)
    # The callers hear first; the recovery follows once the traceback is dropped.
    for _ in range(100):
        if kernel.recovered:
            break
        time.sleep(0.01)
    assert kernel.recovered == 1 and own.stats["failed"] == 1 and own._rate == 0.0
    kernel.die = False
    assert len(own.bands("a", [read(2.0)], [0.5], 1).result(5)) == 1
    assert own.stats["cycles"] == 2 and own._rate > 0
