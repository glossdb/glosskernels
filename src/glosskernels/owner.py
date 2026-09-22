"""One owner per device, fed by a bounded queue.

Handlers do not touch the model: they hand the owner a job and await
it. The owner is one thread. When it wakes it takes what is waiting —
reads from every caller's request — and answers them together, so
under load the forward passes fill on their own (28 -> 877 tables a
second on an L4) and an idle service answers at once. There is no
batching window and no added latency.

The queue pushes back before memory does. Waiting jobs are arrays in
host memory, so the bound is on what they hold — cells, not requests —
with a share of it per caller, so one caller's burst cannot take the
queue from the rest. A job that does not fit is refused with how long
to wait (`Busy`, a 429), and one too large ever to fit is refused
outright (`TooLarge`, a 413). Inside a cycle the owner takes callers in
turn, up to a budget of cells, so a cycle's length — and the device
memory a pass needs — is bounded too."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

BYTES_PER_CELL = 8  # the reads wait as float64


def _mb(name: str, default: int) -> int:
    named = os.environ.get(name, "").strip()
    return int(named) if named.isdigit() and int(named) > 0 else default


class Busy(Exception):
    """The queue is full for this caller: come back in `retry_after` seconds."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class TooLarge(Exception):
    """A request larger than the queue will ever hold."""


@dataclass
class _Job:
    caller: str
    kind: str  # "bands", or an exclusive job's name
    cells: int
    future: Future
    reads: list = field(default_factory=list)
    alphas: tuple = ()
    members: int = 1
    run: Callable[[Any], Any] | None = None  # an exclusive job: the kernel in, the answer out


class Owner:
    def __init__(
        self,
        kernels_get: Callable[[], Any],
        queue_mb: int | None = None,
        cycle_mb: int | None = None,
        caller_share: float = 0.5,
    ):
        self._kernels_get = kernels_get
        # What may wait, and what one cycle takes, in cells.
        self.max_cells = (queue_mb or _mb("GLOSSKERNELS_QUEUE_MB", 512)) * 1024 * 1024 // BYTES_PER_CELL
        self.cycle_cells = (cycle_mb or _mb("GLOSSKERNELS_CYCLE_MB", 64)) * 1024 * 1024 // BYTES_PER_CELL
        self.caller_share = caller_share
        self._waiting: OrderedDict[str, deque[_Job]] = OrderedDict()  # caller -> its jobs, oldest first
        self._cells = 0
        self._cells_of: dict[str, int] = {}
        self._rate = 0.0  # cells answered per second, smoothed
        self._wake = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-owner")
        self._thread.start()
        self.stats = {"cycles": 0, "jobs": 0, "refused_busy": 0, "largest_cycle_jobs": 0}

    # -- the callers' side ---------------------------------------------------

    def bands(self, caller: str, reads: list, alphas: list[float], members: int) -> Future:
        """`reads` are `kernels.Read`s; each comes back `kernels.Answered`."""
        for read in reads:
            read.caller = caller
        cells = sum(
            members * (r.test_x.size + (0 if r.train_x is None else r.train_x.size))
            + (0 if r.history is None else r.history.size)
            for r in reads
        )
        return self._admit(_Job(caller, "bands", cells, Future(), reads, tuple(alphas), members))

    def exclusive(self, caller: str, name: str, cells: int, run: Callable[[Any], Any]) -> Future:
        """A job that has the device to itself for its turn (the misfit read)."""
        return self._admit(_Job(caller, name, cells, Future(), run=run))

    def _admit(self, job: _Job) -> Future:
        with self._wake:
            if job.cells > self.max_cells * self.caller_share:
                raise TooLarge(
                    f"the request holds {job.cells * BYTES_PER_CELL >> 20} MB and one caller's share of the "
                    f"queue is {int(self.max_cells * self.caller_share) * BYTES_PER_CELL >> 20} MB — send it in parts"
                )
            mine = self._cells_of.get(job.caller, 0)
            # A caller alone may fill its share; the queue as a whole is never overfilled.
            if self._cells + job.cells > self.max_cells or mine + job.cells > self.max_cells * self.caller_share:
                self.stats["refused_busy"] += 1
                wait = (self._cells / self._rate) if self._rate > 0 else 1.0
                raise Busy("the kernel's queue is full — retry shortly", retry_after=max(1, min(60, int(wait) + 1)))
            self._waiting.setdefault(job.caller, deque()).append(job)
            self._cells += job.cells
            self._cells_of[job.caller] = mine + job.cells
            self._wake.notify()
        return job.future

    # -- the owner's side ----------------------------------------------------

    def _take(self) -> list[_Job]:
        """The next cycle: callers in turn, one job each per round, until
        the cycle's budget — band jobs of one ensemble size together, an
        exclusive job alone."""
        taken: list[_Job] = []
        budget = self.cycle_cells
        while self._waiting:
            progressed = False
            for caller in list(self._waiting):
                job = self._waiting[caller][0]
                fits = not taken or (
                    job.kind == "bands"
                    and taken[0].kind == "bands"
                    and job.members == taken[0].members
                    and job.cells <= budget
                )
                if not fits:
                    continue
                self._waiting[caller].popleft()
                if not self._waiting[caller]:
                    del self._waiting[caller]
                else:
                    self._waiting.move_to_end(caller)  # the caller just served goes last
                self._cells -= job.cells
                self._cells_of[caller] -= job.cells
                if not self._cells_of[caller]:
                    del self._cells_of[caller]
                budget -= job.cells
                taken.append(job)
                progressed = True
                if job.kind != "bands":
                    return taken
            if not progressed:
                break
        return taken

    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._waiting:
                    self._wake.wait()
                jobs = self._take()
            started = time.perf_counter()
            try:
                self._serve(jobs)
            except BaseException as e:  # the owner must outlive any one cycle
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(e)
            cells = sum(job.cells for job in jobs)
            rate = cells / max(time.perf_counter() - started, 1e-6)
            self._rate = rate if self._rate == 0 else 0.8 * self._rate + 0.2 * rate
            self.stats["cycles"] += 1
            self.stats["jobs"] += len(jobs)
            self.stats["largest_cycle_jobs"] = max(self.stats["largest_cycle_jobs"], len(jobs))

    def _serve(self, jobs: list[_Job]) -> None:
        kernel = self._kernels_get()
        if jobs[0].kind != "bands":
            (job,) = jobs
            job.future.set_result(job.run(kernel))
            return
        reads = [read for job in jobs for read in job.reads]
        alphas = [job.alphas for job in jobs for _ in job.reads]
        answers = kernel.answer(reads, alphas, jobs[0].members)
        at = 0
        for job in jobs:
            mine = answers[at : at + len(job.reads)]
            at += len(job.reads)
            refusal = next((a for a in mine if isinstance(a, Exception)), None)
            if refusal is not None:
                # Its index was the cycle's; say the caller's own.
                local = next(i for i, a in enumerate(mine) if a is refusal)
                text = str(refusal)
                if text.startswith("bands: read "):
                    text = text.split(": ", 2)[-1]
                text = text.removeprefix("bands: ")
                job.future.set_exception(type(refusal)(f"bands: read {local}: {text}"))
            else:
                job.future.set_result(mine)


def cells(*arrays: np.ndarray) -> int:
    return int(sum(a.size for a in arrays))
