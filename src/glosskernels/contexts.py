"""Contexts kept between requests: a caller's, and only that caller's.

The weights are everyone's; a context is a caller's data, and what is
kept of it (the training rows' keys and values at every layer) is
derived from that data. So an entry belongs to the caller that sent it
and is found only under that caller: another caller sending the same id
— or the same bytes — finds nothing, and learns nothing.

Soft state, in memory only. An entry may go at any time: the least
recently read gives way when a caller passes its share of the budget or
the cache passes the whole of it, and a restart or another instance has
none of it. A read naming a context that is not here is refused as
`ContextUnknown`, and the caller sends the rows again — it can always
rebuild a context from its record. A context is a snapshot: a new month
is a new context, and the old one ages out.

Only the owner's thread touches this, so there is no lock."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


class ContextUnknown(Exception):
    """The context is not held here (it never was, it aged out, or this is
    another instance): send the rows again."""


@dataclass
class Held:
    fitted: Any  # the package's estimator, fit with its KV cache built
    nbytes: int
    rows: int
    cols: int


def context_id(train_x: np.ndarray, train_y: np.ndarray, members: int, model: str) -> str:
    """What the context is: its bytes, its shape, how it is read."""
    h = hashlib.sha256()
    h.update(f"{model}|{members}|{train_x.shape}|".encode())
    h.update(np.ascontiguousarray(train_x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(train_y, dtype=np.float64).tobytes())
    return h.hexdigest()[:32]


def budget_bytes(device: str) -> int:
    """`GLOSSKERNELS_CACHE_MB`, else two fifths of the device's memory
    (the rest is the model, a build's working set and the reads' passes),
    else 2 GB off CUDA."""
    named = os.environ.get("GLOSSKERNELS_CACHE_MB", "").strip()
    if named.isdigit():
        return int(named) * 1024 * 1024
    if device == "cuda":
        import torch

        return int(torch.cuda.get_device_properties(0).total_memory * 0.4)
    return 2 * 1024**3


class Contexts:
    def __init__(self, budget: int, caller_share: float = 0.5):
        self.budget, self.caller_share = budget, caller_share
        self._held: OrderedDict[tuple[str, str], Held] = OrderedDict()  # (caller, id), least recently read first
        self.stats = {"hits": 0, "misses": 0, "builds": 0, "evictions": 0}

    @property
    def nbytes(self) -> int:
        return sum(h.nbytes for h in self._held.values())

    def of(self, caller: str) -> int:
        return sum(h.nbytes for (c, _), h in self._held.items() if c == caller)

    def fits(self, nbytes: int) -> bool:
        """Whether one caller's share could ever hold this."""
        return nbytes <= self.budget * self.caller_share

    def get(self, caller: str, id_: str) -> Held:
        held = self._held.get((caller, id_))
        if held is None:
            self.stats["misses"] += 1
            raise ContextUnknown(f"context_unknown: {id_} is not held here — send the rows again")
        self._held.move_to_end((caller, id_))
        self.stats["hits"] += 1
        return held

    def has(self, caller: str, id_: str) -> bool:
        return (caller, id_) in self._held

    def put(self, caller: str, id_: str, held: Held) -> None:
        self._held[(caller, id_)] = held
        self._held.move_to_end((caller, id_))
        self.stats["builds"] += 1
        # The caller's own oldest give way first, then anyone's.
        while self.of(caller) > self.budget * self.caller_share:
            self._evict(next(k for k in self._held if k[0] == caller))
        while self.nbytes > self.budget:
            self._evict(next(iter(self._held)))

    def drop(self, caller: str) -> int:
        """Everything a caller has here — its offboarding, or its wish."""
        mine = [k for k in self._held if k[0] == caller]
        for key in mine:
            self._evict(key)
        return len(mine)

    def _evict(self, key: tuple[str, str]) -> None:
        del self._held[key]
        self.stats["evictions"] += 1
