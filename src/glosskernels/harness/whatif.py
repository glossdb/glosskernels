"""Grading a what-if, which no real panel can: the world the lever was
pulled in never lands. So the panel is simulated from a structure that
is known — and the pulled world can be run.

The structure is the smallest one with the door's two halves in it.
Revenue is price times volume: the recipe, which replay follows
exactly. Volume answers price with an elasticity: behavior, which no
recipe encodes, and which the panel carries as variation — members
priced differently, prices moving over time. Pull price by a factor and
replay says revenue moves by the factor; the truth is factor^(1 -
elasticity). The read under grade is the model's, from the panel as its
context: the member's rows with the price moved, everything else held.

Two dials make the read fail honestly. `factor` past the prices the
panel has seen leaves the support — the bands should widen, not
pretend. `confounding` lets a demand shock no column holds move price
and volume together — the panel's price slope is then not the lever's,
and no context fixes that; the grade shows how far off it lands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .score import BANDS, GRID, score
from .voices import Backend


@dataclass
class World:
    price: np.ndarray  # (members, months)
    base: np.ndarray  # (members, 1) a member's volume at the reference price
    season: np.ndarray  # (months,)
    shock: np.ndarray  # (months,) demand no column holds
    elasticity: float
    noise: float

    def volume(self, price: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eps = rng.normal(0.0, self.noise, size=price.shape)
        return self.base * self.season * np.exp(self.shock) * price ** (-self.elasticity) * np.exp(eps)


def simulate(
    members: int = 40, months: int = 36, elasticity: float = 1.5, confounding: float = 0.0, noise: float = 0.05, seed: int = 0
) -> tuple[World, np.ndarray]:
    """A world and the revenue that landed in it. Prices sit within
    about a quarter of the reference either way: a member's own level,
    a drift over time, and — with `confounding` — a lean into the
    month's demand shock."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    shock = rng.normal(0.0, 0.08, size=months)
    price = np.exp(
        rng.normal(0.0, 0.10, size=(members, 1)) + rng.normal(0.0, 0.05, size=(members, months)) + confounding * shock
    )
    world = World(
        price=price,
        base=rng.uniform(100.0, 5000.0, size=(members, 1)),
        season=1.0 + 0.2 * np.sin(2 * np.pi * (t % 12) / 12.0),
        shock=shock,
        elasticity=elasticity,
        noise=noise,
    )
    return world, price * world.volume(price, rng)


def grade(
    backend: Backend,
    factors: tuple[float, ...] = (1.0, 0.9, 1.1, 1.5),
    held: int = 6,
    draws: int = 400,
    **world_kwargs,
) -> dict:
    """The last `held` months are the scenario's: the context is every
    member's rows before them (the member's size, the month of year, the
    price; revenue in units of the member's size), and the read is those
    months with price moved by each factor. Truth is the pulled world
    run `draws` times: one draw lands as the actual, all of them give
    the true median. `replay` is the arithmetic half alone — volume
    held, price moved. Factor 1.0 is the world as it went: the one read
    whose actual a record would ever see, so the one a what-if's bands
    could be calibrated from."""
    world, revenue = simulate(**world_kwargs)
    members, months = revenue.shape
    cut = months - held
    size = revenue[:, :cut].mean(axis=1, keepdims=True)  # known at the scenario's start
    moy = np.tile(np.arange(months) % 12 + 1, (members, 1))

    def rows(price: np.ndarray, span: slice) -> np.ndarray:
        return np.column_stack([np.log(np.broadcast_to(size, price.shape)[:, span].ravel()), moy[:, span].ravel(), price[:, span].ravel()])

    train_x = rows(world.price, slice(0, cut))
    train_y = (revenue / size)[:, :cut].ravel()
    seen = (float(world.price[:, :cut].min()), float(world.price[:, :cut].max()))

    rng = np.random.default_rng(world_kwargs.get("seed", 0) + 1)
    out = {
        "members": members,
        "train_rows": int(train_x.shape[0]),
        "elasticity": world.elasticity,
        "confounding": world_kwargs.get("confounding", 0.0),
        "prices_seen": [round(v, 2) for v in seen],
        "factors": {},
    }
    for factor in factors:
        pulled = world.price * factor
        truth = np.stack([(pulled * world.volume(pulled, rng))[:, cut:].ravel() for _ in range(draws)])
        actual, true_median = truth[0], np.median(truth, axis=0)
        scale = np.broadcast_to(size, revenue.shape)[:, cut:].ravel()
        q = backend(train_x, train_y, rows(pulled, slice(cut, months)), GRID) * scale[:, None]
        median = np.sort(q, axis=1)[:, GRID.index(0.5)]
        replayed = revenue[:, cut:].ravel() * factor
        inside = float(np.mean((pulled[:, cut:] >= seen[0]) & (pulled[:, cut:] <= seen[1])))
        out["factors"][str(factor)] = {
            "in_support": round(inside, 2),
            "true_move": round(float(factor ** (1.0 - world.elasticity)), 3),
            "replay_move": factor,
            **score(q, actual),
            # How far the called median sits from the pulled world's, as a share of it.
            "median_off": round(float(np.mean(np.abs(median - true_median) / true_median)), 3),
            "replay_off": round(float(np.mean(np.abs(replayed - true_median) / true_median)), 3),
        }
    return out


def grade_worlds(backend: Backend, worlds: int = 5, **kwargs) -> dict:
    """`grade` over several simulated worlds, the scores averaged: six
    held months are six demand shocks, and one world's draw of them says
    more about the draw than about the read."""
    runs = [grade(backend, seed=seed, **kwargs) for seed in range(worlds)]
    out = {"worlds": worlds, **{k: runs[0][k] for k in ("members", "train_rows", "elasticity", "confounding")}, "factors": {}}
    for factor in runs[0]["factors"]:
        out["factors"][factor] = {
            key: round(float(np.mean([run["factors"][factor][key] for run in runs])), 3) for key in runs[0]["factors"][factor]
        }
    return out


__all__ = ["BANDS", "World", "grade", "grade_worlds", "simulate"]
