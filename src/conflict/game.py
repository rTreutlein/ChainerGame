"""Rounds: the reasoner is given each round's claims and states P(up) for every
node, the beliefs are scored against the outcome and the exact posterior under
the true model, and the round's states are revealed: it joins the labelled
history."""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass

from supplynet.game import _mean, score

from .world import SIZES, Round, contested, generate_world, posterior, sample_round, true_rates


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    size: str = "s"
    history: int = 30
    rounds: int = 10
    steps_per_query: float = 4.0  # the round's budget is this times its queries


def run_game(config: GameConfig, backend, on_round=None) -> dict:
    rng = random.Random(config.seed)
    world = generate_world(rng, SIZES[config.size])
    systems = {s.name for s in world.systems}
    rates = true_rates(world)
    history = [sample_round(world, rng, f"h{index + 1}") for index in range(config.history)]
    started = time.perf_counter()
    backend.begin(world, history)
    setup_seconds = time.perf_counter() - started

    rounds = []
    for index in range(config.rounds):
        day = sample_round(world, rng, f"t{index + 1}")
        keys = [(node, day.name) for node in world.nodes]
        exact = posterior(world, rates, day.reports)
        actual = {key: day.truth[key[0]] for key in keys}
        disputed = contested(day.reports)
        budget = max(1, round(config.steps_per_query * len(keys)))
        started = time.perf_counter()
        beliefs = backend.beliefs(Round(day.name, {}, day.reports), keys, budget)  # the claims, not the states
        backend.resolve(day)
        seconds = time.perf_counter() - started
        truth = {key: exact[key[0]] for key in keys}
        record = {
            "type": "round",
            "round": index + 1,
            "seconds": round(seconds, 4),
            **score(beliefs, actual, truth, lambda key: "system" if key[0] in systems else "component"),
            "groups": score(beliefs, actual, truth, lambda key: "contested" if key[0] in disputed else "agreed")["kinds"],
        }
        rounds.append(record)
        if on_round:
            on_round(record)

    def pooled(field):
        names = sorted({kind for r in rounds for kind in r[field]})
        return {name: _mean([r[field][name] for r in rounds if name in r[field]]) for name in names}

    return {
        "type": "run-summary",
        "backend": backend.name,
        "seed": config.seed,
        "size": config.size,
        "steps_per_query": config.steps_per_query,
        "history": config.history,
        "rounds": config.rounds,
        "nodes": len(world.nodes),
        "sources": [f"{s.name}:{s.kind}" for s in world.sources],
        "setup_seconds": round(setup_seconds, 3),
        "seconds": round(sum(r["seconds"] for r in rounds), 3),
        "seconds_per_round": round(sum(r["seconds"] for r in rounds) / len(rounds), 4),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **_mean(rounds),
        "kinds": pooled("kinds"),
        "groups": pooled("groups"),
        **backend.stats(),
    }
