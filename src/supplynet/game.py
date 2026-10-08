"""Rounds of stage 1: each round a new period is observed, the reasoner states
its beliefs, they are scored, and the period is resolved into the history."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .world import Rates, exact_posterior, generate_network, observe, sample_period, truth


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    regions: int = 3
    history: int = 30
    rounds: int = 20
    inspect_rate: float = 0.25
    budget: int = 100


def _brier(beliefs: dict, actual: dict) -> float:
    return sum((beliefs[key] - float(actual[key])) ** 2 for key in beliefs) / len(beliefs) if beliefs else 0.0


def _log_loss(beliefs: dict, actual: dict) -> float:
    eps = 1e-6
    total = 0.0
    for key, p in beliefs.items():
        p = min(1 - eps, max(eps, p))
        total -= math.log(p if actual[key] else 1 - p)
    return total / len(beliefs) if beliefs else 0.0


def score(beliefs: dict, actual: dict, posterior: dict) -> dict:
    """Scores over all keys, an unanswered key read as 0.5, and per kind; the
    exact posterior's own scores alongside."""
    filled = {key: beliefs.get(key, 0.5) for key in actual}
    result = {
        "keys": len(actual),
        "answered": sum(1 for key in actual if key in beliefs),
        "brier": _brier(filled, actual),
        "log_loss": _log_loss(filled, actual),
        "posterior_brier": _brier(posterior, actual),
        "posterior_log_loss": _log_loss(posterior, actual),
        "posterior_error": sum(abs(filled[key] - posterior[key]) for key in actual) / len(actual),
    }
    for kind in ("Storm", "Blocked"):
        keys = [key for key in actual if key[0] == kind]
        if keys:
            result[f"{kind.lower()}_posterior_error"] = sum(abs(filled[key] - posterior[key]) for key in keys) / len(keys)
    return result


def run_game(config: GameConfig, backend, rates: Rates | None = None, on_round=None) -> dict:
    rates = rates or Rates()
    rng = random.Random(config.seed)
    network = generate_network(rng, config.regions)
    history = []
    for index in range(config.history):
        period = sample_period(network, rates, rng, f"h{index + 1}")
        history.append((period, observe(network, period, rng, 1.0)))
    started = time.perf_counter()
    backend.begin(network, rates, history)
    setup_seconds = time.perf_counter() - started
    rounds = []
    for index in range(config.rounds):
        period = sample_period(network, rates, rng, f"t{index + 1}")
        observation = observe(network, period, rng, config.inspect_rate)
        actual = truth(period, observation)
        posterior = exact_posterior(network, rates, observation)
        started = time.perf_counter()
        beliefs = backend.beliefs(observation, sorted(actual), config.budget)
        seconds = time.perf_counter() - started
        backend.resolve(period, observation)
        record = {"type": "round", "round": index + 1, "seconds": round(seconds, 3), **score(beliefs, actual, posterior)}
        rounds.append(record)
        if on_round:
            on_round(record)
    keys = sum(r["keys"] for r in rounds)

    def mean(field):
        return sum(r[field] * r["keys"] for r in rounds) / keys if keys else 0.0

    return {
        "type": "run-summary",
        "backend": backend.name,
        "seed": config.seed,
        "regions": len(network.regions),
        "routes": len(network.routes),
        "shipments": sum(len(route.shipments) for route in network.routes),
        "setup_seconds": round(setup_seconds, 3),
        "seconds": round(sum(r["seconds"] for r in rounds), 3),
        "coverage": sum(r["answered"] for r in rounds) / keys if keys else 0.0,
        **{field: mean(field) for field in ("brier", "log_loss", "posterior_brier", "posterior_log_loss", "posterior_error")},
    }
