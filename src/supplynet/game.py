"""Rounds: each round a new period is observed, the reasoner states its
beliefs, they are scored, and the periods whose shipments have all arrived are
resolved into the history.

Stage 1 periods are independent and shipments arrive at once, so each round's
period resolves at its end. Stage 2 periods follow each other, storms persist,
and a period resolves when its slowest route's shipments arrive. Stage 3 adds
the production cycle: each round also reports the previous period's
production, and the reasoner states the current production before it is
reported. Stage 4 adds a decision per round on top of stage 3's timed cell:
after its beliefs the reasoner chooses a joint action, which is scored by its
regret against the exact reference policy and then enacted."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .decisions import Problem, Values, best, expected_utilities, options, utility
from .world import MINE, Knowledge, Rates, enact, generate_network, observe, sample_period

METRICS = ("coverage", "brier", "log_loss", "posterior_brier", "posterior_log_loss", "posterior_error")


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    regions: int = 3
    history: int = 30
    rounds: int = 20
    inspect_rate: float = 0.25
    budget: int = 100
    stage: int = 1
    cycle: str = "timed"  # stage 3: "timed" or "untimed"; stage 4 is timed
    plants: int = 2  # stages 3 and 4
    values: Values = field(default_factory=Values)  # stage 4


def stage_rates(stage: int) -> Rates:
    return Rates(persist=0.7) if stage >= 2 else Rates()


def _log_loss(p: float, actual: bool) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return -math.log(p if actual else 1 - p)


def _metrics(keys: list, beliefs: dict, actual: dict, posterior: dict) -> dict:
    """The share of keys answered, and scores with an unanswered key read as 0.5."""
    n = len(keys)
    filled = {key: beliefs.get(key, 0.5) for key in keys}
    return {
        "keys": n,
        "coverage": sum(1 for key in keys if key in beliefs) / n,
        "brier": sum((filled[key] - actual[key]) ** 2 for key in keys) / n,
        "log_loss": sum(_log_loss(filled[key], actual[key]) for key in keys) / n,
        "posterior_brier": sum((posterior[key] - actual[key]) ** 2 for key in keys) / n,
        "posterior_log_loss": sum(_log_loss(posterior[key], actual[key]) for key in keys) / n,
        "posterior_error": sum(abs(filled[key] - posterior[key]) for key in keys) / n,
    }


def score(beliefs: dict, actual: dict, posterior: dict, kind_of) -> dict:
    """Scores over all keys and per query kind (``kind_of(key)``); the exact
    posterior's own scores alongside."""
    kinds = {}
    for key in actual:
        kinds.setdefault(kind_of(key), []).append(key)
    return {
        **_metrics(list(actual), beliefs, actual, posterior),
        "kinds": {kind: _metrics(keys, beliefs, actual, posterior) for kind, keys in sorted(kinds.items())},
    }


def _mean(records: list[dict]) -> dict:
    keys = sum(r["keys"] for r in records)
    return {"keys": keys, **{field: sum(r[field] * r["keys"] for r in records) / keys for field in METRICS}}


def describe(action) -> str:
    """An action as the round records name it: "none", or its parts joined by "+"."""
    parts = [*(["stock"] if action.stock else []), *(["reroute"] if action.reroute else []), *(f"repair-{site}" for site in action.repairs)]
    return "+".join(parts) or "none"


def run_game(config: GameConfig, backend, rates: Rates | None = None, on_round=None) -> dict:
    rates = rates or stage_rates(config.stage)
    rng = random.Random(config.seed)
    decisions = config.stage == 4
    cycle = {3: config.cycle, 4: "timed"}.get(config.stage)
    network = generate_network(rng, config.regions, 3 if config.stage >= 2 else 0, cycle, config.plants, decisions)
    cell = network.cell
    choices = options(cell, config.values.capacity) if decisions else ()
    knowledge = Knowledge(network, rates.nodes(network))
    timeline = []

    def next_period(name):
        timeline.append(sample_period(network, rates, rng, name, timeline if rates.persist is not None else []))
        return timeline[-1]

    def reported():
        return timeline[-2].producing if len(timeline) > 1 else {}

    history = []
    for index in range(config.history):
        period = next_period(f"h{index + 1}")
        observation = observe(network, period, rng, 1.0, {(s, period.name): late for s, late in period.late.items()}, reported())
        knowledge.observe(observation)
        if decisions:
            knowledge.act(period.name, period.action(cell))
        knowledge.resolve(period)
        history.append((period, observation))
    started = time.perf_counter()
    backend.begin(network, rates, history)
    setup_seconds = time.perf_counter() - started

    first = len(timeline)
    unresolved = []
    rounds = []
    for index in range(config.rounds):
        period = next_period(f"t{index + 1}")
        now = len(timeline) - 1
        arrivals = {
            (shipment, timeline[now - route.transit].name): timeline[now - route.transit].late[shipment]
            for route in network.routes
            if now - route.transit >= first
            for shipment in route.shipments
        }
        observation = observe(network, period, rng, config.inspect_rate, arrivals, reported())
        knowledge.observe(observation)
        posterior = {
            key: p
            for key, p in knowledge.posterior().items()
            if key[2] == period.name or (key[0] == "Storm" and key[2] == period.previous)
        }
        by_name = {p.name: p for p in timeline[first:]}
        actual = {key: by_name[key[2]].truth(key[0])[key[1]] for key in posterior}

        def kind_of(key):
            """Producing is split by the period's fuel stock: without it,
            the untimed loop cannot start."""
            if key[0] == "Storm":
                return "storm" if key[2] == period.name else "previous_storm"
            if key[0] == "Producing":
                return "producing" if period.stocked else "producing_unstocked"
            return key[0].lower()

        started = time.perf_counter()
        beliefs = backend.beliefs(observation, sorted(actual), config.budget)
        seconds = time.perf_counter() - started
        decision = {}
        if decisions:
            previous = timeline[now - 1]
            problem = Problem(period.name, previous.name, period.inputs, previous.producing[MINE], previous.via, choices, config.values)
            exact = expected_utilities(knowledge, problem)
            started = time.perf_counter()
            action = backend.decide(problem, config.budget)
            decision_seconds = time.perf_counter() - started
            reference = best(exact)
            period = timeline[now] = enact(network, period, previous, action)
            knowledge.act(period.name, action)
            backend.act(period.name, action)
            decision = {
                "decision_seconds": round(decision_seconds, 3),
                "action": describe(action),
                "reference_action": describe(reference),
                "regret": exact[reference] - exact[action],
                "utility": utility(cell, problem, action, period.degraded, previous.blocked[previous.via], period.blocked),
                **(backend.decision_record() if hasattr(backend, "decision_record") else {}),
            }
        unresolved.append((now, period, observation))
        while unresolved and unresolved[0][0] + network.max_transit <= now:
            _, done, seen = unresolved.pop(0)
            knowledge.resolve(done)
            backend.resolve(done, seen)
        record = {"type": "round", "round": index + 1, "seconds": round(seconds, 3), **score(beliefs, actual, posterior, kind_of), **decision}
        rounds.append(record)
        if on_round:
            on_round(record)

    kinds = sorted({kind for r in rounds for kind in r["kinds"]})
    decided = {
        "plants": config.plants,
        "capacity": config.values.capacity,
        "options": len(choices),
        "regret": sum(r["regret"] for r in rounds) / len(rounds),
        "agreement": sum(r["action"] == r["reference_action"] for r in rounds) / len(rounds),
        "utility": sum(r["utility"] for r in rounds) / len(rounds),
        "decision_seconds": round(sum(r["decision_seconds"] for r in rounds), 3),
        **({"decision_queries": sum(r["decision_queries"] for r in rounds) / len(rounds),
            "decision_answered": sum(r["decision_answered"] for r in rounds) / len(rounds)} if "decision_queries" in rounds[0] else {}),
        "actions": [r["action"] for r in rounds],
    } if decisions else {}
    return {
        "type": "run-summary",
        "backend": backend.name,
        "stage": config.stage,
        **({"cycle": cycle, "sites": len(cell.sites)} if cell else {}),
        "seed": config.seed,
        "budget": config.budget,
        "regions": len(network.regions),
        "routes": len(network.routes),
        "shipments": sum(len(route.shipments) for route in network.routes),
        "setup_seconds": round(setup_seconds, 3),
        "seconds": round(sum(r["seconds"] for r in rounds), 3),
        **_mean(rounds),
        "kinds": {kind: _mean([r["kinds"][kind] for r in rounds if kind in r["kinds"]]) for kind in kinds},
        **decided,
        **(backend.summary() if hasattr(backend, "summary") else {}),
    }
