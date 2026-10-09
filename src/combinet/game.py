"""Rounds: each round brings new cases, the reasoner states P(C | observed
parents) for every consequent of every case, the beliefs are scored against
the outcome and the exact posterior, and the outcomes are revealed: the cases
join the labelled history."""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass

from supplynet.game import _mean, score

from .world import RELATIONS, generate_problem, learned_marginals, sample_case, true_marginals


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    parents: int = 3
    relations: tuple[str, ...] = RELATIONS
    per_relation: int = 1
    history: int = 50
    rounds: int = 10
    cases: int = 10
    observe_rate: float = 0.8
    budget: int = 20  # backward steps per query
    rates: str = "true"  # the rules' rates: "true" marginals or "learned" from the initial history


def run_game(config: GameConfig, backend, on_round=None) -> dict:
    rng = random.Random(config.seed)
    problem = generate_problem(rng, config.parents, config.relations, config.per_relation)
    relation = {c.name: c.relation for c in problem}
    history = [sample_case(problem, rng, f"h{index + 1}", config.observe_rate) for index in range(config.history)]
    marginals = true_marginals(problem) if config.rates == "true" else learned_marginals(problem, history)
    started = time.perf_counter()
    backend.begin(problem, marginals, history)
    setup_seconds = time.perf_counter() - started

    rounds = []
    for index in range(config.rounds):
        cases = [sample_case(problem, rng, f"t{index + 1}-{k + 1}", config.observe_rate) for k in range(config.cases)]
        keys = [(c.name, case.name) for case in cases for c in problem]
        posterior = {(c.name, case.name): c.model.posterior(case.observed[c.name]) for case in cases for c in problem}
        actual = {(c.name, case.name): case.truth[c.name][1] for case in cases for c in problem}
        started = time.perf_counter()
        beliefs = backend.beliefs(cases, keys, config.budget)
        backend.resolve(cases)
        seconds = time.perf_counter() - started
        record = {
            "type": "round",
            "round": index + 1,
            "history": config.history + index * config.cases,
            "seconds": round(seconds, 3),
            **score(beliefs, actual, posterior, lambda key: relation[key[0]]),
        }
        rounds.append(record)
        if on_round:
            on_round(record)

    kinds = sorted({kind for r in rounds for kind in r["kinds"]})
    return {
        "type": "run-summary",
        "backend": backend.name,
        "seed": config.seed,
        "parents": config.parents,
        "consequents": len(problem),
        "history": config.history,
        "rounds": config.rounds,
        "cases": config.cases,
        "observe_rate": config.observe_rate,
        "budget": config.budget,
        "rates": config.rates,
        "setup_seconds": round(setup_seconds, 3),
        "seconds": round(sum(r["seconds"] for r in rounds), 3),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **backend.stats(),
        **_mean(rounds),
        "kinds": {kind: _mean([r["kinds"][kind] for r in rounds if kind in r["kinds"]]) for kind in kinds},
        "curve": [{"history": r["history"], "brier": round(r["brier"], 4), "posterior_error": round(r["posterior_error"], 4)} for r in rounds],
    }
