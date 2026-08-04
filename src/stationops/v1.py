"""Deterministic BaseRateTriage-v1 multi-round state machine."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable, TextIO

from .backends import MM2Backend, PeTTaChainerBackend, ReferenceBackend
from .config import Config
from .episode import _utility
from .metta import generate_statements
from .models import EpisodeFixture, HistoryCase, Incident, RoundFixture
from .oracle import belief_error_metrics, empirical_priors, posterior
from .policy import allocate
from .simulator import generate_history


def prior_shift_fixture(config: Config) -> EpisodeFixture:
    """Two rounds: twenty resolved new-cohort leaks shift only the next round."""
    history = tuple(generate_history(config))

    def incident(case_id: str, cohort: str = "new") -> Incident:
        return Incident(case_id, cohort, True)

    first = (incident("r0-new-signal"),) + tuple(
        incident(f"r0-new-evidence-{i:02d}") for i in range(19)
    ) + (incident("r0-old-control", "old"),) + tuple(
        incident(f"r0-old-evidence-{i:02d}", "old") for i in range(19)
    )
    first_resolutions = tuple(
        HistoryCase(
            x.id,
            x.cohort,
            x.cohort == "new" or x.id == "r0-old-evidence-00",
            x.alarm,
        )
        for x in first
    )
    second = (incident("r1-new-signal"), incident("r1-old-control", "old"))
    second_resolutions = tuple(HistoryCase(x.id, x.cohort, False, x.alarm) for x in second)
    return EpisodeFixture(
        "prior-shift",
        history,
        (RoundFixture(first, first_resolutions), RoundFixture(second, second_resolutions)),
    )


def _backend(config, name, mm2_path=None, pettachainer_path=None):
    if name == "reference":
        return ReferenceBackend(config)
    if name == "mm2":
        return MM2Backend(config, mm2_path)
    if name == "pettachainer":
        return PeTTaChainerBackend(config, pettachainer_path)
    raise ValueError(f"unknown backend: {name}")


def _round_result(index, history, round_, beliefs, actions, counters, config, elapsed):
    priors = empirical_priors(history)
    oracle_beliefs = {
        x.id: posterior(priors[x.cohort], x.alarm, config.sensitivity, config.false_positive_rate)
        for x in round_.incidents
    }
    oracle_actions = allocate(list(round_.incidents), oracle_beliefs, config)
    utility = _utility(actions, oracle_beliefs, round_.incidents, config)
    oracle_utility = _utility(oracle_actions, oracle_beliefs, round_.incidents, config)
    return {
        "index": index,
        "history_size_before": len(history),
        "history_size_after": len(history) + len(round_.resolutions),
        "priors_before": priors,
        "visible_incident_ids": [x.id for x in round_.incidents],
        "visible_incidents": [asdict(x) for x in round_.incidents],
        "beliefs": beliefs,
        "belief_error": belief_error_metrics(beliefs, oracle_beliefs),
        "chosen_actions": actions,
        "expected_utility": utility,
        "oracle_utility": oracle_utility,
        "do_nothing_utility": 0.0,
        "regret": oracle_utility - utility,
        "normalized_score": 1.0 if oracle_utility == 0 else max(0.0, utility / oracle_utility),
        "resolution_count": len(round_.resolutions),
        "backend_counters": counters,
        "wall_time_seconds": elapsed,
    }


def _result(config, fixture, backend, budget, rounds, started, status="complete"):
    utility = sum(x["expected_utility"] for x in rounds)
    oracle_utility = sum(x["oracle_utility"] for x in rounds)
    engine_steps = [x["backend_counters"].get("engine_steps") for x in rounds]
    counters = {
        "queries": sum(x["backend_counters"].get("queries", 0) or 0 for x in rounds),
        "engine_steps": (
            sum(engine_steps)
            if engine_steps and all(type(x) in (int, float) for x in engine_steps)
            else None
        ),
    }
    for key in (
        "statements_added",
        "statements_removed",
        "base_rates_updated",
        "forward_seed_facts",
        "forward_steps",
    ):
        if any(key in x["backend_counters"] for x in rounds):
            counters[key] = sum(
                x["backend_counters"].get(key, 0) or 0 for x in rounds
            )
    return {
        "benchmark": "BaseRateTriage-v1",
        "schema_version": 1,
        "config": config.to_dict(),
        "seed": config.seed,
        "fixture": fixture.name,
        "backend": backend,
        "budget_per_round": budget,
        "rounds": rounds,
        "aggregate": {
            "expected_utility": utility,
            "oracle_utility": oracle_utility,
            "do_nothing_utility": 0.0,
            "regret": oracle_utility - utility,
            "normalized_score": 1.0 if oracle_utility == 0 else max(0.0, utility / oracle_utility),
        },
        "aggregate_counters": counters,
        "status": status,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_episode_v1(config: Config, backend_name="reference", budget=100, mm2_path=None,
                   pettachainer_path=None, fixture=None) -> dict:
    fixture = fixture or prior_shift_fixture(config)
    backend = _backend(config, backend_name, mm2_path, pettachainer_path)
    history = list(fixture.history)
    rounds = []
    started = time.perf_counter()
    for index, round_ in enumerate(fixture.rounds):
        statements = generate_statements(history, list(round_.incidents), config)
        round_started = time.perf_counter()
        beliefs, counters = backend.infer(history, list(round_.incidents), budget, statements)
        actions = allocate(list(round_.incidents), beliefs, config)
        rounds.append(_round_result(index, history, round_, beliefs, actions, counters, config,
                                    time.perf_counter() - round_started))
        history.extend(round_.resolutions)
    return _result(config, fixture, backend.name, budget, rounds, started)


def play_episode_v1(config: Config, fixture=None, input_fn: Callable[[str], str] | None = None,
                    output: TextIO | None = None) -> dict:
    import sys
    output = output or sys.stderr
    input_fn = input_fn or input
    fixture = fixture or prior_shift_fixture(config)
    history = list(fixture.history)
    rounds = []
    started = time.perf_counter()
    status = "complete"
    for index, round_ in enumerate(fixture.rounds):
        priors = empirical_priors(history)
        print(f"Round {index}: repair slots={config.repair_slots}; priors={priors}", file=output)
        for item in round_.incidents:
            print(f"  {item.id}: cohort={item.cohort} alarm={item.alarm}", file=output)
        valid = {x.id for x in round_.incidents}
        while True:
            try:
                print("Repair IDs (comma-separated, blank=defer all, quit=stop): ",
                      end="", file=output, flush=True)
                raw = input_fn("")
            except EOFError:
                raw = "quit"
            if raw.strip().lower() in {"quit", "q"}:
                selected, status = set(), "quit"
                break
            ids = [x.strip() for x in raw.split(",") if x.strip()]
            if len(ids) != len(set(ids)):
                print("Invalid selection: duplicate incident ID", file=output)
                continue
            if any(x not in valid for x in ids):
                print("Invalid selection: unknown incident ID", file=output)
                continue
            if len(ids) > config.repair_slots:
                print("Invalid selection: repair slot limit exceeded", file=output)
                continue
            selected = set(ids)
            break
        actions = {x.id: ("repair" if x.id in selected else "defer") for x in round_.incidents}
        rounds.append(_round_result(index, history, round_, {}, actions,
                                    {"queries": 0, "engine_steps": None}, config, 0.0))
        print("Resolutions:", file=output)
        for resolved in round_.resolutions:
            print(f"  {resolved.id}: leak={resolved.leak}", file=output)
        history.extend(round_.resolutions)
        if status == "quit":
            break
    return _result(config, fixture, "human", None, rounds, started, status)
