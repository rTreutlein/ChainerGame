import time

from .backends import MM2Backend, ReferenceBackend
from .config import Config
from .metta import generate_statements
from .oracle import empirical_priors, posterior, repair_increment
from .policy import allocate
from .simulator import generate_history, generate_incidents


def _utility(actions, beliefs, incidents, config):
    return sum(
        repair_increment(beliefs[x.id], config)
        for x in incidents
        if actions[x.id] == "repair"
    )


def run_episode(config: Config, backend_name="reference", budget=100, mm2_path=None) -> dict:
    history, incidents = generate_history(config), generate_incidents(config)
    statements = generate_statements(history, incidents, config)
    backend = ReferenceBackend(config) if backend_name == "reference" else MM2Backend(config, mm2_path)
    started = time.perf_counter()
    beliefs, counters = backend.infer(history, incidents, budget, statements)
    elapsed = time.perf_counter() - started
    priors = empirical_priors(history)
    oracle_beliefs = {
        x.id: posterior(priors[x.cohort], x.alarm, config.sensitivity, config.false_positive_rate)
        for x in incidents
    }
    actions = allocate(incidents, beliefs, config)
    oracle_actions = allocate(incidents, oracle_beliefs, config)
    utility = _utility(actions, oracle_beliefs, incidents, config)
    oracle_utility = _utility(oracle_actions, oracle_beliefs, incidents, config)
    regret = oracle_utility - utility
    return {
        "benchmark": "BaseRateTriage-v0",
        "config": config.to_dict(),
        "seed": config.seed,
        "backend": backend.name,
        "budget": budget,
        "empirical_priors": priors,
        "beliefs": beliefs,
        "chosen_actions": actions,
        "expected_utility": utility,
        "oracle_utility": oracle_utility,
        "do_nothing_utility": 0.0,
        "regret": regret,
        "normalized_score": 1.0 if oracle_utility == 0 else max(0.0, utility / oracle_utility),
        "wall_time_seconds": elapsed,
        "backend_counters": counters,
    }

