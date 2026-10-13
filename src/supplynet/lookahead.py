"""Stage 5: a fully inspected, two-period planning cell.

The observation reveals the storm, degradations and route blocks. Inputs are
guaranteed contracts. Thus next-state expectimax is exactly next-observation
expectimax, rather than a perfect-information upper bound. Only the first
action is enacted; the policy replans next round. No cleared-repair reward is
used. Fuel left at the horizon has stage 4's inventory value.
"""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .decisions import Problem, best, combine, greedy, options, singles, utility
from .game import describe
from .world import MINE, Action, Cell, Network, Rates, Route, production


@dataclass(frozen=True)
class State:
    storm: bool
    degraded: tuple[bool, ...]
    blocked: tuple[bool, bool]
    fuel: bool  # usable fuel arriving from the preceding action


def network(plants: int) -> Network:
    rf = Route("rf", "north", "exposed", (), 1)
    ra = Route("ra", "north", "sheltered", (), 1)
    return Network({"north": "coastal"}, (rf, ra), Cell(rf, {f"plant-{i+1}": f"contract-{i+1}" for i in range(plants)}, ra))


def produced(cell, state, action):
    return production(cell, dict(zip(cell.sites, state.degraded)), action.stock,
                      dict.fromkeys(cell.inputs, True), not state.fuel, True, action.repairs)


def reward(cell, values, state, action, terminal=False):
    output = produced(cell, state, action)
    value = sum(values.output[cell.sites[s]] for s, on in output.items() if on)
    # Fuel sent at t is consumed at t+1, never credited twice.
    if terminal:
        value += values.fuel * (output[MINE] and not state.blocked[int(action.reroute)])
    return value - values.cost_of(action)


def transition(cell, rates, state, action):
    """Enumerate P(next observable state | state, action), normalized."""
    fuel = produced(cell, state, action)[MINE] and not state.blocked[int(action.reroute)]
    probabilities = [rates.degraded_rate(kind, float(d and site not in action.repairs))
                     for (site, kind), d in zip(cell.sites.items(), state.degraded)]
    for storm in (False, True):
        p_storm = rates.storm_rate("coastal", float(state.storm))
        for degraded in itertools.product((False, True), repeat=len(cell.sites)):
            weight = p_storm if storm else 1 - p_storm
            for d, p in zip(degraded, probabilities):
                weight *= p if d else 1 - p
            for blocked in itertools.product((False, True), repeat=2):
                w = weight
                for b, route in zip(blocked, cell.fuel_routes):
                    p = rates.block_rate(route.kind, storm)
                    w *= p if b else 1 - p
                if w:
                    yield State(storm, degraded, blocked, fuel), w


def pair_values(cell, values, state, choices, distributions):
    """Fixed pairs and feedback Q(a)=r(a)+E[max_b r(next,b)]."""
    pairs, feedback = {}, {}
    for first in choices:
        immediate = reward(cell, values, state, first)
        following = [(w, {b: reward(cell, values, s, b, True) for b in choices})
                     for s, w in distributions[first]]
        feedback[first] = immediate + sum(w * max(row.values()) for w, row in following)
        for second in choices:
            pairs[first, second] = immediate + sum(w * row[second] for w, row in following)
    return pairs, feedback


def myopic_values(cell, values, state, choices):
    """Stage 4's one-period policy, including its original repair proxy."""
    problem = Problem("now", "before", dict.fromkeys(cell.inputs, True), True, "rf", choices, values)
    return {a: utility(cell, problem, a, dict(zip(cell.sites, state.degraded)), not state.fuel,
                       dict(zip((r.name for r in cell.fuel_routes), state.blocked))) for a in choices}


def propose_pairs(choices, values, evaluate, cap, beam=2, exhaustive=False):
    """Game proposals: value singles with no-action and stock continuations, extend
    the best singles by one, then extend the best first actions in time.
    All evaluations, including proposal probes, consume the pair cap."""
    evaluated = {}

    def ask(pairs):
        fresh = list(dict.fromkeys(p for p in pairs if p not in evaluated))[:max(0, cap-len(evaluated))]
        if fresh:
            evaluated.update(evaluate(fresh))

    if exhaustive:
        ask(itertools.product(choices, repeat=2))
        return evaluated
    continuations = [Action(), Action(stock=True)]
    ask((a, b) for a in choices if a.units <= 1 for b in continuations)
    layer = list(dict.fromkeys(a for a, b in evaluated if a.units == 1))
    single = [a for a in choices if a.units == 1]
    feasible = set(choices)
    for _ in range(values.capacity - 1):
        scores = {a: max(v for (first, _), v in evaluated.items() if first == a and v is not None)
                  for a in layer if any(first == a and v is not None for (first, _), v in evaluated.items())}
        top = sorted(scores, key=scores.get, reverse=True)[:beam]
        layer = list(dict.fromkeys(combine([a, b]) for a in top for b in single if b not in singles(a)))
        layer = [a for a in layer if a in feasible]
        ask((a, b) for a in layer for b in continuations)
        layer = [a for a in layer if any(first == a for first, _ in evaluated)]
    scores = {a: max(v for (first, _), v in evaluated.items() if first == a and v is not None)
              for a in dict.fromkeys(a for a, _ in evaluated) if any(first == a and v is not None for (first, _), v in evaluated.items())}
    top = sorted(scores, key=scores.get, reverse=True)[:beam]
    # Interleave the first actions so truncation gives each a second-action budget.
    ask((a, b) for b in choices for a in top)
    return evaluated


def problog_distributions(cell, rates, state, choices, engine, timeout):
    """ProbLog computes the joint transitions; Python maximizes after the
    observed next state. No reference probabilities are passed to ProbLog."""
    import tempfile
    from .problog_backend import load, solve
    scratch = Path(__file__).resolve().parents[2] / "benchmarks/supplynet_stage5/scratch"
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir = str(scratch)
    load(engine)
    p = rates.storm_rate("coastal", float(state.storm))
    lines = [f"{p}::storm."]
    for route in cell.fuel_routes:
        lines += [f"{rates.block_given_storm[route.kind]}::block_{route.name} :- storm.",
                  f"{rates.block_without_storm}::block_{route.name} :- \\+storm."]
    for i, (site, kind) in enumerate(cell.sites.items()):
        for repaired in (False, True):
            p = rates.degraded_rate(kind, float(state.degraded[i] and not repaired))
            lines.append(f"{p}::d_{i}_{int(repaired)}.")
    queries, indexed = [], []
    for repair_set in dict.fromkeys(a.repairs for a in choices):
        for bits in itertools.product((False, True), repeat=len(cell.sites)+3):
            atoms = ["storm", "block_rf", "block_ra"] + [f"d_{i}_{int(s in repair_set)}" for i, s in enumerate(cell.sites)]
            name = f"state_{len(queries)}"
            lines.append(f"{name} :- " + ", ".join(atom if on else "\\+"+atom for atom, on in zip(atoms, bits)) + ".")
            lines.append(f"query({name}).")
            queries.append(name)
            indexed.append((repair_set, bits))
    probabilities, record = solve("\n".join(lines), queries, engine, timeout)
    if probabilities is None:
        return None, record
    by_repairs = {}
    for (repair_set, bits), w in zip(indexed, probabilities, strict=True):
        if w:
            by_repairs.setdefault(repair_set, []).append((bits, w))
    result = {}
    for a in choices:
        fuel = produced(cell, state, a)[MINE] and not state.blocked[int(a.reroute)]
        result[a] = [(State(bits[0], tuple(bits[3:]), tuple(bits[1:3]), fuel), w) for bits, w in by_repairs[a.repairs]]
    return result, record


def run(config, backend, proposer="extend", pair_cap=32, beam=2, python_path=None,
        problog_engine="ddnnf", problog_timeout=60, on_round=None):
    cell_network = network(config.plants)
    cell, values = cell_network.cell, config.values
    choices = options(cell, values.capacity)
    rates, rng = Rates(persist=0.7), random.Random(config.seed)
    state = State(rng.random() < rates.stationary("coastal"),
                  tuple(rng.random() < rates.degraded_stationary(k) for k in cell.sites.values()),
                  (False, False), False)
    counts = [0] * (len(cell.sites)+4)
    for _ in range(config.history):
        a = Action(stock=rng.random() < rates.stocked)
        state = draw(transition(cell, rates, state, a), rng)
        row = [state.storm, *state.degraded, *state.blocked, state.fuel]
        counts = [n + b for n, b in zip(counts, row)]
    prior = [(n+1)/(config.history+2) for n in counts]
    rounds, problog_records, prior_choice = [], [], None
    for index in range(config.rounds):
        started = time.perf_counter()
        distributions = {a: list(transition(cell, rates, state, a)) for a in choices}
        pairs, feedback = pair_values(cell, values, state, choices, distributions)
        reference_seconds = time.perf_counter()-started
        reference = best(feedback)
        started = time.perf_counter()
        myopic = myopic_values(cell, values, state, choices)
        myopic_seconds = time.perf_counter()-started
        started = time.perf_counter()
        details = {}
        if backend == "reference":
            action = reference
        elif backend == "myopic":
            action = best(myopic)
        elif backend == "greedy":
            action = greedy(myopic, values.capacity)
        elif backend in ("openloop", "beam"):
            evaluated = pairs if backend == "openloop" else propose_pairs(choices, values, lambda ps: {p: pairs[p] for p in ps}, pair_cap, beam)
            action = max(evaluated, key=evaluated.get)[0]
            details = {"candidate_pairs": len(evaluated)}
        elif backend == "prior":
            if prior_choice is None:
                imagined = {}
                for bits in itertools.product((False, True), repeat=len(prior)):
                    weight = 1.0
                    for b, p in zip(bits, prior):
                        weight *= p if b else 1-p
                    s = State(bits[0], tuple(bits[1:1+len(cell.sites)]), tuple(bits[-3:-1]), bits[-1])
                    # Independent history frequencies, followed by the given dynamics.
                    ps, _ = pair_values(cell, values, s, choices, {a: list(transition(cell, rates, s, a)) for a in choices})
                    for pair, value in ps.items():
                        imagined[pair] = imagined.get(pair, 0.0) + weight * value
                prior_choice = max(imagined, key=imagined.get)[0]
            action = prior_choice
        elif backend == "problog":
            ds, record = problog_distributions(cell, rates, state, choices, problog_engine, problog_timeout)
            problog_records.append(record)
            if ds is None:
                action = Action()
            else:
                _, qs = pair_values(cell, values, state, choices, ds)
                action = best(qs)
                details["problog_reference_max_error"] = max(abs(qs[a]-feedback[a]) for a in choices)
        else:
            payload = {"plants": config.plants, "values": {"output": values.output, "fuel": values.fuel,
                       "repair": values.repair, "cost": values.cost, "capacity": values.capacity},
                       "state": {"storm": state.storm, "degraded": state.degraded, "blocked": state.blocked, "fuel": state.fuel},
                       "budget": config.budget, "pair_cap": pair_cap, "beam": beam, "proposer": proposer,
                       "backend": backend, "python_path": python_path}
            child = subprocess.run([sys.executable, "-m", "supplynet.lookahead_worker"], input=json.dumps(payload),
                                   text=True, capture_output=True, timeout=180)
            if child.returncode:
                raise RuntimeError(child.stderr[-4000:])
            result = json.loads(child.stdout.splitlines()[-1])
            selected = result.pop("chosen")
            action = choices[selected]
            details = result
            errors = [abs(v-pairs[choices[a], choices[b]]) for a, b, v in details["pair_scores"] if v is not None]
            details["pair_mean_error"] = sum(errors)/len(errors) if errors else None
            details["pair_max_error"] = max(errors) if errors else None
            candidates = {choices[a] for a, b, v in details["pair_scores"]}
            details["proposal_regret"] = feedback[reference]-max(feedback[a] for a in candidates)
        seconds = time.perf_counter()-started
        if backend in ("reference", "beam", "openloop"):
            seconds += reference_seconds
        if backend in ("myopic", "greedy"):
            seconds += myopic_seconds
        realized = reward(cell, values, state, action)
        record = {"type": "round", "round": index+1, "action": describe(action), "reference_action": describe(reference),
                  "regret": max(0.0, feedback[reference]-feedback[action]), "utility": realized,
                  "myopic_regret": feedback[reference]-feedback[best(myopic)],
                  "lookahead_gain": feedback[action]-feedback[best(myopic)],
                  "feedback_value_gap": feedback[reference]-max(pairs.values()),
                  "reference_seconds": reference_seconds, "decision_seconds": seconds,
                  "state": {"storm": state.storm, "degraded": state.degraded, "blocked": state.blocked, "fuel": state.fuel}, **details}
        rounds.append(record)
        if on_round:
            on_round(record)
        state = draw(distributions[action], rng)
    result = {"type": "run-summary", "stage": 5, "backend": backend, "seed": config.seed,
              "rounds": config.rounds, "history": config.history, "budget": config.budget, "proposer": proposer,
              "pair_cap": pair_cap, "plants": config.plants, "capacity": values.capacity, "options": len(choices),
              "all_pairs": len(choices)**2, "next_states": 2**(len(cell.sites)+3), "horizon": 2,
              "information": "full state; guaranteed plant inputs; receding horizon", "actions": [r["action"] for r in rounds]}
    fields = ["regret", "utility", "myopic_regret", "lookahead_gain", "feedback_value_gap", "reference_seconds", "decision_seconds"]
    optional = {
        "candidate_pairs", "decision_queries", "decision_answered", "open_answered", "context_bound", "worker_setup_seconds",
        "chainer_wall_seconds", "chainer_inference_seconds", "chainer_overhead_seconds", "problog_reference_max_error",
        "pair_mean_error", "pair_max_error", "proposal_regret"}
    fields += sorted(f for f in optional if all(r.get(f) is not None for r in rounds))
    result.update({field: sum(r[field] for r in rounds)/len(rounds) for field in fields})
    result["unsupported"] = sorted({setting for r in rounds for setting in r.get("unsupported", [])})
    result["early_decision_seconds"] = sum(r["decision_seconds"] for r in rounds[:10])/len(rounds[:10])
    result["late_decision_seconds"] = sum(r["decision_seconds"] for r in rounds[-10:])/len(rounds[-10:])
    if problog_records:
        from .problog_backend import summary
        result.update(summary(problog_records))
    return result


def draw(distribution, rng):
    threshold = rng.random()
    for state, weight in distribution:
        threshold -= weight
        if threshold <= 0:
            return state
    raise ValueError("transition distribution is not normalized")
