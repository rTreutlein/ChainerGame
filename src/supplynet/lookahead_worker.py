"""One bounded stage-5 decision in a fresh process; invoked inside capped.

A process exit discards the planning KB, so neither history nor hypothetical
contexts accumulate. Its cold setup cost is included in seconds per round.
"""
import importlib
import json
import sys
import time

from . import metta
from .decisions import Values, options
from .lookahead import State, network, propose_pairs
from .world import Action, Rates


def action_facts(cell, action, period):
    return [f"(FuelVia {action.via(cell)} {period})", f"({'StockedFuel' if action.stock else 'Unstocked'} {period})",
            *(f"(Repairing {s} {period})" for s in action.repairs),
            *(f"(InService {s} {period})" for s in cell.sites if s not in action.repairs)]


def statements(net, rates, state, values):
    """Snapshot rules, including exclusive positive repair branches. Current
    inspected values are facts; only the next period is predicted."""
    cell = net.cell
    # These are alternative sufficient causes, not exhaustive conditional
    # models of Fuelled. A CTV negative branch would contradict the other
    # cause when its own antecedent is false (e.g. stock with no fuel arrival).
    rules = [r.replace("(CTV (STV 1 1) (STV 0 1))", "(STV 1 1)") if "fuelled-by" in r else r
             for r in metta._cycle_rules(net, positive_actions=True)]
    for site, kind in cell.sites.items():
        current = state.degraded[list(cell.sites).index(site)]
        for repairing, head in [(False, "InService"), (True, "Repairing")]:
            p = rates.degraded_rate(kind, float(current and not repairing))
            rules.append(f"(: (no_inverse degradation-{site}-{head}) (Implication ({head} {site} now) (Degraded {site} later)) (STV {p} 1))")
        rules.append(f"(: d-{site} (Degraded {site} now) (STV {int(current)} 1))")
        if site in cell.inputs:
            rules += [f"(: input-{site}-{t} (InputArrived {site} {t}) (STV 1 1))" for t in ("now", "later")]
        rules.append(f"(: value-{site} (OutputValue {site} {values.output[kind]}) (STV 1 1))")
    p = rates.storm_rate("coastal", float(state.storm))
    rules += [f"(: future-storm (Storm north later) (STV {p} 1))", "(: link (NextPeriod now later) (STV 1 1))"]
    for route, blocked in zip(cell.fuel_routes, state.blocked):
        rules += [f"(: blocked-{route.name} (Blocked {route.name} now) (STV {int(blocked)} 1))",
                  f"(: (no_inverse block-{route.name}) (Implication (Storm north later) (Blocked {route.name} later)) "
                  f"(CTV (STV {rates.block_given_storm[route.kind]} 1) (STV {rates.block_without_storm} 1)))"]
    rules += [f"(: incoming (FuelArrived now) (STV {int(state.fuel)} 1))",
              "(: (no_inverse outputs) (Implication (And (OutputValue $s $v) (Producing $s $t)) (Outcome $t (Output $s) $v)) (CTV (STV 1 1) (STV 0 1)))",
              f"(: (no_inverse inventory) (Implication (And (FuelVia $r later) (Producing mine later) (Not (Blocked $r later))) (Outcome later Inventory {values.fuel})) (CTV (STV 1 1) (STV 0 1)))",
              # Gives a zero-value outcome even when no production is proved.
              "(: empty-now (Outcome now Empty 0) (STV 1 1))",
              "(: empty-later (Outcome later Empty 0) (STV 1 1))"]
    return rules


def pair_query(cell, first, second):
    facts = action_facts(cell, first, "now") + action_facts(cell, second, "later")
    return f"(: $p (Assuming (And {' '.join(facts)}) (Outcome $t $o $v)) $tv)"


def outcomes(answers, goal_index=2):
    values = {}
    for answer in answers:
        goal = answer.statement[goal_index]
        if answer.tv[0] == "STV":
            key = repr(goal[1:-1])
            old = values.get(key)
            if old is None or answer.tv[2] > old[1]:
                values[key] = (float(goal[-1])*answer.tv[1], answer.tv[2])
    return sum(v for v, c in values.values()) if values else None


def decide(payload):
    started = time.perf_counter()
    if payload.get("python_path"):
        sys.path.insert(0, payload["python_path"])
    module = "mm2_chainer.petta" if payload["backend"] == "mm2" else "pettachainer"
    c = importlib.import_module(module).PeTTaChainer()
    clock = c.timing() if hasattr(c, "timing") else None
    c.set_evidence_confidence_k(5)
    net = network(payload["plants"])
    cell, values = net.cell, Values(**payload["values"])
    state = State(**payload["state"])
    choices = options(cell, values.capacity)
    unsupported = []
    for head in metta.complete_predicates(net):
        if hasattr(c, "set_complete_predicate"):
            c.set_complete_predicate(head)
        else:
            unsupported.append(f"set_complete_predicate {head}")
    for head in ("InputArrived", "OutputValue", "FuelVia", "StockedFuel", "Unstocked", "InService", "Repairing", "NextPeriod"):
        if hasattr(c, "set_positive_only_predicate"):
            c.set_positive_only_predicate(head)
        else:
            unsupported.append(f"set_positive_only_predicate {head}")
    c.add_atoms_no_check(statements(net, Rates(persist=0.7), state, values))
    setup = time.perf_counter()-started
    queries = 0

    def evaluate(pairs):
        nonlocal queries
        queries += len(pairs)
        goals = [pair_query(cell, a, b) for a, b in pairs]
        rows = c.query_many_refs(goals, steps=payload["budget"]*len(goals))
        return {pair: None if (v := outcomes(row)) is None else v-values.cost_of(pair[0])-values.cost_of(pair[1])
                for pair, row in zip(pairs, rows, strict=True)}

    cap, beam = payload["pair_cap"], payload["beam"]
    if payload["proposer"] == "open":
        # The game supplies the legal action vocabulary, not its ranking.
        evaluated = evaluate([(a, b) for a in choices if a.units <= 1 for b in (Action(), Action(stock=True))][:cap])
        available = {p: v for p, v in evaluated.items() if v is not None}
        first_scores = {a: max(v for (first, _), v in available.items() if first == a) for a, _ in available}
        firsts = [(a, Action()) for a in sorted(first_scores, key=first_scores.get, reverse=True)[:beam]]
        second_count = min(len(choices), max(0, (cap-len(evaluated))//max(1, len(firsts))))
        rules = []
        for i, b in enumerate(choices[:second_count]):
            facts = " ".join(action_facts(cell, b, "later"))
            rules.append(f"(: (no_inverse choice-{i}) (Implication (Assuming (And {facts}) (Outcome $t $o $v)) "
                         f"(SecondOutcome b{i} $t $o $v)) (CTV (STV 1 1) (STV 0 1)))")
        if rules:
            c.add_atoms_no_check(rules)
        goals = [f"(: $p (Assuming (And {' '.join(action_facts(cell, a, 'now'))}) (SecondOutcome $b $t $o $v)) $tv)" for a, _ in firsts]
        rows = c.query_many_refs(goals, steps=payload["budget"]*max(1, len(firsts)*second_count)) if second_count else []
        queries += len(goals) if second_count else 0
        open_values = {}
        for (a, _), row in zip(firsts, rows):
            by_choice = {}
            for answer in row:
                b = int(answer.statement[2][1][1:])
                by_choice.setdefault(b, []).append(answer)
            for b, answers in by_choice.items():
                v = outcomes(answers)
                if v is not None:
                    open_values[a, choices[b]] = v-values.cost_of(a)-values.cost_of(choices[b])
        ranked = open_values or available
        candidate_pairs = len(evaluated)+len(firsts)*second_count
        context_bound = candidate_pairs + len(firsts)
        open_answered = len(open_values)
    else:
        evaluated = propose_pairs(choices, values, evaluate, cap, beam, payload["proposer"] == "exhaustive")
        ranked = {p: v for p, v in evaluated.items() if v is not None}
        candidate_pairs = context_bound = len(evaluated)
        open_answered = 0
    chosen = max(ranked, key=ranked.get)[0] if ranked else Action()
    result = {"chosen": choices.index(chosen), "candidate_pairs": candidate_pairs, "context_bound": context_bound,
              "decision_queries": queries, "decision_answered": len(ranked), "open_answered": open_answered,
              "worker_setup_seconds": setup, "unsupported": unsupported,
              "pair_scores": [[choices.index(a), choices.index(b), v] for (a, b), v in evaluated.items()]}
    if clock is not None:
        after = c.timing()
        result.update({f"chainer_{part}_seconds": after[part]-clock[part] for part in ("wall", "inference", "overhead")})
    return result


if __name__ == "__main__":
    print(json.dumps(decide(json.load(sys.stdin))), flush=True)
