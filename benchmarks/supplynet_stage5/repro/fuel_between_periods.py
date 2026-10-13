"""Fuel produced under the first assumption supplies the following period.
Run with a query budget argument.
Compare a fresh single query with the same pair after proposal probes.
"""
import sys
from pettachainer import PeTTaChainer
from supplynet import metta
from supplynet.decisions import Values, options
from supplynet.lookahead import State, network, pair_values, transition
from supplynet.lookahead_worker import action_facts, outcomes, pair_query, statements
from supplynet.world import Action, Rates

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 400
net, rates, values = network(1), Rates(persist=0.7), Values()
state = State(False, (False, False, False), (False, False), False)
choices = options(net.cell, 2)
pair = Action(stock=True), Action()
exact, _ = pair_values(net.cell, values, state, choices, {a: list(transition(net.cell, rates, state, a)) for a in choices})
print("exact pair value", exact[pair], "budget", budget, flush=True)
for warm in (False, True):
    c = PeTTaChainer()
    c.set_evidence_confidence_k(5)
    for head in metta.complete_predicates(net):
        c.set_complete_predicate(head)
    for head in ("InputArrived", "OutputValue", "FuelVia", "StockedFuel", "Unstocked", "InService", "Repairing", "NextPeriod"):
        c.set_positive_only_predicate(head)
    c.add_atoms_no_check(statements(net, rates, state, values))
    if warm:
        goals = [pair_query(net.cell, a, b) for a in choices if a.units <= 1 for b in (Action(), Action(stock=True))]
        c.query_many_refs(goals, steps=budget*len(goals))
    rows = c.query_many_refs([pair_query(net.cell, *pair)], steps=budget)[0]
    print("warm", warm, "pair value", outcomes(rows)-values.cost_of(pair[0]), flush=True)
    for answer in rows:
        print(answer.statement, answer.tv, flush=True)
    facts = action_facts(net.cell, pair[0], "now") + action_facts(net.cell, pair[1], "later")
    for target in ("(Producing mine now)", "(FuelArrived later)", "(Producing power-plant later)"):
        print(target, c.query_many_refs([f"(: $p (Assuming (And {' '.join(facts)}) {target}) $tv)"], steps=budget), flush=True)
