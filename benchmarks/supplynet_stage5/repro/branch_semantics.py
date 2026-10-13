"""CTV negative branches must not define alternative production causes."""
from pettachainer import PeTTaChainer
from supplynet import metta
from supplynet.decisions import Values, options
from supplynet.lookahead import State, network, pair_values, transition
from supplynet.lookahead_worker import outcomes, pair_query, statements
from supplynet.world import Action, Rates

net, rates, values = network(1), Rates(persist=0.7), Values()
state = State(False, (False, False, False), (False, False), False)
choices = options(net.cell, 2)
pair = Action(stock=True), Action()
exact, _ = pair_values(net.cell, values, state, choices, {a: list(transition(net.cell, rates, state, a)) for a in choices})
print("exact", exact[pair], flush=True)
for positive in (False, True):
    c = PeTTaChainer()
    c.set_evidence_confidence_k(5)
    for head in metta.complete_predicates(net):
        c.set_complete_predicate(head)
    for head in ("InputArrived", "OutputValue", "FuelVia", "StockedFuel", "Unstocked", "InService", "Repairing", "NextPeriod"):
        c.set_positive_only_predicate(head)
    rules = statements(net, rates, state, values)
    if not positive:
        rules = [r.replace("(STV 1 1))", "(CTV (STV 1 1) (STV 0 1)))") if "fuelled-by" in r else r for r in rules]
    c.add_atoms_no_check(rules)
    for budget in (50, 400):
        answers = c.query_many_refs([pair_query(net.cell, *pair)], steps=budget)[0]
        print("positive alternatives", positive, "budget", budget, "value", outcomes(answers)-2, flush=True)
