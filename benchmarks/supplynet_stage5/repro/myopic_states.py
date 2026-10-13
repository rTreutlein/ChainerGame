"""Enumerate planning states to locate timing tradeoffs (not a rollout score).
Uniform counts here are structural coverage, not a state visitation prior.
"""
import itertools
import json
from supplynet.decisions import Values, best, options
from supplynet.lookahead import State, myopic_values, network, pair_values, transition
from supplynet.world import Rates

cell, rates, values = network(1).cell, Rates(persist=0.7), Values()
choices = options(cell, 2)
witnesses = []
for bits in itertools.product((False, True), repeat=7):
    state = State(bits[0], bits[1:4], bits[4:6], bits[6])
    _, qs = pair_values(cell, values, state, choices, {a: list(transition(cell, rates, state, a)) for a in choices})
    reference, myopic = best(qs), best(myopic_values(cell, values, state, choices))
    regret = qs[reference]-qs[myopic]
    if regret > 1e-8:
        witnesses.append({"state": bits, "reference": repr(reference), "myopic": repr(myopic), "regret": regret})
print(json.dumps({"type": "probe", "states": 128, "myopic_suboptimal": len(witnesses),
                  "max_myopic_regret": max(w["regret"] for w in witnesses), "witnesses": witnesses}), flush=True)
