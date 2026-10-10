"""Repro: the two other ways to write a source's trust rule.

Run with PeTTaChainer's virtual environment (memory-capped, no display):

    PYTHONPATH=/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer \
      /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
      examples/pettachainer_trust_rule_forms_repro.py < /dev/null

1. **A variable consequent**, ``(Implication (Asserts s $x) $x)``: the claim
   is the statement itself. Neither the rule's ``RuleTruth`` nor a query of a
   claimed statement gets any answer.
2. **Two STV rules**, ``(Implication (Asserts s (Up $n $t)) (Up $n $t))`` and
   the same from ``Denies``, refined from 20 labelled claims each (16 and 4
   of them up). Both rule truths come out right (0.798 and 0.202, confidence
   0.80), but applied to a certain claim the views are pulled to the base
   rate: 0.666 and 0.334. A stated STV rule of 0.8 applied to a certain
   antecedent gives 0.775. At PeTTaChainer master b43ad3de.
"""

import pettachainer


def handler():
    h = pettachainer.PeTTaChainer()
    h.set_evidence_confidence_k(5)
    h.set_rule_refinement(True)
    return h


h = handler()
h.add_atoms_no_check(
    ["(: pos (Implication (Asserts s $x) $x) (STV 0.5 0.02))"]
    + [f"(: a{i} (Asserts s (Up c h{i})) (STV 1 1))" for i in range(20)]
    + [f"(: l{i} (Up c h{i}) (STV {int(i < 16)} 1))" for i in range(20)]
    + ["(: qa (Asserts s (Up c n1)) (STV 1 1))"]
)
print("1. variable consequent, rule truth:", h.query_many(["(: $prf (RuleTruth (Implication (Asserts s $x) $x)) $tv)"], steps=20, timeout_sec=0))
print("1. variable consequent, claimed statement:", h.query_many(["(: $prf (Up c n1) $tv)"], steps=40, timeout_sec=0))

h = handler()
statements = [
    "(: pos (Implication (Asserts s (Up $n $t)) (Up $n $t)) (STV 0.5 0.02))",
    "(: neg (Implication (Denies s (Up $n $t)) (Up $n $t)) (STV 0.5 0.02))",
]
for i in range(20):
    statements += [f"(: a{i} (Asserts s (Up c h{i})) (STV 1 1))", f"(: la{i} (Up c h{i}) (STV {int(i < 16)} 1))"]
for i in range(20, 40):
    statements += [f"(: d{i} (Denies s (Up c h{i})) (STV 1 1))", f"(: ld{i} (Up c h{i}) (STV {int(i < 24)} 1))"]
h.add_atoms_no_check(statements)
for claim in ("Asserts", "Denies"):
    (answer,) = h.query_many([f"(: $prf (RuleTruth (Implication ({claim} s (Up $n $t)) (Up $n $t))) $tv)"], steps=20, timeout_sec=0)
    print(f"2. {claim} rule truth:", answer[0][-50:])
h.add_atoms_no_check(["(: qa (Asserts s (Up c n1)) (STV 1 1))", "(: qd (Denies s (Up c n2)) (STV 1 1))"])
for answer in h.query_many(["(: $prf (Up c n1) $tv)", "(: $prf (Up c n2) $tv)"], steps=40, timeout_sec=0):
    print("2. applied:", answer)
