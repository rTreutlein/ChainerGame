"""An event that was not observed at a past time point was no negative sample.

Anticipation (ONA): an agent that acted on a rule predicting an event at the
next time point, and then observed that it did not happen, has negative
evidence for the rule. In PeTTaChainer the rule's instance tally counts an
instance only where the consequent's truth is known, and an unobserved
statement is unknown (open world), so the failed prediction is not a sample.

World-model rule, refined, k = 1:

    (And (NextPeriod $t $u) (Ball right $t) (Did right $t)) -> (Ball equal $u)

Three instances: at d2 the ball was observed equal (a positive sample); at
d3 and d4 it was observed right, so (Ball equal d3) and (Ball equal d4) did
not happen. The truth should rest on 1 positive and 2 negative samples
(strength 0.33, confidence 0.75).

- open: (Ball equal d3) reads 0.995 (the rule's own prediction) and the
  rule's truth rests on the positive sample alone (0.995, c 0.50).
- complete: set-complete-predicate Ball changes nothing. A complete
  statement is read false only where no rule application derives it, and
  the rule itself derives it; and an instance fold does not see closed
  reads (docs/metta/complete_predicates.md, "Partial folds").
- explicit: the harness states (Ball equal d3) and (Ball equal d4) false:
  0.33, c 0.75. This is what the Pong agent did as an interim.
- observed: PeTTaChainer's observed predicates (set_observed_predicate,
  branch miner-failures, docs/metta/observed_predicates.md): the link into
  a time point stores false each unobserved statement there that a rule
  predicts, here (Ball equal d3) and (Ball equal d4): 0.33, c 0.75. Each
  time point's observations are stated before the link into it.

    python closed_world_past_events.py [open|complete|explicit|observed]
"""
import sys

from pettachainer import PeTTaChainer

mode = sys.argv[1] if len(sys.argv) > 1 else "open"
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
h.set_positive_only_predicate("Did")
h.set_positive_only_predicate("NextPeriod")
if mode == "complete":
    h.set_complete_predicate("Ball")
if mode == "observed":
    h.set_predicate_time("Ball", 2, 0)
    h.set_time_link("NextPeriod")
    h.set_observed_predicate("Ball")
rule = "(Implication (And (NextPeriod $t $u) (Ball right $t) (Did right $t)) (Ball equal $u))"
h.add_atoms_no_check([f"(: w {rule} (STV 0.5 0.01))"])
facts = ["(: b1 (Ball right d1) (STV 1 1))", "(: o1 (Did right d1) (STV 1 1))",
         "(: b2 (Ball equal d2) (STV 1 1))", "(: b2r (Ball right d2) (STV 1 1))", "(: n1 (NextPeriod d1 d2) (STV 1 1))",
         "(: o2 (Did right d2) (STV 1 1))",
         "(: b3 (Ball right d3) (STV 1 1))", "(: n2 (NextPeriod d2 d3) (STV 1 1))", "(: o3 (Did right d3) (STV 1 1))",
         "(: b4 (Ball right d4) (STV 1 1))", "(: n3 (NextPeriod d3 d4) (STV 1 1))"]
if mode == "explicit":
    facts += ["(: f3 (Ball equal d3) (STV 0 1))", "(: f4 (Ball equal d4) (STV 0 1))"]
h.add_atoms_no_check(facts)
for d in ("d3", "d4"):
    (rows,) = h.query_many_refs([f"(: $p (Ball equal {d}) $tv)"], steps=200)
    print(f"(Ball equal {d})", [a.tv for a in rows])
(rows,) = h.query_many_refs([f"(: $p (RuleTruth {rule}) $tv)"], steps=200)
print(mode, "rule truth", [a.tv for a in rows], "expected 0.33, c 0.75")
