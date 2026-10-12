"""An event that was not observed at a past time point is no negative sample.

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
  0.33, c 0.75. This is what the Pong agent does (onabench.pong, interim).

The missing feature is a closed-world reading of observed predicates at
time points that have passed: for a predicate declared observed (its time
argument from set-predicate-time), a ground statement at a time point that
has a successor (a time-link fact (NextPeriod d d')) and no stored fact
reads as a stored (STV 0 1) observation, a leaf that applies no rules, and
instance folds read it as their samples' consequent.

    python closed_world_past_events.py [open|complete|explicit]
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
rule = "(Implication (And (NextPeriod $t $u) (Ball right $t) (Did right $t)) (Ball equal $u))"
h.add_atoms_no_check([f"(: w {rule} (STV 0.5 0.01))"])
facts = [f"(: n{i} (NextPeriod d{i} d{i + 1}) (STV 1 1))" for i in (1, 2, 3)]
facts += ["(: b1 (Ball right d1) (STV 1 1))", "(: o1 (Did right d1) (STV 1 1))",
          "(: b2 (Ball equal d2) (STV 1 1))",
          "(: b2r (Ball right d2) (STV 1 1))", "(: o2 (Did right d2) (STV 1 1))",
          "(: b3 (Ball right d3) (STV 1 1))", "(: o3 (Did right d3) (STV 1 1))",
          "(: b4 (Ball right d4) (STV 1 1))"]
if mode == "explicit":
    facts += ["(: f3 (Ball equal d3) (STV 0 1))", "(: f4 (Ball equal d4) (STV 0 1))"]
h.add_atoms_no_check(facts)
for d in ("d3", "d4"):
    (rows,) = h.query_many_refs([f"(: $p (Ball equal {d}) $tv)"], steps=200)
    print(f"(Ball equal {d})", [a.tv for a in rows])
(rows,) = h.query_many_refs([f"(: $p (RuleTruth {rule}) $tv)"], steps=200)
print(mode, "rule truth", [a.tv for a in rows], "expected 0.33, c 0.75")
