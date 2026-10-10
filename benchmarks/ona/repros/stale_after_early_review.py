"""Two refined hypotheses with the same consequent and co-extensive antecedents.
A review made before any consequent is observed (so each rule's samples are the
other rule's derivations) freezes both rule truths: later observations never
reach them. Without the early review the same final query learns them."""
import re, sys
from pettachainer import PeTTaChainer
tv = lambda r: re.findall(r"\(STV [^()]*\)\)$", r[0])[0] if r else "none"
early_review = sys.argv[1] == "early-review"
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
for p in ("A", "C"): h.set_positive_only_predicate(p)
h.add_atoms_no_check(["(: hA (Implication (A $x) (B $x)) (STV 0.5 0.01))",
                      "(: hC (Implication (C $x) (B $x)) (STV 0.5 0.01))"])
rts = ["(: $p (RuleTruth (Implication (A $x) (B $x))) $tv)", "(: $p (RuleTruth (Implication (C $x) (B $x))) $tv)"]
h.add_atoms_no_check(["(: a0 (A e0) (STV 1 1))", "(: c0 (C e0) (STV 1 1))"])
if early_review:
    print("early review", [tv(r) for r in h.query_many(rts, steps=200, timeout_sec=None)])
for i in range(1, 5):
    h.add_atoms_no_check([f"(: a{i} (A i{i}) (STV 1 1))", f"(: c{i} (C i{i}) (STV 1 1))", f"(: b{i} (B i{i}) (STV 0 1))"])
print("after 4 observations of B false:", [tv(r) for r in h.query_many(rts, steps=200, timeout_sec=None)])
h.add_atoms_no_check(["(: anew (A new) (STV 1 1))", "(: cnew (C new) (STV 1 1))"])
print("applied (B new):", tv(h.query("(: $p (B new) $tv)", steps=200, timeout_sec=None)))
