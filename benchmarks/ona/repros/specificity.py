"""Overlapping hypotheses about one outcome: the specific one is right, the
general one is right on average only. The merge for a new case follows the
general rule instead of the more specific one (no specificity / no
conditioning of one rule on the other)."""
import re
from pettachainer import PeTTaChainer
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
for p in ("S", "L"): h.set_positive_only_predicate(p)
h.add_atoms_no_check(["(: gen (Implication (S a $t) (Win $t)) (STV 0.5 0.01))",
                      "(: spec (Implication (And (S a $t) (L b $t)) (Win $t)) (STV 0.5 0.01))"])
n = 0
for l, win, k in (("b", 1, 3), ("c", 0, 9)):     # with L b: always wins (3 cases); with L c: never (9 cases)
    for _ in range(k):
        n += 1
        h.add_atoms_no_check([f"(: s{n} (S a t{n}) (STV 1 1))", f"(: l{n} (L {l} t{n}) (STV 1 1))", f"(: w{n} (Win t{n}) (STV {win} 1))"])
h.add_atoms_no_check(["(: sx (S a new) (STV 1 1))", "(: lx (L b new) (STV 1 1))"])
tv = lambda r: [re.findall(r"\(STV [^()]*\)\)$", x)[0] for x in r]
print("review", [tv(r) for r in h.query_many(["(: $p (RuleTruth (Implication (S a $t) (Win $t))) $tv)",
      "(: $p (RuleTruth (Implication (And (S a $t) (L b $t)) (Win $t))) $tv)"], steps=200, timeout_sec=None)])
r = h.query("(: $p (Win new) $tv)", steps=200, timeout_sec=None)
print("(Win new), true P = 1.0:", [x[:160] + " ... " + x[-45:] for x in r])
