"""Three-part antecedents get no instance fold. Without stored rules, the
implication query answers nothing for (And P Q R) while (And P Q) is folded;
a refined rule with three parts keeps its axiom while one with two parts
learns. (ONA's hypotheses are sequences of up to three events plus the
operation.)"""
import re
from pettachainer import PeTTaChainer
tv = lambda r: [re.findall(r"\((?:STV|CTV).*\)\)$", x)[0] for x in r]
def facts(h):
    for i in range(1, 4):
        h.add_atoms_no_check([f"(: p{i} (P t{i}) (STV 1 1))", f"(: q{i} (Q t{i}) (STV 1 1))",
                              f"(: r{i} (R t{i}) (STV 1 1))", f"(: g{i} (G t{i}) (STV 1 1))"])
plain = PeTTaChainer()
plain.set_evidence_confidence_k(1)
facts(plain)
for q in ["(: $p (Implication (And (P $t) (Q $t)) (G $t)) $tv)",
          "(: $p (Implication (And (P $t) (Q $t) (R $t)) (G $t)) $tv)"]:
    print("no rules:", q, "->", tv(plain.query(q, steps=200, timeout_sec=None)))
h = PeTTaChainer()
h.set_rule_refinement(True)
h.add_atoms_no_check(["(: h3 (Implication (And (P $t) (Q $t) (R $t)) (G $t)) (STV 0.5 0.01))",
                      "(: h2 (Implication (And (P $t) (Q $t)) (G $t)) (STV 0.5 0.01))"])
facts(h)
for q in ["(: $p (RuleTruth (Implication (And (P $t) (Q $t)) (G $t))) $tv)",
          "(: $p (RuleTruth (Implication (And (P $t) (Q $t) (R $t)) (G $t))) $tv)"]:
    print("refined:", q, "->", tv(h.query(q, steps=200, timeout_sec=None)))
