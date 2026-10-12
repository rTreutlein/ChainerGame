"""Gap 2 (docs/supply_network.md, "Stage 4 as built"): a negated statement of
a complete predicate that has no rules and no fact. The root reads (R t1)
closed (0), but neither (Not (R t1)) nor B, whose rule concludes an open
predicate, has an answer."""
from pettachainer import PeTTaChainer
for declare in (True, False):
    h = PeTTaChainer()
    if declare:
        h.set_complete_predicate("R")
    h.add_atoms_no_check([
        "(: a1 (A t1) (STV 1 1))",
        "(: r-other (R t9) (STV 1 1))",
        "(: rule (Implication (And (A $t) (Not (R $t))) (B $t)) (CTV (STV 0.8 1) (STV 0.1 1)))",
    ])
    for q in ("(: $prf (R t1) $tv)", "(: $prf (Not (R t1)) $tv)", "(: $prf (B t1) $tv)"):
        print("complete" if declare else "open", q, h.query(q, steps=100, timeout_sec=0))
