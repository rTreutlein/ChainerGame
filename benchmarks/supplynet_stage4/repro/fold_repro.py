"""Gap 4 (docs/supply_network.md, "Stage 4 as built"): a FoldAllTruth over
one decision's outcomes, on SupplyNet stage 4's round-1 knowledge base, with
the outcomes themselves for comparison. Each budget on a fresh knowledge
base. Usage: fold_repro.py stage4_kb.json STEPS"""
import json, re, sys
from pettachainer import PeTTaChainer
d = json.load(open(sys.argv[1]))
h = PeTTaChainer()
h.set_evidence_confidence_k(d["k"])
for head in d["complete"]:
    h.set_complete_predicate(head)
fold = ("(FoldAllTruth (Outcome $t $o $v) (STV $s $_c) (Weighted $s $v) 0.0 "
        "(|-> ($acc (Weighted $p $x)) (+ $acc (* $p $x))) -> $g)")
h.add_atoms_no_check(d["atoms"] + [f"(: (no_inverse decision-value) (Implication (And (Decision $t) {fold}) (DecisionValue $t $g)) (CTV (STV 1 1) (STV 0 1)))"])
facts = "(Decision t1) (FuelVia rf t1) (StockedFuel t1) (Repairing mine t1) (InService power-plant t1) (InService plant-a t1) (InService plant-b t1)"
steps = int(sys.argv[2])
answers = h.query_many_refs([f"(: $prf (Assuming (And {facts}) (DecisionValue t1 $g)) $tv)"], steps=steps)[0]
proof = h.expand_proof(answers[0].proof) if answers else ""
print(steps, "fold", [a.statement[2][2] for a in answers], "confidence", [round(a.tv[2], 3) for a in answers],
      re.findall(r"partial-fold [0-9]+ [0-9]+", proof)[:1], flush=True)
outcomes = h.query_many_refs([f"(: $prf (Assuming (And {facts}) (Outcome t1 $o $v)) $tv)"], steps=steps)[0]
print(steps, "outcomes summed", round(sum(a.statement[2][3] * a.tv[1] for a in outcomes), 4), len(outcomes), "members", flush=True)
