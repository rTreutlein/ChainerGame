"""Per-decision cost of the Pong agent grows with the history.

A refined rule (Ball right $t) -> (Reward right $t) (STV 0.5 0.01), k = 1,
positive-only Ball. Each "decision" adds (Ball right dN) and reads the rule's
truth, (RuleTruth ...) with budget 0 (200 at the first read, which folds the
instances), as the Pong agent does; every 15th decision also gets an outcome
(Reward right dN). The rule set and the work per decision stay the same:
the instance tallies (docs/metta/hypothesis_rules.md, "Instance tallies")
should make the read cost constant, yet it grows with the number of
decisions so far.

    python pong_decision_growth.py [N] [--instance] [--metrics] [--forget HORIZON]

--instance also queries the instance (Reward right dN) with budget 200, as
the first Pong agent did (its answer equals the truth). --forget sets
set-forgetting(HORIZON, 1) with one advance-epoch per decision.
"""
import random
import sys

from pettachainer import PeTTaChainer

n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
instance = "--instance" in sys.argv
metrics = "--metrics" in sys.argv
forget = int(sys.argv[sys.argv.index("--forget") + 1]) if "--forget" in sys.argv else 0
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
h.set_positive_only_predicate("Ball")
if metrics:
    h.set_query_metrics(True)
if forget:
    h.set_forgetting(forget, 1.0)
h.add_atoms_no_check(["(: h1 (Implication (Ball right $t) (Reward right $t)) (STV 0.5 0.01))"])
rng = random.Random(1)
window, outcome = [], []
for d in range(1, n + 1):
    c0 = h.timing()
    h.add_atoms_no_check([f"(: b{d} (Ball right d{d}) (STV 1 1))"])
    c1 = h.timing()
    (truth,) = h.query_many_refs(["(: $p (RuleTruth (Implication (Ball right $t) (Reward right $t))) $tv)"],
                                 steps=200 if d == 1 else 0)
    c2 = h.timing()
    if instance:
        h.query_many_refs([f"(: $p (Reward right d{d}) $tv)"], steps=200)
    c3 = h.timing()
    if d % 15 == 0:
        h.add_atoms_no_check([f"(: r{d} (Reward right d{d}) (STV {int(rng.random() < 0.7)} 1))"])
        outcome.append(h.timing()["wall"] - c3["wall"])
    if forget:
        h.advance_epoch()
    window.append((h.timing()["wall"] - c0["wall"], c1["wall"] - c0["wall"], c2["wall"] - c1["wall"],
                   c2["inference"] - c1["inference"], c3["wall"] - c2["wall"]))
    if d % 500 == 0:
        def mean(i):
            return 1000 * sum(w[i] for w in window) / len(window)
        print(f"decisions {d}: {mean(0):.1f} ms per decision; add {mean(1):.2f} ms; truth read {mean(2):.2f} ms"
              f" (inference {mean(3):.2f}); instance query {mean(4):.2f} ms;"
              f" outcome add {1000 * sum(outcome) / len(outcome):.2f} ms; truth {truth[0].tv if truth else None}",
              flush=True)
        if metrics:
            print("   ", {k: v for k, v in h.last_query_metrics().items() if v}, flush=True)
        window, outcome = [], []
