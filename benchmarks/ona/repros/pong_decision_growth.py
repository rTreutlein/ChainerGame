"""Per-decision cost of the Pong agent grows with the history.

A refined rule (Ball right $t) -> (Reward right $t) (STV 0.5 0.01), k = 1,
positive-only Ball. Each "decision" adds (Ball right dN) and queries
(Reward right dN) with budget 200; every 15th decision also gets an outcome
(Reward right dN). The queries per decision stay the same, the rule set stays
the same, yet the time per decision grows with the number of decisions so far.

    python pong_decision_growth.py [N] [--metrics]
"""
import random
import sys
import time

from pettachainer import PeTTaChainer

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
h.set_positive_only_predicate("Ball")
metrics = "--metrics" in sys.argv
if metrics:
    h.set_query_metrics(True)
h.add_atoms_no_check(["(: h1 (Implication (Ball right $t) (Reward right $t)) (STV 0.5 0.01))"])
rng = random.Random(1)
window, outcome = [], []
for d in range(1, n + 1):
    t0 = time.perf_counter()
    c0 = h.timing()
    h.add_atoms_no_check([f"(: b{d} (Ball right d{d}) (STV 1 1))"])
    clock = h.timing()
    add = (clock["wall"] - c0["wall"], clock["inference"] - c0["inference"])
    (rows,) = h.query_many_refs([f"(: $p (Reward right d{d}) $tv)"], steps=200)
    now = h.timing()
    if d % 15 == 0:
        c1 = h.timing()
        h.add_atoms_no_check([f"(: r{d} (Reward right d{d}) (STV {int(rng.random() < 0.7)} 1))"])
        outcome.append(h.timing()["wall"] - c1["wall"])
    q = (now["wall"] - clock["wall"], now["inference"] - clock["inference"])
    window.append((time.perf_counter() - t0, add, q))
    if d % 500 == 0:
        def mean(f):
            return 1000 * sum(f(w) for w in window) / len(window)
        tv = rows[0].tv if rows else None
        print(f"decisions {d}: {mean(lambda w: w[0]):.1f} ms per decision; add {mean(lambda w: w[1][0]):.1f} ms"
              f" (inference {mean(lambda w: w[1][1]):.1f}); query {mean(lambda w: w[2][0]):.1f} ms"
              f" (inference {mean(lambda w: w[2][1]):.1f}); outcome add {1000 * sum(outcome) / len(outcome):.1f} ms;"
              f" answer {tv}", flush=True)
        if metrics:
            print("   ", {k: v for k, v in h.last_query_metrics().items() if v}, flush=True)
        window, outcome = [], []
