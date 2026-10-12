"""The miner never proposes a hypothesis for an action that never succeeds.

Pong agent with hypothesis mining: in context (Ball right t), the operation
"right" is rewarded 4 times in 6, "left" never (6 failures). Mining proposes
(Ball $x $t) -> (Reward $x $t)-style rules from the true rewards only; a
false fact proposes only where a live hypothesis already predicted it. So
"left" gets no hypothesis: the agent cannot learn that it fails, and its
desire stays at the neutral 0.5 (or the population prior of the stored
rewards, which the agent's own choices bias), above the learned rate of
"right" whenever that is below 0.5. ONA learns the failure through
anticipation (negative evidence when a predicted consequence does not come).

Prototype on PeTTaChainer WIP branch miner-failures (frozen, not landed;
design in docs/pong_one_step.md): a false fact also proposes the candidates
concluding it whose consequent pattern no live hypothesis concluded before
it. "left" then gets (Ball right $t) -> (Reward left $t), which reads 0.0008
after its six failures, and the query answers it instead of the population
prior (0.34 on master 2307ca50).

    python mined_never_rewarded.py
"""
from pettachainer import PeTTaChainer

h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_positive_only_predicate("Ball")
h.set_hypothesis_mining(4, 8)
h.set_predicate_time("Ball", 2, 0)
h.set_predicate_time("Reward", 2, 1)
for d in range(1, 13):
    op = "right" if d % 2 else "left"
    hit = op == "right" and d % 3 != 0
    h.add_atoms_no_check([f"(: b{d} (Ball right d{d}) (STV 1 1))",
                          f"(: r{d} (Reward {op} d{d}) (STV {int(hit)} 1))"])
h.add_atoms_no_check(["(: b13 (Ball right d13) (STV 1 1))"])
for v in h.review_hypotheses("(Reward $op d13)", steps=200):
    print(v)
for op in ("right", "left"):
    (rows,) = h.query_many_refs([f"(: $p (Reward {op} d13) $tv)"], steps=200)
    print(op, [(r.proof if isinstance(r.proof, str) else r.proof[0], r.tv) for r in rows])
