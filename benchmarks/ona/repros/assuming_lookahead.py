"""Lookahead by an Assuming query per operation does not chain through the
one-step world rules: on a trained one-step Pong agent (onabench.pong, 400
steps of pong2, seed 1, PeTTaChainer 2307ca50 or later), the agent's truth
reads separate the operations (P(hit next | stop) 0.080, | left 0.0005), while
(Assuming (Did op d) (Hit d+1)), with (NextPeriod d d+1) stored, answers the
same (Hit 0.0163) for every operation, at 1 to 12 s per query. Measured with
the goal rules over Ball, (And (NextPeriod $t $u) (Ball y $t)) -> (Hit $t),
before the agent split its condition into a positive-only At
(one_step_rule_growth.py); with At the goal rules read no world rule's
conclusion, so the knowledge base has no chain for the query to follow and
it answers from Hit's own rules.

    PYTHONPATH=src python benchmarks/ona/repros/assuming_lookahead.py pong2 400
"""
import sys, time
sys.path.insert(0, "src")
from onabench.pong import Pong, PLNAgent, play, BALL, world_rule, goal_rule
variant, n = sys.argv[1], int(sys.argv[2])
agent = PLNAgent(variant, 1, 200)
env = Pong(variant, 1)
play(env, agent, n)
obs = env.observe()
agent.decide(obs)          # states the last outcome and this observation, reads the truths
d, x = agent.d, agent.timing[-1]["x"]
print("x", x, "agent desire", agent.timing[-1]["desire"])
reads = [world_rule(x, op, y) for op in agent.ops for y in BALL] + [goal_rule(y, g) for y in BALL for g in ("Hit", "Miss")]
ans = agent.h.query_many_refs([f"(: $p (RuleTruth {r}) $tv)" for r in reads], steps=0)
s = {r: max((float(a.tv[2]), float(a.tv[1])) for a in rows)[1] for r, rows in zip(reads, ans)}
agent.h.add_atoms_no_check([f"(: nextp (NextPeriod d{d} d{d + 1}) (STV 1 1))"])
for op in agent.ops:
    p = [s[world_rule(x, op, y)] for y in BALL]
    q = {g: sum(pi * s[goal_rule(y, g)] for pi, y in zip(p, BALL)) / sum(p) for g in ("Hit", "Miss")}
    t0 = time.perf_counter()
    rows = agent.h.query_many_refs([f"(: $p (Assuming (Did {op} d{d}) ({g} d{d + 1})) $tv)" for g in ("Hit", "Miss")], steps=200)
    dt = time.perf_counter() - t0
    ch = [[(round(float(a.tv[1]), 4), round(float(a.tv[2]), 3)) for a in r if a.tv[0] == "STV"] for r in rows]
    print(f"{op:8s} P(y'|x,op) {[round(v / sum(p), 3) for v in p]}  mixture Hit {q['Hit']:.4f} Miss {q['Miss']:.4f}"
          f"  | Assuming Hit {ch[0]} Miss {ch[1]}  ({1000 * dt:.0f} ms)")
