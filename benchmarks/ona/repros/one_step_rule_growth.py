"""Per-decision cost of refined one-step rules grows with the history when
their condition predicate also holds false facts.

The one-step Pong agent's world rules, refined, k = 1, over time points:

    (And (NextPeriod $t $u) (C x $t) (Did op $t)) -> (Ball y $u)
    (And (NextPeriod $t $u) (C y $t)) -> (Hit $t)

with every past (Ball y d) and (Hit d) stated true or false (the harness's
closed-world interim, closed_world_past_events.py). Each decision adds one
time point's facts and reads the truths of the rules of its observation
(budget at each rule's first read, as the agent does).

- ball: the condition is Ball itself (C = Ball), which holds false facts, so
  completions of the refined rules are compiled onto it. Under the open
  nodes they search, a row is built for every past time point's certain
  (Ball x d), each reading the rule's truth, which every sample changes, and
  joins wait on that truth for good.
- at: the condition is a positive-only At holding the same observation
  (only true facts): nothing is compiled onto it. This is what onabench.pong
  does.

A closed-world reading of observed predicates at past time points would let
one predicate serve both roles; the chainer would also need to build no
refined-rule row concluding a certain statement, and no waiter on a certain
premise (docs/pong_one_step.md, "Chainer gaps").

    python one_step_rule_growth.py [ball|at] [DECISIONS]
"""
import random
import statistics
import sys
import time

from pettachainer import PeTTaChainer

cond = "Ball" if (sys.argv[1] if len(sys.argv) > 1 else "ball") == "ball" else "At"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BALL, OPS = ("right", "left", "equal"), ("left", "right", "stop", "nothing")
h = PeTTaChainer()
h.set_evidence_confidence_k(1)
h.set_rule_refinement(True)
for p in ("Did", "NextPeriod") + (("At",) if cond == "At" else ()):
    h.set_positive_only_predicate(p)
world = {(x, o, y): f"(Implication (And (NextPeriod $t $u) ({cond} {x} $t) (Did {o} $t)) (Ball {y} $u))"
         for x in BALL for o in OPS for y in BALL}
goal = {y: f"(Implication (And (NextPeriod $t $u) ({cond} {y} $t)) (Hit $t))" for y in BALL}
h.add_atoms_no_check([f"(: r{i} {r} (STV 0.5 0.01))" for i, r in enumerate(list(world.values()) + list(goal.values()))])
rng, read, ms = random.Random(1), set(), []
for d in range(1, n + 1):
    x = rng.choice(BALL)
    facts = [f"(: h{d - 1} (Hit d{d - 1}) (STV {int(rng.random() < 0.05)} 1))"] if d > 1 else []
    facts += [f"(: b{d}{y} (Ball {y} d{d}) (STV {int(y == x)} 1))" for y in BALL]
    facts += [f"(: a{d} (At {x} d{d}) (STV 1 1))"] if cond == "At" else []
    facts += [f"(: n{d} (NextPeriod d{d - 1} d{d}) (STV 1 1))"] if d > 1 else []
    t0 = time.perf_counter()
    h.add_atoms_no_check(facts)
    rules = [world[(x, o, y)] for o in OPS for y in BALL] + list(goal.values())
    h.query_many_refs([f"(: $p (RuleTruth {r}) $tv)" for r in rules], steps=0 if read.issuperset(rules) else 200)
    read.update(rules)
    h.add_atoms_no_check([f"(: o{d} (Did {rng.choice(OPS)} d{d}) (STV 1 1))"])
    ms.append(1000 * (time.perf_counter() - t0))
    if d % 100 == 0:
        print(f"{cond}: decisions {d - 99}-{d}: mean {statistics.fmean(ms[-100:]):.0f} ms", flush=True)
