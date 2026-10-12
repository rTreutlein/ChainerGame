"""ONA's Pong and Pong2 (src/system_tests/Pong_Test.h, Pong2_Test.h) with a
pluggable agent, so ONA and PeTTaChainer play the same game.

The game is a port of ONA's C loop, rule for rule: the same field (50 x 20),
bat (half width 4, Pong2 6), speeds, clamping, resets, the same generator
(ONA's myrand, POSIX's example rand) drawn at the same points, and the same
step order:

1. observe: ``ball_right`` when the bat is left of the ball, ``ball_left``
   when right of it, nothing when level (Pong2: ``ball_equal`` when the ball
   is within the bat's reach, the other two outside it);
2. the goal ``good_nar!`` is input;
3. the ball moves (Pong2: vertically only, as ONA's ``t%2 == -1`` never
   holds) and is scored when it reaches the bat's row (y = 0): a hit within
   the bat's half width, which inputs ``good_nar.``, else a miss; at the bat's
   row or a side wall it restarts at a random place;
4. the operations executed since the last step set the bat's velocity
   (left, then right, then stop: the last one wins) and the bat moves. The
   velocity persists: without an operation the bat keeps moving.

The hit ratio is hits / (hits + misses). ``./NAR pong N`` runs N + 1 steps
(``t++ > iterations``); ``play`` does the same.

``trace`` replays the scripted policy of ``benchmarks/ona/pong/pong_trace.c``,
which compiles ONA's game loops unchanged against stubbed NAR calls, so the
port can be compared with the C game line for line (tests/test_pong.py).

ONA's C game draws its random restarts from the generator its reasoner also
draws from, so no port can replay ``./NAR pong`` bit for bit; the port has
its own generator, seeded per run, and ONA playing it through NAR.py is
checked against the C runs statistically.

Run (PeTTaChainer's venv for the PLN agent; ``--show`` draws the game):

    PYTHONPATH=src python -m onabench.pong pong --agent pln --seed 11 --out run.json
    PYTHONPATH=src python -m onabench.pong pong --agent pln --deadline-ms 50 --show   # real time, watch it
    PYTHONPATH=src python -m onabench.pong pong2 --agent ona --show
    benchmarks/ona/scripts/run_pong.sh benchmarks/ona/scripts/pong_jobs.txt; python -m onabench.pong_report
"""
import argparse
import json
import os
import random
import re
import sys
import time

SZ_X, SZ_Y = 50, 20


class MyRand:
    """ONA's myrand/mysrand (Globals.c) with a 64-bit unsigned long."""

    def __init__(self, seed=1):
        self.next = seed

    def __call__(self):
        self.next = (self.next * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        return ((self.next // 65536) & 0xFFFFFFFF) % 32768


class Pong:
    """One game. ``variant`` is "pong" or "pong2"."""

    def __init__(self, variant="pong", seed=1):
        self.variant = variant
        self.rand = MyRand(seed)
        self.ball_x, self.ball_y = SZ_X // 2, SZ_Y // 5
        self.bat_x, self.bat_vx = 20, 0
        self.bat_width = 4 if variant == "pong" else 6
        self.speed = 2 if variant == "pong" else 3
        self.vx = self.vy = 1
        self.hits = self.misses = 0
        self.ops = ("left", "right") if variant == "pong" else ("left", "right", "stop")

    def observe(self):
        if self.variant == "pong":
            if self.bat_x < self.ball_x:
                return "ball_right"
            if self.ball_x < self.bat_x:
                return "ball_left"
            return None
        if self.bat_x <= self.ball_x - self.bat_width:
            return "ball_right"
        if self.ball_x + self.bat_width < self.bat_x:
            return "ball_left"
        return "ball_equal"

    def advance(self):
        """Move the ball, score it and restart it; "hit", "miss" or None."""
        if self.ball_x <= 0:
            self.vx = 1
        if self.ball_x >= SZ_X - 1:
            self.vx = -1
        if self.ball_y <= 0:
            self.vy = 1
        if self.ball_y >= SZ_Y - 1:
            self.vy = -1
        if self.variant == "pong":
            self.ball_x += self.vx
        self.ball_y += self.vy
        result = None
        if self.ball_y == 0:
            if abs(self.ball_x - self.bat_x) <= self.bat_width:
                self.hits += 1
                result = "hit"
            else:
                self.misses += 1
                result = "miss"
        if self.ball_y == 0 or self.ball_x == 0 or self.ball_x >= SZ_X - 1:
            self.ball_y = SZ_Y // 2 + self.rand() % (SZ_Y // 2)
            self.ball_x = self.rand() % SZ_X
            self.vx = 1 if self.rand() % 2 == 0 else -1
        return result

    def act(self, executed):
        """Apply the operations executed since the last step and move the bat."""
        for op, v in (("left", -self.speed), ("right", self.speed), ("stop", 0)):
            if op in executed and op in self.ops:
                self.bat_vx = v
        lo, hi = (0, SZ_X - 1) if self.variant == "pong" else (-2 * self.bat_width, SZ_X - 1 + self.bat_width)
        self.bat_x = max(lo, min(hi, self.bat_x + int(self.bat_vx * self.bat_width / 2)))

    def ratio(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def render(self):
        """The field as ONA's C game draws it (bat row on top)."""
        w = self.bat_width
        bat_len = 2 * w - 1 if self.variant == "pong" else 2 * w - 1 + min(0, self.bat_x)
        rows = [" " * max(0, self.bat_x - w + 1) + "@" * max(0, bat_len)]
        for y in range(SZ_Y):
            rows.append((" " * self.ball_x + "#" + " " * (SZ_X - self.ball_x - 1) if y == self.ball_y
                         else " " * SZ_X) + "|")
        return "\n".join(rows)


def trace(variant, iterations, seed):
    """The lines pong_trace.c prints for the same arguments."""
    env = Pong(variant, seed)
    out, executed, t = [], set(), 0
    while True:
        t += 1
        if t - 1 > iterations:
            break
        obs = env.observe()
        if obs:
            out.append(f"TRACE belief {obs}")
        out.append("TRACE goal good_nar")
        res = env.advance()
        if res == "hit":
            out.append("TRACE belief good_nar")
        if res:
            out.append("good" if res == "hit" else "bad")
        for op in ("left", "right", "stop"):
            if op in executed and (op in env.ops):
                out.append(f"Exec: op_{op}")
        env.act(executed)
        out.append(f"Hits={env.hits} misses={env.misses} t={t}")
        r = env.rand() % 4
        out.append(f"TRACE policy {r}")
        executed = {("left", "right", "stop", None)[r]} - {None}
    return out


class ONAAgent:
    """ONA through its own misc/Python/NAR.py, given what ONA's C game gives
    it: the observation and the goal as events, ``good_nar.`` at a hit, then
    5 cycles. The operations it executes in any of these cycles reach the bat
    at the step's end, as the C game's flags do."""
    name = "ona"

    def __init__(self, variant, ona_dir, ona_seed=None):
        if ona_seed is not None:
            os.environ["ONA_SEED"] = str(ona_seed)  # read by a build that seeds myrand (run_c.sh); ONA ignores it
        os.chdir(os.path.join(ona_dir, "misc", "Python"))
        sys.path.insert(0, os.getcwd())
        import NAR  # spawns ./../../NAR shell
        self.nar = NAR
        setup = ["*volume=0", "*motorbabbling=0.2"]
        setup += (["*babblingops=2"] if variant == "pong" else ["*babblingops=3", "*setopname 3 ^stop"])
        for line in setup:
            NAR.AddInput(line, Print=False)
        self.pending = set()
        self.timing = []

    def _input(self, line):
        return {e["operator"][1:] for e in self.nar.AddInput(line, Print=False)["executions"]}

    def decide(self, obs):
        t0 = time.perf_counter()
        ops = self.pending | (self._input(f"{obs}. :|:") if obs else set()) | self._input("good_nar! :|:")
        self.pending = set()
        self.timing.append(time.perf_counter() - t0)
        return None, ops

    def feedback(self, result, credited):
        t0 = time.perf_counter()
        ops = self._input("good_nar. :|:") if result == "hit" else set()
        self.timing[-1] += time.perf_counter() - t0
        return ops

    def idle(self):
        t0 = time.perf_counter()
        self.pending = self._input("5")
        self.timing[-1] += time.perf_counter() - t0


class PLNAgent:
    """PeTTaChainer learning which operation gets the ball hit.

    Events are time-indexed facts, one time point d per decision: the
    observation ``(Ball right d)`` (``left``, Pong2 also ``equal``; nothing
    when ONA's game inputs nothing), and, once the ball reaches the bat's row
    within ``credit`` steps after the decision's operation moved the bat, its
    outcome ``(Reward op d)``, (STV 1 1) for a hit and (STV 0 1) for a miss.
    The operations are left, right (Pong2 also stop) and nothing; nothing
    leaves the bat moving as it was. A decision whose operation reached no
    scored ball has no outcome (open world). The chainer has no intervals
    between events, so the harness assigns the delayed outcome: every
    decision of the ``credit`` steps before a score gets it (6 by default,
    about the steps the bat needs to cross to the ball; on seeds 1-3 at 2000
    steps 1 step scored 0.57 on average, 3 0.77, 6 0.81, 10 0.77: one step
    cannot tell tracking the ball from drifting with it).

    Hypotheses are ``(Implication (Ball x $t) (Reward op $t)) (STV 0.5
    0.01)``: refined rules the harness states at an operation's first outcome
    in a context, as the matching-to-sample agent does (onabench.mts), or,
    with ``mined``, that the knowledge base proposes itself
    (``set-hypothesis-mining``; Ball facts are layer 0 and Reward facts layer
    1 of their time point). Evidence k = 1; ``Ball`` is positive-only.

    Decision: the truths of the hypotheses that apply are read (their
    ``RuleTruth`` goals in one query, with budget only at a hypothesis's first
    read: the instance tallies keep the truths current, so later reads are
    reads). With one hypothesis per observation and operation and a stored
    fact as its antecedent, the answer to ``(Reward op d)`` is that truth
    (equal within 2e-8 over 4269 decisions, seed 12), so the read is the
    desire, c(f - 0.5) + 0.5. Mined hypotheses can overlap (a relational and
    a specific one), so that agent reviews them (``review-hypotheses``) and
    then queries ``(Reward op d)`` for the merge. The choice is ONA's
    ``Decision_Suggest`` babbling (probability 0.2 of one of ONA's operations
    unless the best desire exceeds 0.55) and then, with ``policy`` "argmax",
    the best operation, "nothing" included; with "ona", the best only if its
    desire is at least 0.501, else nothing. ONA's threshold presumes a
    rewarded hypothesis reads above 0.5; here every scored ball is a full
    sample and hits start near 0.17, so no operation would pass it and the
    agent would only babble.

    An operation no live hypothesis applies to has desire 0.5 and
    is not queried, as ONA's Decision_Suggest reads desire only through
    hypotheses that match: the query would answer the population prior of
    the stored rewards, which is biased by the agent's own choices (an
    operation taken only where it worked reads near 1) and can beat a learned
    rule in a near tie. Without an observation no hypothesis applies.
    """
    name = "pln"

    def __init__(self, variant, seed, steps, pettachainer_path=None, mined=None, change_hazard=None,
                 policy="argmax"):
        if pettachainer_path:
            sys.path.insert(0, pettachainer_path)
        from pettachainer import PeTTaChainer
        self.h = PeTTaChainer()
        self.h.set_evidence_confidence_k(1)
        self.h.set_rule_refinement(True)
        if change_hazard is not None:
            self.h.set_rule_change_hazard(change_hazard)
        self.h.set_positive_only_predicate("Ball")
        self.mined = mined
        if mined:
            self.h.set_hypothesis_mining(*mined)
            self.h.set_predicate_time("Ball", 2, 0)
            self.h.set_predicate_time("Reward", 2, 1)
        self.babble_ops = ("left", "right") if variant == "pong" else ("left", "right", "stop")
        self.ops = self.babble_ops + ("nothing",)
        self.rng = random.Random(seed * 7919 + 1)
        self.steps = steps
        self.policy = policy
        self.d = 0
        self.context = {}
        self.hyps = set()
        self.reviewed = set()
        self.dropped = 0
        self.timing = []

    def decide(self, obs):
        self.d += 1
        d = self.d
        clock = self.h.timing()
        t0 = time.perf_counter()
        x = obs[len("ball_"):] if obs else None
        desire = dict.fromkeys(self.ops, 0.5)
        matching = []
        if x:
            self.h.add_atoms_no_check([f"(: b{d} (Ball {x} d{d}) (STV 1 1))"])
            if self.mined:
                verdicts = self.h.review_hypotheses(f"(Reward $op d{d})", steps=self.steps)
                self.dropped += sum(v.endswith(" dropped)") for v in verdicts)
                matching = sorted({op for v in verdicts if v.endswith(" kept)") for op in applies(v, x, self.ops)})
                answers = self.h.query_many_refs([f"(: $p (Reward {op} d{d}) $tv)" for op in matching],
                                                 steps=self.steps) if matching else []
            else:
                matching = [op for op in self.ops if (x, op) in self.hyps]
                steps = self.steps if any((x, op) not in self.reviewed for op in matching) else 0
                self.reviewed.update((x, op) for op in matching)
                answers = self.h.query_many_refs(
                    [f"(: $p (RuleTruth (Implication (Ball {x} $t) (Reward {op} $t))) $tv)" for op in matching],
                    steps=steps) if matching else []
            for op, rows in zip(matching, answers):
                tvs = [(float(a.tv[1]), float(a.tv[2])) for a in rows if a.tv[0] == "STV"]
                f, c = max(tvs, key=lambda tv: tv[1]) if tvs else (0.5, 0.0)
                desire[op] = c * (f - 0.5) + 0.5
        best = max(self.ops, key=lambda o: (desire[o], self.rng.random()))
        babble = self.rng.choice(self.babble_ops) if self.rng.random() < 0.2 else None
        if babble and desire[best] <= 0.55:
            op = babble
        elif desire[best] >= 0.501 or self.policy == "argmax":
            op = best
        else:
            op = "nothing"
        self.context[d] = (x, op)
        now = self.h.timing()
        self.timing.append({"decide": time.perf_counter() - t0, "queried": len(matching),
                            "desire": {o: round(v, 4) for o, v in desire.items() if v != 0.5},
                            "inference": now["inference"] - clock["inference"],
                            "overhead": now["overhead"] - clock["overhead"]})
        return d, ({op} - {"nothing"})

    def feedback(self, result, credited):
        if result is None:
            return set()
        clock = self.h.timing()
        t0 = time.perf_counter()
        atoms = []
        for d in credited:
            x, op = self.context[d]
            if x is None:
                continue
            if not self.mined and (x, op) not in self.hyps:
                self.hyps.add((x, op))
                atoms.append(f"(: h{len(self.hyps)} (Implication (Ball {x} $t) (Reward {op} $t)) (STV 0.5 0.01))")
            atoms.append(f"(: r{d} (Reward {op} d{d}) (STV {1 if result == 'hit' else 0} 1))")
        if atoms:
            self.h.add_atoms_no_check(atoms)
        now = self.h.timing()
        last = self.timing[-1]
        last["feedback"] = time.perf_counter() - t0
        last["inference"] += now["inference"] - clock["inference"]
        last["overhead"] += now["overhead"] - clock["overhead"]
        return set()

    def idle(self):
        pass


def applies(verdict, x, ops):
    """The operations a reviewed hypothesis (review-hypotheses verdict)
    predicts for an observation ``x``: its consequent's operation, the
    observation itself where the operation is the antecedent's variable;
    none when its antecedent names another observation."""
    ball = re.search(r"\(Ball (\S+) \$\w+\)", verdict)
    op = re.search(r"\(Reward (\S+) \$\w+\)", verdict)
    if not ball or not op or (not ball[1].startswith("$") and ball[1] != x):
        return []
    if op[1].startswith("$"):
        return [x] if op[1] == ball[1] and x in ops else list(ops)
    return [op[1]] if op[1] in ops else []


def play(env, agent, steps, credit=1, deadline=None, show=None):
    """Play ``steps`` game steps. Each step the agent decides on the step's
    observation; ``feedback`` gets the step's result and the decisions whose
    operations moved the bat in the ``credit`` steps before it.

    With ``deadline`` (seconds per step) the game runs on a clock of its own:
    step i starts at i * deadline whatever the agent does. An agent free at
    a step's start decides on its observation; the operation reaches the bat
    at the end of the step in which the decision (measured in wall time)
    finishes, and steps before that get none (the bat keeps its velocity).
    Feedback keeps the agent busy for as long as it takes, too. Steps the
    agent spends busy are not observed by it. Idle time is not slept away
    (the outcome is the same) unless ``show`` asks to watch the game.
    """
    applied = {}
    curve = []
    busy_until = 0.0
    pending = None
    late = 0
    decisions = 0
    for i in range(steps):
        tick = i * (deadline or 0.0)
        obs = env.observe()
        ops = set()
        if deadline is None:
            d, ops = agent.decide(obs)
            applied[i] = d
            decisions += 1
        else:
            if pending is None and busy_until <= tick:
                t0 = time.perf_counter()
                d, o = agent.decide(obs)
                busy_until = tick + time.perf_counter() - t0
                pending = (d, o)
                decisions += 1
            if pending is not None and busy_until <= tick + deadline:
                applied[i] = pending[0]
                ops = pending[1]
                pending = None
            elif pending is not None:
                late += 1
        result = env.advance()
        credited = [applied[j] for j in range(i - credit, i) if j in applied and applied[j] is not None] if result else []
        t0 = time.perf_counter()
        ops |= agent.feedback(result, credited)
        if deadline is not None and result:
            busy_until = max(busy_until, tick) + time.perf_counter() - t0
        env.act(ops)
        agent.idle()
        if (i + 1) % 100 == 0 or i + 1 == steps:
            curve.append((i + 1, env.hits, env.misses))
        if show:
            show(env, i + 1, ops, result)
            if deadline:
                time.sleep(max(0.0, deadline - (time.perf_counter() - show.last)))
                show.last = time.perf_counter()
    return {"hits": env.hits, "misses": env.misses, "ratio": env.ratio(), "curve": curve,
            "decisions": decisions, "late_steps": late}


class Show:
    """Draws the game in the terminal after every step."""

    def __init__(self, name):
        self.name = name
        self.last = time.perf_counter()

    def __call__(self, env, t, ops, result):
        sys.stdout.write("\033[H\033[2J" + env.render() + "\n" +
                         f"{self.name}  t={t}  hits={env.hits} misses={env.misses} ratio={env.ratio():.3f}"
                         f"  op={','.join(sorted(ops)) or '-'}  {result or ''}\n")
        sys.stdout.flush()


def summarize_timing(timing):
    """Mean and percentiles (ms) of a PLN agent's per-decision record."""
    import statistics
    total = sorted(1000 * (r["decide"] + r.get("feedback", 0.0)) for r in timing)
    out = {"decisions": len(total), "mean_ms": statistics.fmean(total),
           "p50_ms": total[len(total) // 2], "p95_ms": total[int(len(total) * 0.95)],
           "p99_ms": total[int(len(total) * 0.99)], "max_ms": total[-1],
           "inference_ms": 1000 * statistics.fmean(r["inference"] for r in timing),
           "overhead_ms": 1000 * statistics.fmean(r["overhead"] for r in timing)}
    queried = [r for r in timing if r["queried"]]
    if queried:
        out["queried_decisions"] = len(queried)
        out["queried_mean_ms"] = 1000 * statistics.fmean(r["decide"] for r in queried)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant", choices=("pong", "pong2"))
    ap.add_argument("--agent", choices=("ona", "pln"), required=True)
    ap.add_argument("--iterations", type=int, default=10000, help="as ./NAR pong ITERATIONS (runs one step more)")
    ap.add_argument("--seed", type=int, default=1, help="the game's generator seed (PLN's babbling too)")
    ap.add_argument("--steps", type=int, default=200, help="PLN query budget")
    ap.add_argument("--credit", type=int, default=6, help="PLN: steps after an operation whose score is its outcome")
    ap.add_argument("--mined", nargs=2, type=int, metavar=("MAX_NEW", "MAX_LIVE"),
                    help="PLN: the knowledge base mines its own hypotheses (set-hypothesis-mining)")
    ap.add_argument("--change-hazard", type=float, metavar="HAZARD",
                    help="PLN: prior probability that a sample starts a new regime of a refined rule's rate "
                         "(PeTTaChainer set-rule-change-hazard; default PeTTaChainer's, 0 keeps one regime)")
    ap.add_argument("--policy", choices=("argmax", "ona"), default="argmax",
                    help="PLN: take the best operation (argmax) or only one whose desire is at least 0.501 (ona)")
    ap.add_argument("--deadline-ms", type=float, help="real-time mode: the game advances every DEADLINE ms")
    ap.add_argument("--show", action="store_true", help="draw the game in the terminal")
    ap.add_argument("--ona-dir", default="/nexus/Dev/OpenCog/ONA")
    ap.add_argument("--ona-seed", type=int, help="ONA_SEED for an ONA build that seeds its generator from it")
    ap.add_argument("--pettachainer-path")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.out else None
    agent = (ONAAgent(a.variant, a.ona_dir, a.ona_seed) if a.agent == "ona"
             else PLNAgent(a.variant, a.seed, a.steps, a.pettachainer_path, a.mined, a.change_hazard,
                      a.policy))
    env = Pong(a.variant, a.seed)
    deadline = a.deadline_ms / 1000 if a.deadline_ms else None
    t0 = time.perf_counter()
    res = play(env, agent, a.iterations + 1, a.credit, deadline, Show(a.agent) if a.show else None)
    res.update({"variant": a.variant, "agent": a.agent, "seed": a.seed, "iterations": a.iterations,
                "deadline_ms": a.deadline_ms, "wall": time.perf_counter() - t0})
    if a.agent == "pln":
        res.update({"budget": a.steps, "credit": a.credit, "policy": a.policy, "mined": a.mined, "change_hazard": a.change_hazard,
                    "hypotheses": len(agent.hyps), "dropped": agent.dropped,
                    "timing_summary": summarize_timing(agent.timing), "timing": agent.timing})
    else:
        res["ona_seed"] = a.ona_seed
        ms = sorted(1000 * x for x in agent.timing)
        res["timing_summary"] = {"mean_ms": round(sum(ms) / len(ms), 3), "p95_ms": round(ms[int(len(ms) * 0.95)], 3),
                                 "max_ms": round(ms[-1], 3)}
    if out:
        with open(out, "w") as f:
            json.dump(res, f)
    print(json.dumps({k: res[k] for k in ("variant", "agent", "seed", "ratio", "hits", "misses", "late_steps",
                                          "wall", "timing_summary")}))


if __name__ == "__main__":
    main()
