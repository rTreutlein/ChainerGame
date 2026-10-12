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

    def feedback(self, result):
        t0 = time.perf_counter()
        ops = self._input("good_nar. :|:") if result == "hit" else set()
        self.timing[-1] += time.perf_counter() - t0
        return ops

    def idle(self):
        t0 = time.perf_counter()
        self.pending = self._input("5")
        self.timing[-1] += time.perf_counter() - t0


BALL = ("right", "left", "equal")


def expectation(truth):
    """A truth's expectation, c (f - 0.5) + 0.5, from (confidence, strength)."""
    c, f = truth
    return c * (f - 0.5) + 0.5


def world_rule(x, op, y):
    """The one-step world-model rule: the ball at x and operation op at a time
    point put the ball at y at the next one."""
    return f"(Implication (And (NextPeriod $t $u) (At {x} $t) (Did {op} $t)) (Ball {y} $u))"


def goal_rule(y, goal):
    """The goal rule: the ball at y when the step's outcome is ``goal``, at a
    time point that has passed (see PLNAgent)."""
    return f"(Implication (And (NextPeriod $t $u) (At {y} $t)) ({goal} $t))"


class PLNAgent:
    """PeTTaChainer learning a one-step world model and the goal's states.

    Time points: one per decision, d1, d2, ..., linked by ``(NextPeriod d
    d+1)``. At each the agent observes where the ball is, x right, left or
    equal (Pong's level bat, where ONA's game inputs nothing, is equal: the
    step happened and showed neither side), as ``(At x d)``, the condition
    rules read, and as ``(Ball y d)`` for every y, the event rules predict
    (true for x, false for the others; see Anticipation), records
    the operation it took, ``(Did op d)`` (left, right, Pong2 also stop, or
    nothing, which leaves the bat moving as it was), and, once the step has
    played, its outcome: ``(Hit d)`` when the ball was hit, and with
    ``misses`` also ``(Miss d)`` when it was missed.

    Anticipation. Every event statement of a time point that has passed is
    known: ``(Ball y d)`` for the positions not observed, and the outcomes
    that did not happen, are stated false (STV 0 1). So a rule that
    predicted an event that did not come gets a negative sample. Interim:
    the harness states these facts; PeTTaChainer has no closed-world reading
    of observed predicates at past time points (a complete predicate is
    read closed only where no rule concludes it, and an instance fold does
    not see closed reads: benchmarks/ona/repros/closed_world_past_events.py,
    design in docs/pong_one_step.md). The observation is stated twice for
    the same reason: as a condition, ``At`` is positive-only (only true
    facts), so no completion or inversion is compiled onto it; read as
    ``Ball``, which has false facts, the refined rules' completions onto it
    built rows for every past time point, each reading the rule's truth,
    and a decision took 0.44 s at the start and 5 s by the 300th (PeTTaChainer
    2307ca50; 45 ms throughout with ``At``).

    Rules, all refined, stated at the start with a weak prior (STV 0.5,
    the evidence of 0.01 observations, so the first samples decide):

    - world model, one step, for every x, op and y:
      ``(And (NextPeriod $t $u) (At x $t) (Did op $t)) -> (Ball y $u)``;
    - goal rules, for every y: ``(And (NextPeriod $t $u) (At y $t)) ->
      (Hit $t)`` and, with ``misses``, the failure rule ``... -> (Miss
      $t)``.

    ``At``, ``Did`` and ``NextPeriod`` are positive-only. Evidence k = 1. A time
    point has passed once the link to the next one is stated, and each
    decision states the last time point's outcome and the new position
    before that link: every rule's instances are joined from the links, so
    an instance exists only once all its statements are known facts. A goal
    rule anchored on the link into its time point read ``(Hit d)`` before
    the step played, and the rules started for it while it was unknown
    stayed waiting on their truth once it was stated (250 such joins a
    decision, 1.1 MB). A closed-world reading of past observations would
    make the ordering unnecessary.

    Decision: chaining the one-step rules. One query reads the truths of the
    world rules from the observed x (every operation, every y) and of the
    goal rules (``RuleTruth`` goals, with budget only when one of them is
    read for the first time, which folds its instances: the instance tallies
    keep the truths current, so later reads compute nothing). The predicted
    next state of op is the world rules' strengths normalized over y. Each
    state y has a value per goal g, V_g(y): the outcomes expected after a
    step at y, discounted by ``gamma`` per step. An operation's value is
    what its predicted next state leads to,

        Q_g(op) = sum_y P(y | x, op) (E(g | y) + gamma V_g(y)),

    where E(g | y) is the goal rule's truth expectation, c (f - 0.5) + 0.5,
    as ONA reads desire: a state seen rarely counts near 0.5, so the agent
    goes to find out, and its estimate takes over as samples come. With the
    strength alone a state that missed its first few chances read 0, and the
    agent never went back (Pong without misses, seeds 2 and 3, 0.06 and
    0.12 at 2000 steps: the bat stayed at a wall).

    and the best operation's value becomes V(x) (asynchronous value
    iteration, one state per decision): the goal's desire flows back through
    the one-step rules, from a hit through equal to the moves that reach
    equal, so multi-step credit comes from chaining, at a constant cost per
    decision (12 to 18 truth reads and 9 to 12 sums). With one step of
    lookahead alone (V = 0) the agent in Pong stopped at the left wall: one
    move right does not reach a state whose immediate hit rate is higher.
    The best operation maximizes Q_Hit - Q_Miss (Q_Hit without misses). Its
    desire, on ONA's scale, is the probability that the next scored ball is
    hit, Q_Hit / (Q_Hit + Q_Miss), and without misses Q_Hit, the discounted
    expectation of hits, about the probability of a hit within 1 / (1 -
    gamma) steps. The choice is ONA's ``Decision_Suggest`` babbling
    (probability 0.2 of one of ONA's operations unless the best desire
    exceeds 0.55) and otherwise the best operation, nothing included, ties
    broken at random.
    """
    name = "pln"

    def __init__(self, variant, seed, steps, pettachainer_path=None, change_hazard=None, misses=True, gamma=0.9):
        if pettachainer_path:
            sys.path.insert(0, pettachainer_path)
        from pettachainer import PeTTaChainer
        self.h = PeTTaChainer()
        self.h.set_evidence_confidence_k(1)
        self.h.set_rule_refinement(True)
        if change_hazard is not None:
            self.h.set_rule_change_hazard(change_hazard)
        self.h.set_positive_only_predicate("Did")
        self.h.set_positive_only_predicate("NextPeriod")
        self.h.set_positive_only_predicate("At")
        self.babble_ops = ("left", "right") if variant == "pong" else ("left", "right", "stop")
        self.ops = self.babble_ops + ("nothing",)
        self.goals = ("Hit", "Miss") if misses else ("Hit",)
        rules = [world_rule(x, op, y) for x in BALL for op in self.ops for y in BALL]
        rules += [goal_rule(y, g) for y in BALL for g in self.goals]
        self.h.add_atoms_no_check([f"(: rule{i} {r} (STV 0.5 0.01))" for i, r in enumerate(rules)])
        self.rng = random.Random(seed * 7919 + 1)
        self.steps = steps
        self.d = 0
        self.gamma = gamma
        self.value = {y: dict.fromkeys(self.goals, 0.0) for y in BALL}
        self.outcome = None
        self.read = set()
        self.timing = []

    def decide(self, obs):
        clock = self.h.timing()
        t0 = time.perf_counter()
        d = self.d = self.d + 1
        x = obs[len("ball_"):] if obs else "equal"
        facts = [f"(: {g.lower()}{d - 1} ({g} d{d - 1}) (STV {int(self.outcome == g)} 1))" for g in self.goals if d > 1]
        facts += [f"(: ball{d}{y} (Ball {y} d{d}) (STV {int(y == x)} 1))" for y in BALL]
        facts.append(f"(: at{d} (At {x} d{d}) (STV 1 1))")
        facts += [f"(: next{d} (NextPeriod d{d - 1} d{d}) (STV 1 1))"] if d > 1 else []
        self.outcome = None
        self.h.add_atoms_no_check(facts)
        reads = [world_rule(x, op, y) for op in self.ops for y in BALL] + [goal_rule(y, g) for y in BALL for g in self.goals]
        steps = self.steps if not self.read.issuperset(reads) else 0
        self.read.update(reads)
        answers = self.h.query_many_refs([f"(: $p (RuleTruth {r}) $tv)" for r in reads], steps=steps)
        truth = {}
        for r, rows in zip(reads, answers):
            truth[r] = max(((float(a.tv[2]), float(a.tv[1])) for a in rows if a.tv[0] == "STV"), default=(0.0, 0.5))
        q = {}
        for op in self.ops:
            p = [truth[world_rule(x, op, y)][1] for y in BALL]
            q[op] = {g: sum(pi * (expectation(truth[goal_rule(y, g)]) + self.gamma * self.value[y][g]) for pi, y in zip(p, BALL))
                     / sum(p) for g in self.goals}
        best = max(self.ops, key=lambda o: (q[o]["Hit"] - q[o].get("Miss", 0.0), self.rng.random()))
        self.value[x] = q[best]
        value = {o: v["Hit"] / (v["Hit"] + v["Miss"]) if "Miss" in v and v["Hit"] + v["Miss"] else v["Hit"] for o, v in q.items()}
        babble = self.rng.choice(self.babble_ops) if self.rng.random() < 0.2 else None
        op = babble if babble and value[best] <= 0.55 else best
        self.h.add_atoms_no_check([f"(: did{d} (Did {op} d{d}) (STV 1 1))"])
        now = self.h.timing()
        self.timing.append({"decide": time.perf_counter() - t0, "x": x, "op": op,
                            "desire": {o: round(v, 4) for o, v in value.items()},
                            "inference": now["inference"] - clock["inference"],
                            "overhead": now["overhead"] - clock["overhead"]})
        return d, ({op} - {"nothing"})

    def feedback(self, result):
        """The outcome of a step played since the last decision; it becomes the
        decision's time point's fact at the next decision, when the time
        point has passed."""
        if result:
            self.outcome = "Hit" if result == "hit" else "Miss"
        return set()

    def idle(self):
        pass


def play(env, agent, steps, deadline=None, show=None):
    """Play ``steps`` game steps. Each step the agent decides on the step's
    observation; ``feedback`` gets the step's result ("hit", "miss" or None).

    With ``deadline`` (seconds per step) the game runs on a clock of its own:
    step i starts at i * deadline whatever the agent does. An agent free at
    a step's start decides on its observation; the operation reaches the bat
    at the end of the step in which the decision (measured in wall time)
    finishes, and steps before that get none (the bat keeps its velocity).
    Feedback keeps the agent busy for as long as it takes, too. Steps the
    agent spends busy are not observed by it. Idle time is not slept away
    (the outcome is the same) unless ``show`` asks to watch the game.
    """
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
            _, ops = agent.decide(obs)
            decisions += 1
        else:
            if pending is None and busy_until <= tick:
                t0 = time.perf_counter()
                d, o = agent.decide(obs)
                busy_until = tick + time.perf_counter() - t0
                pending = (d, o)
                decisions += 1
            if pending is not None and busy_until <= tick + deadline:
                ops = pending[1]
                pending = None
            elif pending is not None:
                late += 1
        result = env.advance()
        t0 = time.perf_counter()
        ops |= agent.feedback(result)
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
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant", choices=("pong", "pong2"))
    ap.add_argument("--agent", choices=("ona", "pln"), required=True)
    ap.add_argument("--iterations", type=int, default=10000, help="as ./NAR pong ITERATIONS (runs one step more)")
    ap.add_argument("--seed", type=int, default=1, help="the game's generator seed (PLN's babbling too)")
    ap.add_argument("--steps", type=int, default=200, help="PLN query budget")
    ap.add_argument("--anticipation", action="store_true",
                    help="PLN: no miss facts; a step without a hit is a step without a hit (ONA's information)")
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="PLN: discount per step of the outcomes a state leads to (chaining the one-step rules)")
    ap.add_argument("--change-hazard", type=float, metavar="HAZARD",
                    help="PLN: prior probability that a sample starts a new regime of a refined rule's rate "
                         "(PeTTaChainer set-rule-change-hazard; default PeTTaChainer's, 0 keeps one regime)")
    ap.add_argument("--deadline-ms", type=float, help="real-time mode: the game advances every DEADLINE ms")
    ap.add_argument("--show", action="store_true", help="draw the game in the terminal")
    ap.add_argument("--ona-dir", default="/nexus/Dev/OpenCog/ONA")
    ap.add_argument("--ona-seed", type=int, help="ONA_SEED for an ONA build that seeds its generator from it")
    ap.add_argument("--pettachainer-path")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.out else None
    agent = (ONAAgent(a.variant, a.ona_dir, a.ona_seed) if a.agent == "ona"
             else PLNAgent(a.variant, a.seed, a.steps, a.pettachainer_path, a.change_hazard, not a.anticipation,
                      a.gamma))
    env = Pong(a.variant, a.seed)
    deadline = a.deadline_ms / 1000 if a.deadline_ms else None
    t0 = time.perf_counter()
    res = play(env, agent, a.iterations + 1, deadline, Show(a.agent) if a.show else None)
    res.update({"variant": a.variant, "agent": a.agent, "seed": a.seed, "iterations": a.iterations,
                "deadline_ms": a.deadline_ms, "wall": time.perf_counter() - t0})
    if a.agent == "pln":
        res.update({"budget": a.steps, "outcomes": "anticipation" if a.anticipation else "misses", "gamma": a.gamma,
                    "change_hazard": a.change_hazard, "timing_summary": summarize_timing(agent.timing), "timing": agent.timing})
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
