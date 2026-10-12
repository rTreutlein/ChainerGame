import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest

from onabench.pong import MyRand, Pong, play, trace

ONA_SRC = os.environ.get("ONA_DIR", "/nexus/Dev/OpenCog/ONA") + "/src/system_tests"
TRACE_C = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ona", "pong", "pong_trace.c")


def c_lines(text):
    """pong_trace.c's output without the drawn field, in trace()'s form."""
    out = []
    for line in text.splitlines():
        m = re.match(r"Hits=(\d+) misses=(\d+) ratio=\S+ time=(\d+)", line)
        if m:
            out.append(f"Hits={m[1]} misses={m[2]} t={m[3]}")
        elif re.match(r"^(TRACE|good$|bad$|Exec: )", line):
            out.append(line)
    return out


class PortTest(unittest.TestCase):
    def test_generator_is_posix_example_rand(self):
        r = MyRand(1)
        self.assertEqual([r(), r(), r()], [16838, 5758, 10113])

    def test_port_replays_the_c_trace(self):
        """Digests of pong_trace.c's output for 5000 iterations, seed 3,
        compiled against ONA v0.9.3 (dc4efd0)."""
        for variant, digest, last in [
            ("pong", "ad6d79fae0a0971006d1fbd9d96fdf3681cfc34375d2a5731b042ef1be890d24", "Hits=37 misses=263 t=5001"),
            ("pong2", "1b4c2d1f1638278cddacce36d3b6bbcf227a8a05c262a0227bff3364b8e56c74", "Hits=49 misses=293 t=5001"),
        ]:
            lines = trace(variant, 5000, 3)
            self.assertEqual(lines[-2], last)
            self.assertEqual(hashlib.sha256("\n".join(lines).encode()).hexdigest(), digest)

    @unittest.skipUnless(shutil.which("cc") and os.path.exists(ONA_SRC + "/Pong_Test.h"), "no C compiler or ONA source")
    def test_port_matches_the_c_game_compiled_from_onas_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "pong_trace")
            subprocess.run(["cc", "-O1", "-I", ONA_SRC, "-o", exe, TRACE_C], check=True)
            for variant in ("pong", "pong2"):
                for seed in (1, 42):
                    c = subprocess.run([exe, variant, "20000", str(seed)], capture_output=True, text=True,
                                       errors="replace", check=True).stdout
                    self.assertEqual(c_lines(c), trace(variant, 20000, seed), (variant, seed))

    def test_bat_keeps_its_velocity_and_is_clamped(self):
        env = Pong("pong", 1)
        env.act({"right"})
        self.assertEqual(env.bat_x, 24)
        env.act(set())
        self.assertEqual(env.bat_x, 28)
        env.act({"left", "right"})  # right is applied after left, as in the C game
        self.assertEqual(env.bat_x, 32)
        for _ in range(10):
            env.act(set())
        self.assertEqual(env.bat_x, 49)
        env2 = Pong("pong2", 1)
        env2.act({"left"})
        self.assertEqual(env2.bat_x, 11)
        env2.act({"stop"})
        self.assertEqual(env2.bat_x, 11)


class Scripted:
    """An agent that numbers its decisions by step, optionally slowly."""

    def __init__(self, op=None, delay=0.0):
        self.op, self.delay, self.d = op, delay, 0
        self.results = []

    def decide(self, obs):
        self.d += 1
        time.sleep(self.delay)
        return self.d, {self.op} - {None}

    def feedback(self, result):
        self.results.append(result)
        return set()

    def idle(self):
        pass


class PlayTest(unittest.TestCase):
    def test_feedback_gets_every_steps_result(self):
        agent = Scripted()
        res = play(Pong("pong", 5), agent, 600)
        self.assertEqual(res["decisions"], 600)
        self.assertEqual(len(agent.results), 600)
        self.assertEqual(agent.results.count("hit"), res["hits"])
        self.assertEqual(agent.results.count("miss"), res["misses"])

    def test_the_curve_counts_steps(self):
        res = play(Pong("pong", 3), Scripted(), 250)
        self.assertEqual([s for s, _, _ in res["curve"]], [100, 200, 250])
        self.assertEqual(res["curve"][-1][1:], (res["hits"], res["misses"]))

    def test_a_slow_agent_misses_steps_and_acts_late(self):
        agent = Scripted("right", delay=0.025)
        res = play(Pong("pong", 5), agent, 40, deadline=0.01)
        self.assertLess(res["decisions"], 20)
        self.assertGreater(res["late_steps"], 0)
        fast = play(Pong("pong", 5), Scripted("right"), 400, deadline=0.01)
        self.assertEqual((fast["decisions"], fast["late_steps"]), (400, 0))
        self.assertEqual(fast["curve"], play(Pong("pong", 5), Scripted("right"), 400)["curve"])


@unittest.skipUnless(importlib.util.find_spec("pettachainer"), "run in a PeTTaChainer venv")
class PLNAgentTest(unittest.TestCase):
    def truth(self, agent, rule):
        (rows,) = agent.h.query_many_refs([f"(: $p (RuleTruth {rule}) $tv)"], steps=0)
        return max((float(a.tv[2]), float(a.tv[1])) for a in rows)[1]

    def test_rule_truths_count_the_observed_transitions_and_outcomes(self):
        """With one regime (hazard 0) a world rule's truth is the share of
        its context's next observations at y, every position not observed
        counting as a failed prediction, and a goal rule's the share of its
        state's steps with that outcome, misses included."""
        from onabench.pong import BALL, PLNAgent, goal_rule, world_rule
        agent = PLNAgent("pong2", 1, 200, change_hazard=0)
        outcomes = []
        play(Pong("pong2", 1), _Recorder(agent, outcomes), 150)
        steps = [(r["x"], r["op"]) for r in agent.timing]
        agent.decide("ball_left")  # the last step's outcome and position become facts
        nxt = [r["x"] for r in agent.timing[1:]]
        (x, op), = [max(set(steps[:-1]), key=steps[:-1].count)]
        after = [n for (c, n) in zip(steps, nxt) if c == (x, op)]
        for y in BALL:
            self.assertAlmostEqual(self.truth(agent, world_rule(x, op, y)),
                                   (after.count(y) + 0.005) / (len(after) + 0.01), delta=2e-4)
        for y in BALL:
            at = [o for (c, _), o in zip(steps, outcomes) if c == y]
            for g in ("Hit", "Miss"):
                self.assertAlmostEqual(self.truth(agent, goal_rule(y, g)),
                                       (at.count(g) + 0.005) / (len(at) + 0.01), delta=2e-4)

    def test_anticipation_states_no_misses(self):
        from onabench.pong import PLNAgent
        agent = PLNAgent("pong", 1, 200, misses=False)
        res = play(Pong("pong", 1), agent, 120)
        self.assertEqual(len(agent.timing), 120)
        self.assertGreater(res["hits"] + res["misses"], 0)
        self.assertEqual(agent.h.query_many_refs(["(: $p (Miss $d) $tv)"], steps=0), [[]])
        self.assertTrue(all(set(r["desire"]) == {"left", "right", "nothing"} for r in agent.timing))


class _Recorder:
    """Passes the agent's calls through and records each decision's outcome."""

    def __init__(self, agent, outcomes):
        self.agent, self.outcomes = agent, outcomes

    def decide(self, obs):
        self.outcomes.append(None)
        return self.agent.decide(obs)

    def feedback(self, result):
        if result:
            self.outcomes[-1] = "Hit" if result == "hit" else "Miss"
        return self.agent.feedback(result)

    def idle(self):
        self.agent.idle()


if __name__ == "__main__":
    unittest.main()
