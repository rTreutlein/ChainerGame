import itertools
import math
import os
import random
import shutil
import unittest
from pathlib import Path

from conflict import metta
from conflict.backends import ReferenceBackend
from conflict.game import GameConfig, run_game
from conflict.world import (
    ARCHETYPES,
    SIZES,
    Report,
    Round,
    Size,
    contested,
    generate_world,
    learned_rates,
    posterior,
    sample_round,
    true_rates,
)


def brute_force(world, rates, reports):
    """P(node up | claims) by enumerating every node's state, systems included."""
    nodes = world.nodes
    total, on = 0.0, dict.fromkeys(nodes, 0.0)
    for values in itertools.product((True, False), repeat=len(nodes)):
        state = dict(zip(nodes, values))
        weight = math.prod(p if state[c] else 1 - p for c, p in rates.priors.items())
        for s in world.systems:
            rate = s.given_both if all(state[p] for p in s.parents) else s.given_not
            weight *= rate if state[s.name] else 1 - rate
        for r in reports:
            up, down = rates.reliability[r.source]
            p = up if state[r.node] else down
            weight *= p if r.up else 1 - p
        total += weight
        for node in nodes:
            on[node] += weight * state[node]
    return {node: on[node] / total for node in nodes}


class WorldTests(unittest.TestCase):
    def test_posterior_matches_brute_force(self):
        rng = random.Random(3)
        for size in (Size(2, 2, 5), Size(3, 3, 8)):
            world = generate_world(rng, size)
            for name in ("a", "b", "c"):
                day = sample_round(world, rng, name)
                for rates in (true_rates(world), learned_rates(world, [sample_round(world, rng, f"h{i}") for i in range(10)])):
                    exact, brute = posterior(world, rates, day.reports), brute_force(world, rates, day.reports)
                    for node in world.nodes:
                        self.assertAlmostEqual(exact[node], brute[node], places=12)

    def test_copies_carry_no_evidence(self):
        world = generate_world(random.Random(1), SIZES["s"])
        rates = true_rates(world)
        once = (Report("s1", "c1", True, 1), Report("s4", "w1", False, 1))
        loud = (Report("s1", "c1", True, 15), Report("s4", "w1", False, 3))
        self.assertEqual(posterior(world, rates, once), posterior(world, rates, loud))

    def test_sources_follow_their_archetype(self):
        world = generate_world(random.Random(2), SIZES["l"])
        for source in world.sources:
            up, down, coverage, repeats = ARCHETYPES[source.kind]
            self.assertTrue(up[0] <= source.given_up <= up[1] and down[0] <= source.given_down <= down[1], source)
            self.assertTrue(repeats[0] <= source.repeats <= repeats[1])

    def test_learned_rates_converge(self):
        rng = random.Random(4)
        world = generate_world(rng, SIZES["m"])
        rates = learned_rates(world, [sample_round(world, rng, f"h{i}") for i in range(4000)])
        for source in world.sources:
            if source.kind != "rare":
                self.assertAlmostEqual(rates.reliability[source.name][0], source.given_up, delta=0.03)
                self.assertAlmostEqual(rates.reliability[source.name][1], source.given_down, delta=0.03)
        for c, p in world.priors.items():
            self.assertAlmostEqual(rates.priors[c], p, delta=0.03)

    def test_contested_nodes(self):
        reports = (Report("s1", "c1", True, 1), Report("s2", "c1", False, 4), Report("s1", "c2", True, 1), Report("s2", "c2", True, 1))
        self.assertEqual(contested(reports), {"c1"})


class GameTests(unittest.TestCase):
    def test_exact_scores_zero_and_baselines_do_not(self):
        config = GameConfig(seed=5, size="s", history=20, rounds=4)
        exact = run_game(config, ReferenceBackend("exact"))
        self.assertEqual(exact["posterior_error"], 0.0)
        self.assertEqual(set(exact["kinds"]), {"component", "system"})
        for name in ("exact-learned", "prior", "vote", "last-wins", "loudest", "trust-mean"):
            summary = run_game(config, ReferenceBackend(name))
            self.assertGreater(summary["posterior_error"], 0.0, name)
            self.assertEqual(summary["coverage"], 1.0)
            self.assertAlmostEqual(summary["posterior_brier"], exact["posterior_brier"])

    def test_backends_do_not_see_the_states(self):
        seen = []

        class Spy(ReferenceBackend):
            def beliefs(self, day, keys, budget):
                seen.append(day.truth)
                return super().beliefs(day, keys, budget)

        run_game(GameConfig(seed=1, history=3, rounds=2), Spy("vote"))
        self.assertEqual(seen, [{}, {}])

    def test_vote_counts_copies(self):
        world = generate_world(random.Random(1), SIZES["s"])
        backend = ReferenceBackend("vote")
        backend.begin(world, [])
        day = Round("t1", {}, (Report("s5", "c1", True, 12), Report("s1", "c1", False, 1)))
        self.assertAlmostEqual(backend.beliefs(day, [("c1", "t1")], 1)[("c1", "t1")], 12 / 13)


class MettaTests(unittest.TestCase):
    def test_statements(self):
        world = generate_world(random.Random(1), SIZES["s"])
        system = world.systems[0]
        self.assertTrue(metta.system_rules(world)[0].startswith(f"(: sys-w1 (Implication (And (Up {system.parents[0]} $t) (Up {system.parents[1]} $t)) (Up w1 $t)) (CTV (STV "))
        reports = (Report("s5", "c1", True, 15), Report("s4", "w2", False, 1))
        self.assertEqual(metta.raw_facts(reports, "t3", 5), ["(: rep-s5-c1-t3 (Up c1 t3) (STV 1 0.75))", "(: rep-s4-w2-t3 (Up w2 t3) (STV 0 0.166667))"])
        self.assertEqual(metta.claims(reports, "t3")[1], "(: claim-s4-w2-t3 (Claims s4 (Up w2 t3)) (STV 0 1))")
        self.assertEqual(metta.trust_rules(world)[0], "(: trust-s1 (Implication (Claims s1 (Up $n $t)) (Up $n $t)) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))")
        self.assertEqual(metta.labels(Round("h1", {"c1": True, "w1": False}, ())), ["(: up-c1-h1 (Up c1 h1) (STV 1 1))", "(: up-w1-h1 (Up w1 h1) (STV 0 1))"])
        self.assertEqual(metta.reliability_rules(reports, "t3"), [
            "(: (no_inverse claim-s5-c1-t3) (Implication (Reliable s5 up) (Up c1 t3)) (CTV (STV 1 1) (STV 0 1)))",
            "(: (no_inverse claim-s4-w2-t3) (Implication (Reliable s4 down) (Up w2 t3)) (CTV (STV 0 1) (STV 1 1)))",
        ])
        day = Round("h2", {"c1": True, "w2": True}, (*reports, Report("s5", "w2", True, 15)))
        self.assertEqual(metta.reliability_evidence(day, 5), [
            "(: reliability-s4-down-h2 (Reliable s4 down) (STV 0 0.166667))",
            "(: reliability-s5-up-h2 (Reliable s5 up) (STV 1 0.285714))",
        ])
        self.assertEqual(metta.likelihood_rules(world)[0], "(: report-s1 (Implication (Up $n $t) (Claims s1 (Up $n $t))) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))")
        self.assertEqual(metta.reliability_priors(world, 5)[:2], [
            "(: reliability-s1-up-prior (Reliable s1 up) (STV 0.5 0.285714))",
            "(: reliability-s1-down-prior (Reliable s1 down) (STV 0.5 0.285714))",
        ])


def _problog_available() -> bool:
    try:
        import problog  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_problog_available(), "ProbLog is not installed (docs/problog_backend.md)")
class ProblogTests(unittest.TestCase):
    def test_oracle_is_exact_and_learned_is_exact_learned(self):
        from conflict.problog_backend import ProblogBackend

        for size, seed in (("s", 1), ("m", 2)):
            config = GameConfig(seed=seed, size=size, history=15, rounds=3)
            for model, reference in (("oracle", "exact"), ("learned", "exact-learned")):
                records = {}
                for backend in (ProblogBackend(model), ReferenceBackend(reference)):
                    beliefs = []
                    original = backend.beliefs
                    backend.beliefs = lambda day, keys, budget, original=original, beliefs=beliefs: beliefs.append(original(day, keys, budget)) or beliefs[-1]
                    run_game(config, backend)
                    records[backend.name] = beliefs
                problog, python = records[f"problog-{model}"], records[reference]
                for a, b in zip(problog, python, strict=True):
                    self.assertEqual(set(a), set(b))
                    for key in a:
                        self.assertAlmostEqual(a[key], b[key], places=9, msg=(size, model, key))

    def test_naive_noisy_or(self):
        from conflict.problog_backend import naive_program
        from supplynet.problog_backend import solve

        world = generate_world(random.Random(1), Size(2, 0, 2))
        reports = (Report("s1", "c1", True, 2), Report("s2", "c1", False, 1))
        probabilities, record = solve(naive_program(world, {"c1": 0.5, "c2": 0.6}, 0.7, reports), ["up(c1)", "up(c2)"], "ddnnf", 60)
        self.assertEqual(record["outcome"], "answered")
        self.assertAlmostEqual(probabilities[0], (1 - 0.5 * 0.3 * 0.3) * 0.3)  # supported by prior or two copies, not denied
        self.assertAlmostEqual(probabilities[1], 0.6)


class NarsTranslationTests(unittest.TestCase):
    def test_kb(self):
        from conflict import nars

        backend = nars.NarsBackend.__new__(nars.NarsBackend)
        backend.encoding, backend.evidence_k = "sources", 5
        world = generate_world(random.Random(1), Size(2, 1, 1))
        history = [Round("h1", {"c1": True, "c2": True, "w1": True}, (Report("s1", "c1", True, 1),)),
                   Round("h2", {"c1": False, "c2": True, "w1": False}, (Report("s1", "c1", True, 1), Report("s1", "c2", False, 1)))]
        backend.begin(world, history)
        lines = backend.kb(Round("t1", {}, (Report("s1", "c1", False, 3), Report("s1", "c1", False, 3))))
        self.assertIn("<<(s1 * $n * $t) --> claims> ==> <($n * $t) --> up>>. %0.5;0.666667%", lines)
        self.assertIn("<(! <(s1 * $n * $t) --> claims>) ==> <($n * $t) --> up>>. %1;0.5%", lines)
        self.assertEqual(lines.count("(! <(s1 * c1 * t1) --> claims>). %1;0.99%"), 1)
        self.assertIn("<t1 --> period>. %1;0.99%", lines)
        backend.encoding = "raw"
        self.assertIn("(! <(c1 * t1) --> up>). %1;0.375%", backend.kb(Round("t1", {}, (Report("s1", "c1", False, 3),))))

    @unittest.skipUnless(os.access(os.environ.get("NARS_PATH", "/nexus/Dev/OpenCog/ONA/NAR"), os.X_OK), "ONA is not built")
    def test_run(self):
        from conflict.nars import NarsBackend

        summary = run_game(GameConfig(seed=1, history=5, rounds=1, steps_per_query=1), NarsBackend("sources"))
        self.assertGreater(summary["coverage"], 0.5)


def _pettachainer_path() -> str | None:
    path = os.environ.get("PETTACHAINER_PYTHONPATH")
    return path if path and Path(path, "pettachainer").is_dir() and shutil.which("swipl") else None


@unittest.skipUnless(_pettachainer_path(), "set PETTACHAINER_PYTHONPATH to a PeTTaChainer checkout and run in its venv")
class PeTTaChainerTests(unittest.TestCase):
    def test_encodings_answer(self):
        from conflict.backends import PeTTaChainerBackend

        # Fifteen labelled rounds: from eight, a source has one or two claims of
        # a polarity, and its rates learned from them decide a node alone.
        for encoding in ("raw", "sources", "stated", "given", "reliable", "likelihood"):
            summary = run_game(GameConfig(seed=1, history=15, rounds=1, steps_per_query=4), PeTTaChainerBackend(encoding, _pettachainer_path()))
            self.assertGreaterEqual(summary["coverage"], 0.75, encoding)
            self.assertLess(summary["posterior_error"], 0.5, encoding)

    def test_raw_revises_copies(self):
        """Two sources' facts on one node revise by their copy counts."""
        from conflict.backends import PeTTaChainerBackend

        backend = PeTTaChainerBackend("raw", _pettachainer_path())
        world = generate_world(random.Random(1), Size(1, 0, 2))
        backend.begin(world, [])
        day = Round("t1", {}, (Report("s1", "c1", True, 6), Report("s2", "c1", False, 2)))
        self.assertAlmostEqual(backend.beliefs(day, [("c1", "t1")], 4)[("c1", "t1")], 6 / 8, places=2)


if __name__ == "__main__":
    unittest.main()
