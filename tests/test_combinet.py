import itertools
import math
import os
import random
import unittest

from combinet import metta
from combinet.backends import CombinationBackend, ExactBackend, PeTTaChainerBackend, combine, views
from combinet.game import GameConfig, run_game
from combinet.world import (
    RELATIONS,
    Case,
    Causes,
    Consequent,
    Redundant,
    Signs,
    draw_model,
    generate_problem,
    sample_case,
    true_marginals,
)


def joint(model, parents, c):
    """P(parents, C = c) under ``model``, written out per relation."""
    def bern(p, value):
        return p if value else 1 - p

    if isinstance(model, Signs):
        return bern(model.base, c) * math.prod(bern(t if c else f, a) for a, t, f in zip(parents, model.given_c, model.given_not_c))
    if isinstance(model, Causes):
        on = 1 - (1 - model.leak) * math.prod(1 - w for a, w in zip(parents, model.weight) if a)
        return math.prod(bern(p, a) for a, p in zip(parents, model.prior)) * bern(on, c)
    if isinstance(model, Redundant):
        return sum(
            bern(model.latent, latent)
            * math.prod(bern(f, a == latent) for a, f in zip(parents, model.fidelity))
            * bern(model.given_l if latent else model.given_not_l, c)
            for latent in (True, False)
        )
    return math.prod(bern(p, a) for a, p in zip(parents, model.prior)) * bern(model.chance(parents), c)


def brute_force_posterior(model, n, observed):
    total = on = 0.0
    for parents in itertools.product((True, False), repeat=n):
        if any(parents[i] != v for i, v in observed.items()):
            continue
        for c in (True, False):
            weight = joint(model, parents, c)
            total += weight
            on += weight * c
    return on / total


def problem_of(relation, model, n):
    return (Consequent("C1", relation, tuple(f"A1_{i + 1}" for i in range(n)), model),)


class WorldTests(unittest.TestCase):
    def test_posterior_matches_brute_force(self):
        rng = random.Random(5)
        for relation in RELATIONS:
            for n in (1, 2, 4):
                model = draw_model(relation, n, rng)
                for values in itertools.product((None, True, False), repeat=n):
                    observed = {i: v for i, v in enumerate(values) if v is not None}
                    self.assertAlmostEqual(model.posterior(observed), brute_force_posterior(model, n, observed), places=12, msg=(relation, observed))

    def test_samples_follow_the_model(self):
        rng = random.Random(9)
        for relation in RELATIONS:
            model = draw_model(relation, 3, rng)
            samples = [model.sample(rng) for _ in range(40000)]
            with_first = [c for parents, c in samples if parents[0] and not parents[1]]
            self.assertAlmostEqual(sum(with_first) / len(with_first), model.posterior({0: True, 1: False}), delta=0.02, msg=relation)

    def test_interacting_pairs_are_antagonistic(self):
        model = draw_model("interacting", 2, random.Random(1))
        both, first, second = model.posterior({0: True, 1: True}), model.posterior({0: True, 1: False}), model.posterior({0: False, 1: True})
        self.assertLess(both, min(first, second))
        self.assertGreater(min(first, second), model.posterior({0: False, 1: False}))

    def test_problem_and_cases(self):
        rng = random.Random(2)
        problem = generate_problem(rng, 3, RELATIONS, 2)
        self.assertEqual([c.relation for c in problem], [r for r in RELATIONS for _ in range(2)])
        self.assertEqual(problem[2].parents, ("A3_1", "A3_2", "A3_3"))
        case = sample_case(problem, rng, "e1", 0.5)
        self.assertEqual(set(case.observed), {c.name for c in problem})
        for c in problem:
            parents, _ = case.truth[c.name]
            self.assertTrue(all(parents[i] == v for i, v in case.observed[c.name].items()))


class CombinationTests(unittest.TestCase):
    def test_each_mode_is_exact_for_its_relation(self):
        """Odds for signs, noisy-OR for causes, revision for perfect copies."""
        rng = random.Random(3)
        perfect = Redundant(0.4, (1.0, 1.0, 1.0), 0.8, 0.1)
        for mode, model in (("odds", draw_model("signs", 3, rng)), ("noisy-or", draw_model("causes", 3, rng)), ("revision", perfect)):
            marginals = true_marginals(problem_of("x", model, 3))
            for values in itertools.product((None, True, False), repeat=3):
                observed = {i: v for i, v in enumerate(values) if v is not None}
                if mode == "revision" and len(set(observed.values())) > 1:
                    continue
                got = combine(mode, marginals.base["C1"], views(marginals, "C1", observed))
                self.assertAlmostEqual(got, model.posterior(observed), places=9, msg=(mode, observed))

    def history(self, relation, model, n, count):
        problem = problem_of(relation, model, n)
        rng = random.Random(4)
        return problem, [sample_case(problem, rng, f"h{i}", 0.8) for i in range(count)]

    def test_learned_mode_finds_the_relation(self):
        rng = random.Random(6)
        for relation, mode in (("signs", "odds"), ("causes", "noisy-or"), ("redundant", "revision")):
            problem, history = self.history(relation, draw_model(relation, 3, rng), 3, 3000)
            backend = CombinationBackend("mode")
            backend.begin(problem, true_marginals(problem), history)
            backend.beliefs([], [], 0)
            self.assertEqual(backend.modes["C1"], mode, relation)

    def test_cells_converge_to_the_posterior_of_an_interaction(self):
        model = draw_model("interacting", 2, random.Random(7))
        problem, history = self.history("interacting", model, 2, 4000)
        backend = CombinationBackend("cells")
        backend.begin(problem, true_marginals(problem), history)
        case = Case("q", {"C1": ((True, True), True)}, {"C1": {0: True, 1: True}})
        belief = backend.beliefs([case], [("C1", "q")], 0)[("C1", "q")]
        self.assertAlmostEqual(belief, model.posterior({0: True, 1: True}), delta=0.04)
        revision = CombinationBackend("revision")
        revision.begin(problem, true_marginals(problem), history)
        self.assertGreater(abs(revision.beliefs([case], [("C1", "q")], 0)[("C1", "q")] - belief), 0.1)


class StatementTests(unittest.TestCase):
    def setUp(self):
        self.problem = problem_of("signs", Signs(0.3, (0.8, 0.7), (0.2, 0.1)), 2)
        self.case = Case("e1", {"C1": ((True, False), True)}, {"C1": {0: True, 1: False}})

    def test_rules_state_both_branches(self):
        marginals = true_marginals(self.problem)
        line = metta.rules(self.problem, marginals)[0]
        t, f = marginals.given["C1"][0]
        self.assertEqual(line, f"(: r-A1_1 (Implication (A1_1 $x) (C1 $x)) (CTV (STV {t:.6g} 1) (STV {f:.6g} 1)))")

    def test_complement_facts_share_the_observation_name(self):
        self.assertEqual(metta.case_facts(self.problem, self.case), ["(: o-A1_1-e1 (A1_1 e1) (STV 1 1))", "(: o-A1_2-e1 (A1_2 e1) (STV 0 1))"])
        self.assertEqual(
            metta.case_facts(self.problem, self.case, complement=True)[2:],
            ["(: o-A1_2-e1 (A1_2 e1) (STV 0 1))", "(: o-A1_2-e1 (NotA1_2 e1) (STV 1 1))"],
        )
        self.assertEqual(metta.label_facts(self.problem, self.case), ["(: l-C1-e1 (C1 e1) (STV 1 1))"])

    def test_hypothesis_forms(self):
        pattern = ((0, True), (1, False))
        self.assertEqual(
            metta.hypothesis(self.problem[0], pattern, 0.45, "complement"),
            "(: h-C1-1T-2F (Implication (And (A1_1 $x) (NotA1_2 $x)) (C1 $x)) (STV 0.45 0.02))",
        )
        self.assertEqual(
            metta.review(self.problem[0], pattern, "not"),
            "(: $prf (RuleTruth (Implication (And (A1_1 $x) (Not (A1_2 $x))) (C1 $x))) $tv)",
        )


class GameTests(unittest.TestCase):
    config = GameConfig(seed=3, parents=3, history=30, rounds=4, cases=8)

    def test_exact_matches_the_posterior_and_combinations_trail_it(self):
        exact = run_game(self.config, ExactBackend())
        self.assertEqual(exact["posterior_error"], 0.0)
        self.assertEqual(exact["coverage"], 1.0)
        self.assertEqual(set(exact["kinds"]), set(RELATIONS))
        revision = run_game(self.config, CombinationBackend("revision"))
        self.assertGreater(revision["posterior_error"], 0.02)
        self.assertAlmostEqual(run_game(self.config, CombinationBackend("odds"))["kinds"]["signs"]["posterior_error"], 0.0, places=12)
        self.assertLess(revision["brier"], run_game(self.config, CombinationBackend("base-rate"))["brier"])

    def test_runs_are_deterministic(self):
        first = run_game(self.config, CombinationBackend("cell-mode"))
        second = run_game(self.config, CombinationBackend("cell-mode"))
        for field in ("brier", "log_loss", "posterior_error", "modes", "curve"):
            self.assertEqual(first[field], second[field])
        self.assertEqual([point["history"] for point in first["curve"]], [30, 38, 46, 54])


LIVE = os.environ.get("COMBINET_LIVE_PETTACHAINER") == "1"


@unittest.skipUnless(LIVE, "set COMBINET_LIVE_PETTACHAINER=1")
class PeTTaChainerTests(unittest.TestCase):
    config = GameConfig(seed=3, parents=2, history=40, rounds=2, cases=6)

    def test_marginal_rules_are_revised(self):
        """With two parents the chainer's revision is the views' mean."""
        chainer = run_game(self.config, PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH")))
        revision = run_game(self.config, CombinationBackend("revision"))
        self.assertAlmostEqual(chainer["posterior_error"], revision["posterior_error"], places=3)

    def test_cell_hypotheses_are_written_and_reviewed(self):
        result = run_game(self.config, PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"), hypotheses="cells"))
        self.assertGreater(result["hypotheses"], 0)
        self.assertGreater(result["folds"], 0)
        self.assertGreater(result["coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
