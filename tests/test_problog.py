import importlib.util
import itertools
import unittest
from types import SimpleNamespace

from supplynet import scale
from supplynet.backends import LearnedReferenceBackend, ReferenceBackend
from supplynet.game import GameConfig, run_game
from supplynet.problog_backend import ProblogBackend, rule, solve

HAVE_PROBLOG = importlib.util.find_spec("problog") is not None


def recording(backend):
    """``backend`` keeping every round's beliefs."""

    class Recording(backend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.answers = []

        def beliefs(self, observation, keys, budget):
            self.answers.append(super().beliefs(observation, keys, budget))
            return self.answers[-1]

    return Recording


RecordingReference, RecordingLearnedReference, RecordingProblog = map(recording, (ReferenceBackend, LearnedReferenceBackend, ProblogBackend))


class TranslationTests(unittest.TestCase):
    def test_a_ctv_rule_is_two_exclusive_clauses(self):
        persist = rule("(: persist-north (Implication (And (NextPeriod $p $t) (Storm north $p)) (Storm north $t)) (CTV (STV 0.7 1) (STV 0.3 1)))", 0)
        self.assertEqual(persist.head, ("Storm", ("north",)))
        self.assertEqual(
            persist.clauses(*persist.rates),
            [
                "ant_0(Vt) :- nextperiod(Vp, Vt), storm('north', Vp).",
                "0.7::storm('north', Vt) :- live(Vt), ant_0(Vt).",
                "0.3::storm('north', Vt) :- live(Vt), \\+ant_0(Vt).",
            ],
        )

    def test_or_negation_and_certain_rules(self):
        fuelled = rule("(: fuelled (Implication (Or (StockedFuel $t) (FuelArrived $t)) (Fuelled power-plant $t)) (CTV (STV 1 1) (STV 0 1)))", 3)
        self.assertEqual(
            fuelled.clauses(*fuelled.rates),
            [
                "ant_3(Vt) :- stockedfuel(Vt).",
                "ant_3(Vt) :- fuelarrived(Vt).",
                "fuelled('power-plant', Vt) :- live(Vt), ant_3(Vt).",
            ],
        )
        runs = rule("(: runs (Implication (And (Not (Degraded mine $t)) (Producing power-plant $t)) (Producing mine $t)) (CTV (STV 1 1) (STV 0 1)))", 4)
        self.assertEqual(runs.clauses(*runs.rates)[0], "ant_4(Vt) :- producing('power-plant', Vt), \\+degraded('mine', Vt).")


@unittest.skipUnless(HAVE_PROBLOG, "problog is not installed")
class ExactnessTests(unittest.TestCase):
    def assert_matches_reference(self, play, reference, problog):
        play(reference)
        play(problog)
        self.assertTrue(problog.answers)
        for exact, answer in zip(reference.answers, problog.answers, strict=True):
            self.assertEqual(exact.keys(), answer.keys())
            for key, p in exact.items():
                self.assertAlmostEqual(answer[key], p, places=9, msg=key)

    def test_stages_match_the_exact_reference(self):
        for (stage, cycle), engine in itertools.product(((1, "timed"), (2, "timed"), (3, "timed"), (3, "untimed")), ("ddnnf", "sdd")):
            with self.subTest(stage=stage, cycle=cycle, engine=engine):
                config = GameConfig(seed=2, rounds=6, stage=stage, cycle=cycle)
                self.assert_matches_reference(lambda backend: run_game(config, backend), RecordingReference(), RecordingProblog(engine=engine))

    def test_learned_rates_match_the_exact_posterior_under_counted_rates(self):
        """With learned rules ProbLog counts every rate from the labelled
        periods' facts; the reference counts them from the simulator's labels."""
        for stage, cycle in ((1, "timed"), (2, "timed"), (3, "timed"), (3, "untimed")):
            with self.subTest(stage=stage, cycle=cycle):
                config = GameConfig(seed=3, rounds=8, stage=stage, cycle=cycle)
                self.assert_matches_reference(lambda backend: run_game(config, backend), RecordingLearnedReference(), RecordingProblog(learned=True))

    def test_scale_matches_the_exact_reference(self):
        config = scale.GameConfig(1, scale.SIZES["s"], rounds=4)
        self.assert_matches_reference(
            lambda backend: scale.run_game(config, backend),
            RecordingReference(scale.Knowledge),
            RecordingProblog(scale, priors=lambda network, rates: {}),
        )

    def test_a_round_past_its_timeout_is_unanswered(self):
        program = "0.5::a(X) :- between(1, 22, X).\nb :- a(X).\nquery(b).\n"
        probabilities, record = solve(program, ["b"], "ddnnf", 60)
        self.assertAlmostEqual(probabilities[0], 1 - 0.5**22)
        probabilities, record = solve(program, ["b"], "ddnnf", 0.001)
        self.assertIsNone(probabilities)
        self.assertEqual(record["outcome"], "timeout")


@unittest.skipUnless(HAVE_PROBLOG, "problog is not installed")
class StationOpsTests(unittest.TestCase):
    def test_shortfall_and_persistence_are_exact(self):
        """Two live units, one of which leaks by the shortfall's loss, and
        the first unit's successor in the next shift, against enumeration."""
        from stationops.problog_backend import ProblogBackend as StationBackend

        h, sensitivity, false_positive = 0.2, 0.9, 0.1
        statements = "\n".join(
            [
                "(: persist-old-pump (Implication (And (NextState $previous $unit) (SealLeakAtShiftEnd old pump $previous)) "
                f"(LeakPredicted old pump $unit)) (CTV (STV 1 1) (STV {h} 1)))",
                "(: alarmGivenLeak-pump (ForAll ($cohort) (WithPrior (LeakPredicted $cohort pump $unit) "
                f"(Implication (SealLeak $cohort pump $unit) (PressureAlarm $cohort pump $unit)))) (CTV (STV {sensitivity} 1) (STV {false_positive} 1)))",
                "(: alarm-a (PressureAlarm old pump shift-01-a) (STV 1 1))",
                "(: alarm-b (PressureAlarm old pump shift-01-b) (STV 0 1))",
                "(: loss (ShortfallLoss shortfall-s01 40) (STV 1 1))",
                "(: cand-a (ShortfallCandidate shortfall-s01 old pump shift-01-a 40) (STV 1 1))",
                "(: cand-b (ShortfallCandidate shortfall-s01 old pump shift-01-b 40) (STV 1 1))",
                "(: next-c (NextState shift-01-a shift-02-c) (STV 1 1))",
                "(: alarm-c (PressureAlarm old pump shift-02-c) (STV 0 1))",
            ]
        )
        weights = {}
        for a, b, c in itertools.product((True, False), repeat=3):
            if a + b != 1:  # the loss is one impact
                continue
            weight = (h if a else 1 - h) * (h if b else 1 - h) * ((1.0 if c else 0.0) if a else (h if c else 1 - h))
            weight *= sensitivity if a else false_positive  # a alarmed
            weight *= (1 - sensitivity) if b else (1 - false_positive)
            weight *= (1 - sensitivity) if c else (1 - false_positive)
            weights[(a, b, c)] = weight
        total = sum(weights.values())
        exact = [sum(w for state, w in weights.items() if state[i]) / total for i in range(3)]
        incidents = [SimpleNamespace(id=unit, cohort="old", equipment_type="pump") for unit in ("shift-01-a", "shift-01-b", "shift-02-c")]
        beliefs, counters = StationBackend().infer([], incidents, 50, statements)
        self.assertEqual(counters["problog_outcome"], "answered")
        for item, p in zip(incidents, exact, strict=True):
            self.assertAlmostEqual(beliefs[item.id].strength, p, places=9, msg=item.id)


if __name__ == "__main__":
    unittest.main()
