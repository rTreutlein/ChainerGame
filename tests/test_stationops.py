import json
import os
import unittest
from types import SimpleNamespace

from stationops.backends import BackendUnavailable, MM2Backend
from stationops.config import Config
from stationops.episode import run_episode
from stationops.metta import generate_statements
from stationops.models import HistoryCase, Incident
from stationops.oracle import empirical_priors, posterior, resolve
from stationops.policy import allocate
from stationops.simulator import generate_history, generate_incidents


class BenchmarkTests(unittest.TestCase):
    def test_fixed_episode_and_coherent_explicit_history(self):
        cfg = Config()
        history, incidents = generate_history(cfg), generate_incidents(cfg)
        self.assertEqual(len(history), 2000)
        self.assertEqual(len(incidents), 100)
        self.assertEqual({"old": 0.05, "new": 0.005}, empirical_priors(history))
        source = generate_statements(history, incidents, cfg)
        self.assertIn("(Not (SealLeak", source)
        self.assertIn("(Not (PressureAlarm", source)
        self.assertIn("(CTV", source)

    def test_posterior_threshold_and_decisions(self):
        cfg = Config(repair_slots=100)
        old = posterior(.05, True, .9, .12)
        new = posterior(.005, True, .9, .12)
        self.assertAlmostEqual(old, .2830188679)
        self.assertAlmostEqual(new, .0363196126)
        self.assertLess(new, cfg.repair_threshold)
        self.assertGreater(old, cfg.repair_threshold)
        cases = [Incident("z-old", "old", True), Incident("a-new", "new", True)]
        actions = allocate(cases, {"z-old": old, "a-new": new}, cfg)
        self.assertEqual(actions, {"z-old": "repair", "a-new": "defer"})

    def test_allocator_slots_and_tie_break(self):
        cfg = Config(repair_slots=1)
        cases = [Incident("b", "old", True), Incident("a", "old", True)]
        self.assertEqual(allocate(cases, {"a": .5, "b": .5}, cfg), {"b": "defer", "a": "repair"})

    def test_irrelevant_invariance(self):
        a = run_episode(Config(irrelevant_statements=0), budget=100)
        b = run_episode(Config(irrelevant_statements=50), budget=100)
        self.assertEqual(a["beliefs"], b["beliefs"])
        self.assertEqual(a["chosen_actions"], b["chosen_actions"])

    def test_resolved_case_updates_only_one_cohort(self):
        history = generate_history(Config())
        before = empirical_priors(history)
        after = empirical_priors(resolve(history, HistoryCase("resolved", "new", True, True)))
        self.assertEqual(before["old"], after["old"])
        self.assertGreater(after["new"], before["new"])
        self.assertNotEqual(posterior(before["new"], True, .9, .12), posterior(after["new"], True, .9, .12))

    def test_json_contract(self):
        result = run_episode(Config(), budget=100)
        for key in ("config", "seed", "backend", "budget", "empirical_priors", "beliefs",
                    "chosen_actions", "expected_utility", "oracle_utility", "do_nothing_utility",
                    "regret", "normalized_score", "wall_time_seconds", "backend_counters"):
            self.assertIn(key, result)
        json.dumps(result)

    def test_mm2_conformance_when_binding_available(self):
        path = os.environ.get("MM2_CHAINER_PYTHONPATH")
        try:
            MM2Backend(Config(), path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        result = run_episode(Config(repair_slots=100), "mm2", 100, path)
        old_alarm = next(x for x in generate_incidents(Config(repair_slots=100)) if x.cohort == "old" and x.alarm)
        new_alarm = next(x for x in generate_incidents(Config(repair_slots=100)) if x.cohort == "new" and x.alarm)
        self.assertAlmostEqual(
            result["beliefs"][old_alarm.id],
            posterior(.05, True, .9, .12),
            places=6,
        )
        self.assertAlmostEqual(
            result["beliefs"][new_alarm.id],
            posterior(.005, True, .9, .12),
            places=6,
        )
        self.assertGreater(result["beliefs"][old_alarm.id], Config().repair_threshold)
        self.assertLess(result["beliefs"][new_alarm.id], Config().repair_threshold)
        self.assertEqual(result["chosen_actions"][old_alarm.id], "repair")
        self.assertEqual(result["chosen_actions"][new_alarm.id], "defer")

    def test_mm2_beliefs_are_derived_from_engine_results(self):
        class Engine:
            def add_many(self, *args):
                pass

            def set_base_rate(self, *args):
                pass

            def query_many(self, kb, queries, budget):
                if budget < 2:
                    return [(tag, []) for tag, _ in queries]
                values = {"old-alarm": .8, "new-alarm": .01}
                return [
                    (tag, [{"truth_value": {"strength": values[tag], "confidence": 1.0}}])
                    for tag, _ in queries
                ]

        cfg = Config(repair_slots=2)
        backend = MM2Backend(cfg, module=SimpleNamespace(Engine=Engine))
        history = generate_history(cfg)
        incidents = [
            Incident("old-alarm", "old", True),
            Incident("new-alarm", "new", True),
        ]
        beliefs, _ = backend.infer(history, incidents, 2, "")
        self.assertEqual(beliefs, {"old-alarm": .8, "new-alarm": .01})
        self.assertEqual(
            allocate(incidents, beliefs, cfg),
            {"old-alarm": "repair", "new-alarm": "defer"},
        )
        missing, _ = backend.infer(history, incidents, 1, "")
        self.assertEqual(missing, {})
        self.assertEqual(
            allocate(incidents, missing, cfg),
            {"old-alarm": "defer", "new-alarm": "defer"},
        )
        zero, _ = backend.infer(history, incidents, 0, "")
        self.assertEqual(zero, {})

    def test_mm2_wrong_results_change_decisions(self):
        class Engine:
            def add_many(self, *args):
                pass

            def set_base_rate(self, *args):
                pass

            def query_many(self, kb, queries, budget):
                return [
                    (tag, [{"truth_value": {"strength": .01, "confidence": 1.0}}])
                    for tag, _ in queries
                ]

        cfg = Config(repair_slots=1)
        incident = Incident("old-alarm", "old", True)
        backend = MM2Backend(cfg, module=SimpleNamespace(Engine=Engine))
        beliefs, _ = backend.infer(generate_history(cfg), [incident], 100, "")
        self.assertEqual(beliefs, {"old-alarm": .01})
        self.assertEqual(allocate([incident], beliefs, cfg), {"old-alarm": "defer"})


if __name__ == "__main__":
    unittest.main()
