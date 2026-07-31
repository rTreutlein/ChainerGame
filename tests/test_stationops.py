import json
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stationops.backends import BackendUnavailable, MM2Backend, PeTTaChainerBackend
from stationops.config import Config
from stationops.episode import run_episode
from stationops.metta import generate_statements
from stationops.models import HistoryCase, Incident
from stationops.models import EpisodeFixture, RoundFixture
from stationops.oracle import empirical_priors, posterior, resolve
from stationops.policy import allocate
from stationops.simulator import generate_history, generate_incidents
from stationops.v1 import play_episode_v1, prior_shift_fixture, run_episode_v1


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

    def test_pettachainer_beliefs_drive_decisions_and_preserve_order(self):
        class Handler:
            instances = []

            def __init__(self):
                self.queries = []
                self.atoms = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                self.atoms = atoms

            def query(self, query, steps, timeout_sec):
                self.queries.append((query, steps, timeout_sec))
                if "old-alarm" in query:
                    return [
                        "(: weak (PatchPaysOff old old-alarm) (STV 0.2 1))",
                        "(: strong (PatchPaysOff old old-alarm) (STV 0.8 1))",
                    ]
                return []

        cfg = Config(repair_slots=2)
        backend = PeTTaChainerBackend(
            cfg, module=SimpleNamespace(PeTTaChainer=Handler)
        )
        incidents = [
            Incident("new-alarm", "new", True),
            Incident("old-alarm", "old", True),
        ]
        beliefs, counters = backend.infer(
            generate_history(cfg), incidents, 17, "(: fact (A) (STV 1 1))"
        )

        self.assertEqual(beliefs, {"old-alarm": .8})
        self.assertEqual(counters, {"queries": 2, "engine_steps": None})
        self.assertEqual(
            allocate(incidents, beliefs, cfg),
            {"new-alarm": "defer", "old-alarm": "repair"},
        )
        self.assertEqual(
            [query for query, _, _ in Handler.instances[-1].queries],
            [
                "(: $prf (PatchPaysOff new new-alarm) $tv)",
                "(: $prf (PatchPaysOff old old-alarm) $tv)",
            ],
        )
        self.assertEqual(Handler.instances[-1].atoms, ["(: fact (A) (STV 1 1))"])
        self.assertTrue(
            all(
                steps == 17 and timeout == 0
                for _, steps, timeout in Handler.instances[-1].queries
            )
        )

    def test_pettachainer_batches_are_isolated_and_zero_budget_is_empty(self):
        class Handler:
            instances = []

            def __init__(self):
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                pass

            def query(self, query, steps, timeout_sec):
                incident_id = "second" if "second" in query else "first"
                return [f"(: proof (PatchPaysOff old {incident_id}) (STV .4 1e0))"]

        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=Handler)
        )
        first = [Incident("first", "old", True), Incident("second", "old", True)]
        reversed_batch = list(reversed(first))
        beliefs, _ = backend.infer([], first, 10, "")
        reversed_beliefs, _ = backend.infer([], reversed_batch, 10, "")
        zero, _ = backend.infer([], first, 0, "")

        self.assertEqual(beliefs, {"first": .4, "second": .4})
        self.assertEqual(reversed_beliefs, {"second": .4, "first": .4})
        self.assertEqual(zero, {})
        self.assertEqual(len(Handler.instances), 2)

    def test_pettachainer_unavailable_error_is_actionable(self):
        with patch("stationops.backends.importlib.import_module", side_effect=ImportError):
            with self.assertRaises(BackendUnavailable) as caught:
                PeTTaChainerBackend(Config(), "/definitely/not/pettachainer")
        message = str(caught.exception)
        self.assertIn("--pettachainer-path", message)
        self.assertIn("PETTACHAINER_PYTHONPATH", message)

    def test_pettachainer_runtime_dependency_error_is_actionable(self):
        class Handler:
            def __init__(self):
                raise ModuleNotFoundError("No module named 'janus_swi'", name="janus_swi")

        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=Handler)
        )
        with self.assertRaises(BackendUnavailable) as caught:
            backend.infer(generate_history(Config()), [Incident("x", "new", True)], 10, "")
        self.assertIn("runtime handler", str(caught.exception))
        self.assertIn("janus_swi", str(caught.exception))

    def test_run_episode_explicitly_selects_pettachainer(self):
        class Backend:
            name = "pettachainer"

            def __init__(self, config, python_path):
                self.python_path = python_path

            def infer(self, history, incidents, budget, statements):
                return {incident.id: .8 for incident in incidents}, {
                    "queries": len(incidents),
                    "engine_steps": None,
                }

        cfg = Config(incidents=2, repair_slots=2)
        with patch("stationops.episode.PeTTaChainerBackend", Backend):
            result = run_episode(
                cfg,
                "pettachainer",
                20,
                pettachainer_path="/controlled/pettachainer",
            )

        self.assertEqual(result["backend"], "pettachainer")
        self.assertEqual(len(result["beliefs"]), 2)
        self.assertEqual(set(result["chosen_actions"].values()), {"repair"})

    def test_pettachainer_conformance_when_available(self):
        if os.environ.get("STATIONOPS_LIVE_PETTACHAINER") != "1":
            self.skipTest("set STATIONOPS_LIVE_PETTACHAINER=1 for live conformance")
        path = os.environ.get("PETTACHAINER_PYTHONPATH")
        try:
            backend = PeTTaChainerBackend(Config(), path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))

        cfg = Config()
        history = generate_history(cfg)
        all_incidents = generate_incidents(cfg)
        incidents = [
            next(x for x in all_incidents if x.cohort == cohort and x.alarm)
            for cohort in ("old", "new")
        ]
        try:
            beliefs, _ = backend.infer(
                history,
                incidents,
                200,
                generate_statements(history, incidents, cfg),
            )
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        priors = empirical_priors(history)
        for incident in incidents:
            expected = posterior(
                priors[incident.cohort],
                incident.alarm,
                cfg.sensitivity,
                cfg.false_positive_rate,
            )
            self.assertAlmostEqual(beliefs[incident.id], expected, places=6)


class V1BenchmarkTests(unittest.TestCase):
    def test_fixture_transition_threshold_and_control_invariance(self):
        cfg = Config(repair_slots=50)
        fixture = prior_shift_fixture(cfg)
        self.assertEqual(len(fixture.history), 2000)
        self.assertEqual(len({x.id for x in fixture.history} | {
            x.id for r in fixture.rounds for x in r.incidents
        }), 2042)
        result = run_episode_v1(cfg, fixture=fixture)
        first, second = result["rounds"]
        self.assertEqual(first["history_size_after"], second["history_size_before"])
        self.assertEqual(first["priors_before"], {"new": .005, "old": .05})
        self.assertAlmostEqual(second["priors_before"]["new"], 25 / 1020)
        self.assertEqual(second["priors_before"]["old"], .05)
        self.assertEqual(first["chosen_actions"]["r0-new-signal"], "defer")
        self.assertEqual(second["chosen_actions"]["r1-new-signal"], "repair")
        self.assertEqual(first["chosen_actions"]["r0-old-control"], "repair")
        self.assertEqual(second["chosen_actions"]["r1-old-control"], "repair")

    def test_private_resolutions_never_reach_current_backend(self):
        seen = []

        class Backend:
            name = "reference"
            def __init__(self, config): pass
            def infer(self, history, incidents, budget, statements):
                seen.append((list(history), list(incidents), statements))
                return {}, {"queries": len(incidents), "engine_steps": None}

        with patch("stationops.v1.ReferenceBackend", Backend):
            run_episode_v1(Config())
        fixture = prior_shift_fixture(Config())
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(seen[0][0]), len(fixture.history))
        for resolution in fixture.rounds[0].resolutions:
            self.assertNotIn(f"leak-{resolution.id}", seen[0][2])
        self.assertEqual(
            [x.id for x in seen[1][0][-len(fixture.rounds[0].resolutions):]],
            [x.id for x in fixture.rounds[0].resolutions],
        )

    def test_schema_aggregate_repeatability_zero_and_irrelevant_invariance(self):
        cfg = Config(repair_slots=50)
        a = run_episode_v1(cfg)
        b = run_episode_v1(cfg)
        for result in (a, b):
            for key in ("benchmark", "schema_version", "config", "seed", "fixture", "backend",
                        "budget_per_round", "rounds", "aggregate", "aggregate_counters",
                        "wall_time_seconds"):
                self.assertIn(key, result)
            self.assertAlmostEqual(result["aggregate"]["expected_utility"],
                                   sum(x["expected_utility"] for x in result["rounds"]))
            self.assertAlmostEqual(result["aggregate"]["regret"],
                                   sum(x["regret"] for x in result["rounds"]))
            json.dumps(result)
        a.pop("wall_time_seconds"); b.pop("wall_time_seconds")
        for result in (a, b):
            for round_ in result["rounds"]:
                round_.pop("wall_time_seconds")
        self.assertEqual(a, b)
        zero = run_episode_v1(cfg, budget=0)
        self.assertTrue(all(not r["beliefs"] for r in zero["rounds"]))
        self.assertTrue(all(set(r["chosen_actions"].values()) == {"defer"} for r in zero["rounds"]))
        irrelevant = run_episode_v1(Config(repair_slots=50, irrelevant_statements=25))
        self.assertEqual([r["beliefs"] for r in a["rounds"]],
                         [r["beliefs"] for r in irrelevant["rounds"]])

    def test_aggregate_engine_steps_sum_only_when_all_rounds_are_numeric(self):
        class Backend:
            name = "reference"
            calls = 0
            def __init__(self, config): pass
            def infer(self, history, incidents, budget, statements):
                self.__class__.calls += 1
                return {}, {"queries": len(incidents), "engine_steps": self.calls * 3}

        with patch("stationops.v1.ReferenceBackend", Backend):
            result = run_episode_v1(Config())
        self.assertEqual(result["aggregate_counters"]["engine_steps"], 9)

        class Partial(Backend):
            calls = 0
            def infer(self, history, incidents, budget, statements):
                self.__class__.calls += 1
                steps = 3 if self.calls == 1 else None
                return {}, {"queries": len(incidents), "engine_steps": steps}

        with patch("stationops.v1.ReferenceBackend", Partial):
            result = run_episode_v1(Config())
        self.assertIsNone(result["aggregate_counters"]["engine_steps"])

    def test_controlled_pettachainer_uses_fresh_handler_each_round(self):
        class Handler:
            instances = []
            def __init__(self):
                self.__class__.instances.append(self)
            def add_atoms_no_check(self, atoms): pass
            def query(self, query, steps, timeout_sec):
                return ["(: proof (PatchPaysOff x y) (STV .2 1))"]
        backend_module = SimpleNamespace(PeTTaChainer=Handler)
        real = PeTTaChainerBackend
        with patch("stationops.v1.PeTTaChainerBackend",
                   lambda config, path: real(config, module=backend_module)):
            result = run_episode_v1(Config(), "pettachainer")
        self.assertEqual(len(Handler.instances), len(result["rounds"]))
        self.assertEqual(result["aggregate_counters"]["queries"], 42)

    def test_human_validation_scoring_parity_and_no_early_reveal(self):
        cfg = Config(repair_slots=50)
        automated = run_episode_v1(cfg)
        choices = iter([
            ",".join(k for k, v in automated["rounds"][0]["chosen_actions"].items() if v == "repair"),
            ",".join(k for k, v in automated["rounds"][1]["chosen_actions"].items() if v == "repair"),
        ])
        output = io.StringIO()
        snapshots = []
        def input_fn(prompt):
            snapshots.append(output.getvalue())
            return next(choices)
        human = play_episode_v1(cfg, input_fn=input_fn, output=output)
        self.assertNotIn("leak=", snapshots[0])
        self.assertIn("r0-new-signal: leak=", snapshots[1])
        self.assertEqual(human["backend"], "human")
        self.assertEqual(human["aggregate"], automated["aggregate"])

    def test_human_reprompts_invalid_duplicate_and_capacity_and_handles_quit(self):
        output = io.StringIO()
        values = iter(["unknown", "r0-new-signal,r0-new-signal", "r0-new-signal,r0-new-evidence-00", "quit"])
        result = play_episode_v1(Config(repair_slots=1), input_fn=lambda prompt: next(values), output=output)
        self.assertEqual(result["status"], "quit")
        self.assertEqual(len(result["rounds"]), 1)
        self.assertIn("unknown incident", output.getvalue())
        self.assertIn("duplicate incident", output.getvalue())
        self.assertIn("slot limit", output.getvalue())

    def test_live_pettachainer_v1_conformance_when_available(self):
        if os.environ.get("STATIONOPS_LIVE_PETTACHAINER") != "1":
            self.skipTest("set STATIONOPS_LIVE_PETTACHAINER=1 for live conformance")
        path = os.environ.get("PETTACHAINER_PYTHONPATH")
        try:
            PeTTaChainerBackend(Config(), path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        cfg = Config(repair_slots=50)
        try:
            result = run_episode_v1(cfg, "pettachainer", 200, pettachainer_path=path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        for round_ in result["rounds"]:
            for incident in round_["visible_incidents"]:
                expected = posterior(
                    round_["priors_before"][incident["cohort"]], incident["alarm"],
                    cfg.sensitivity, cfg.false_positive_rate,
                )
                self.assertAlmostEqual(round_["beliefs"][incident["id"]], expected, places=6)
        self.assertEqual(result["rounds"][0]["chosen_actions"]["r0-new-signal"], "defer")
        self.assertEqual(result["rounds"][1]["chosen_actions"]["r1-new-signal"], "repair")
        self.assertEqual(result["rounds"][0]["chosen_actions"]["r0-old-control"], "repair")
        self.assertEqual(result["rounds"][1]["chosen_actions"]["r1-old-control"], "repair")

    def test_fixture_rejects_mismatch_and_duplicate_ids(self):
        incident = Incident("x", "new", True)
        with self.assertRaises(ValueError):
            RoundFixture((incident,), (HistoryCase("x", "old", True, True),))
        round_ = RoundFixture((incident,), (HistoryCase("x", "new", True, True),))
        with self.assertRaises(ValueError):
            EpisodeFixture("bad", (HistoryCase("x", "new", False, False),), (round_,))


if __name__ == "__main__":
    unittest.main()
