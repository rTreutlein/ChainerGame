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
from stationops.models import Belief, HistoryCase, Incident
from stationops.models import EpisodeFixture, RoundFixture
from stationops.oracle import belief_error_metrics, empirical_priors, posterior, resolve
from stationops.policy import allocate
from stationops.simulator import generate_history, generate_incidents
from stationops.v1 import play_episode_v1, prior_shift_fixture, run_episode_v1


LIVE_MAX_ABSOLUTE_BELIEF_ERROR = 0.05
LIVE_MIN_NORMALIZED_SCORE = 0.95


def compact_live_prior_shift_fixture(config: Config) -> EpisodeFixture:
    """Cheaper live fixture with the same new-cohort threshold transition."""
    history = tuple(generate_history(config))
    first = (
        Incident("r0-new-signal", "new", True),
        Incident("r0-new-evidence-00", "new", True),
        Incident("r0-new-evidence-01", "new", True),
        Incident("r0-new-evidence-02", "new", True),
        Incident("r0-old-control", "old", True),
    )
    first_resolutions = tuple(
        HistoryCase(item.id, item.cohort, item.cohort == "new", item.alarm)
        for item in first
    )
    second = (
        Incident("r1-new-signal", "new", True),
        Incident("r1-old-control", "old", True),
    )
    second_resolutions = tuple(
        HistoryCase(item.id, item.cohort, False, item.alarm) for item in second
    )
    return EpisodeFixture(
        "compact-live-prior-shift",
        history,
        (RoundFixture(first, first_resolutions), RoundFixture(second, second_resolutions)),
    )


class BenchmarkTests(unittest.TestCase):
    def test_fixed_episode_and_coherent_explicit_history(self):
        cfg = Config()
        history, incidents = generate_history(cfg), generate_incidents(cfg)
        self.assertEqual(len(history), 2000)
        self.assertEqual(len(incidents), 100)
        self.assertEqual({"old": 0.05, "new": 0.005}, empirical_priors(history))
        source = generate_statements(history, incidents, cfg)
        self.assertIn("(SealLeak old h-old-0000) (STV 0 1)", source)
        self.assertIn("(PressureAlarm old h-old-0000) (STV 0 1)", source)
        self.assertNotIn("(Not ", source)
        self.assertIn("(CTV", source)
        self.assertIn(
            "(Implication (SealLeak old $unit) (PressureAlarm old $unit))",
            source,
        )
        self.assertIn(
            "(Implication (SealLeak new $unit) (PressureAlarm new $unit))",
            source,
        )
        self.assertIn(
            "(Implication (SealLeak old $unit) (PatchPaysOff old $unit))",
            source,
        )
        self.assertIn(
            "(Implication (SealLeak new $unit) (PatchPaysOff new $unit))",
            source,
        )
        self.assertNotIn("(SealLeak $cohort $unit) (PressureAlarm", source)
        self.assertNotIn("(SealLeak $cohort $unit) (PatchPaysOff", source)
        self.assertNotIn("(Premises ", source)
        self.assertNotIn("(Conclusions ", source)

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
                    "belief_error",
                    "chosen_actions", "expected_utility", "oracle_utility", "do_nothing_utility",
                    "regret", "normalized_score", "wall_time_seconds", "backend_counters"):
            self.assertIn(key, result)
        json.dumps(result)
        self.assertEqual(result["belief_error"]["missing_count"], 0)
        self.assertEqual(result["belief_error"]["max_absolute_error"], 0.0)

    def test_belief_error_metrics_measure_approximation_and_missing_results(self):
        metrics = belief_error_metrics(
            {"exact": 0.25, "approximate": 0.4, "invalid": float("nan")},
            {"exact": 0.25, "approximate": 0.5, "missing": 0.75, "invalid": 0.1},
        )
        self.assertEqual(metrics["expected_count"], 4)
        self.assertEqual(metrics["evaluated_count"], 2)
        self.assertEqual(metrics["missing_count"], 2)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["mean_confidence"], 1.0)
        self.assertEqual(metrics["confidence_weighted_coverage"], 0.5)
        self.assertAlmostEqual(metrics["mean_absolute_error"], 0.05)
        self.assertAlmostEqual(metrics["max_absolute_error"], 0.1)

        uncertain = belief_error_metrics(
            {"only": Belief(0.5, 0.2)}, {"only": 0.5, "missing": 0.5}
        )
        self.assertEqual(uncertain["coverage"], 0.5)
        self.assertEqual(uncertain["mean_confidence"], 0.2)
        self.assertEqual(uncertain["confidence_weighted_coverage"], 0.1)

    def test_mm2_conformance_when_binding_available(self):
        path = os.environ.get("MM2_CHAINER_PYTHONPATH")
        try:
            MM2Backend(Config(), path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        result = run_episode(Config(repair_slots=100), "mm2", 100, path)
        old_alarm = next(x for x in generate_incidents(Config(repair_slots=100)) if x.cohort == "old" and x.alarm)
        new_alarm = next(x for x in generate_incidents(Config(repair_slots=100)) if x.cohort == "new" and x.alarm)
        self.assertEqual(result["belief_error"]["missing_count"], 0)
        self.assertLessEqual(
            result["belief_error"]["max_absolute_error"],
            LIVE_MAX_ABSOLUTE_BELIEF_ERROR,
        )
        self.assertGreater(result["beliefs"][old_alarm.id], Config().repair_threshold)
        self.assertLess(result["beliefs"][new_alarm.id], Config().repair_threshold)
        self.assertEqual(result["chosen_actions"][old_alarm.id], "repair")
        self.assertEqual(result["chosen_actions"][new_alarm.id], "defer")

    def test_mm2_beliefs_are_derived_from_engine_results(self):
        class Engine:
            instances = []

            def __init__(self):
                self.queries = []
                self.__class__.instances.append(self)

            def add_many(self, *args):
                pass

            def set_base_rate(self, *args):
                pass

            def query_many(self, kb, queries, budget):
                self.queries = list(queries)
                if budget < 2:
                    return [(tag, []) for tag, _ in queries]
                values = {"old-alarm": .8, "new-alarm": .01}
                return [
                    (
                        tag,
                        [{
                            "truth_value": {
                                "strength": values[tag],
                                "confidence": .6 if tag == "old-alarm" else .9,
                            }
                        }],
                    )
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
        self.assertEqual(beliefs["old-alarm"].confidence, .6)
        self.assertEqual(beliefs["new-alarm"].confidence, .9)
        self.assertEqual(
            Engine.instances[-1].queries,
            [
                ("old-alarm", "(SealLeak old old-alarm)"),
                ("new-alarm", "(SealLeak new new-alarm)"),
            ],
        )
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

    def test_mm2_reuses_engine_and_keeps_knowledge_append_only(self):
        class Engine:
            instances = []

            def __init__(self):
                self.added = []
                self.base_rates = []
                self.forwarded = []
                self.__class__.instances.append(self)

            def add_many(self, kb, statements):
                self.added.append((kb, statements))

            def set_base_rate(self, kb, pattern, value):
                self.base_rates.append((kb, pattern, value))

            def forward_chain(self, kb, seeds, steps):
                self.forwarded.append((kb, list(seeds), steps))
                return []

            def query_many(self, kb, queries, budget):
                return [(tag, []) for tag, _ in queries]

            def remove_statement(self, *args):
                raise AssertionError("append-only adapter must not remove statements")

        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))
        history = [HistoryCase("history", "old", True, True)]
        first = "\n".join((
            "(: stable (A) (STV 1 1))",
            "(: old-view (B) (STV 1 1))",
        ))
        second = "\n".join((
            "(: stable (A) (STV 1 1))",
            "(: new-fact (C) (STV 1 1))",
            "(: second-new-fact (D) (STV 1 1))",
        ))

        _, initial = backend.infer(history, [], 1, first)
        _, updated = backend.infer(history, [], 1, second)

        self.assertEqual(len(Engine.instances), 1)
        self.assertEqual(
            Engine.instances[0].added,
            [
                ("stationops", first),
                (
                    "stationops",
                    "(: new-fact (C) (STV 1 1))\n"
                    "(: second-new-fact (D) (STV 1 1))",
                ),
            ],
        )
        self.assertEqual(initial["statements_added"], 2)
        self.assertEqual(updated["statements_added"], 2)
        self.assertEqual(updated["statements_removed"], 0)
        self.assertEqual(updated["base_rates_updated"], 0)
        self.assertEqual(updated["forward_seed_facts"], 2)
        self.assertEqual(updated["forward_steps"], 2)
        self.assertEqual(
            Engine.instances[0].forwarded,
            [("stationops", ["(C)", "(D)"], 2)],
        )

        with self.assertRaisesRegex(ValueError, "append-only MM2"):
            backend.infer(history, [], 1, "(: stable (A) (STV .5 1))")

    def test_mm2_prefers_native_scheduler_when_binding_exposes_it(self):
        class Engine:
            constructors = []

            def __init__(self, scheduler):
                self.scheduler = scheduler

            @classmethod
            def native(cls):
                cls.constructors.append("native")
                return cls("native")

        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))

        self.assertEqual(backend._new_engine().scheduler, "native")
        self.assertEqual(Engine.constructors, ["native"])

    def test_mm2_materializes_dependency_problem_roots_before_diagnosis(self):
        class Engine:
            def __init__(self):
                self.query_batches = []
                self.query_budgets = []

            def add_many(self, *args):
                pass

            def set_base_rate(self, *args):
                pass

            def query_many(self, kb, queries, budget):
                self.query_batches.append(list(queries))
                self.query_budgets.append((kb, budget))
                return [(tag, []) for tag, _ in queries]

        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))
        incidents = [
            Incident(
                "shift-01-M01", "old", False, "coolant-pump", "M01",
                ("M03",), "shift-01",
            ),
            Incident(
                "shift-01-M03", "old", False, "power-converter", "M03",
                (), "shift-01",
            ),
        ]
        _, counters = backend.infer([], incidents, 600, "")

        self.assertEqual(counters["dependency_support_queries"], 2)
        self.assertEqual(counters["diagnosis_budget_requested"], 600)
        self.assertEqual(counters["diagnosis_budget_effective"], 100)
        self.assertEqual(
            backend._engine.query_batches[0],
            [
                (
                    "dependency-support:shift-01-M01",
                    "(Problem old coolant-pump shift-01-M01)",
                ),
                (
                    "dependency-support:shift-01-M03",
                    "(Problem old power-converter shift-01-M03)",
                ),
            ],
        )
        self.assertEqual(
            [tag for tag, _ in backend._engine.query_batches[1]],
            ["shift-01-M01", "shift-01-M03"],
        )
        self.assertEqual(
            [budget for _, budget in backend._engine.query_budgets],
            [100, 100],
        )

    def test_mm2_graph_budget_scales_with_batch_width(self):
        class Engine:
            def __init__(self):
                self.query_budgets = []

            def add_many(self, *args):
                pass

            def set_base_rate(self, *args):
                pass

            def query_many(self, _kb, queries, budget):
                self.query_budgets.append(budget)
                return [(tag, []) for tag, _ in queries]

        incidents = [
            Incident(
                f"shift-01-M0{index}", "old", False, "coolant-pump",
                f"M0{index}", ("M09",), "shift-01",
            )
            for index in range(1, 6)
        ]
        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))

        _, counters = backend.infer([], incidents, 600, "")

        self.assertEqual(counters["diagnosis_budget_effective"], 125)
        self.assertEqual(backend._engine.query_budgets, [125, 125])

    def test_pettachainer_beliefs_drive_decisions_and_preserve_order(self):
        class Handler:
            instances = []

            def __init__(self):
                self.queries = []
                self.atoms = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                self.atoms.extend(atoms)

            def remove_statement(self, name):
                raise AssertionError(f"unexpected removal: {name}")

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                self.forwarded = (list(facts), steps)
                return []

            def query_many(self, queries, steps, timeout_sec):
                self.queries.append((list(queries), steps, timeout_sec))
                return [
                    [],
                    [
                        "(: weak (PatchPaysOff old old-alarm) (STV 0.2 1))",
                        "(: strong (PatchPaysOff old old-alarm) (STV 0.8 1))",
                    ],
                ]

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
        self.assertEqual(
            counters,
            {
                "statements_added": 1,
                "statements_removed": 0,
                "forward_seed_facts": 1,
                "forward_steps": 2,
                "queries": 2,
                "induced_queries": 0,
                "learned_relation_queries": 0,
                "engine_steps": None,
            },
        )
        self.assertEqual(
            allocate(incidents, beliefs, cfg),
            {"new-alarm": "defer", "old-alarm": "repair"},
        )
        self.assertEqual(
            Handler.instances[-1].queries,
            [
                ([
                    "(: $prf (SealLeak new new-alarm) $tv)",
                    "(: $prf (SealLeak old old-alarm) $tv)",
                ], 17, 0),
            ],
        )
        self.assertEqual(Handler.instances[-1].atoms, ["(: fact (A) (STV 1 1))"])
        self.assertEqual(Handler.instances[-1].forwarded, (["(A)"], 2))

    def test_pettachainer_snapshots_reuse_handler_and_zero_budget_is_empty(self):
        class Handler:
            instances = []

            def __init__(self):
                self.removed = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                pass

            def remove_statement(self, name):
                self.removed.append(name)

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                return []

            def query_many(self, queries, steps, timeout_sec):
                return [
                    [
                        f"(: proof (PatchPaysOff old "
                        f"{'second' if 'second' in query else 'first'}) (STV .4 1e0))"
                    ]
                    for query in queries
                ]

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
        self.assertEqual(len(Handler.instances), 1)

    def test_pettachainer_induced_query_uses_shared_state_relation(self):
        class Handler:
            instances = []

            def __init__(self):
                self.queries = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                pass

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                return []

            def query_many(self, queries, steps, timeout_sec):
                self.queries.append(list(queries))
                return [["(: induced relation (STV .42 .7))"] for _ in queries]

        backend = PeTTaChainerBackend(
            Config(),
            module=SimpleNamespace(PeTTaChainer=Handler),
            sensor_knowledge={"thermal-loop-pump": "induced"},
        )
        incident = Incident(
            "current", "old", True, "thermal-loop-pump"
        )
        beliefs, counters = backend.infer([], [incident], 100, "")
        self.assertEqual(beliefs, {"current": .42})
        self.assertEqual(beliefs["current"].confidence, .7)
        self.assertEqual(counters["induced_queries"], 1)
        self.assertEqual(counters["learned_relation_queries"], 1)
        self.assertEqual(
            Handler.instances[0].queries,
            [
                [
                    "(: $prf (Inheritance (PressureAlarm old thermal-loop-pump) "
                    "(SealLeak old thermal-loop-pump)) $tv)"
                ]
            ],
        )

        positive_backend = PeTTaChainerBackend(
            Config(),
            module=SimpleNamespace(PeTTaChainer=Handler),
            sensor_knowledge={"thermal-loop-pump": "positive"},
        )
        _, positive_counters = positive_backend.infer([], [incident], 100, "")
        self.assertEqual(positive_counters["induced_queries"], 0)
        self.assertEqual(positive_counters["learned_relation_queries"], 1)

    def test_pettachainer_batches_diagnoses_in_one_shared_query_budget(self):
        class Handler:
            instances = []

            def __init__(self):
                self.batches = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                pass

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                return []

            def query_many(self, queries, steps, timeout_sec):
                self.batches.append((list(queries), steps, timeout_sec))
                return [
                    ["(: first-proof (SealLeak old first) (STV .25 1))"],
                    ["(: second-proof (SealLeak new second) (STV .75 1))"],
                ]

        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=Handler)
        )
        incidents = [
            Incident("first", "old", True),
            Incident("second", "new", False),
        ]

        beliefs, counters = backend.infer([], incidents, 23, "")

        self.assertEqual(beliefs, {"first": .25, "second": .75})
        self.assertEqual(counters["queries"], 2)
        self.assertEqual(
            Handler.instances[0].batches,
            [
                ([
                    "(: $prf (SealLeak old first) $tv)",
                    "(: $prf (SealLeak new second) $tv)",
                ], 23, 0)
            ],
        )

    def test_graph_diagnosis_queries_context_scoped_local_evidence(self):
        graph_incident = Incident(
            "shift-01-M09",
            "old",
            True,
            "thermal-loop-pump",
            "M09",
            ("M06",),
            "shift-01",
        )
        mm2 = MM2Backend(
            Config(),
            module=SimpleNamespace(),
            sensor_knowledge={"thermal-loop-pump": "full"},
        )
        self.assertEqual(
            mm2._incident_query(graph_incident),
            "(LocalProblem shift-01 M09 old "
            "thermal-loop-pump shift-01-M09)",
        )

        induced = MM2Backend(
            Config(),
            module=SimpleNamespace(),
            sensor_knowledge={"thermal-loop-pump": "induced"},
        )
        self.assertEqual(
            induced._incident_query(graph_incident),
            "(Inheritance (PressureAlarm old thermal-loop-pump) "
            "(SealLeak old thermal-loop-pump))",
        )

    def test_pettachainer_graph_query_uses_local_problem_evidence(self):
        class Handler:
            def __init__(self):
                self.queries = []

            def add_atoms_no_check(self, atoms):
                pass

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                return []

            def query_many(self, queries, steps, timeout_sec):
                self.queries.append(list(queries))
                return [[
                    "(: proof (LocalProblem shift-01 M09 old "
                    "thermal-loop-pump shift-01-M09) (STV .7 1))"
                ]]

        backend = PeTTaChainerBackend(
            Config(),
            module=SimpleNamespace(PeTTaChainer=Handler),
            sensor_knowledge={"thermal-loop-pump": "full"},
        )
        incident = Incident(
            "shift-01-M09",
            "old",
            True,
            "thermal-loop-pump",
            "M09",
            ("M06",),
            "shift-01",
        )
        beliefs, _ = backend.infer([], [incident], 20, "")
        self.assertEqual(beliefs, {"shift-01-M09": .7})
        self.assertEqual(
            backend._handler.queries,
            [[
                "(: $prf (LocalProblem shift-01 M09 old "
                "thermal-loop-pump shift-01-M09) $tv)"
            ]],
        )

    def test_pettachainer_adds_named_deltas_rules_first_without_retractions(self):
        class Handler:
            instances = []

            def __init__(self):
                self.added = []
                self.removed = []
                self.forwarded = []
                self.__class__.instances.append(self)

            def add_atoms_no_check(self, atoms):
                self.added.append(list(atoms))

            def remove_statement(self, name):
                self.removed.append(name)

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                self.forwarded.append((list(facts), steps))
                return []

            def query(self, query, steps, timeout_sec):
                return []

        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=Handler)
        )
        first = "\n".join((
            "(: rule (Implication (A) (Goal)) (CTV (STV 1 1) (STV 0 1)))",
            "(: bi-rule (BiImplication (B) (OtherGoal)) "
            "(CTV (STV 1 1) (STV 0 1)))",
            "(: fact-a (A) (STV 1 1))",
            "(: fact-b (Not (B)) (STV 1 1))",
        ))
        second = "\n".join((
            "(: rule (Implication (A) (Goal)) (CTV (STV 1 1) (STV 0 1)))",
            "(: bi-rule (BiImplication (B) (OtherGoal)) "
            "(CTV (STV 1 1) (STV 0 1)))",
            "(: fact-a (A) (STV 1 1))",
            "(: fact-c (C) (STV 1 1))",
        ))

        _, initial = backend.infer([], [], 1, first)
        _, revised = backend.infer([], [], 1, second)

        handler = Handler.instances[0]
        self.assertEqual(
            handler.added,
            [
                [
                    "(: rule (Implication (A) (Goal)) "
                    "(CTV (STV 1 1) (STV 0 1)))",
                    "(: bi-rule (BiImplication (B) (OtherGoal)) "
                    "(CTV (STV 1 1) (STV 0 1)))",
                ],
                ["(: fact-a (A) (STV 1 1))", "(: fact-b (Not (B)) (STV 1 1))"],
                ["(: fact-c (C) (STV 1 1))"],
            ],
        )
        self.assertEqual(handler.removed, [])
        self.assertEqual(
            handler.forwarded,
            [(["(A)", "(B)"], 4), (["(C)"], 2)],
        )
        self.assertEqual(initial["statements_added"], 4)
        self.assertEqual(initial["statements_removed"], 0)
        self.assertEqual(revised["statements_added"], 1)
        self.assertEqual(revised["statements_removed"], 0)

        with self.assertRaisesRegex(ValueError, "append-only PeTTaChainer"):
            backend.infer([], [], 1, "(: fact-a (A) (STV .5 1))")

    def test_pettachainer_keeps_resolved_shift_leaks_out_of_forward_replay(self):
        class Handler:
            def __init__(self):
                self.forwarded = []

            def add_atoms_no_check(self, atoms):
                pass

            def select_facts(self, terms):
                return list(terms)

            def forward_chain(self, facts, steps):
                self.forwarded.extend(facts)

            def query_many(self, queries, steps, timeout_sec):
                return [[] for _ in queries]

        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=Handler)
        )
        backend.infer([], [], 1, "(: initial (A) (STV 1 1))")
        backend._handler.forwarded.clear()
        backend.infer(
            [],
            [],
            1,
            "\n".join((
                "(: initial (A) (STV 1 1))",
                "(: leak-shift-01-M01 (SealLeak old pump shift-01-M01) (STV 0 1))",
                "(: state-leak-shift-01-M01 (Inheritance (State shift-01-M01) "
                "(SealLeak old pump)) (STV 0 1))",
            )),
        )
        self.assertEqual(
            backend._handler.forwarded,
            ["(Inheritance (State shift-01-M01) (SealLeak old pump))"],
        )

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
        oracle_beliefs = {
            incident.id: posterior(
                priors[incident.cohort],
                incident.alarm,
                cfg.sensitivity,
                cfg.false_positive_rate,
            )
            for incident in incidents
        }
        metrics = belief_error_metrics(beliefs, oracle_beliefs)
        self.assertEqual(metrics["missing_count"], 0)
        self.assertLessEqual(
            metrics["max_absolute_error"], LIVE_MAX_ABSOLUTE_BELIEF_ERROR
        )
        self.assertEqual(
            allocate(incidents, beliefs, cfg),
            allocate(incidents, oracle_beliefs, cfg),
        )


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
        a.pop("wall_time_seconds")
        b.pop("wall_time_seconds")
        for result in (a, b):
            for round_ in result["rounds"]:
                round_.pop("wall_time_seconds")
        self.assertEqual(a, b)
        zero = run_episode_v1(cfg, budget=0)
        self.assertTrue(all(not r["beliefs"] for r in zero["rounds"]))
        self.assertTrue(all(r["belief_error"]["coverage"] == 0 for r in zero["rounds"]))
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

    def test_controlled_pettachainer_reuses_handler_and_forwards_round_deltas(self):
        class Handler:
            instances = []
            def __init__(self):
                self.added = []
                self.removed = []
                self.forwarded = []
                self.__class__.instances.append(self)
            def add_atoms_no_check(self, atoms):
                self.added.append(list(atoms))
            def remove_statement(self, name):
                self.removed.append(name)
            def select_facts(self, terms):
                return list(terms)
            def forward_chain(self, facts, steps):
                self.forwarded.append((list(facts), steps))
                return []
            def query_many(self, queries, steps, timeout_sec):
                return [
                    ["(: proof (PatchPaysOff x y) (STV .2 1))"]
                    for _ in queries
                ]
        backend_module = SimpleNamespace(PeTTaChainer=Handler)
        real = PeTTaChainerBackend
        with patch("stationops.v1.PeTTaChainerBackend",
                   lambda config, path: real(config, module=backend_module)):
            result = run_episode_v1(Config(), "pettachainer")
        self.assertEqual(len(Handler.instances), 1)
        handler = Handler.instances[0]
        self.assertEqual(handler.removed, [])
        self.assertEqual(
            result["rounds"][1]["backend_counters"]["statements_added"],
            42,
        )
        self.assertEqual(
            result["rounds"][1]["backend_counters"]["statements_removed"],
            0,
        )
        self.assertEqual(
            result["rounds"][1]["backend_counters"]["forward_seed_facts"],
            42,
        )
        self.assertTrue(handler.forwarded)
        self.assertEqual(result["aggregate_counters"]["queries"], 42)
        self.assertEqual(result["aggregate_counters"]["statements_added"], 4086)
        self.assertEqual(result["aggregate_counters"]["statements_removed"], 0)
        self.assertEqual(result["aggregate_counters"]["forward_seed_facts"], 4082)
        self.assertEqual(result["aggregate_counters"]["forward_steps"], 8164)

    def test_mm2_v1_conformance_when_binding_available(self):
        path = os.environ.get("MM2_CHAINER_PYTHONPATH")
        try:
            MM2Backend(Config(), path)
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        cfg = Config(history_size=200, repair_slots=10)
        result = run_episode_v1(
            cfg,
            "mm2",
            100,
            mm2_path=path,
            fixture=compact_live_prior_shift_fixture(cfg),
        )

        for round_ in result["rounds"]:
            self.assertEqual(round_["belief_error"]["missing_count"], 0)
            self.assertLessEqual(
                round_["belief_error"]["max_absolute_error"],
                LIVE_MAX_ABSOLUTE_BELIEF_ERROR,
            )
        self.assertGreaterEqual(
            result["aggregate"]["normalized_score"], LIVE_MIN_NORMALIZED_SCORE
        )
        self.assertEqual(result["rounds"][0]["chosen_actions"]["r0-new-signal"], "defer")
        self.assertEqual(result["rounds"][1]["chosen_actions"]["r1-new-signal"], "repair")
        self.assertEqual(result["rounds"][0]["chosen_actions"]["r0-old-control"], "repair")
        self.assertEqual(result["rounds"][1]["chosen_actions"]["r1-old-control"], "repair")

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
        cfg = Config(history_size=200, repair_slots=10)
        fixture = compact_live_prior_shift_fixture(cfg)
        try:
            result = run_episode_v1(
                cfg,
                "pettachainer",
                10,
                pettachainer_path=path,
                fixture=fixture,
            )
        except BackendUnavailable as exc:
            self.skipTest(str(exc))
        for round_ in result["rounds"]:
            self.assertEqual(round_["belief_error"]["missing_count"], 0)
            self.assertLessEqual(
                round_["belief_error"]["max_absolute_error"],
                LIVE_MAX_ABSOLUTE_BELIEF_ERROR,
            )
        self.assertGreaterEqual(
            result["aggregate"]["normalized_score"], LIVE_MIN_NORMALIZED_SCORE
        )
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
