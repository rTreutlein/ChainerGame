import json
import os
import unittest
from unittest.mock import patch

from stationops.actions import (
    ActionCandidate,
    generate_action_statements,
    reference_action_proposals,
)
from stationops.backends import ReferenceBackend
from stationops.config import Config
from stationops.game import (
    GameConfig,
    GameSession,
    _action_candidates,
    allocate_repairs,
    decision_beliefs,
    diagnostic_plan,
    new_fault_probability,
    run_action_loop,
    run_game_episode,
    sensor_rates,
)
from stationops.models import Belief, HistoryCase, Incident
from stationops.web import GameApplication, INDEX_HTML


def without_timing(value):
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if key != "wall_time_seconds"
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


class GameSessionTests(unittest.TestCase):
    def test_action_rules_distinguish_observation_and_intervention_values(self):
        candidates = [ActionCandidate("shift-01-M01", 0.5, 40)]
        proposals = reference_action_proposals(
            "decision-s01-step00",
            candidates,
            inspection_cost=2,
            repair_cost=5,
            unnecessary_repair_penalty=8,
        )
        values = {
            (proposal.action, proposal.rationale): proposal.utility
            for proposal in proposals
        }
        self.assertEqual(
            values,
            {("Repair", "Intervention"): 11.0, ("Inspect", "Diagnostic"): 4.5},
        )

        source = generate_action_statements(
            "decision-s01-step00",
            candidates,
            inspection_cost=2,
            repair_cost=5,
            unnecessary_repair_penalty=8,
        )
        self.assertIn("(RepairValue $context $incident $utility)", source)
        self.assertIn("(InspectionValue $context $incident $utility)", source)
        self.assertIn(
            "(ActionProposal $context (Repair $incident) $utility $confidence Intervention)",
            source,
        )
        self.assertIn(
            "(ActionProposal $context (Inspect $incident) $utility $confidence Diagnostic)",
            source,
        )
        learning = reference_action_proposals(
            "decision-s01-step00",
            [ActionCandidate("shift-01-M02", None, 65, learning_samples=3)],
            inspection_cost=2,
            repair_cost=5,
            unnecessary_repair_penalty=8,
        )
        self.assertEqual(
            [(proposal.action, proposal.rationale) for proposal in learning],
            [("Inspect", "Learning")],
        )

    def test_action_loop_requeries_after_observation_before_repair(self):
        config = GameConfig(
            shifts=1,
            modules=1,
            diagnostic_slots=1,
            repair_slots=1,
        )
        session = GameSession(config)
        incident = session._incidents[0]
        session._module_faults[incident.module.id] = True
        session._faults[incident.id] = True
        backend = ReferenceBackend(session.logic_config)

        inspections, _, repairs, counters, trace, effective = run_action_loop(
            backend, session, {incident.id: 0.5}, 1
        )

        self.assertEqual(inspections, {incident.id: "seal leak confirmed"})
        self.assertEqual(effective[incident.id], 1.0)
        self.assertEqual(repairs, [incident.id])
        self.assertEqual(counters["action_queries"], 2)
        self.assertEqual(
            [row["context"] for row in trace],
            ["decision-s01-step00", "decision-s01-step01"],
        )
        final_repair = next(
            row
            for row in trace[-1]["proposals"]
            if row["action"] == "Repair" and row["incident_id"] == incident.id
        )
        self.assertEqual(final_repair["utility"], incident.module.value_at_risk - 5)

    def test_hidden_rates_are_shared_by_visible_equipment_features(self):
        session = GameSession(GameConfig(modules=10))
        old_rates = {
            module.equipment_type: new_fault_probability(session.config, module)
            for module in session.modules
            if module.cohort == "old"
        }
        self.assertGreater(len(set(old_rates.values())), 1)
        self.assertNotEqual(
            sensor_rates(session.config, "coolant-pump"),
            sensor_rates(session.config, "ore-feed-pump"),
        )

    def test_mixed_knowledge_separates_graph_likelihood_from_history_learning(self):
        session = GameSession(GameConfig(modules=10))
        source = session.logic_statements()
        self.assertNotIn("alarmGivenLeak-", source)
        self.assertIn("problem-alarm-old-coolant-pump", source)
        self.assertIn("(CTV (STV 0.92 1) (STV 0.12 1))", source)
        self.assertIn("problem-alarm-old-oxygen-scrubber", source)
        self.assertIn("(STV 0.75 1)", source)
        self.assertNotIn("problem-alarm-old-thermal-loop-pump", source)
        resolved = session.history[0]
        self.assertIn(
            f"(Inheritance (State {resolved.id}) "
            f"(EquipmentState {resolved.cohort} {resolved.equipment_type}))",
            source,
        )
        self.assertIn(
            f"(Inheritance (State {resolved.id}) "
            f"(SealLeak {resolved.cohort} {resolved.equipment_type}))",
            source,
        )
        propagated = next(
            case for case in session.history if case.problem and not case.leak
        )
        self.assertIn(
            f"(Problem {propagated.cohort} {propagated.equipment_type} "
            f"{propagated.id}) (STV 1 1)",
            source,
        )
        self.assertIn(
            f"(SealLeak {propagated.cohort} {propagated.equipment_type} "
            f"{propagated.id}) (STV 0 1)",
            source,
        )
        self.assertNotIn(f"(State {session.incidents[0].id})", source)

    def test_zero_action_limits_are_valid_benchmark_dimensions(self):
        session = GameSession(
            GameConfig(shifts=1, modules=2, diagnostic_slots=0, repair_slots=0)
        )
        self.assertEqual(session.public_state()["diagnostics_remaining"], 0)
        self.assertEqual(session.repair_capacity(), 0)
        self.assertEqual(session.commit([])["state"]["status"], "complete")

    def test_public_state_does_not_expose_hidden_faults(self):
        session = GameSession()
        state = session.public_state()
        self.assertTrue(any(session._faults.values()), "default seed should exercise a real fault")
        for incident in state["incidents"]:
            self.assertNotIn("fault", incident)
            self.assertNotIn("leak", incident)
            self.assertNotIn("inspection", incident)
            self.assertIn("sensor_knowledge", incident)
        json.dumps(state)

    def test_learning_contains_only_discovered_cases(self):
        session = GameSession()
        initial = len(session.history)
        inspected_id = next(key for key, fault in session._faults.items() if fault)
        hidden_healthy = next(key for key, fault in session._faults.items() if not fault)

        result = session.inspect(inspected_id)
        inspected = next(x for x in result["state"]["incidents"] if x["id"] == inspected_id)
        still_hidden = next(x for x in result["state"]["incidents"] if x["id"] == hidden_healthy)
        self.assertEqual(inspected["inspection"], "seal leak confirmed")
        self.assertNotIn("inspection", still_hidden)

        session.commit([])
        learned = {case.id for case in session.history[initial:]}
        self.assertEqual(learned, {inspected_id})
        self.assertNotIn(hidden_healthy, learned)

    def test_undiagnosed_leak_persists_without_report_attribution(self):
        session = GameSession(
            GameConfig(shifts=2, modules=10, old_prior=0.0, new_prior=0.0)
        )
        incident = next(item for item in session._incidents if item.module.id == "M09")
        session._module_faults["M09"] = True
        session._faults[incident.id] = True

        resolution = session.commit([])
        report_text = json.dumps(resolution["report"])
        self.assertGreater(resolution["report"]["production_loss"], 0)
        self.assertEqual(
            resolution["report"]["production_event"],
            "one or more unresolved equipment faults reduced station production",
        )
        self.assertNotIn("M09", report_text)
        self.assertNotIn(incident.id, report_text)
        self.assertTrue(session._module_faults["M09"])

        next_row = next(
            item for item in resolution["state"]["incidents"] if item["module_id"] == "M09"
        )
        self.assertNotIn("known_condition", next_row)
        next_incident = next(item for item in session._incidents if item.module.id == "M09")
        self.assertTrue(session._faults[next_incident.id])

    def test_anonymous_loss_conditions_next_shift_beliefs_by_module_impact(self):
        session = GameSession(
            GameConfig(
                shifts=2,
                modules=10,
                old_prior=0.0,
                new_prior=0.0,
                dependency_graph=False,
            )
        )
        incident = next(item for item in session._incidents if item.module.id == "M09")
        session._module_faults["M09"] = True
        session._faults[incident.id] = True
        snapshot = {item.id: 0.1 for item in session._incidents}
        session.commit([], snapshot)

        raw = {item.id: 0.1 for item in session._incidents}
        adjusted, evidence = decision_beliefs(session, raw)
        current_m09 = next(item for item in session._incidents if item.module.id == "M09")
        self.assertGreater(adjusted[current_m09.id], 0.999)
        self.assertEqual(evidence[current_m09.id]["shortfall_shifts"], [1])
        for item in session._incidents:
            if item.module.id != "M09":
                self.assertLess(adjusted[item.id], 0.001)

    def test_service_age_and_chainer_belief_drive_diagnostic_order(self):
        session = GameSession(GameConfig(modules=3, diagnostic_slots=1))
        session.shift_index = 2
        session._module_records["M01"] = {
            "shift": 1,
            "status": "seal inspected intact",
        }
        raw = {item.id: 0.01 for item in session._incidents}
        target = next(item for item in session._incidents if item.module.id == "M01")
        raw[target.id] = 0.1

        adjusted, evidence = decision_beliefs(session, raw)
        self.assertGreater(adjusted[target.id], raw[target.id])
        self.assertEqual(evidence[target.id]["shifts_since_clear"], 2)
        plan, priorities = diagnostic_plan(session, adjusted)
        self.assertEqual(plan, [target.id])
        self.assertGreater(priorities[target.id], 0)

    def test_low_confidence_proof_barely_moves_public_base_rate(self):
        session = GameSession(GameConfig(seed=7, shifts=1, modules=10))
        target = next(
            item for item in session._incidents if item.module.id == "M03"
        )
        missing, missing_evidence = decision_beliefs(session, {})
        weak, weak_evidence = decision_beliefs(
            session,
            {target.id: Belief(0.24883470583173792, 1.0e-6)},
        )

        self.assertEqual(missing[target.id], 0.25)
        self.assertEqual(missing[target.id].confidence, 0.0)
        self.assertLess(abs(weak[target.id] - missing[target.id]), 2e-9)
        self.assertEqual(weak[target.id].confidence, 1.0e-6)
        self.assertEqual(
            missing_evidence[target.id]["decision_source"], "public-base-rate"
        )
        self.assertEqual(
            weak_evidence[target.id]["decision_source"], "reasoner-proof"
        )
        self.assertAlmostEqual(
            weak_evidence[target.id]["confidence_adjusted_belief"],
            1.0e-6 * 0.24883470583173792 + (1.0 - 1.0e-6) * 0.25,
        )
        candidate = next(
            item
            for item in _action_candidates(session, weak)
            if item.incident_id == target.id
        )
        self.assertEqual(candidate.confidence, 1.0e-6)
        self.assertEqual(
            diagnostic_plan(session, missing)[0], diagnostic_plan(session, weak)[0]
        )
        self.assertEqual(
            allocate_repairs(
                session._incidents, missing, session.config, session.repair_capacity()
            ),
            allocate_repairs(
                session._incidents, weak, session.config, session.repair_capacity()
            ),
        )

    def test_low_confidence_proof_without_public_prior_is_not_actionable(self):
        session = GameSession(GameConfig(shifts=1, modules=1))
        session.history.clear()
        target = session._incidents[0]
        adjusted, evidence = decision_beliefs(
            session, {target.id: Belief(0.9, 0.01)}
        )
        self.assertNotIn(target.id, adjusted)
        self.assertFalse(evidence[target.id]["usable_for_decision"])

    def test_induced_types_receive_exploration_diagnostics_without_a_proof(self):
        session = GameSession(
            GameConfig(modules=10, diagnostic_slots=1, sensor_knowledge="induced")
        )
        plan, _ = diagnostic_plan(session, {})
        self.assertEqual(len(plan), 1)
        self.assertIn(plan[0], {item.id for item in session._incidents})

    def test_diagnosed_unrepaired_leak_remains_known_until_repaired(self):
        session = GameSession(
            GameConfig(shifts=3, modules=2, old_prior=0.0, new_prior=0.0)
        )
        first = session._incidents[0]
        session._module_faults[first.module.id] = True
        session._faults[first.id] = True
        session.inspect(first.id)
        next_state = session.commit([])["state"]

        known = next(item for item in next_state["incidents"] if item["module_id"] == first.module.id)
        self.assertEqual(
            known["known_condition"], "seal leak previously confirmed; still unrepaired"
        )
        current = next(item for item in session._incidents if item.module.id == first.module.id)
        self.assertIn(
            f"(SealLeak {current.module.cohort} "
            f"{current.module.equipment_type} {current.id})",
            session.logic_statements(),
        )

        final_state = session.commit([current.id])["state"]
        repaired = next(
            item for item in final_state["incidents"] if item["module_id"] == first.module.id
        )
        self.assertNotIn("known_condition", repaired)
        self.assertFalse(session._module_faults[first.module.id])

    def test_inspections_and_repairs_consume_real_resources(self):
        config = GameConfig(
            shifts=1,
            modules=4,
            initial_credits=12,
            diagnostic_slots=1,
            repair_slots=2,
            initial_seal_kits=2,
        )
        session = GameSession(config)
        ids = [item["id"] for item in session.public_state()["incidents"]]
        session.inspect(ids[0])
        self.assertEqual(session.credits, 10)
        self.assertEqual(session.repair_capacity(), 2)
        result = session.commit(ids[1:3])
        self.assertEqual(result["state"]["credits"], 0)
        self.assertEqual(result["state"]["seal_kits"], 0)
        self.assertEqual(result["state"]["status"], "complete")

    def test_invalid_actions_are_rejected_without_mutation(self):
        session = GameSession()
        before = session.public_state()
        with self.assertRaisesRegex(ValueError, "unknown incident"):
            session.inspect("missing")
        with self.assertRaisesRegex(ValueError, "twice"):
            session.commit([before["incidents"][0]["id"]] * 2)
        self.assertEqual(session.public_state(), before)

    def test_allocator_uses_production_impact_not_only_alarm_or_id(self):
        session = GameSession(GameConfig(modules=3, repair_slots=1))
        beliefs = {item.id: 0.5 for item in session._incidents}
        chosen = allocate_repairs(session._incidents, beliefs, session.config, 1)
        highest_value = max(
            session._incidents, key=lambda item: item.production_at_risk
        )
        self.assertEqual(chosen, [highest_value.id])

    def test_public_dependency_graph_and_transitive_risk_are_visible(self):
        session = GameSession(
            GameConfig(modules=5, old_prior=0.0, new_prior=0.0)
        )
        state = session.public_state()
        self.assertEqual(
            state["dependency_graph"],
            [
                {"upstream": "M03", "downstream": "M01"},
                {"upstream": "M03", "downstream": "M02"},
                {"upstream": "M01", "downstream": "M04"},
                {"upstream": "M04", "downstream": "M05"},
            ],
        )
        rows = {row["module_id"]: row for row in state["incidents"]}
        self.assertEqual(rows["M03"]["production_value"], 90)
        self.assertEqual(rows["M03"]["production_at_risk"], 380)
        self.assertEqual(rows["M01"]["production_at_risk"], 245)
        self.assertEqual(rows["M05"]["production_at_risk"], 55)

    def test_root_repair_restores_downstream_production_but_symptom_repair_does_not(self):
        config = GameConfig(
            shifts=1,
            modules=5,
            old_prior=0.0,
            new_prior=0.0,
            diagnostic_slots=1,
            repair_slots=1,
        )
        symptom_session = GameSession(config)
        symptom_session._module_faults["M03"] = True
        symptom = next(
            item for item in symptom_session._incidents if item.module.id == "M05"
        )
        symptom_session.inspect(symptom.id)
        symptom_report = symptom_session.commit([symptom.id])["report"]
        self.assertEqual(symptom_report["symptom_repairs"], 1)
        self.assertEqual(symptom_report["production_recovered"], 0)
        self.assertEqual(symptom_report["production_loss"], 380)

        root_session = GameSession(config)
        root_session._module_faults["M03"] = True
        root = next(item for item in root_session._incidents if item.module.id == "M03")
        root_session._faults[root.id] = True
        root_report = root_session.commit([root.id])["report"]
        self.assertEqual(root_report["root_cause_repairs"], 1)
        self.assertEqual(root_report["production_recovered"], 380)
        self.assertEqual(root_report["production_loss"], 0)

    def test_dependency_rules_form_a_multi_hop_causal_chain(self):
        session = GameSession(GameConfig(modules=5, sensor_knowledge="full"))
        source = session.logic_statements()
        self.assertIn(
            "(BiImplication (Inheritance (ModuleState shift-01 M03) "
            "(SealLeak old power-converter)) (LocalProblem shift-01 M03 "
            "old power-converter shift-01-M03))",
            source,
        )
        self.assertIn(
            "(Or (LocalProblem shift-01 M01 old coolant-pump "
            "shift-01-M01) (ProblemDependency shift-01 M01 "
            "old coolant-pump shift-01-M01))",
            source,
        )
        self.assertIn(
            "(BiImplication (Problem new thermal-loop-pump shift-01-M04) "
            "(ProblemDependency shift-01 M05 old ore-feed-pump "
            "shift-01-M05))",
            source,
        )
        self.assertIn(
            "(Implication (Problem old ore-feed-pump $unit) "
            "(PressureAlarm old ore-feed-pump $unit))",
            source,
        )

    def test_reference_oracle_conditions_jointly_on_dependency_chain(self):
        history = [
            HistoryCase(f"h-{index}", "old", index < 10, index < 10, "pump")
            for index in range(100)
        ]
        graph = [
            Incident("a", "old", True, "pump", "M1"),
            Incident("b", "old", True, "pump", "M2", ("M1",)),
            Incident("c", "old", True, "pump", "M3", ("M2",)),
        ]
        independent = [
            Incident(item.id, item.cohort, item.alarm, item.equipment_type)
            for item in graph
        ]
        backend = ReferenceBackend(
            Config(sensitivity=0.9, false_positive_rate=0.1)
        )
        graph_beliefs, counters = backend.infer(history, graph, 1, "")
        independent_beliefs, _ = backend.infer(history, independent, 1, "")
        self.assertEqual(counters["exact_dependency_components"], 1)
        self.assertGreater(graph_beliefs["a"], independent_beliefs["a"])
        self.assertLess(graph_beliefs["c"], independent_beliefs["c"])


class AutomatedGameTests(unittest.TestCase):
    def test_long_episode_reports_windowed_learning_metrics(self):
        result = run_game_episode(
            GameConfig(shifts=6, modules=4, learning_window=2),
            "reference",
        )
        curve = result["aggregate"]["learning_curve"]
        self.assertEqual(
            [(window["shift_start"], window["shift_end"]) for window in curve],
            [(1, 2), (3, 4), (5, 6)],
        )
        self.assertTrue(all(window["mean_coverage"] == 1.0 for window in curve))

    def test_reference_episode_is_deterministic_and_complete(self):
        first = run_game_episode()
        second = run_game_episode()
        self.assertEqual(without_timing(first), without_timing(second))
        self.assertEqual(first["benchmark"], "StationOps-v2")
        self.assertEqual(first["status"], "complete")
        self.assertEqual(len(first["rounds"]), 5)
        self.assertEqual(first["aggregate"]["regret"], 0.0)
        self.assertEqual(first["aggregate"]["normalized_score"], 1.0)
        self.assertGreater(first["aggregate"]["confirmed_history_size"], 80)
        self.assertLess(first["aggregate"]["confirmed_history_size"], 130)
        for round_ in first["rounds"]:
            self.assertEqual(round_["belief_metrics"]["coverage"], 1.0)
            self.assertEqual(
                round_["belief_metrics"]["confidence_weighted_coverage"], 1.0
            )
            self.assertEqual(round_["decision_belief_metrics"]["coverage"], 1.0)
            self.assertTrue(all(
                value
                == {
                    "strength": round_["beliefs"][incident_id],
                    "confidence": 1.0,
                }
                for incident_id, value in round_["belief_truth_values"].items()
            ))
            self.assertIsNotNone(round_["belief_metrics"]["brier_score"])
            self.assertNotIn("fault", json.dumps(round_["visible_incidents"]))

    def test_zero_diagnosis_budget_uses_public_base_rates(self):
        result = run_game_episode(budget=0, action_budget=1)
        for round_ in result["rounds"]:
            self.assertEqual(round_["belief_metrics"]["coverage"], 0.0)
            self.assertTrue(round_["decision_beliefs"])
            self.assertTrue(all(
                evidence["decision_source"] == "public-base-rate"
                for incident_id, evidence in round_["belief_evidence"].items()
                if incident_id not in round_["inspections"]
                and not evidence.get("known_unrepaired_fault")
            ))

    def test_zero_action_budget_returns_no_action_proposals(self):
        result = run_game_episode(
            GameConfig(shifts=1, modules=4),
            budget=1,
            action_budget=0,
        )
        self.assertTrue(
            all(
                not row["proposals"]
                for row in result["rounds"][0]["action_trace"]
            )
        )
        self.assertEqual(result["rounds"][0]["chosen_repairs"], [])

    def test_diagnosis_action_and_shortfall_budgets_are_independent(self):
        class RecordingBackend:
            name = "recording"

            def __init__(self, config):
                self.reference = ReferenceBackend(config)
                self.diagnosis_budgets = []
                self.action_budgets = []
                self.shortfall_budgets = []

            def infer(self, history, incidents, budget, statements):
                self.diagnosis_budgets.append(budget)
                return self.reference.infer(history, incidents, budget, statements)

            def condition_shortfalls(self, events, budget):
                self.shortfall_budgets.append(budget)
                return self.reference.condition_shortfalls(events, budget)

            def propose_actions(self, context, candidates, budget, **costs):
                self.action_budgets.append(budget)
                return self.reference.propose_actions(
                    context, candidates, budget, **costs
                )

        config = GameConfig(shifts=2, modules=4)
        backend = RecordingBackend(config.logic_config())
        with patch("stationops.game.create_backend", return_value=backend):
            result = run_game_episode(
                config,
                "reference",
                budget=3,
                shortfall_budget=7,
                action_budget=11,
            )
        self.assertEqual(backend.diagnosis_budgets, [3, 3])
        self.assertEqual(backend.shortfall_budgets, [7, 7])
        self.assertTrue(backend.action_budgets)
        self.assertTrue(all(value == 11 for value in backend.action_budgets))
        self.assertEqual(result["diagnosis_budget_per_shift"], 3)
        self.assertEqual(result["action_budget_per_query"], 11)
        self.assertEqual(result["shortfall_budget_per_shift"], 7)

    def test_backend_receives_only_pre_shift_history_and_visible_incidents(self):
        class RecordingBackend:
            name = "recording"

            def __init__(self, config):
                self.reference = ReferenceBackend(config)
                self.calls = []

            def infer(self, history, incidents, budget, statements):
                self.calls.append((list(history), list(incidents), statements))
                return self.reference.infer(history, incidents, budget, statements)

        config = GameConfig(shifts=3)
        backend = RecordingBackend(config.logic_config())
        with patch("stationops.game.create_backend", return_value=backend):
            result = run_game_episode(config, "reference")
        self.assertEqual(result["backend"], "recording")
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(len(backend.calls[0][0]), 80)
        for history, incidents, statements in backend.calls:
            current_ids = {item.id for item in incidents}
            self.assertTrue(all(case.id not in current_ids for case in history))
            self.assertNotIn("production_loss", statements)
            self.assertNotIn("fault_found", statements)

    def test_live_mm2_game_conformance_when_binding_available(self):
        path = os.environ.get("MM2_CHAINER_PYTHONPATH")
        if not path:
            self.skipTest("set MM2_CHAINER_PYTHONPATH for live StationOps-v2 conformance")
        result = run_game_episode(
            GameConfig(shifts=2, modules=4),
            "mm2",
            600,
            mm2_path=path,
            action_budget=1,
        )
        self.assertGreaterEqual(
            min(round_["belief_metrics"]["coverage"] for round_ in result["rounds"]),
            0.75,
        )
        self.assertEqual(result["aggregate"]["regret"], 0.0)

    def test_live_pettachainer_game_conformance_when_enabled(self):
        if os.environ.get("STATIONOPS_LIVE_PETTACHAINER") != "1":
            self.skipTest("set STATIONOPS_LIVE_PETTACHAINER=1 for live StationOps-v2 conformance")
        result = run_game_episode(
            GameConfig(shifts=2, modules=4),
            "pettachainer",
            600,
            pettachainer_path=os.environ.get("PETTACHAINER_PYTHONPATH"),
            action_budget=1,
        )
        self.assertGreaterEqual(
            min(round_["belief_metrics"]["coverage"] for round_ in result["rounds"]),
            0.75,
        )
        self.assertEqual(result["aggregate"]["regret"], 0.0)


class WebGameTests(unittest.TestCase):
    def test_dashboard_and_restart_use_the_shared_session(self):
        self.assertIn("Current station state", INDEX_HTML)
        self.assertIn("/api/inspect", INDEX_HTML)
        app = GameApplication(GameConfig(shifts=1, modules=2))
        original = app.session
        app.session.commit([])
        self.assertEqual(app.session.status, "complete")
        restarted = app.restart()
        self.assertIsNot(app.session, original)
        self.assertEqual(restarted["status"], "active")
        self.assertEqual(restarted["shift"], 1)


if __name__ == "__main__":
    unittest.main()
