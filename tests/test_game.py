import json
import os
import unittest
from unittest.mock import patch

from stationops.backends import ReferenceBackend
from stationops.game import (
    GameConfig,
    GameSession,
    allocate_repairs,
    decision_beliefs,
    diagnostic_plan,
    run_game_episode,
)
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
            GameConfig(shifts=2, modules=10, old_prior=0.0, new_prior=0.0)
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
            f"(SealLeak {current.module.cohort} {current.id})",
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
        highest_value = max(session._incidents, key=lambda item: item.module.value_at_risk)
        self.assertEqual(chosen, [highest_value.id])


class AutomatedGameTests(unittest.TestCase):
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
            self.assertIsNotNone(round_["belief_metrics"]["brier_score"])
            self.assertNotIn("fault", json.dumps(round_["visible_incidents"]))

    def test_zero_budget_has_no_inferred_diagnostics_or_repairs(self):
        result = run_game_episode(budget=0)
        for round_ in result["rounds"]:
            self.assertEqual(round_["belief_metrics"]["coverage"], 0.0)
            self.assertTrue(set(round_["chosen_repairs"]).issubset(round_["inspections"]))

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
        with patch("stationops.game._backend", return_value=backend):
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
            GameConfig(shifts=2, modules=4), "mm2", 100, mm2_path=path
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
            10,
            pettachainer_path=os.environ.get("PETTACHAINER_PYTHONPATH"),
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
