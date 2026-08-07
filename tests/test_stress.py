import unittest

from stationops.game import GameConfig
from stationops.stress import (
    run_stress_sweep,
    summarize_budget_curve,
    summarize_stress_episode,
)


class StressBenchmarkTests(unittest.TestCase):
    def test_reference_sweep_preserves_partial_budget_runs(self):
        streamed = []
        result = run_stress_sweep(
            GameConfig(shifts=2, modules=4, initial_history_per_cohort=4),
            "reference",
            [0, 1],
            [0, 1],
            [7],
            action_budgets=[5],
            on_run=streamed.append,
        )
        self.assertEqual(result["benchmark"], "StationOps-v2-stress")
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(len(result["budget_curve"]), 4)
        self.assertEqual(streamed, result["runs"])
        by_budget = {
            (row["diagnosis_budget"], row["shortfall_budget"]): row
            for row in result["runs"]
        }
        self.assertEqual(by_budget[(0, 0)]["diagnosis_coverage"], 0.0)
        self.assertEqual(by_budget[(1, 1)]["diagnosis_coverage"], 1.0)
        self.assertTrue(all(row["action_budget"] == 5 for row in result["runs"]))
        self.assertIn("shifts", by_budget[(0, 0)])

    def test_summary_counts_shortfall_answers_and_native_work(self):
        result = {
            "config": {"seed": 3},
            "backend": "controlled",
            "diagnosis_budget_per_shift": 2,
            "action_budget_per_query": 3,
            "shortfall_budget_per_shift": 5,
            "wall_time_seconds": 1.5,
            "aggregate": {
                "normalized_score": 0.75,
                "regret": 2.0,
                "station_score": 9,
                "total_production": 10,
                "learning_curve": [],
            },
            "rounds": [{
                "shift": 1,
                "belief_metrics": {
                    "expected_count": 4,
                    "evaluated_count": 3,
                    "coverage": 0.75,
                    "brier_score": 0.2,
                    "log_loss": 0.3,
                },
                "backend_counters": {
                    "shortfall_queries": 3,
                    "shortfall_engine_queries": 2,
                    "shortfall_cache_hits": 1,
                    "diagnosis_engine_stats": {
                        "steps": 4,
                        "transitions": 5,
                        "unifications": 6,
                    },
                    "shortfall_engine_stats": {
                        "steps": 7,
                        "transitions": 8,
                        "unifications": 9,
                    },
                    "action_engine_steps": 10,
                },
                "action_trace": [
                    {"proposals": [{"action": "Repair"}]},
                    {"proposals": []},
                ],
                "shortfall_marginals": {1: {"M01": 0.4, "M02": 0.6}},
                "regret": 2.0,
                "wall_time_seconds": 1.5,
            }],
        }
        summary = summarize_stress_episode(result)
        self.assertEqual(summary["diagnosis_coverage"], 0.75)
        self.assertEqual(summary["shortfall_coverage"], 2 / 3)
        self.assertEqual(summary["shortfall_engine_queries"], 2)
        self.assertEqual(summary["shortfall_cache_hits"], 1)
        self.assertEqual(summary["action_budget"], 3)
        self.assertEqual(summary["action_queries"], 2)
        self.assertEqual(summary["action_empty_queries"], 1)
        self.assertEqual(summary["backend_work"]["action_steps"], 10)
        self.assertEqual(summary["shifts"][0]["shortfall_engine_queries"], 2)
        self.assertEqual(summary["shifts"][0]["shortfall_cache_hits"], 1)
        self.assertEqual(summary["backend_work"]["diagnosis_steps"], 4)
        self.assertEqual(summary["backend_work"]["shortfall_unifications"], 9)

    def test_budget_curve_reduces_seed_distributions(self):
        curve = summarize_budget_curve([
            {
                "diagnosis_budget": 1,
                "action_budget": 30,
                "action_empty_queries": 1,
                "action_proposals": 5,
                "shortfall_budget": 20,
                "diagnosis_coverage": 1.0,
                "shortfall_coverage": 0.25,
                "normalized_score": 0.8,
                "regret": 10.0,
                "wall_time_seconds": 2.0,
            },
            {
                "diagnosis_budget": 1,
                "action_budget": 30,
                "action_empty_queries": 0,
                "action_proposals": 7,
                "shortfall_budget": 20,
                "diagnosis_coverage": 0.8,
                "shortfall_coverage": 0.75,
                "normalized_score": 0.6,
                "regret": 30.0,
                "wall_time_seconds": 4.0,
            },
        ])
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve[0]["runs"], 2)
        self.assertEqual(curve[0]["action_budget"], 30)
        self.assertEqual(curve[0]["action_empty_queries"]["max"], 1)
        self.assertEqual(curve[0]["action_proposals"]["mean"], 6)
        self.assertAlmostEqual(curve[0]["diagnosis_coverage"]["mean"], 0.9)
        self.assertEqual(curve[0]["shortfall_coverage"]["min"], 0.25)
        self.assertEqual(curve[0]["regret"]["max"], 30.0)
        self.assertEqual(curve[0]["wall_time_seconds"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
