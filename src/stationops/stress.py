"""Budget and workload stress sweeps for the interactive StationOps system."""
from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Callable

from .game import GameConfig, run_game_episode


def _sum_optional(values):
    present = [value for value in values if isinstance(value, (int, float))]
    return sum(present) if present else None


def summarize_stress_episode(result: dict) -> dict:
    rounds = result["rounds"]
    expected = sum(row["belief_metrics"]["expected_count"] for row in rounds)
    evaluated = sum(row["belief_metrics"]["evaluated_count"] for row in rounds)
    confidence_mass = sum(
        row["belief_metrics"].get(
            "confidence_weighted_coverage", row["belief_metrics"]["coverage"]
        )
        * row["belief_metrics"]["expected_count"]
        for row in rounds
    )
    shortfall_requested = sum(
        row["backend_counters"].get("shortfall_queries", 0) for row in rounds
    )
    shortfall_returned = sum(
        len(marginals)
        for row in rounds
        for marginals in row["shortfall_marginals"].values()
    )

    def work_total(scope: str, field: str):
        return _sum_optional(
            (row["backend_counters"].get(scope) or {}).get(field)
            for row in rounds
        )

    return {
        "seed": result["config"]["seed"],
        "backend": result["backend"],
        "diagnosis_budget": result["diagnosis_budget_per_shift"],
        "action_budget": result["action_budget_per_query"],
        "shortfall_budget": result["shortfall_budget_per_shift"],
        "wall_time_seconds": result["wall_time_seconds"],
        "diagnosis_expected": expected,
        "diagnosis_returned": evaluated,
        "diagnosis_coverage": evaluated / expected if expected else 1.0,
        "diagnosis_confidence_weighted_coverage": (
            confidence_mass / expected if expected else 1.0
        ),
        "shortfall_requested": shortfall_requested,
        "shortfall_returned": shortfall_returned,
        "shortfall_coverage": (
            shortfall_returned / shortfall_requested if shortfall_requested else 1.0
        ),
        "shortfall_engine_queries": _sum_optional(
            row["backend_counters"].get("shortfall_engine_queries")
            for row in rounds
        ),
        "shortfall_cache_hits": _sum_optional(
            row["backend_counters"].get("shortfall_cache_hits") for row in rounds
        ),
        "action_queries": sum(len(row["action_trace"]) for row in rounds),
        "action_empty_queries": sum(
            not trace["proposals"]
            for row in rounds
            for trace in row["action_trace"]
        ),
        "action_proposals": sum(
            len(trace["proposals"])
            for row in rounds
            for trace in row["action_trace"]
        ),
        "normalized_score": result["aggregate"]["normalized_score"],
        "regret": result["aggregate"]["regret"],
        "station_score": result["aggregate"]["station_score"],
        "total_production": result["aggregate"]["total_production"],
        "production_recovered": result["aggregate"].get("production_recovered", 0),
        "root_cause_repairs": result["aggregate"].get("root_cause_repairs", 0),
        "symptom_repairs": result["aggregate"].get("symptom_repairs", 0),
        "causal_diagnosis": {
            label: {
                "modules": sum(
                    row.get("causal_diagnosis_metrics", {}).get(label, {}).get("modules", 0)
                    for row in rounds
                ),
                "diagnoses": sum(
                    row.get("causal_diagnosis_metrics", {}).get(label, {}).get("diagnoses", 0)
                    for row in rounds
                ),
            }
            for label in sorted({
                label
                for row in rounds
                for label in row.get("causal_diagnosis_metrics", {})
            })
        },
        "learning_curve": result["aggregate"]["learning_curve"],
        "backend_work": {
            "diagnosis_steps": work_total("diagnosis_engine_stats", "steps"),
            "diagnosis_transitions": work_total(
                "diagnosis_engine_stats", "transitions"
            ),
            "diagnosis_unifications": work_total(
                "diagnosis_engine_stats", "unifications"
            ),
            "shortfall_steps": work_total("shortfall_engine_stats", "steps"),
            "shortfall_transitions": work_total(
                "shortfall_engine_stats", "transitions"
            ),
            "shortfall_unifications": work_total(
                "shortfall_engine_stats", "unifications"
            ),
            "action_steps": _sum_optional(
                row["backend_counters"].get("action_engine_steps")
                for row in rounds
            ),
        },
        "shifts": [
            {
                "shift": row["shift"],
                "diagnosis_coverage": row["belief_metrics"]["coverage"],
                "diagnosis_confidence_weighted_coverage": row[
                    "belief_metrics"
                ].get(
                    "confidence_weighted_coverage",
                    row["belief_metrics"]["coverage"],
                ),
                "shortfall_requested": row["backend_counters"].get(
                    "shortfall_queries", 0
                ),
                "shortfall_engine_queries": row["backend_counters"].get(
                    "shortfall_engine_queries"
                ),
                "shortfall_cache_hits": row["backend_counters"].get(
                    "shortfall_cache_hits"
                ),
                "shortfall_returned": sum(
                    len(values) for values in row["shortfall_marginals"].values()
                ),
                "brier_score": row["belief_metrics"]["brier_score"],
                "log_loss": row["belief_metrics"]["log_loss"],
                "regret": row["regret"],
                "wall_time_seconds": row["wall_time_seconds"],
            }
            for row in rounds
        ],
    }


def summarize_budget_curve(runs: list[dict]) -> list[dict]:
    grouped = {}
    for run in runs:
        key = (
            run["diagnosis_budget"],
            run["action_budget"],
            run["shortfall_budget"],
        )
        grouped.setdefault(key, []).append(run)

    curve = []
    for (diagnosis_budget, action_budget, shortfall_budget), points in grouped.items():
        def distribution(field: str) -> dict:
            values = [point[field] for point in points]
            return {"mean": mean(values), "min": min(values), "max": max(values)}

        curve.append({
            "diagnosis_budget": diagnosis_budget,
            "action_budget": action_budget,
            "shortfall_budget": shortfall_budget,
            "runs": len(points),
            "diagnosis_coverage": distribution("diagnosis_coverage"),
            "diagnosis_confidence_weighted_coverage": distribution(
                "diagnosis_confidence_weighted_coverage"
            ),
            "shortfall_coverage": distribution("shortfall_coverage"),
            "action_empty_queries": distribution("action_empty_queries"),
            "action_proposals": distribution("action_proposals"),
            "normalized_score": distribution("normalized_score"),
            "regret": distribution("regret"),
            "wall_time_seconds": distribution("wall_time_seconds"),
        })
    return curve


def run_stress_sweep(
    config: GameConfig,
    backend: str,
    diagnosis_budgets: list[int],
    shortfall_budgets: list[int],
    seeds: list[int],
    mm2_path: str | None = None,
    pettachainer_path: str | None = None,
    action_budgets: list[int] | None = None,
    on_run: Callable[[dict], None] | None = None,
) -> dict:
    if (
        not diagnosis_budgets
        or not shortfall_budgets
        or not seeds
        or action_budgets == []
    ):
        raise ValueError("stress budgets and seeds cannot be empty")
    if any(
        value < 0
        for value in diagnosis_budgets + shortfall_budgets + (action_budgets or [])
    ):
        raise ValueError("stress budgets cannot be negative")

    runs = []
    for seed in seeds:
        seeded_config = replace(config, seed=seed)
        for diagnosis_budget in diagnosis_budgets:
            selected_action_budgets = action_budgets or [diagnosis_budget]
            for action_budget in selected_action_budgets:
                for shortfall_budget in shortfall_budgets:
                    result = run_game_episode(
                        seeded_config,
                        backend,
                        diagnosis_budget,
                        mm2_path,
                        pettachainer_path,
                        shortfall_budget=shortfall_budget,
                        action_budget=action_budget,
                    )
                    summary = summarize_stress_episode(result)
                    runs.append(summary)
                    if on_run is not None:
                        on_run(summary)
    return {
        "benchmark": "StationOps-v2-stress",
        "schema_version": 2,
        "config": config.to_dict(),
        "backend": backend,
        "diagnosis_budgets": diagnosis_budgets,
        "action_budgets": action_budgets,
        "action_budget_mode": (
            "explicit" if action_budgets is not None else "same-as-diagnosis"
        ),
        "shortfall_budgets": shortfall_budgets,
        "seeds": seeds,
        "budget_curve": summarize_budget_curve(runs),
        "runs": runs,
    }
