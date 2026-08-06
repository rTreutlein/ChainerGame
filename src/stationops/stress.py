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
        "shortfall_budget": result["shortfall_budget_per_shift"],
        "wall_time_seconds": result["wall_time_seconds"],
        "diagnosis_expected": expected,
        "diagnosis_returned": evaluated,
        "diagnosis_coverage": evaluated / expected if expected else 1.0,
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
        "normalized_score": result["aggregate"]["normalized_score"],
        "regret": result["aggregate"]["regret"],
        "station_score": result["aggregate"]["station_score"],
        "total_production": result["aggregate"]["total_production"],
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
        },
        "shifts": [
            {
                "shift": row["shift"],
                "diagnosis_coverage": row["belief_metrics"]["coverage"],
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
        key = (run["diagnosis_budget"], run["shortfall_budget"])
        grouped.setdefault(key, []).append(run)

    curve = []
    for (diagnosis_budget, shortfall_budget), points in grouped.items():
        def distribution(field: str) -> dict:
            values = [point[field] for point in points]
            return {"mean": mean(values), "min": min(values), "max": max(values)}

        curve.append({
            "diagnosis_budget": diagnosis_budget,
            "shortfall_budget": shortfall_budget,
            "runs": len(points),
            "diagnosis_coverage": distribution("diagnosis_coverage"),
            "shortfall_coverage": distribution("shortfall_coverage"),
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
    on_run: Callable[[dict], None] | None = None,
) -> dict:
    if not diagnosis_budgets or not shortfall_budgets or not seeds:
        raise ValueError("stress budgets and seeds cannot be empty")
    if any(value < 0 for value in diagnosis_budgets + shortfall_budgets):
        raise ValueError("stress budgets cannot be negative")

    runs = []
    for seed in seeds:
        seeded_config = replace(config, seed=seed)
        for diagnosis_budget in diagnosis_budgets:
            for shortfall_budget in shortfall_budgets:
                result = run_game_episode(
                    seeded_config,
                    backend,
                    diagnosis_budget,
                    mm2_path,
                    pettachainer_path,
                    shortfall_budget=shortfall_budget,
                )
                summary = summarize_stress_episode(result)
                runs.append(summary)
                if on_run is not None:
                    on_run(summary)
    return {
        "benchmark": "StationOps-v2-stress",
        "schema_version": 1,
        "config": config.to_dict(),
        "backend": backend,
        "diagnosis_budgets": diagnosis_budgets,
        "shortfall_budgets": shortfall_budgets,
        "seeds": seeds,
        "budget_curve": summarize_budget_curve(runs),
        "runs": runs,
    }
