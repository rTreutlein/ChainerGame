from collections import Counter
import math

from .config import Config
from .models import HistoryCase


def empirical_priors(history: list[HistoryCase]) -> dict[str, float]:
    totals = Counter(x.cohort for x in history)
    leaks = Counter(x.cohort for x in history if x.leak)
    return {c: leaks[c] / totals[c] for c in sorted(totals)}


def empirical_feature_priors(history: list[HistoryCase]) -> dict[tuple[str, str | None], float]:
    """Estimate leak rates at the most specific visible feature level."""
    totals = Counter((x.cohort, x.equipment_type) for x in history)
    leaks = Counter((x.cohort, x.equipment_type) for x in history if x.leak)
    ordered = sorted(
        totals.items(), key=lambda row: (row[0][0], row[0][1] or "")
    )
    return {key: leaks[key] / total for key, total in ordered}


def posterior(prior: float, alarm: bool, sensitivity: float, fpr: float) -> float:
    like_leak = sensitivity if alarm else 1.0 - sensitivity
    like_sound = fpr if alarm else 1.0 - fpr
    numerator = like_leak * prior
    return numerator / (numerator + like_sound * (1.0 - prior))


def belief_error_metrics(
    beliefs: dict[str, float], oracle_beliefs: dict[str, float]
) -> dict[str, int | float | None]:
    """Measure approximation error without making oracle equality a contract."""
    signed_errors = [
        beliefs[key] - expected
        for key, expected in oracle_beliefs.items()
        if isinstance(beliefs.get(key), (int, float))
        and math.isfinite(beliefs[key])
    ]
    expected_count = len(oracle_beliefs)
    evaluated_count = len(signed_errors)
    absolute_errors = [abs(error) for error in signed_errors]
    confidences = [
        min(max(float(getattr(beliefs[key], "confidence", 1.0)), 0.0), 1.0)
        for key in oracle_beliefs
        if isinstance(beliefs.get(key), (int, float))
        and math.isfinite(beliefs[key])
    ]
    return {
        "expected_count": expected_count,
        "evaluated_count": evaluated_count,
        "missing_count": expected_count - evaluated_count,
        "coverage": 1.0 if expected_count == 0 else evaluated_count / expected_count,
        "mean_confidence": (
            sum(confidences) / evaluated_count if evaluated_count else None
        ),
        "confidence_weighted_coverage": (
            1.0 if expected_count == 0 else sum(confidences) / expected_count
        ),
        "mean_absolute_error": (
            sum(absolute_errors) / evaluated_count if evaluated_count else None
        ),
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
    }


def repair_increment(p: float, config: Config) -> float:
    return p * config.avoided_loss - (1.0 - p) * config.unnecessary_penalty - config.action_cost


def resolve(history: list[HistoryCase], case: HistoryCase) -> list[HistoryCase]:
    return [*history, case]
