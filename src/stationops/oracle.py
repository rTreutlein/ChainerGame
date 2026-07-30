from collections import Counter

from .config import Config
from .models import HistoryCase


def empirical_priors(history: list[HistoryCase]) -> dict[str, float]:
    totals = Counter(x.cohort for x in history)
    leaks = Counter(x.cohort for x in history if x.leak)
    return {c: leaks[c] / totals[c] for c in sorted(totals)}


def posterior(prior: float, alarm: bool, sensitivity: float, fpr: float) -> float:
    like_leak = sensitivity if alarm else 1.0 - sensitivity
    like_sound = fpr if alarm else 1.0 - fpr
    numerator = like_leak * prior
    return numerator / (numerator + like_sound * (1.0 - prior))


def repair_increment(p: float, config: Config) -> float:
    return p * config.avoided_loss - (1.0 - p) * config.unnecessary_penalty - config.action_cost


def resolve(history: list[HistoryCase], case: HistoryCase) -> list[HistoryCase]:
    return [*history, case]

