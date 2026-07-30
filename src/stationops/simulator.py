import random

from .config import Config
from .models import HistoryCase, Incident


def _table(n: int, prior: float, sensitivity: float, fpr: float) -> list[tuple[bool, bool]]:
    """Construct one coherent integer contingency table deterministically."""
    leaks = round(n * prior)
    leak_alarm = round(leaks * sensitivity)
    sound_alarm = round((n - leaks) * fpr)
    return (
        [(True, True)] * leak_alarm
        + [(True, False)] * (leaks - leak_alarm)
        + [(False, True)] * sound_alarm
        + [(False, False)] * (n - leaks - sound_alarm)
    )


def generate_history(config: Config) -> list[HistoryCase]:
    rng = random.Random(config.seed)
    result: list[HistoryCase] = []
    for cohort, prior in (("old", config.old_prior), ("new", config.new_prior)):
        rows = _table(config.history_size, prior, config.sensitivity, config.false_positive_rate)
        rng.shuffle(rows)
        result.extend(
            HistoryCase(f"h-{cohort}-{i:04}", cohort, leak, alarm)
            for i, (leak, alarm) in enumerate(rows)
        )
    return result


def generate_incidents(config: Config) -> list[Incident]:
    rng = random.Random(config.seed + 1)
    result = []
    for i in range(config.incidents):
        cohort = "old" if i % 2 == 0 else "new"
        prior = config.old_prior if cohort == "old" else config.new_prior
        leak = rng.random() < prior
        alarm = rng.random() < (config.sensitivity if leak else config.false_positive_rate)
        result.append(Incident(f"current-{i:03}", cohort, alarm))
    return result
