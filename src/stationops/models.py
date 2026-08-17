import math
from dataclasses import dataclass
from typing import Tuple


class Belief(float):
    """Numeric belief strength carrying the confidence of its source STV.

    Subclassing ``float`` preserves the existing public JSON and metric
    contracts while keeping confidence available at the decision boundary.
    """

    def __new__(cls, strength: float, confidence: float = 1.0):
        strength = float(strength)
        confidence = float(confidence)
        if not math.isfinite(strength) or not math.isfinite(confidence):
            raise ValueError("belief strength and confidence must be finite")
        value = super().__new__(cls, strength)
        value.confidence = min(max(confidence, 0.0), 1.0)
        return value

    @property
    def strength(self) -> float:
        return float(self)

    def truth_value(self) -> dict[str, float]:
        return {"strength": self.strength, "confidence": self.confidence}


def as_belief(value: float | Belief, confidence: float | None = None) -> Belief:
    if isinstance(value, Belief) and confidence is None:
        return value
    return Belief(
        float(value),
        getattr(value, "confidence", 1.0) if confidence is None else confidence,
    )


def belief_truth_values(values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {key: as_belief(value).truth_value() for key, value in values.items()}


@dataclass(frozen=True)
class HistoryCase:
    id: str
    cohort: str
    leak: bool
    alarm: bool
    equipment_type: str | None = None
    problem: bool | None = None


@dataclass(frozen=True)
class Incident:
    id: str
    cohort: str
    alarm: bool
    equipment_type: str | None = None
    module_id: str | None = None
    upstream_module_ids: Tuple[str, ...] = ()
    problem_context: str | None = None


@dataclass(frozen=True)
class RoundFixture:
    incidents: Tuple[Incident, ...]
    resolutions: Tuple[HistoryCase, ...]

    def __post_init__(self):
        visible = [(x.id, x.cohort, x.alarm) for x in self.incidents]
        resolved = [(x.id, x.cohort, x.alarm) for x in self.resolutions]
        if visible != resolved:
            raise ValueError("round resolutions must match visible incidents in stable order")


@dataclass(frozen=True)
class EpisodeFixture:
    name: str
    history: Tuple[HistoryCase, ...]
    rounds: Tuple[RoundFixture, ...]

    def __post_init__(self):
        ids = [x.id for x in self.history]
        ids.extend(x.id for round_ in self.rounds for x in round_.incidents)
        if len(ids) != len(set(ids)):
            raise ValueError("fixture IDs must be unique across the episode")
