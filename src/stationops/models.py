from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HistoryCase:
    id: str
    cohort: str
    leak: bool
    alarm: bool
    equipment_type: str | None = None


@dataclass(frozen=True)
class Incident:
    id: str
    cohort: str
    alarm: bool
    equipment_type: str | None = None
    module_id: str | None = None
    upstream_module_ids: Tuple[str, ...] = ()


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
