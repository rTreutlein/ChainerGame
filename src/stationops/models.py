from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryCase:
    id: str
    cohort: str
    leak: bool
    alarm: bool


@dataclass(frozen=True)
class Incident:
    id: str
    cohort: str
    alarm: bool

