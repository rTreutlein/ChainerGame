"""Shared hidden-state StationOps simulation for humans and reasoners.

The public surface deliberately contains only observations and facts learned by
the player.  Hidden fault state is owned by :class:`GameSession` and is used
only while resolving actions and calculating benchmark metrics.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass

from .actions import ActionCandidate, ActionProposal, reference_action_proposals
from .backends import MM2Backend, PeTTaChainerBackend, ReferenceBackend
from .config import Config
from .metta import generate_dependency_statements, generate_statements
from .models import HistoryCase, Incident
from .oracle import (
    belief_error_metrics,
    empirical_feature_priors,
    posterior,
)
from .shortfall import oracle_event_marginals


EQUIPMENT_TYPES = (
    ("coolant-pump", "coolant pump"),
    ("oxygen-scrubber", "oxygen scrubber"),
    ("power-converter", "power converter"),
    ("thermal-loop-pump", "thermal-loop pump"),
    ("ore-feed-pump", "ore-feed pump"),
)

# These are hidden simulator parameters shared by every module with the same
# visible equipment type.  They are deliberately not arbitrary per-module
# probabilities: resolved cases can therefore generalize to future states.
HAZARD_MULTIPLIERS = {
    "coolant-pump": 1.35,
    "oxygen-scrubber": 0.70,
    "power-converter": 1.05,
    "thermal-loop-pump": 1.20,
    "ore-feed-pump": 0.80,
}
SENSOR_DELTAS = {
    "coolant-pump": (0.07, -0.03),
    "oxygen-scrubber": (-0.10, -0.07),
    "power-converter": (0.00, 0.05),
    "thermal-loop-pump": (0.05, 0.03),
    "ore-feed-pump": (-0.15, -0.10),
}
MIXED_SENSOR_KNOWLEDGE = {
    "coolant-pump": "full",
    "oxygen-scrubber": "positive",
    "power-converter": "positive",
    "thermal-loop-pump": "induced",
    "ore-feed-pump": "induced",
}


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    shifts: int = 5
    modules: int = 10
    initial_history_per_cohort: int = 40
    diagnostic_slots: int = 2
    repair_slots: int = 2
    initial_credits: int = 30
    credit_income_per_shift: int = 10
    initial_seal_kits: int = 2
    seal_kits_per_shift: int = 1
    maximum_seal_kits: int = 3
    inspection_cost: int = 2
    repair_cost: int = 5
    unnecessary_repair_penalty: int = 8
    old_prior: float = 0.20
    new_prior: float = 0.04
    sensitivity: float = 0.85
    false_positive_rate: float = 0.15
    sensor_knowledge: str = "mixed"
    learning_window: int = 10
    dependency_graph: bool = True

    def __post_init__(self):
        positive = {
            "shifts": self.shifts,
            "modules": self.modules,
            "initial_history_per_cohort": self.initial_history_per_cohort,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("shifts, modules, and initial history must be positive")
        nonnegative = {
            "diagnostic_slots": self.diagnostic_slots,
            "repair_slots": self.repair_slots,
            "initial_credits": self.initial_credits,
            "credit_income_per_shift": self.credit_income_per_shift,
            "initial_seal_kits": self.initial_seal_kits,
            "seal_kits_per_shift": self.seal_kits_per_shift,
            "maximum_seal_kits": self.maximum_seal_kits,
            "inspection_cost": self.inspection_cost,
            "repair_cost": self.repair_cost,
            "unnecessary_repair_penalty": self.unnecessary_repair_penalty,
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError("resources, action limits, and action costs cannot be negative")
        if self.initial_seal_kits > self.maximum_seal_kits:
            raise ValueError("initial seal kits cannot exceed maximum seal kits")
        if self.sensor_knowledge not in {"full", "positive", "induced", "mixed"}:
            raise ValueError("sensor_knowledge must be full, positive, induced, or mixed")
        if self.learning_window <= 0:
            raise ValueError("learning_window must be positive")
        for name in ("old_prior", "new_prior", "sensitivity", "false_positive_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    def to_dict(self) -> dict:
        return asdict(self)

    def logic_config(self) -> Config:
        """Return the one-fault logic contract shared by all current backends."""
        return Config(
            seed=self.seed,
            history_size=self.initial_history_per_cohort,
            incidents=self.modules,
            repair_slots=self.repair_slots,
            old_prior=self.old_prior,
            new_prior=self.new_prior,
            sensitivity=self.sensitivity,
            false_positive_rate=self.false_positive_rate,
            avoided_loss=100.0,
            unnecessary_penalty=float(self.unnecessary_repair_penalty),
            action_cost=float(self.repair_cost),
        )


@dataclass(frozen=True)
class StationModule:
    id: str
    cohort: str
    equipment_type: str
    description: str
    criticality: str
    value_at_risk: int


@dataclass(frozen=True)
class StationIncident:
    id: str
    module: StationModule
    alarm: bool
    production_at_risk: int

    def logic_incident(
        self,
        upstream_module_ids: tuple[str, ...] = (),
        problem_context: str | None = None,
    ) -> Incident:
        return Incident(
            self.id,
            self.module.cohort,
            self.alarm,
            self.module.equipment_type,
            self.module.id,
            upstream_module_ids,
            problem_context,
        )


def _modules(count: int) -> tuple[StationModule, ...]:
    values = (120, 45, 90, 70, 55, 110, 40, 80, 65, 100)
    result = []
    for index in range(count):
        value = values[index % len(values)]
        equipment_type, description = EQUIPMENT_TYPES[index % len(EQUIPMENT_TYPES)]
        criticality = "HIGH" if value >= 90 else "MEDIUM" if value >= 60 else "LOW"
        cohort = "old" if index % 2 == 0 else "new"
        result.append(
            StationModule(
                f"M{index + 1:02d}",
                cohort,
                equipment_type,
                description,
                criticality,
                value,
            )
        )
    return tuple(result)


def dependency_edges(
    modules: tuple[StationModule, ...], enabled: bool = True
) -> tuple[tuple[str, str], ...]:
    """Return ``(downstream, upstream)`` edges for a public acyclic forest.

    Every five-module equipment train has one power source feeding coolant and
    oxygen, with coolant feeding the thermal loop and then the ore feed.  The
    repeated bounded component keeps exact oracle inference linear in station
    size while causal proof depth grows to four rules.
    """
    if not enabled:
        return ()
    ids = {module.id for module in modules}
    edges = []
    for offset in range(0, len(modules), len(EQUIPMENT_TYPES)):
        block = modules[offset:offset + len(EQUIPMENT_TYPES)]
        if len(block) < 2:
            continue
        by_type = {module.equipment_type: module.id for module in block}
        planned = (
            ("coolant-pump", "power-converter"),
            ("oxygen-scrubber", "power-converter"),
            ("thermal-loop-pump", "coolant-pump"),
            ("ore-feed-pump", "thermal-loop-pump"),
        )
        for downstream_type, upstream_type in planned:
            downstream = by_type.get(downstream_type)
            upstream = by_type.get(upstream_type)
            if downstream in ids and upstream in ids:
                edges.append((downstream, upstream))
    return tuple(edges)


def _bounded_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def sensor_rates(config: GameConfig, equipment_type: str) -> tuple[float, float]:
    sensitivity_delta, false_positive_delta = SENSOR_DELTAS[equipment_type]
    return (
        _bounded_probability(config.sensitivity + sensitivity_delta),
        _bounded_probability(config.false_positive_rate + false_positive_delta),
    )


def new_fault_probability(config: GameConfig, module: StationModule) -> float:
    cohort_rate = config.old_prior if module.cohort == "old" else config.new_prior
    return 1.0 - (1.0 - cohort_rate) ** HAZARD_MULTIPLIERS[module.equipment_type]


def sensor_knowledge(config: GameConfig) -> dict[str, str]:
    if config.sensor_knowledge == "mixed":
        return dict(MIXED_SENSOR_KNOWLEDGE)
    return {
        equipment_type: config.sensor_knowledge
        for equipment_type, _ in EQUIPMENT_TYPES
    }


def public_sensor_description(config: GameConfig, equipment_type: str) -> str:
    mode = sensor_knowledge(config)[equipment_type]
    sensitivity, false_positive = sensor_rates(config, equipment_type)
    if mode == "full":
        return (
            f"calibrated: {sensitivity:.0%} detection / "
            f"{false_positive:.0%} false alarm"
        )
    if mode == "positive":
        return f"partial spec: {sensitivity:.0%} detection / false alarm unknown"
    return "uncharacterized sensor"


def _generate_game_history(config: GameConfig) -> list[HistoryCase]:
    """Generate resolved public cases from the same feature-conditioned world."""
    rng = random.Random(config.seed)
    result = []
    for cohort in ("old", "new"):
        cohort_modules = [module for module in _modules(10) if module.cohort == cohort]
        for index in range(config.initial_history_per_cohort):
            template = cohort_modules[index % len(cohort_modules)]
            leak = rng.random() < new_fault_probability(config, template)
            sensitivity, false_positive = sensor_rates(config, template.equipment_type)
            alarm = rng.random() < (sensitivity if leak else false_positive)
            result.append(
                HistoryCase(
                    f"h-{cohort}-{index:04d}",
                    cohort,
                    leak,
                    alarm,
                    template.equipment_type,
                )
            )
    return result


class GameSession:
    """Mutable game state with an explicit public/private information boundary."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.logic_config = self.config.logic_config()
        self.modules = _modules(self.config.modules)
        self.dependencies = dependency_edges(
            self.modules, self.config.dependency_graph
        )
        self._upstream_by_module: dict[str, tuple[str, ...]] = {
            module.id: tuple(
                upstream
                for downstream, upstream in self.dependencies
                if downstream == module.id
            )
            for module in self.modules
        }
        self._downstream_by_module: dict[str, tuple[str, ...]] = {
            module.id: tuple(
                downstream
                for downstream, upstream in self.dependencies
                if upstream == module.id
            )
            for module in self.modules
        }
        self.history = _generate_game_history(self.config)
        self.shift_index = 0
        self.credits = self.config.initial_credits
        self.seal_kits = self.config.initial_seal_kits
        self.station_score = 0.0
        self.total_production = 0
        self._rng = random.Random(self.config.seed + 10_003)
        self._module_faults = {module.id: False for module in self.modules}
        self._known_fault_modules: set[str] = set()
        self._module_records: dict[str, dict[str, int | str]] = {}
        self._shortfall_log: list[dict[str, int]] = []
        self._shortfall_events: list[dict] = []
        self._incidents: tuple[StationIncident, ...] = ()
        self._faults: dict[str, bool] = {}
        self._unavailable_modules: set[str] = set()
        self._inspected: dict[str, bool] = {}
        self._inspection_spend = 0
        self._last_report: dict | None = None
        self.status = "active"
        self._begin_shift()

    def _begin_shift(self):
        if self.shift_index:
            self.credits += self.config.credit_income_per_shift
            self.seal_kits = min(
                self.config.maximum_seal_kits,
                self.seal_kits + self.config.seal_kits_per_shift,
            )
        incidents = []
        faults = {}
        for module in self.modules:
            prior = new_fault_probability(self.config, module)
            fault = self._module_faults[module.id] or self._rng.random() < prior
            self._module_faults[module.id] = fault
        self._unavailable_modules = self._availability_closure(self._module_faults)
        for module in self.modules:
            fault = self._module_faults[module.id]
            sensitivity, false_positive = sensor_rates(
                self.config, module.equipment_type
            )
            alarm_rate = (
                sensitivity
                if module.id in self._unavailable_modules
                else false_positive
            )
            alarm = self._rng.random() < alarm_rate
            incident_id = f"shift-{self.shift_index + 1:02d}-{module.id}"
            incidents.append(StationIncident(
                incident_id,
                module,
                alarm,
                self.production_at_risk(module.id),
            ))
            faults[incident_id] = fault
        self._incidents = tuple(incidents)
        self._faults = faults
        self._inspected = {}
        self._inspection_spend = 0

    @property
    def incidents(self) -> list[Incident]:
        return [
            item.logic_incident(
                self._upstream_by_module[item.module.id],
                f"shift-{self.shift_index + 1:02d}" if self.dependencies else None,
            )
            for item in self._incidents
        ]

    @property
    def maximum_production(self) -> int:
        return sum(module.value_at_risk for module in self.modules)

    def _availability_closure(self, faults: dict[str, bool]) -> set[str]:
        unavailable = {module_id for module_id, fault in faults.items() if fault}
        changed = True
        while changed:
            changed = False
            for downstream, upstream in self.dependencies:
                if upstream in unavailable and downstream not in unavailable:
                    unavailable.add(downstream)
                    changed = True
        return unavailable

    def affected_modules(self, module_id: str) -> set[str]:
        """Return the module plus every transitive consumer of its output."""
        return self._availability_closure({
            module.id: module.id == module_id for module in self.modules
        })

    def production_at_risk(self, module_id: str) -> int:
        affected = self.affected_modules(module_id)
        return sum(
            module.value_at_risk for module in self.modules if module.id in affected
        )

    def causal_distance(self, module_id: str) -> int | None:
        """Distance from the closest active local fault to this module."""
        frontier = [(module_id, 0)]
        visited = set()
        while frontier:
            current, distance = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if self._module_faults.get(current, False):
                return distance
            frontier.extend(
                (upstream, distance + 1)
                for upstream in self._upstream_by_module.get(current, ())
            )
        return None

    def _incident(self, incident_id: str) -> StationIncident:
        for item in self._incidents:
            if item.id == incident_id:
                return item
        raise ValueError(f"unknown incident: {incident_id}")

    def maintenance_summary(self) -> dict[str, dict[str, int]]:
        totals = Counter(case.cohort for case in self.history)
        leaks = Counter(case.cohort for case in self.history if case.leak)
        return {
            cohort: {"confirmed_cases": totals[cohort], "confirmed_leaks": leaks[cohort]}
            for cohort in sorted(totals)
        }

    def public_state(self) -> dict:
        """Return everything a human controller is allowed to observe."""
        incidents = []
        for item in self._incidents if self.status == "active" else ():
            upstream = self._upstream_by_module[item.module.id]
            downstream = self._downstream_by_module[item.module.id]
            row = {
                "id": item.id,
                "module_id": item.module.id,
                "description": item.module.description,
                "equipment_type": item.module.equipment_type,
                "cohort": "old / Orion" if item.module.cohort == "old" else "new / Vesta",
                "alarm": item.alarm,
                "sensor": "PRESSURE ALARM" if item.alarm else "pressure normal",
                "sensor_knowledge": public_sensor_description(
                    self.config, item.module.equipment_type
                ),
                "criticality": item.module.criticality,
                "value_at_risk": item.module.value_at_risk,
                "production_value": item.module.value_at_risk,
                "production_at_risk": item.production_at_risk,
                "upstream_modules": list(upstream),
                "downstream_modules": list(downstream),
            }
            if item.id in self._inspected:
                row["inspection"] = "seal leak confirmed" if self._inspected[item.id] else "seal intact"
            elif item.module.id in self._known_fault_modules:
                row["known_condition"] = "seal leak previously confirmed; still unrepaired"
            if item.module.id in self._module_records:
                row["last_service"] = dict(self._module_records[item.module.id])
            incidents.append(row)
        return {
            "game": "StationOps-v2",
            "status": self.status,
            "shift": min(self.shift_index + 1, self.config.shifts),
            "total_shifts": self.config.shifts,
            "credits": self.credits,
            "seal_kits": self.seal_kits,
            "diagnostics_remaining": self.config.diagnostic_slots - len(self._inspected),
            "inspection_cost": self.config.inspection_cost,
            "repair_slots": self.config.repair_slots,
            "repair_capacity": self.repair_capacity() if self.status == "active" else 0,
            "repair_cost": self.config.repair_cost,
            "maximum_production": self.maximum_production,
            "station_score": self.station_score,
            "total_production": self.total_production,
            "maintenance_log": self.maintenance_summary(),
            "production_shortfalls": list(self._shortfall_log),
            "dependency_graph": [
                {"upstream": upstream, "downstream": downstream}
                for downstream, upstream in self.dependencies
            ],
            "incidents": incidents,
            "last_report": self._last_report,
        }

    def inspect(self, incident_id: str) -> dict:
        if self.status != "active":
            raise ValueError("the game is complete")
        incident = self._incident(incident_id)
        if incident.module.id in self._known_fault_modules:
            raise ValueError("that module already has a confirmed unrepaired leak")
        if incident_id in self._inspected:
            raise ValueError("that module has already been inspected this shift")
        if len(self._inspected) >= self.config.diagnostic_slots:
            raise ValueError("no diagnostic slots remain this shift")
        if self.credits < self.config.inspection_cost:
            raise ValueError("not enough credits for that inspection")
        self.credits -= self.config.inspection_cost
        self._inspection_spend += self.config.inspection_cost
        self._inspected[incident_id] = self._faults[incident_id]
        if self._faults[incident_id]:
            self._known_fault_modules.add(incident.module.id)
            status = "seal leak confirmed"
        else:
            status = "seal inspected intact"
        self._module_records[incident.module.id] = {
            "shift": self.shift_index + 1,
            "status": status,
        }
        self._resolve_shortfall_candidate(incident.module, self._faults[incident_id])
        return {
            "incident_id": incident_id,
            "result": "seal leak confirmed" if self._faults[incident_id] else "seal intact",
            "state": self.public_state(),
        }

    def repair_capacity(self) -> int:
        affordable = self.credits // self.config.repair_cost if self.config.repair_cost else self.config.repair_slots
        return min(self.config.repair_slots, self.seal_kits, affordable)

    def _resolve_shortfall_candidate(self, module: StationModule, fault: bool):
        """Apply newly public diagnostic evidence to older ambiguous shortfalls."""
        remaining = []
        for event in self._shortfall_events:
            candidate = event["candidates"].pop(module.id, None)
            if candidate is not None and fault:
                event["remaining_loss"] = max(
                    0, event["remaining_loss"] - module.value_at_risk
                )
            if event["remaining_loss"] and event["candidates"]:
                remaining.append(event)
        self._shortfall_events = remaining

    def _record_shortfall(
        self,
        production_loss: int,
        repairs: set[str],
        known_before: set[str],
        belief_snapshot: dict[str, float] | None,
    ):
        if not production_loss:
            return
        self._shortfall_log.append({
            "shift": self.shift_index + 1,
            "production_loss": production_loss,
        })
        # WeightedSubsetPosteriorDP models additive independent losses.  In a
        # dependency graph, two faults can have overlapping downstream outage
        # closures, so conditioning this report with that fold would be wrong.
        # Keep the anonymous public report, but do not create a legacy event.
        if self.dependencies:
            return
        known_loss = sum(
            item.module.value_at_risk
            for item in self._incidents
            if item.id not in repairs
            and self._faults[item.id]
            and (item.id in self._inspected or item.module.id in known_before)
        )
        remaining_loss = production_loss - known_loss
        if remaining_loss <= 0:
            return

        priors = empirical_feature_priors(self.history)
        candidates = {}
        for item in self._incidents:
            if (
                item.id in repairs
                or item.id in self._inspected
                or item.module.id in known_before
            ):
                continue
            probability = (belief_snapshot or {}).get(item.id)
            if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                probability = posterior(
                    priors[(item.module.cohort, item.module.equipment_type)],
                    item.alarm,
                    *sensor_rates(self.config, item.module.equipment_type),
                )
            candidates[item.module.id] = {
                "impact": item.module.value_at_risk,
                "prior": min(max(float(probability), 1e-6), 1.0 - 1e-6),
            }
        if candidates:
            self._shortfall_events.append({
                "shift": self.shift_index + 1,
                "remaining_loss": remaining_loss,
                "candidates": candidates,
            })

    def commit(
        self,
        repair_ids: list[str],
        belief_snapshot: dict[str, float] | None = None,
    ) -> dict:
        if self.status != "active":
            raise ValueError("the game is complete")
        if len(repair_ids) != len(set(repair_ids)):
            raise ValueError("a module cannot be repaired twice")
        known = {item.id for item in self._incidents}
        unknown = sorted(set(repair_ids) - known)
        if unknown:
            raise ValueError("unknown incidents: " + ", ".join(unknown))
        if len(repair_ids) > self.repair_capacity():
            raise ValueError("repair selection exceeds available slots, kits, or credits")

        repairs = set(repair_ids)
        known_before = set(self._known_fault_modules)
        confirmed: dict[str, HistoryCase] = {}
        outcomes = []
        unavailable_before = self._availability_closure(self._module_faults)
        unnecessary_repairs = 0
        for item in self._incidents:
            fault = self._faults[item.id]
            repaired = item.id in repairs
            if repaired:
                self.credits -= self.config.repair_cost
                self.seal_kits -= 1
                if fault:
                    result = "leak patched; production protected"
                else:
                    result = "seal was intact; repair was unnecessary"
                    unnecessary_repairs += 1
                outcomes.append({"module_id": item.module.id, "incident_id": item.id, "result": result})
                confirmed[item.id] = HistoryCase(
                    item.id,
                    item.module.cohort,
                    fault,
                    item.alarm,
                    item.module.equipment_type,
                )
                self._resolve_shortfall_candidate(item.module, fault)
                self._module_faults[item.module.id] = False
                self._known_fault_modules.discard(item.module.id)
                self._module_records[item.module.id] = {
                    "shift": self.shift_index + 1,
                    "status": "leak patched" if fault else "seal serviced intact",
                }
            elif fault:
                if item.id in self._inspected:
                    confirmed[item.id] = HistoryCase(
                        item.id,
                        item.module.cohort,
                        True,
                        item.alarm,
                        item.module.equipment_type,
                    )
            elif item.id in self._inspected:
                confirmed[item.id] = HistoryCase(
                    item.id,
                    item.module.cohort,
                    False,
                    item.alarm,
                    item.module.equipment_type,
                )

        unavailable_after = self._availability_closure(self._module_faults)
        self._unavailable_modules = unavailable_after
        production_loss = sum(
            module.value_at_risk
            for module in self.modules
            if module.id in unavailable_after
        )
        loss_before_repairs = sum(
            module.value_at_risk
            for module in self.modules
            if module.id in unavailable_before
        )
        production_recovered = loss_before_repairs - production_loss
        production = self.maximum_production - production_loss
        maintenance_cost = len(repairs) * self.config.repair_cost + self._inspection_spend
        false_repair_cost = unnecessary_repairs * self.config.unnecessary_repair_penalty
        actual_incremental_utility = (
            production_recovered
            - len(repairs) * self.config.repair_cost
            - false_repair_cost
        )
        shift_score = production - maintenance_cost - false_repair_cost
        self.total_production += production
        self.station_score += shift_score
        self.history.extend(confirmed[key] for key in sorted(confirmed))
        self._record_shortfall(production_loss, repairs, known_before, belief_snapshot)

        report = {
            "shift": self.shift_index + 1,
            "production": production,
            "maximum_production": self.maximum_production,
            "production_loss": production_loss,
            "maintenance_cost": maintenance_cost,
            "unnecessary_repair_penalty": false_repair_cost,
            "shift_score": shift_score,
            "actual_incremental_utility": actual_incremental_utility,
            "production_recovered": production_recovered,
            "root_cause_repairs": sum(
                self._faults[item.id] and item.id in repairs
                for item in self._incidents
            ),
            "symptom_repairs": sum(
                not self._faults[item.id]
                and item.module.id in unavailable_before
                and item.id in repairs
                for item in self._incidents
            ),
            "repairs": sorted(repairs),
            "production_event": (
                "one or more unresolved equipment faults reduced station production"
                if production_loss else None
            ),
            "new_confirmed_cases": len(confirmed),
            "outcomes": outcomes,
        }
        self._last_report = report
        self.shift_index += 1
        if self.shift_index >= self.config.shifts:
            self.status = "complete"
            self._incidents = ()
            self._faults = {}
            self._inspected = {}
        else:
            self._begin_shift()
        return {"report": report, "state": self.public_state()}

    @property
    def known_fault_incident_ids(self) -> set[str]:
        return {
            item.id
            for item in self._incidents
            if item.module.id in self._known_fault_modules
        }

    def logic_statements(self, history: list[HistoryCase] | None = None) -> str:
        """Return exactly the current public knowledge made available to a backend."""
        history = self.history if history is None else history
        lines = [
            generate_statements(
                history,
                self.incidents,
                self.logic_config,
                sensor_knowledge=sensor_knowledge(self.config),
                sensor_models={
                    equipment_type: sensor_rates(self.config, equipment_type)
                    for equipment_type, _ in EQUIPMENT_TYPES
                },
            )
        ]
        if self.dependencies:
            lines.append(generate_dependency_statements(
                self.incidents,
                shift=self.shift_index + 1,
                sensor_knowledge=sensor_knowledge(self.config),
                sensor_models={
                    equipment_type: sensor_rates(self.config, equipment_type)
                    for equipment_type, _ in EQUIPMENT_TYPES
                },
            ))
        lines.extend(
            f"(: known-leak-{item.id} "
            f"(SealLeak {item.module.cohort} {item.module.equipment_type} {item.id}) "
            f"(STV 1 1))"
            for item in self._incidents
            if item.module.id in self._known_fault_modules
        )
        return "\n".join(lines)


def repair_increment(probability: float, incident: StationIncident, config: GameConfig) -> float:
    return (
        probability * incident.production_at_risk
        - (1.0 - probability) * config.unnecessary_repair_penalty
        - config.repair_cost
    )


def allocate_repairs(
    incidents: tuple[StationIncident, ...],
    beliefs: dict[str, float],
    config: GameConfig,
    capacity: int,
) -> list[str]:
    candidates = sorted(
        (
            (repair_increment(beliefs[item.id], item, config), item.id)
            for item in incidents
            if item.id in beliefs and repair_increment(beliefs[item.id], item, config) > 0
        ),
        key=lambda row: (-row[0], row[1]),
    )
    return [incident_id for _, incident_id in candidates[:capacity]]


def _odds(probability: float) -> float:
    probability = min(max(probability, 1e-9), 1.0 - 1e-9)
    return probability / (1.0 - probability)


def _from_odds(value: float) -> float:
    return value / (1.0 + value)


def decision_beliefs(
    session: GameSession,
    beliefs: dict[str, float],
    shortfall_marginals: dict[int, dict[str, float]] | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Combine chainer output with public recency and aggregate-loss evidence."""
    if shortfall_marginals is None:
        shortfall_marginals = oracle_event_marginals(session._shortfall_events)
    adjusted = dict(beliefs)
    evidence = {}
    by_module = {item.module.id: item for item in session._incidents}

    for item in session._incidents:
        if item.id not in adjusted:
            continue
        raw = adjusted[item.id]
        current_odds = _odds(raw)
        details = {"raw_belief": raw}
        record = session._module_records.get(item.module.id)
        if record and record["status"] in {
            "seal inspected intact",
            "leak patched",
            "seal serviced intact",
        }:
            age = session.shift_index + 1 - int(record["shift"])
            cohort_prior = new_fault_probability(session.config, item.module)
            age_prior = 1.0 - (1.0 - cohort_prior) ** max(age, 0)
            if age_prior <= 0:
                current_odds = 0.0
            elif cohort_prior > 0:
                current_odds *= _odds(age_prior) / _odds(cohort_prior)
            details.update({
                "last_service_shift": record["shift"],
                "shifts_since_clear": age,
            })
        adjusted[item.id] = _from_odds(current_odds) if current_odds else 0.0
        evidence[item.id] = details

    for event in session._shortfall_events:
        marginals = shortfall_marginals.get(int(event["shift"]), {})
        for module_id, shortfall_probability in marginals.items():
            item = by_module.get(module_id)
            if item is None or item.id not in adjusted:
                continue
            stored_prior = event["candidates"][module_id]["prior"]
            likelihood_ratio = _odds(shortfall_probability) / _odds(stored_prior)
            adjusted[item.id] = _from_odds(_odds(adjusted[item.id]) * likelihood_ratio)
            evidence[item.id].setdefault("shortfall_shifts", []).append(event["shift"])

    for item in session._incidents:
        if item.module.id in session._known_fault_modules:
            adjusted[item.id] = 1.0
            evidence[item.id] = {
                "raw_belief": beliefs.get(item.id),
                "known_unrepaired_fault": True,
            }
        if item.id in evidence:
            evidence[item.id]["decision_belief"] = adjusted[item.id]
    return adjusted, evidence


def diagnostic_value(
    probability: float, incident: StationIncident, config: GameConfig
) -> float:
    """Expected gain from a perfect seal inspection before the repair decision."""
    act_now = max(0.0, repair_increment(probability, incident, config))
    act_after_inspection = probability * max(
        0.0, incident.production_at_risk - config.repair_cost
    )
    return act_after_inspection - act_now - config.inspection_cost


def diagnostic_plan(
    session: GameSession, beliefs: dict[str, float]
) -> tuple[list[str], dict[str, float]]:
    """Allocate diagnostics by decision value, then explore unknown sensor types."""
    priorities = {
        item.id: diagnostic_value(beliefs[item.id], item, session.config)
        for item in session._incidents
        if item.id in beliefs and item.module.id not in session._known_fault_modules
    }
    candidates = sorted(
        (
            (value, item.production_at_risk, item.id)
            for item in session._incidents
            if (value := priorities.get(item.id, float("-inf"))) > 0
        ),
        key=lambda row: (-row[0], -row[1], row[2]),
    )
    affordable = (
        session.credits // session.config.inspection_cost
        if session.config.inspection_cost else len(candidates)
    )
    count = min(session.config.diagnostic_slots, affordable)
    selected = [incident_id for _, _, incident_id in candidates[:count]]

    # Pure exploitation can permanently starve induction: a type with no proof
    # would never be inspected and could therefore never acquire one.  Fill any
    # remaining diagnostic capacity with the least-sampled feature groups that
    # need a learned inverse, preferring higher production impact on ties.
    remaining = count - len(selected)
    if remaining:
        sample_counts = Counter(
            (case.cohort, case.equipment_type) for case in session.history
        )
        modes = sensor_knowledge(session.config)
        exploration = sorted(
            (
                sample_counts[(item.module.cohort, item.module.equipment_type)],
                -item.production_at_risk,
                item.id,
            )
            for item in session._incidents
            if item.id not in selected
            and item.module.id not in session._known_fault_modules
            and modes[item.module.equipment_type] in {"positive", "induced"}
        )
        selected.extend(incident_id for _, _, incident_id in exploration[:remaining])
    return selected, priorities


def _action_candidates(
    session: GameSession, beliefs: dict[str, float]
) -> list[ActionCandidate]:
    """Build the explicit numeric bridge from diagnosis TVs to action rules."""
    sample_counts = Counter(
        (case.cohort, case.equipment_type) for case in session.history
    )
    modes = sensor_knowledge(session.config)
    return [
        ActionCandidate(
            incident_id=item.id,
            probability=(
                beliefs.get(item.id)
                if isinstance(beliefs.get(item.id), (int, float))
                and math.isfinite(beliefs[item.id])
                else None
            ),
            production_at_risk=item.production_at_risk,
            inspection_eligible=(
                item.id not in session._inspected
                and item.module.id not in session._known_fault_modules
            ),
            learning_samples=(
                sample_counts[(item.module.cohort, item.module.equipment_type)]
                if item.id not in session._inspected
                and item.module.id not in session._known_fault_modules
                and modes[item.module.equipment_type] in {"positive", "induced"}
                else None
            ),
        )
        for item in session._incidents
    ]


def _query_action_proposals(
    backend,
    session: GameSession,
    context: str,
    beliefs: dict[str, float],
    budget: int,
) -> tuple[list[ActionProposal], dict[str, object]]:
    candidates = _action_candidates(session, beliefs)
    proposer = getattr(backend, "propose_actions", None)
    arguments = {
        "inspection_cost": session.config.inspection_cost,
        "repair_cost": session.config.repair_cost,
        "unnecessary_repair_penalty": session.config.unnecessary_repair_penalty,
    }
    if callable(proposer):
        return proposer(context, candidates, budget, **arguments)
    # Small test/demonstration backends written against the older diagnosis-only
    # protocol retain the exact reference action semantics.
    proposals = (
        reference_action_proposals(context, candidates, **arguments)
        if budget > 0
        else []
    )
    return proposals, {
        "action_queries": 1 if candidates else 0,
        "action_proposals": len(proposals),
        "action_statements_added": 0,
        "action_engine_steps": None,
        "action_fallback": True,
    }


def _merge_action_counters(
    total: dict[str, object], update: dict[str, object]
) -> None:
    for key, value in update.items():
        if key in {
            "action_queries",
            "action_proposals",
            "action_statements_added",
            "action_forward_seed_facts",
            "action_forward_steps",
            "action_engine_steps",
        } and isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def run_action_loop(
    backend,
    session: GameSession,
    beliefs: dict[str, float],
    budget: int,
) -> tuple[
    dict[str, str],
    dict[str, float],
    list[str],
    dict,
    list[dict],
    dict[str, float],
]:
    """Observe, re-query, then select interventions from generic proposals."""
    effective_beliefs = dict(beliefs)
    inspection_results: dict[str, str] = {}
    diagnostic_priorities: dict[str, float] = {}
    counters: dict[str, object] = {}
    trace = []
    step = 0
    final_proposals: list[ActionProposal] | None = None

    while (
        len(session._inspected) < session.config.diagnostic_slots
        and (
            session.config.inspection_cost == 0
            or session.credits >= session.config.inspection_cost
        )
    ):
        context = f"decision-s{session.shift_index + 1:02d}-step{step:02d}"
        proposals, action_counters = _query_action_proposals(
            backend, session, context, effective_beliefs, budget
        )
        _merge_action_counters(counters, action_counters)
        for proposal in proposals:
            if proposal.action == "Inspect" and proposal.rationale == "Diagnostic":
                diagnostic_priorities.setdefault(
                    proposal.incident_id, proposal.utility
                )
        valued = sorted(
            (
                proposal.utility,
                session._incident(proposal.incident_id).production_at_risk,
                proposal.incident_id,
            )
            for proposal in proposals
            if proposal.action == "Inspect"
            and proposal.rationale == "Diagnostic"
            and proposal.utility > 0
            and proposal.incident_id not in session._inspected
            and session._incident(proposal.incident_id).module.id
            not in session._known_fault_modules
        )
        if valued:
            _, _, incident_id = min(
                valued, key=lambda row: (-row[0], -row[1], row[2])
            )
            selection_kind = "value-of-information"
        else:
            learning_probes = [
                proposal
                for proposal in proposals
                if proposal.action == "Inspect"
                and proposal.rationale == "Learning"
                and proposal.incident_id not in session._inspected
                and session._incident(proposal.incident_id).module.id
                not in session._known_fault_modules
            ]
            selected_probe = min(
                learning_probes,
                key=lambda proposal: (-proposal.utility, proposal.incident_id),
                default=None,
            )
            incident_id = (
                selected_probe.incident_id if selected_probe is not None else None
            )
            selection_kind = "learning-probe"
        trace.append({
            "context": context,
            "proposals": [
                {
                    "action": proposal.action,
                    "incident_id": proposal.incident_id,
                    "utility": proposal.utility,
                    "confidence": proposal.confidence,
                    "rationale": proposal.rationale,
                }
                for proposal in proposals
            ],
            "selected": (
                {
                    "action": "Inspect",
                    "incident_id": incident_id,
                    "reason": selection_kind,
                }
                if incident_id is not None
                else None
            ),
        })
        if incident_id is None:
            final_proposals = proposals
            break
        result = session.inspect(incident_id)
        inspection_results[incident_id] = result["result"]
        effective_beliefs[incident_id] = (
            1.0 if session._faults[incident_id] else 0.0
        )
        step += 1

    if final_proposals is None:
        context = f"decision-s{session.shift_index + 1:02d}-step{step:02d}"
        final_proposals, action_counters = _query_action_proposals(
            backend, session, context, effective_beliefs, budget
        )
        _merge_action_counters(counters, action_counters)
        trace.append({
            "context": context,
            "proposals": [
                {
                    "action": proposal.action,
                    "incident_id": proposal.incident_id,
                    "utility": proposal.utility,
                    "confidence": proposal.confidence,
                    "rationale": proposal.rationale,
                }
                for proposal in final_proposals
            ],
            "selected": None,
        })

    repair_candidates = sorted(
        (
            proposal.utility,
            proposal.incident_id,
        )
        for proposal in final_proposals
        if proposal.action == "Repair" and proposal.utility > 0
    )
    repairs = [
        incident_id
        for _, incident_id in sorted(
            repair_candidates, key=lambda row: (-row[0], row[1])
        )[:session.repair_capacity()]
    ]
    if trace:
        trace[-1]["selected_repairs"] = repairs
    return (
        inspection_results,
        diagnostic_priorities,
        repairs,
        counters,
        trace,
        effective_beliefs,
    )


def _decision_utility(
    repairs: list[str],
    beliefs: dict[str, float],
    incidents: tuple[StationIncident, ...],
    config: GameConfig,
) -> float:
    by_id = {item.id: item for item in incidents}
    return sum(repair_increment(beliefs[item_id], by_id[item_id], config) for item_id in repairs)


def _probability_metrics(beliefs: dict[str, float], truth: dict[str, bool]) -> dict:
    valid = [
        (beliefs[key], float(value))
        for key, value in truth.items()
        if isinstance(beliefs.get(key), (int, float)) and math.isfinite(beliefs[key])
    ]
    clipped = [(min(max(probability, 1e-12), 1.0 - 1e-12), outcome) for probability, outcome in valid]
    return {
        "brier_score": (
            sum((probability - outcome) ** 2 for probability, outcome in valid) / len(valid)
            if valid else None
        ),
        "log_loss": (
            -sum(
                outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability)
                for probability, outcome in clipped
            ) / len(clipped)
            if clipped else None
        ),
    }


def _causal_diagnosis_metrics(
    session: GameSession, beliefs: dict[str, float]
) -> dict[str, dict]:
    """Break diagnosis quality out by root, propagated symptom, and healthy state."""
    groups: dict[str, dict[str, bool]] = {}
    for item in session._incidents:
        distance = session.causal_distance(item.module.id)
        label = (
            "healthy" if distance is None else "root" if distance == 0
            else f"downstream-{distance}"
        )
        groups.setdefault(label, {})[item.id] = session._faults[item.id]
    result = {}
    for label, truth in sorted(groups.items()):
        valid = {
            incident_id: beliefs[incident_id]
            for incident_id in truth
            if isinstance(beliefs.get(incident_id), (int, float))
            and math.isfinite(beliefs[incident_id])
        }
        result[label] = {
            "modules": len(truth),
            "diagnoses": len(valid),
            "coverage": len(valid) / len(truth) if truth else 1.0,
            **_probability_metrics(valid, truth),
        }
    return result


def _backend(
    config: Config,
    name: str,
    mm2_path=None,
    pettachainer_path=None,
    *,
    sensor_models=None,
    sensor_knowledge_map=None,
):
    if name == "reference":
        return ReferenceBackend(config, sensor_models)
    if name == "mm2":
        return MM2Backend(
            config, mm2_path, sensor_knowledge=sensor_knowledge_map
        )
    if name == "pettachainer":
        return PeTTaChainerBackend(
            config,
            pettachainer_path,
            sensor_knowledge=sensor_knowledge_map,
        )
    raise ValueError(f"unknown backend: {name}")


def _mean(values: list[float | None]) -> float | None:
    finite = [
        value
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    return sum(finite) / len(finite) if finite else None


def _learning_curve(rounds: list[dict], window_size: int) -> list[dict]:
    curve = []
    for offset in range(0, len(rounds), window_size):
        window = rounds[offset:offset + window_size]
        curve.append({
            "shift_start": window[0]["shift"],
            "shift_end": window[-1]["shift"],
            "mean_brier_score": _mean([
                round_["belief_metrics"]["brier_score"] for round_ in window
            ]),
            "mean_log_loss": _mean([
                round_["belief_metrics"]["log_loss"] for round_ in window
            ]),
            "mean_coverage": _mean([
                round_["belief_metrics"]["coverage"] for round_ in window
            ]),
            "mean_regret": _mean([round_["regret"] for round_ in window]),
            "total_regret": sum(round_["regret"] for round_ in window),
            "confirmed_cases_added": sum(
                round_["history_size_after"] - round_["history_size_before"]
                for round_ in window
            ),
            "station_score": sum(
                round_["resolution"]["shift_score"] for round_ in window
            ),
        })
    return curve


def _resolved_model(history: list[HistoryCase]) -> dict[str, dict]:
    """Summarize the public labels from which induction is allowed to learn."""
    grouped: dict[tuple[str, str | None], list[HistoryCase]] = {}
    for case in history:
        grouped.setdefault((case.cohort, case.equipment_type), []).append(case)
    result = {}
    for (cohort, equipment_type), cases in sorted(
        grouped.items(), key=lambda row: (row[0][0], row[0][1] or "")
    ):
        leaks = [case for case in cases if case.leak]
        intact = [case for case in cases if not case.leak]
        alarms = [case for case in cases if case.alarm]
        normals = [case for case in cases if not case.alarm]
        key = f"{cohort}|{equipment_type or 'untyped'}"
        result[key] = {
            "cases": len(cases),
            "leak_rate": len(leaks) / len(cases),
            "alarm_given_leak": (
                sum(case.alarm for case in leaks) / len(leaks) if leaks else None
            ),
            "alarm_given_intact": (
                sum(case.alarm for case in intact) / len(intact) if intact else None
            ),
            "leak_given_alarm": (
                sum(case.leak for case in alarms) / len(alarms) if alarms else None
            ),
            "leak_given_normal": (
                sum(case.leak for case in normals) / len(normals) if normals else None
            ),
        }
    return result


def run_game_episode(
    config: GameConfig | None = None,
    backend_name: str = "reference",
    budget: int = 100,
    mm2_path: str | None = None,
    pettachainer_path: str | None = None,
    shortfall_budget: int | None = None,
    action_budget: int | None = None,
) -> dict:
    """Run the standard controller through the same simulation used by humans."""
    config = config or GameConfig()
    shortfall_budget = budget if shortfall_budget is None else shortfall_budget
    action_budget = budget if action_budget is None else action_budget
    session = GameSession(config)
    models = {
        equipment_type: sensor_rates(config, equipment_type)
        for equipment_type, _ in EQUIPMENT_TYPES
    }
    knowledge = sensor_knowledge(config)
    backend = _backend(
        session.logic_config,
        backend_name,
        mm2_path,
        pettachainer_path,
        sensor_models=models,
        sensor_knowledge_map=knowledge,
    )
    reference = ReferenceBackend(session.logic_config, models)
    initial_resolved_model = _resolved_model(session.history)
    rounds = []
    started = time.perf_counter()

    while session.status == "active":
        incidents = session.incidents
        station_incidents = session._incidents
        truth = dict(session._faults)
        history_before = list(session.history)
        known_faults = session.known_fault_incident_ids
        statements = session.logic_statements(history_before)
        round_started = time.perf_counter()
        beliefs, counters = backend.infer(history_before, incidents, budget, statements)
        oracle_beliefs, _ = reference.infer(history_before, incidents, 1, statements)
        oracle_beliefs.update((incident_id, 1.0) for incident_id in known_faults)
        beliefs.update((incident_id, 1.0) for incident_id in known_faults)

        conditioner = getattr(backend, "condition_shortfalls", None)
        if conditioner is None:
            shortfall_marginals = oracle_event_marginals(session._shortfall_events)
            shortfall_counters = {
                "shortfall_queries": 0,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": None,
            }
        else:
            shortfall_marginals, shortfall_counters = conditioner(
                session._shortfall_events, shortfall_budget
            )
        oracle_shortfalls, _ = reference.condition_shortfalls(
            session._shortfall_events, 1
        )
        counters.update(shortfall_counters)

        controller_beliefs, belief_evidence = decision_beliefs(
            session, beliefs, shortfall_marginals
        )
        oracle_controller_beliefs, _ = decision_beliefs(
            session, oracle_beliefs, oracle_shortfalls
        )
        oracle_inspection_ids, _ = diagnostic_plan(
            session, oracle_controller_beliefs
        )
        (
            inspection_results,
            diagnostic_priorities,
            repairs,
            action_counters,
            action_trace,
            effective_beliefs,
        ) = run_action_loop(backend, session, controller_beliefs, action_budget)
        counters.update(action_counters)

        effective_oracle = dict(oracle_controller_beliefs)
        for incident_id in inspection_results:
            observed = 1.0 if truth[incident_id] else 0.0
            effective_oracle[incident_id] = observed

        oracle_repairs = allocate_repairs(
            station_incidents, effective_oracle, config, session.repair_capacity()
        )
        expected_utility = _decision_utility(repairs, effective_oracle, station_incidents, config)
        oracle_utility = _decision_utility(
            oracle_repairs, effective_oracle, station_incidents, config
        )
        causal_metrics = _causal_diagnosis_metrics(session, beliefs)
        resolution = session.commit(repairs, effective_beliefs)
        metrics = belief_error_metrics(beliefs, oracle_beliefs)
        metrics.update(_probability_metrics(beliefs, truth))
        rounds.append({
            "shift": resolution["report"]["shift"],
            "visible_incidents": [
                {
                    "id": item.id,
                    "module_id": item.module.id,
                    "cohort": item.module.cohort,
                    "equipment_type": item.module.equipment_type,
                    "alarm": item.alarm,
                    "criticality": item.module.criticality,
                    "value_at_risk": item.module.value_at_risk,
                    "production_at_risk": item.production_at_risk,
                    "upstream_modules": list(
                        session._upstream_by_module[item.module.id]
                    ),
                }
                for item in station_incidents
            ],
            "history_size_before": len(history_before),
            "history_size_after": len(session.history),
            "beliefs": beliefs,
            "decision_beliefs": controller_beliefs,
            "belief_evidence": belief_evidence,
            "shortfall_marginals": shortfall_marginals,
            "belief_metrics": metrics,
            "causal_diagnosis_metrics": causal_metrics,
            "inspections": inspection_results,
            "diagnostic_priorities": diagnostic_priorities,
            "action_trace": action_trace,
            "oracle_inspection_plan": oracle_inspection_ids,
            "chosen_repairs": repairs,
            "oracle_repairs": oracle_repairs,
            "expected_utility": expected_utility,
            "oracle_utility": oracle_utility,
            "regret": oracle_utility - expected_utility,
            "backend_counters": counters,
            "resolution": resolution["report"],
            "wall_time_seconds": time.perf_counter() - round_started,
        })

    learning_curve = _learning_curve(rounds, config.learning_window)
    expected = sum(round_["expected_utility"] for round_ in rounds)
    oracle = sum(round_["oracle_utility"] for round_ in rounds)
    return {
        "benchmark": "StationOps-v2",
        "schema_version": 2,
        "config": config.to_dict(),
        "backend": backend.name,
        "budget_per_shift": budget,
        "diagnosis_budget_per_shift": budget,
        "action_budget_per_query": action_budget,
        "shortfall_budget_per_shift": shortfall_budget,
        "rounds": rounds,
        "aggregate": {
            "expected_utility": expected,
            "oracle_utility": oracle,
            "regret": oracle - expected,
            "normalized_score": 1.0 if oracle == 0 else max(0.0, expected / oracle),
            "station_score": session.station_score,
            "total_production": session.total_production,
            "production_recovered": sum(
                round_["resolution"]["production_recovered"] for round_ in rounds
            ),
            "root_cause_repairs": sum(
                round_["resolution"]["root_cause_repairs"] for round_ in rounds
            ),
            "symptom_repairs": sum(
                round_["resolution"]["symptom_repairs"] for round_ in rounds
            ),
            "confirmed_history_size": len(session.history),
            "learning_curve": learning_curve,
            "early_to_late_brier_improvement": (
                learning_curve[0]["mean_brier_score"]
                - learning_curve[-1]["mean_brier_score"]
                if len(learning_curve) > 1
                and learning_curve[0]["mean_brier_score"] is not None
                and learning_curve[-1]["mean_brier_score"] is not None
                else None
            ),
            "early_to_late_log_loss_improvement": (
                learning_curve[0]["mean_log_loss"]
                - learning_curve[-1]["mean_log_loss"]
                if len(learning_curve) > 1
                and learning_curve[0]["mean_log_loss"] is not None
                and learning_curve[-1]["mean_log_loss"] is not None
                else None
            ),
            "early_to_late_regret_improvement": (
                learning_curve[0]["mean_regret"]
                - learning_curve[-1]["mean_regret"]
                if len(learning_curve) > 1
                else None
            ),
            "early_to_late_coverage_change": (
                learning_curve[-1]["mean_coverage"]
                - learning_curve[0]["mean_coverage"]
                if len(learning_curve) > 1
                else None
            ),
            "resolved_model_before": initial_resolved_model,
            "resolved_model_after": _resolved_model(session.history),
        },
        "sensor_knowledge_by_type": knowledge,
        "scoring_only_hidden_model": {
            "new_fault_probability": {
                f"{module.cohort}|{module.equipment_type}": new_fault_probability(
                    config, module
                )
                for module in session.modules
            },
            "sensor_rates": {
                equipment_type: {
                    "sensitivity": rates[0],
                    "false_positive_rate": rates[1],
                }
                for equipment_type, rates in models.items()
            },
        },
        "status": session.status,
        "wall_time_seconds": time.perf_counter() - started,
    }
