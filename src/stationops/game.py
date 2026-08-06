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

from .backends import MM2Backend, PeTTaChainerBackend, ReferenceBackend
from .config import Config
from .metta import generate_statements
from .models import HistoryCase, Incident
from .oracle import belief_error_metrics, empirical_priors, posterior
from .shortfall import oracle_event_marginals
from .simulator import generate_history


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
    description: str
    criticality: str
    value_at_risk: int


@dataclass(frozen=True)
class StationIncident:
    id: str
    module: StationModule
    alarm: bool

    def logic_incident(self) -> Incident:
        return Incident(self.id, self.module.cohort, self.alarm)


def _modules(count: int) -> tuple[StationModule, ...]:
    descriptions = (
        "coolant pump",
        "oxygen scrubber",
        "power converter",
        "thermal-loop pump",
        "ore-feed pump",
    )
    values = (120, 45, 90, 70, 55, 110, 40, 80, 65, 100)
    result = []
    for index in range(count):
        value = values[index % len(values)]
        criticality = "HIGH" if value >= 90 else "MEDIUM" if value >= 60 else "LOW"
        cohort = "old" if index % 2 == 0 else "new"
        result.append(
            StationModule(
                f"M{index + 1:02d}",
                cohort,
                descriptions[index % len(descriptions)],
                criticality,
                value,
            )
        )
    return tuple(result)


class GameSession:
    """Mutable game state with an explicit public/private information boundary."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.logic_config = self.config.logic_config()
        self.modules = _modules(self.config.modules)
        self.history = generate_history(self.logic_config)
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
            prior = self.config.old_prior if module.cohort == "old" else self.config.new_prior
            fault = self._module_faults[module.id] or self._rng.random() < prior
            self._module_faults[module.id] = fault
            alarm_rate = self.config.sensitivity if fault else self.config.false_positive_rate
            alarm = self._rng.random() < alarm_rate
            incident_id = f"shift-{self.shift_index + 1:02d}-{module.id}"
            incidents.append(StationIncident(incident_id, module, alarm))
            faults[incident_id] = fault
        self._incidents = tuple(incidents)
        self._faults = faults
        self._inspected = {}
        self._inspection_spend = 0

    @property
    def incidents(self) -> list[Incident]:
        return [item.logic_incident() for item in self._incidents]

    @property
    def maximum_production(self) -> int:
        return sum(module.value_at_risk for module in self.modules)

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
            row = {
                "id": item.id,
                "module_id": item.module.id,
                "description": item.module.description,
                "cohort": "old / Orion" if item.module.cohort == "old" else "new / Vesta",
                "alarm": item.alarm,
                "sensor": "PRESSURE ALARM" if item.alarm else "pressure normal",
                "criticality": item.module.criticality,
                "value_at_risk": item.module.value_at_risk,
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

        priors = empirical_priors(self.history)
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
                    priors[item.module.cohort],
                    item.alarm,
                    self.config.sensitivity,
                    self.config.false_positive_rate,
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
        production_loss = 0
        unnecessary_repairs = 0
        actual_incremental_utility = 0.0
        for item in self._incidents:
            fault = self._faults[item.id]
            repaired = item.id in repairs
            if repaired:
                self.credits -= self.config.repair_cost
                self.seal_kits -= 1
                if fault:
                    result = "leak patched; production protected"
                    actual_incremental_utility += item.module.value_at_risk - self.config.repair_cost
                else:
                    result = "seal was intact; repair was unnecessary"
                    unnecessary_repairs += 1
                    actual_incremental_utility -= (
                        self.config.repair_cost + self.config.unnecessary_repair_penalty
                    )
                outcomes.append({"module_id": item.module.id, "incident_id": item.id, "result": result})
                confirmed[item.id] = HistoryCase(item.id, item.module.cohort, fault, item.alarm)
                self._resolve_shortfall_candidate(item.module, fault)
                self._module_faults[item.module.id] = False
                self._known_fault_modules.discard(item.module.id)
                self._module_records[item.module.id] = {
                    "shift": self.shift_index + 1,
                    "status": "leak patched" if fault else "seal serviced intact",
                }
            elif fault:
                production_loss += item.module.value_at_risk
                if item.id in self._inspected:
                    confirmed[item.id] = HistoryCase(
                        item.id, item.module.cohort, True, item.alarm
                    )
            elif item.id in self._inspected:
                confirmed[item.id] = HistoryCase(item.id, item.module.cohort, False, item.alarm)

        production = self.maximum_production - production_loss
        maintenance_cost = len(repairs) * self.config.repair_cost + self._inspection_spend
        false_repair_cost = unnecessary_repairs * self.config.unnecessary_repair_penalty
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
        lines = [generate_statements(history, self.incidents, self.logic_config)]
        lines.extend(
            f"(: known-leak-{item.id} (SealLeak {item.module.cohort} {item.id}) (STV 1 1))"
            for item in self._incidents
            if item.module.id in self._known_fault_modules
        )
        return "\n".join(lines)


def repair_increment(probability: float, incident: StationIncident, config: GameConfig) -> float:
    return (
        probability * incident.module.value_at_risk
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
            cohort_prior = (
                session.config.old_prior
                if item.module.cohort == "old"
                else session.config.new_prior
            )
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
        0.0, incident.module.value_at_risk - config.repair_cost
    )
    return act_after_inspection - act_now - config.inspection_cost


def diagnostic_plan(
    session: GameSession, beliefs: dict[str, float]
) -> tuple[list[str], dict[str, float]]:
    """Allocate diagnostics from chainer beliefs by expected value of information."""
    priorities = {
        item.id: diagnostic_value(beliefs[item.id], item, session.config)
        for item in session._incidents
        if item.id in beliefs and item.module.id not in session._known_fault_modules
    }
    candidates = sorted(
        (
            (value, item.module.value_at_risk, item.id)
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
    return [incident_id for _, _, incident_id in candidates[:count]], priorities


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


def _backend(config: Config, name: str, mm2_path=None, pettachainer_path=None):
    if name == "reference":
        return ReferenceBackend(config)
    if name == "mm2":
        return MM2Backend(config, mm2_path)
    if name == "pettachainer":
        return PeTTaChainerBackend(config, pettachainer_path)
    raise ValueError(f"unknown backend: {name}")


def run_game_episode(
    config: GameConfig | None = None,
    backend_name: str = "reference",
    budget: int = 100,
    mm2_path: str | None = None,
    pettachainer_path: str | None = None,
) -> dict:
    """Run the standard controller through the same simulation used by humans."""
    config = config or GameConfig()
    session = GameSession(config)
    backend = _backend(session.logic_config, backend_name, mm2_path, pettachainer_path)
    reference = ReferenceBackend(session.logic_config)
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
        if backend.name == "reference":
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
                session._shortfall_events, budget
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
        inspection_ids, diagnostic_priorities = diagnostic_plan(
            session, controller_beliefs
        )
        oracle_inspection_ids, _ = diagnostic_plan(
            session, oracle_controller_beliefs
        )
        inspection_results = {}
        for incident_id in inspection_ids:
            inspection_results[incident_id] = session.inspect(incident_id)["result"]

        effective_beliefs = dict(controller_beliefs)
        effective_oracle = dict(oracle_controller_beliefs)
        for incident_id in inspection_ids:
            observed = 1.0 if truth[incident_id] else 0.0
            effective_beliefs[incident_id] = observed
            effective_oracle[incident_id] = observed

        repairs = allocate_repairs(
            station_incidents, effective_beliefs, config, session.repair_capacity()
        )
        oracle_repairs = allocate_repairs(
            station_incidents, effective_oracle, config, session.repair_capacity()
        )
        expected_utility = _decision_utility(repairs, effective_oracle, station_incidents, config)
        oracle_utility = _decision_utility(
            oracle_repairs, effective_oracle, station_incidents, config
        )
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
                    "alarm": item.alarm,
                    "criticality": item.module.criticality,
                    "value_at_risk": item.module.value_at_risk,
                }
                for item in station_incidents
            ],
            "history_size_before": len(history_before),
            "beliefs": beliefs,
            "decision_beliefs": controller_beliefs,
            "belief_evidence": belief_evidence,
            "shortfall_marginals": shortfall_marginals,
            "belief_metrics": metrics,
            "inspections": inspection_results,
            "diagnostic_priorities": diagnostic_priorities,
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

    expected = sum(round_["expected_utility"] for round_ in rounds)
    oracle = sum(round_["oracle_utility"] for round_ in rounds)
    return {
        "benchmark": "StationOps-v2",
        "schema_version": 1,
        "config": config.to_dict(),
        "backend": backend.name,
        "budget_per_shift": budget,
        "rounds": rounds,
        "aggregate": {
            "expected_utility": expected,
            "oracle_utility": oracle,
            "regret": oracle - expected,
            "normalized_score": 1.0 if oracle == 0 else max(0.0, expected / oracle),
            "station_score": session.station_score,
            "total_production": session.total_production,
            "confirmed_history_size": len(session.history),
        },
        "status": session.status,
        "wall_time_seconds": time.perf_counter() - started,
    }
