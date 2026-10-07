"""Reference, MM2, and PeTTaChainer reasoner adapters."""
from __future__ import annotations

import importlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .actions import (
    BELIEF_ACTION_RULES,
    ActionCandidate,
    ActionProposal,
    decision_assumptions,
    generate_action_statements,
    leak_belief_assumption,
    reference_action_proposals,
)
from .config import Config
from .models import Belief, HistoryCase, Incident
from .oracle import empirical_feature_priors, posterior
from .shortfall import (
    SHORTFALL_RULES,
    event_atom,
    finite_probability,
    generate_shortfall_statements,
    load_weighted_subset_posterior,
    marginal_queries,
    oracle_event_marginals,
)


class BackendUnavailable(RuntimeError):
    pass


def _engine_execution_stats(engine) -> dict | None:
    snapshot = getattr(engine, "last_execution_stats", None)
    if not callable(snapshot):
        return None
    try:
        result = snapshot()
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def _shortfall_query_plan(events, cache):
    queries = marginal_queries(events)
    shifts = {event_atom(event): int(event["shift"]) for event in events}
    active_tags = {tag for tag, _ in queries}
    for stale_tag in cache.keys() - active_tags:
        del cache[stale_tag]

    marginals: dict[int, dict[str, float]] = {}
    pending = []
    cache_hits = 0
    for tag, query in queries:
        event_name, unit = tag.split("|", 1)
        if tag in cache:
            marginals.setdefault(shifts[event_name], {})[unit] = cache[tag]
            cache_hits += 1
        else:
            pending.append((tag, query))
    return queries, shifts, marginals, pending, cache_hits


def _fields(expression: str) -> list[str]:
    """Split one restricted MeTTa expression into top-level fields."""
    expression = expression.strip()
    if len(expression) < 2 or expression[0] != "(" or expression[-1] != ")":
        raise ValueError(f"expected a parenthesized MeTTa expression: {expression}")
    fields: list[str] = []
    start = None
    depth = 0
    for index, char in enumerate(expression[1:-1], start=1):
        if char == "(":
            if start is None:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced MeTTa expression: {expression}")
        elif char.isspace() and depth == 0:
            if start is not None:
                fields.append(expression[start:index])
                start = None
        elif start is None:
            start = index
    if depth != 0:
        raise ValueError(f"unbalanced MeTTa expression: {expression}")
    if start is not None:
        fields.append(expression[start:-1])
    return fields


def _statement_parts(atom: str) -> tuple[str, str]:
    fields = _fields(atom)
    if len(fields) != 4 or fields[0] != ":":
        raise ValueError(f"expected one named MeTTa statement: {atom}")
    return fields[1], fields[2]


def _is_rule(type_expression: str) -> bool:
    fields = _fields(type_expression)
    return bool(fields) and fields[0] in {"Implication", "BiImplication", "ForAll"}


def _fact_seed(type_expression: str) -> str:
    """Return the positive surface term whose canonical fact seeds forward work."""
    fields = _fields(type_expression)
    if len(fields) == 2 and fields[0] == "Not":
        return fields[1]
    return type_expression


def _forward_fact(entry: tuple[str, str, str]) -> bool:
    """Whether a newly public fact should enter incremental forward work.

    A resolved shift leak is a historical label.  It remains queryable and is
    represented in the shared-state induction table, but replaying it through
    that shift's now-obsolete availability graph creates irrelevant old outage
    proofs.  PeTTaChainer also rejects one such mixed ``no-tv`` proof during a
    later multi-root query, so keep these labels lazy.
    """
    name, _, _ = entry
    return not name.startswith("leak-shift-")


_number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_action_proposal_re = re.compile(
    rf"\(ActionProposal\s+([^()\s]+)\s+"
    rf"\((Inspect|Repair)\s+([^()\s]+)\)\s+"
    rf"({_number_pattern})\s+({_number_pattern})\s+([^()\s]+)\)"
)


def _term_source(term) -> str:
    """Render the small public term shape returned by either Python adapter."""
    if isinstance(term, str):
        return term
    if isinstance(term, dict):
        kind = term.get("kind")
        value = term.get("value")
        if kind == "atom":
            return str(value)
        if kind == "variable":
            return f"${value}"
        if kind == "expression" and isinstance(value, (list, tuple)):
            return "(" + " ".join(_term_source(item) for item in value) + ")"
    return str(term)


def _action_proposals_from_proofs(proofs) -> list[ActionProposal]:
    proposals = []
    for proof in proofs or ():
        term = proof.get("term") if isinstance(proof, dict) else proof
        match = _action_proposal_re.search(_term_source(term))
        if match is None:
            continue
        context, action, incident_id, utility, confidence, rationale = match.groups()
        values = (float(utility), float(confidence))
        if not all(math.isfinite(value) for value in values):
            continue
        proposals.append(ActionProposal(
            context=context,
            action=action,
            incident_id=incident_id,
            utility=values[0],
            confidence=values[1],
            rationale=rationale,
        ))
    return proposals


class ReasonerBackend(Protocol):
    name: str

    def infer(
        self, history: list[HistoryCase], incidents: list[Incident], budget: int, statements: str
    ) -> tuple[dict[str, Belief], dict[str, object]]: ...

    def propose_actions(
        self,
        context: str,
        candidates: list[ActionCandidate],
        budget: int,
        *,
        inspection_cost: float,
        repair_cost: float,
        unnecessary_repair_penalty: float,
    ) -> tuple[list[ActionProposal], dict[str, object]]: ...


@dataclass
class ReferenceBackend:
    config: Config
    sensor_models: dict[str, tuple[float, float]] | None = None
    name: str = "reference"

    def _sensor_rates(self, equipment_type: str | None) -> tuple[float, float]:
        default = (self.config.sensitivity, self.config.false_positive_rate)
        return (
            self.sensor_models.get(equipment_type, default)
            if self.sensor_models
            else default
        )

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}
        if any(item.module_id is not None for item in incidents):
            beliefs = {
                key: Belief(value, 1.0)
                for key, value in self._graph_beliefs(history, incidents).items()
            }
            return beliefs, {
                "queries": len(incidents),
                "engine_steps": None,
                "exact_dependency_components": self._dependency_component_count(incidents),
            }
        priors = empirical_feature_priors(history)
        beliefs = {
            x.id: Belief(
                posterior(
                    priors[(x.cohort, x.equipment_type)],
                    x.alarm,
                    *self._sensor_rates(x.equipment_type),
                ),
                1.0,
            )
            for x in incidents
        }
        return beliefs, {"queries": len(incidents), "engine_steps": None}

    @staticmethod
    def _components(incidents: list[Incident]) -> list[list[Incident]]:
        by_module = {
            item.module_id: item for item in incidents if item.module_id is not None
        }
        neighbors = {module_id: set() for module_id in by_module}
        for item in incidents:
            if item.module_id is None:
                continue
            for upstream in item.upstream_module_ids:
                if upstream in neighbors:
                    neighbors[item.module_id].add(upstream)
                    neighbors[upstream].add(item.module_id)
        components = []
        remaining = set(by_module)
        while remaining:
            start = min(remaining)
            stack = [start]
            found = set()
            while stack:
                module_id = stack.pop()
                if module_id in found:
                    continue
                found.add(module_id)
                stack.extend(neighbors[module_id] - found)
            remaining -= found
            components.append([by_module[module_id] for module_id in sorted(found)])
        return components

    @classmethod
    def _dependency_component_count(cls, incidents: list[Incident]) -> int:
        return len(cls._components(incidents))

    def _graph_beliefs(self, history, incidents) -> dict[str, float]:
        """Exact inference by enumerating each bounded equipment train.

        Dependency components contain at most five local fault variables, so
        this is O(number of modules) despite jointly conditioning every alarm
        in a train.  It is the scoring oracle, not an implementation strategy
        offered to the tested chainers.
        """
        priors = empirical_feature_priors(history)
        result = {}
        for component in self._components(incidents):
            indexes = {item.module_id: index for index, item in enumerate(component)}
            denominator = 0.0
            numerators = [0.0] * len(component)
            for mask in range(1 << len(component)):
                local_fault = [bool(mask & (1 << index)) for index in range(len(component))]
                unavailable: dict[str, bool] = {}

                def is_unavailable(item: Incident) -> bool:
                    module_id = item.module_id
                    if module_id in unavailable:
                        return unavailable[module_id]
                    value = local_fault[indexes[module_id]] or any(
                        is_unavailable(component[indexes[upstream]])
                        for upstream in item.upstream_module_ids
                        if upstream in indexes
                    )
                    unavailable[module_id] = value
                    return value

                weight = 1.0
                for index, item in enumerate(component):
                    prior = priors[(item.cohort, item.equipment_type)]
                    weight *= prior if local_fault[index] else 1.0 - prior
                    sensitivity, false_positive = self._sensor_rates(item.equipment_type)
                    alarm_probability = (
                        sensitivity if is_unavailable(item) else false_positive
                    )
                    weight *= alarm_probability if item.alarm else 1.0 - alarm_probability
                denominator += weight
                for index, fault in enumerate(local_fault):
                    if fault:
                        numerators[index] += weight
            for item, numerator in zip(component, numerators, strict=True):
                if denominator > 0:
                    result[item.id] = numerator / denominator
        return result

    def condition_shortfalls(self, events, budget):
        queries = marginal_queries(events)
        return (
            oracle_event_marginals(events) if budget > 0 else {},
            {
                "shortfall_queries": len(queries),
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": None,
            },
        )

    def propose_actions(
        self,
        context,
        candidates,
        budget,
        *,
        inspection_cost,
        repair_cost,
        unnecessary_repair_penalty,
    ):
        proposals = (
            reference_action_proposals(
                context,
                candidates,
                inspection_cost=inspection_cost,
                repair_cost=repair_cost,
                unnecessary_repair_penalty=unnecessary_repair_penalty,
            )
            if budget > 0
            else []
        )
        return proposals, {
            "action_queries": 1 if candidates else 0,
            "action_proposals": len(proposals),
            "action_statements_added": 0,
            "action_engine_steps": None,
        }


class MM2Backend:
    """Persistent adapter for MM2's incremental named-statement API."""

    name = "mm2"
    _forward_depth = 2
    # StationOps graph diagnosis reaches complete coverage and stable actions
    # within 25 MM2 steps per incident (with a 100-step setup floor).
    # Continuing to the PeTTa-sized ceiling repeatedly revises low-confidence
    # alternative proofs without changing a diagnosis or controller decision.
    # Scale wider graphs instead of imposing a benchmark-size global cap, keep
    # the caller's value as a maximum, and expose the effective window.
    _graph_diagnosis_steps_per_query = 25
    _graph_diagnosis_budget_floor = 100

    def _new_engine(self):
        native = getattr(self.module.Engine, "native", None)
        return native() if callable(native) else self.module.Engine()

    def __init__(
        self,
        config: Config,
        python_path: str | None = None,
        module=None,
        sensor_knowledge: dict[str, str] | None = None,
    ):
        self.config = config
        self.sensor_knowledge = sensor_knowledge or {}
        self._engine = None
        self._action_engine = None
        self._action_atoms_by_name: dict[str, str] = {}
        self._action_context: str | None = None
        self._atoms_by_name: dict[str, str] = {}
        self._base_rates: dict[str, float] = {}
        self._shortfall_supported: bool | None = None
        self._shortfall_unsupported_reason: str | None = None
        self._shortfall_cache: dict[str, float] = {}
        if module is not None:
            self.module = module
            return
        if python_path:
            sys.path.insert(0, str(Path(python_path).expanduser().resolve()))
        try:
            self.module = importlib.import_module("mm2_chainer")
        except (ImportError, OSError) as exc:
            raise BackendUnavailable(
                "mm2_chainer is not importable; build its Python >=3.13 binding with "
                "`maturin develop --release`, then pass --mm2-path or set "
                "MM2_CHAINER_PYTHONPATH"
            ) from exc

    def _reconcile(self, statements: str, forward_facts: bool = True) -> dict[str, int]:
        cold_start = not self._atoms_by_name
        entries: list[tuple[str, str, str]] = []
        desired: dict[str, str] = {}
        for atom in (line.strip() for line in statements.splitlines() if line.strip()):
            name, type_expression = _statement_parts(atom)
            if name in desired:
                raise ValueError(f"duplicate MeTTa statement name: {name}")
            desired[name] = atom
            entries.append((name, type_expression, atom))

        changed_names = [
            name
            for name, _, atom in entries
            if name in self._atoms_by_name and self._atoms_by_name[name] != atom
        ]
        if changed_names:
            raise ValueError(
                "cannot replace named statements in append-only MM2 knowledge base: "
                + ", ".join(changed_names)
            )

        additions = [entry for entry in entries if entry[0] not in self._atoms_by_name]
        if additions:
            self._engine.add_many(
                "stationops",
                "\n".join(atom for _, _, atom in additions),
            )
            self._atoms_by_name.update((name, atom) for name, _, atom in additions)

        facts = [] if cold_start or not forward_facts else [
            entry
            for entry in additions
            if not _is_rule(entry[1]) and _forward_fact(entry)
        ]
        seeds = [_fact_seed(type_expression) for _, type_expression, _ in facts]
        forward_steps = self._forward_depth if seeds else 0
        if seeds:
            self._engine.forward_chain("stationops", seeds, forward_steps)

        return {
            "statements_added": len(additions),
            "statements_removed": 0,
            "forward_seed_facts": len(seeds),
            "forward_steps": forward_steps,
        }

    @staticmethod
    def _result_probability(result) -> float | None:
        term = result.get("term") if isinstance(result, dict) else None
        if isinstance(term, dict) and term.get("kind") == "expression":
            fields = term.get("value") or ()
            if fields:
                value = fields[-1]
                if isinstance(value, dict) and value.get("kind") == "atom":
                    return finite_probability(value.get("value"))
        if isinstance(term, str):
            fields = _fields(term)
            if len(fields) == 4 and fields[0] == "ShortfallMarginal":
                return finite_probability(fields[-1])
        return None

    def condition_shortfalls(self, events, budget):
        queries = marginal_queries(events)
        if not queries or budget <= 0:
            return {}, {
                "shortfall_queries": len(queries),
                "shortfall_engine_queries": 0,
                "shortfall_cache_hits": 0,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": None,
            }
        (
            queries,
            shifts,
            marginals,
            pending,
            cache_hits,
        ) = _shortfall_query_plan(
            events,
            self._shortfall_cache,
        )
        if self._engine is None:
            self._engine = self._new_engine()
        if self._shortfall_supported is None:
            try:
                probe = self._new_engine()
                probe.add_many("stationops-shortfall-probe", "\n".join(SHORTFALL_RULES))
            except Exception as exc:
                self._shortfall_supported = False
                self._shortfall_unsupported_reason = str(exc)
            else:
                self._shortfall_supported = True
        if not self._shortfall_supported:
            return {}, {
                "shortfall_queries": len(queries),
                "shortfall_engine_queries": 0,
                "shortfall_cache_hits": cache_hits,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": None,
                "shortfall_supported": False,
                "shortfall_unsupported_reason": self._shortfall_unsupported_reason,
            }

        if not pending:
            return marginals, {
                "shortfall_queries": len(queries),
                "shortfall_marginal_queries": len(queries),
                "shortfall_engine_queries": 0,
                "shortfall_cache_hits": cache_hits,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": 0,
                "shortfall_engine_stats": None,
                "shortfall_supported": True,
            }

        pending_events = {tag.split("|", 1)[0] for tag, _ in pending}
        try:
            counters = self._reconcile(
                generate_shortfall_statements([
                    event for event in events if event_atom(event) in pending_events
                ]),
                forward_facts=False,
            )
        except Exception:
            self._engine = None
            self._atoms_by_name.clear()
            self._base_rates.clear()
            raise
        for tag, proofs in self._engine.query_many("stationops", pending, budget):
            event_name, unit = tag.split("|", 1)
            probability = next(
                (
                    value
                    for value in (self._result_probability(proof) for proof in proofs)
                    if value is not None
                ),
                None,
            )
            if probability is not None:
                self._shortfall_cache[tag] = probability
                marginals.setdefault(shifts[event_name], {})[unit] = probability
        execution_stats = _engine_execution_stats(self._engine)
        return marginals, {
            "shortfall_queries": len(queries),
            "shortfall_marginal_queries": len(queries),
            "shortfall_engine_queries": len(pending),
            "shortfall_cache_hits": cache_hits,
            "shortfall_statements_added": counters["statements_added"],
            "shortfall_engine_steps": (
                execution_stats.get("steps") if execution_stats else None
            ),
            "shortfall_engine_stats": execution_stats,
            "shortfall_supported": True,
        }

    def _update_base_rates(self, history) -> int:
        priors = empirical_feature_priors(history)
        desired = {}
        for (cohort, equipment_type), prior in priors.items():
            fields = f"{cohort} " + (f"{equipment_type} " if equipment_type else "")
            desired[f"(SealLeak {fields}$unit)"] = prior
        missing_patterns = sorted(self._base_rates.keys() - desired.keys())
        if missing_patterns:
            raise ValueError(
                "history lost feature groups in append-only MM2 episode: "
                + ", ".join(missing_patterns)
            )
        updated = 0
        for pattern, prior in desired.items():
            if self._base_rates.get(pattern) == prior:
                continue
            self._engine.set_base_rate("stationops", pattern, f"(STV {prior} 1)")
            self._base_rates[pattern] = prior
            updated += 1
        return updated

    def _incident_query(self, incident: Incident) -> str:
        if (
            incident.problem_context is not None
            and incident.module_id is not None
            and self.sensor_knowledge.get(incident.equipment_type, "full") == "full"
        ):
            return (
                f"(LocalProblem {incident.problem_context} "
                f"{incident.module_id} {incident.cohort} "
                f"{incident.equipment_type} {incident.id})"
            )
        if (
            incident.equipment_type is not None
            and self.sensor_knowledge.get(incident.equipment_type)
            in {"positive", "induced"}
        ):
            observation = "PressureAlarm" if incident.alarm else "PressureNormal"
            return (
                f"(Inheritance ({observation} {incident.cohort} {incident.equipment_type}) "
                f"(SealLeak {incident.cohort} {incident.equipment_type}))"
            )
        fields = f"{incident.cohort} "
        if incident.equipment_type is not None:
            fields += f"{incident.equipment_type} "
        return f"(SealLeak {fields}{incident.id})"

    @staticmethod
    def _beliefs_from_results(results) -> dict[str, Belief]:
        """Map each proven SealLeak goal to its strongest returned STV."""
        beliefs = {}
        for tag, proofs in results:
            truth_values = [
                Belief(
                    proof["truth_value"]["strength"],
                    proof["truth_value"].get("confidence", 1.0),
                )
                for proof in proofs
                if isinstance(proof, dict)
                and isinstance(proof.get("truth_value"), dict)
                and isinstance(proof["truth_value"].get("strength"), (int, float))
                and isinstance(
                    proof["truth_value"].get("confidence", 1.0), (int, float)
                )
                and math.isfinite(proof["truth_value"]["strength"])
                and math.isfinite(proof["truth_value"].get("confidence", 1.0))
            ]
            if truth_values:
                beliefs[tag] = max(truth_values, key=float)
        return beliefs

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}
        if self._engine is None:
            self._engine = self._new_engine()
        try:
            counters = self._reconcile(statements)
            counters["base_rates_updated"] = self._update_base_rates(history)
        except Exception:
            # Reconciliation can partially mutate the native engine. Rebuild
            # from the next complete snapshot rather than retaining mixed state.
            self._engine = None
            self._atoms_by_name.clear()
            self._base_rates.clear()
            raise
        queries = [(x.id, self._incident_query(x)) for x in incidents]
        dependency_support = (
            [
                (
                    f"dependency-support:{x.id}",
                    f"(Problem {x.cohort} {x.equipment_type} {x.id})",
                )
                for x in incidents
                if x.problem_context is not None
                and x.module_id is not None
                and x.equipment_type is not None
            ]
            if any(x.upstream_module_ids for x in incidents)
            else []
        )
        effective_budget = (
            min(
                budget,
                max(
                    self._graph_diagnosis_budget_floor,
                    len(queries) * self._graph_diagnosis_steps_per_query,
                ),
            )
            if dependency_support
            else budget
        )
        if dependency_support:
            # MM2 retains query proofs. Materialize the current Problem roots
            # once so dependency BiImplications and strict Or projections have
            # the same reachable proof frontier as PeTTaChainer's best-first
            # nested search. These are proof-producing support queries, not
            # adapter-side beliefs.
            self._engine.query_many(
                "stationops", dependency_support, effective_budget
            )
        results = self._engine.query_many("stationops", queries, effective_budget)
        execution_stats = _engine_execution_stats(self._engine)
        beliefs = self._beliefs_from_results(results)
        counters.update({
            "queries": len(queries),
            "induced_queries": sum(
                self.sensor_knowledge.get(x.equipment_type) == "induced"
                for x in incidents
            ),
            "learned_relation_queries": sum(
                self.sensor_knowledge.get(x.equipment_type)
                in {"positive", "induced"}
                for x in incidents
            ),
            "dependency_support_queries": len(dependency_support),
            "diagnosis_budget_requested": budget,
            "diagnosis_budget_effective": effective_budget,
            "engine_steps": execution_stats.get("steps") if execution_stats else None,
            "diagnosis_engine_stats": execution_stats,
        })
        return beliefs, counters

    def propose_actions(
        self,
        context,
        candidates,
        budget,
        *,
        inspection_cost,
        repair_cost,
        unnecessary_repair_penalty,
    ):
        if not candidates or budget <= 0:
            return [], {
                "action_queries": 1 if candidates else 0,
                "action_proposals": 0,
                "action_statements_added": 0,
                "action_engine_steps": None,
            }
        if self._action_engine is None or self._action_context != context:
            self._action_engine = self._new_engine()
            self._action_atoms_by_name.clear()
            self._action_context = context
        statements = generate_action_statements(
            context,
            candidates,
            inspection_cost=inspection_cost,
            repair_cost=repair_cost,
            unnecessary_repair_penalty=unnecessary_repair_penalty,
        )
        entries = []
        for atom in (line.strip() for line in statements.splitlines() if line.strip()):
            name, type_expression = _statement_parts(atom)
            previous = self._action_atoms_by_name.get(name)
            if previous is not None and previous != atom:
                raise ValueError(f"cannot replace action statement: {name}")
            if previous is None:
                entries.append((name, type_expression, atom))
        try:
            if entries:
                self._action_engine.add_many(
                    "stationops-actions",
                    "\n".join(atom for _, _, atom in entries),
                )
                self._action_atoms_by_name.update(
                    (name, atom) for name, _, atom in entries
                )
            facts = [entry for entry in entries if not _is_rule(entry[1])]
            seeds = [_fact_seed(type_expression) for _, type_expression, _ in facts]
            forward_steps = self._forward_depth if seeds else 0
            if seeds:
                self._action_engine.forward_chain(
                    "stationops-actions", seeds, forward_steps
                )
        except Exception:
            self._action_engine = None
            self._action_atoms_by_name.clear()
            self._action_context = None
            raise
        query = (
            f"(ActionProposal {context} $action $utility $confidence $rationale)"
        )
        results = self._action_engine.query_many(
            "stationops-actions", [(context, query)], budget
        )
        proposals = []
        for _, proofs in results:
            proposals.extend(_action_proposals_from_proofs(proofs))
        execution_stats = _engine_execution_stats(self._action_engine)
        return proposals, {
            "action_queries": 1,
            "action_proposals": len(proposals),
            "action_statements_added": len(entries),
            "action_forward_seed_facts": len(seeds),
            "action_forward_steps": forward_steps,
            "action_engine_steps": (
                execution_stats.get("steps") if execution_stats else None
            ),
            "action_engine_stats": execution_stats,
        }


class PeTTaChainerBackend:
    """Persistent StationOps adapter for PeTTaChainer's supported Python API.

    ``statements`` is the current round's public view. The adapter treats each
    view as an append-only contribution, incrementally forwards newly named
    facts, and retains all prior knowledge and caches for later rounds.
    """

    name = "pettachainer"
    _forward_batch_size = 100
    # Forward chaining spends a fixed number of row computations per round,
    # however many facts the round brings: new facts are queued without
    # units and one run per round spends the slice on them first, then on
    # what earlier runs left. 150 reaches PeTTaChainer's plateau on the
    # replayed temporal model (Brier 0.0590 at 11 s per 20-shift seed).
    _forward_slice = 150
    # PLN's evidence parameter: n resolved cases give confidence n/(n+k).
    # StationOps groups hold a handful of cases each, and k = 1 weighs them
    # against the public base rate as one pseudo-observation of prior, which
    # the confidence blend of decision_beliefs relies on.
    _evidence_confidence_k = 1
    _stv_re = re.compile(
        r"\((?:STV|stv)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(
        self,
        config: Config,
        python_path: str | None = None,
        module=None,
        sensor_knowledge: dict[str, str] | None = None,
        temporal_model: bool = False,
    ):
        self.config = config
        self.sensor_knowledge = sensor_knowledge or {}
        # The temporal model asks every incident for its own seal state, learns
        # its induced sensors from the resolved cases, and reads rates over a
        # handful of cases as uncertain. Its beliefs are used as they are, not
        # blended with public rates by confidence, and k = 5 calibrates them
        # best (Brier 0.079 against 0.097 at k = 1, 4 seeds x 20 shifts).
        self.temporal_model = temporal_model
        if temporal_model:
            self._evidence_confidence_k = 5
        self._handler = None
        self._atoms_by_name: dict[str, str] = {}
        # The statement each incident's belief was queried as, which its
        # decisions read through a LeakBelief link.
        self._belief_goals: dict[str, str] = {}
        self._shortfall_cache: dict[str, float] = {}
        self._shortfall_formula_handler = None
        if module is not None:
            self.module = module
            return
        if python_path:
            sys.path.insert(0, str(Path(python_path).expanduser().resolve()))
        try:
            self.module = importlib.import_module("pettachainer")
            getattr(self.module, "PeTTaChainer")
        except (AttributeError, ImportError, OSError) as exc:
            raise BackendUnavailable(
                "PeTTaChainer is not importable; install its PeTTa dependency and package, "
                "then pass --pettachainer-path pointing to the checkout/package parent or "
                "set PETTACHAINER_PYTHONPATH"
            ) from exc

    @classmethod
    def _strongest_belief(cls, proofs) -> Belief | None:
        truth_values = []
        for proof in proofs or ():
            # An answer is (: proof statement tv): its own truth value is the
            # last one; a proof may hold others (a prior's, for instance).
            matches = cls._stv_re.findall(str(proof))
            if matches:
                strength = float(matches[-1][0])
                confidence = float(matches[-1][1])
                if math.isfinite(strength) and math.isfinite(confidence):
                    truth_values.append(Belief(strength, confidence))
        return max(truth_values, key=float) if truth_values else None

    def _new_handler(self):
        try:
            handler = self.module.PeTTaChainer()
        except (ImportError, OSError) as exc:
            dependency = getattr(exc, "name", None) or str(exc)
            raise BackendUnavailable(
                "PeTTaChainer could not construct its runtime handler; install its "
                f"PeTTa/Janus dependencies (missing or unavailable: {dependency}) and ensure "
                "both source roots and native libraries are available"
            ) from exc
        handler.set_evidence_confidence_k(self._evidence_confidence_k)
        if self.temporal_model:
            handler.set_rule_refinement(True)
            handler.set_base_rate_smoothing(True)
        return handler

    def _load_shortfall_formulas(self) -> None:
        if self._shortfall_formula_handler is self._handler:
            return
        load_weighted_subset_posterior(self._handler)
        self._shortfall_formula_handler = self._handler

    def _reconcile(self, statements: str, forward_facts: bool = True) -> dict[str, int]:
        entries: list[tuple[str, str, str]] = []
        desired: dict[str, str] = {}
        for atom in (line.strip() for line in statements.splitlines() if line.strip()):
            name, type_expression = _statement_parts(atom)
            if name in desired:
                raise ValueError(f"duplicate MeTTa statement name: {name}")
            desired[name] = atom
            entries.append((name, type_expression, atom))

        changed_names = [
            name
            for name, _, atom in entries
            if name in self._atoms_by_name and self._atoms_by_name[name] != atom
        ]
        if changed_names:
            raise ValueError(
                "cannot replace named statements in append-only PeTTaChainer knowledge base: "
                + ", ".join(changed_names)
            )

        additions = [entry for entry in entries if entry[0] not in self._atoms_by_name]
        rules = [entry for entry in additions if _is_rule(entry[1])]
        facts = [entry for entry in additions if not _is_rule(entry[1])]

        if rules:
            self._handler.add_atoms_no_check([atom for _, _, atom in rules])
            self._atoms_by_name.update((name, atom) for name, _, atom in rules)

        forward_seeds = 0
        forward_steps = 0
        for offset in range(0, len(facts), self._forward_batch_size):
            batch = facts[offset:offset + self._forward_batch_size]
            self._handler.add_atoms_no_check([atom for _, _, atom in batch])
            self._atoms_by_name.update((name, atom) for name, _, atom in batch)
            seed_entries = [entry for entry in batch if _forward_fact(entry)]
            if forward_facts and seed_entries:
                seeds = self._handler.select_facts(
                    [_fact_seed(type_expression) for _, type_expression, _ in seed_entries]
                )
                self._handler.forward_chain(seeds, steps=0)
                forward_seeds += len(seeds)
        if forward_facts and forward_seeds:
            self._handler.forward_chain([], steps=self._forward_slice)
            forward_steps = self._forward_slice

        return {
            "statements_added": len(additions),
            "statements_removed": 0,
            "forward_seed_facts": forward_seeds,
            "forward_steps": forward_steps,
        }

    _shortfall_re = re.compile(
        r"\(ShortfallMarginal\s+[^()\s]+\s+[^()\s]+\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    @classmethod
    def _shortfall_probability(cls, proofs) -> float | None:
        for proof in proofs or ():
            match = cls._shortfall_re.search(str(proof))
            if match is not None:
                probability = finite_probability(match.group(1))
                if probability is not None:
                    return probability
        return None

    def condition_shortfalls(self, events, budget):
        queries = marginal_queries(events)
        if not queries or budget <= 0:
            return {}, {
                "shortfall_queries": len(queries),
                "shortfall_engine_queries": 0,
                "shortfall_cache_hits": 0,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": None,
            }
        (
            queries,
            shifts,
            marginals,
            pending,
            cache_hits,
        ) = _shortfall_query_plan(
            events,
            self._shortfall_cache,
        )
        if self._handler is None:
            self._handler = self._new_handler()
        if not pending:
            return marginals, {
                "shortfall_queries": len(queries),
                "shortfall_marginal_queries": len(queries),
                "shortfall_engine_queries": 0,
                "shortfall_cache_hits": cache_hits,
                "shortfall_statements_added": 0,
                "shortfall_engine_steps": 0,
                "shortfall_supported": True,
            }
        pending_events = {tag.split("|", 1)[0] for tag, _ in pending}
        try:
            self._load_shortfall_formulas()
            counters = self._reconcile(
                generate_shortfall_statements([
                    event for event in events if event_atom(event) in pending_events
                ]),
                forward_facts=False,
            )
        except Exception:
            self._handler = None
            self._shortfall_formula_handler = None
            self._atoms_by_name.clear()
            raise
        for tag, query in pending:
            event_name, unit = tag.split("|", 1)
            proofs = self._handler.query(
                f"(: $prf {query} $tv)", steps=budget, timeout_sec=0
            )
            probability = self._shortfall_probability(proofs)
            if probability is not None:
                self._shortfall_cache[tag] = probability
                marginals.setdefault(shifts[event_name], {})[unit] = probability
        return marginals, {
            "shortfall_queries": len(queries),
            "shortfall_marginal_queries": len(queries),
            "shortfall_engine_queries": len(pending),
            "shortfall_cache_hits": cache_hits,
            "shortfall_statements_added": counters["statements_added"],
            "shortfall_engine_steps": None,
            "shortfall_supported": True,
        }

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}

        if self._handler is None:
            self._handler = self._new_handler()
        try:
            if self.temporal_model:
                self._load_shortfall_formulas()
            counters = self._reconcile(statements)
        except Exception:
            # A failed batch may have partially mutated the external runtime.
            # Discard it so a later call can rebuild from its complete snapshot.
            self._handler = None
            self._atoms_by_name.clear()
            raise

        goals = []
        for incident in incidents:
            if (
                incident.problem_context is not None
                and incident.module_id is not None
                and self.sensor_knowledge.get(incident.equipment_type, "full")
                == "full"
            ):
                goal = (
                    f"(LocalProblem {incident.problem_context} "
                    f"{incident.module_id} {incident.cohort} "
                    f"{incident.equipment_type} {incident.id})"
                )
            elif (
                not self.temporal_model
                and incident.equipment_type is not None
                and self.sensor_knowledge.get(incident.equipment_type)
                in {"positive", "induced"}
            ):
                observation = "PressureAlarm" if incident.alarm else "PressureNormal"
                goal = (
                    f"(Inheritance ({observation} {incident.cohort} "
                    f"{incident.equipment_type}) "
                    f"(SealLeak {incident.cohort} {incident.equipment_type}))"
                )
            else:
                fields = f"{incident.cohort} "
                if incident.equipment_type is not None:
                    fields += f"{incident.equipment_type} "
                goal = f"(SealLeak {fields}{incident.id})"
            self._belief_goals[incident.id] = goal
            goals.append(f"(: $prf {goal} $tv)")

        proof_batches = (
            self._handler.query_many(goals, steps=budget, timeout_sec=0)
            if goals
            else []
        )

        beliefs = {}
        for incident, proofs in zip(incidents, proof_batches, strict=True):
            belief = self._strongest_belief(proofs)
            if belief is not None:
                beliefs[incident.id] = belief
        counters.update({
            "queries": len(incidents),
            "induced_queries": sum(
                self.sensor_knowledge.get(x.equipment_type) == "induced"
                for x in incidents
            ),
            "learned_relation_queries": sum(
                self.sensor_knowledge.get(x.equipment_type)
                in {"positive", "induced"}
                for x in incidents
            ),
            "engine_steps": None,
        })
        return beliefs, counters

    def propose_actions(
        self,
        context,
        candidates,
        budget,
        *,
        inspection_cost,
        repair_cost,
        unnecessary_repair_penalty,
    ):
        if not candidates or budget <= 0:
            return [], {
                "action_queries": 1 if candidates else 0,
                "action_proposals": 0,
                "action_statements_added": 0,
                "action_engine_steps": None,
            }
        if self._handler is None:
            self._handler = self._new_handler()
        # The action rules are added once; a decision step adds nothing: its
        # facts, and the links from each candidate's belief to the LeakBelief
        # the rules read, are assumed for the step's query alone.
        rules = [
            (name, atom)
            for atom in BELIEF_ACTION_RULES
            for name in (_statement_parts(atom)[0],)
            if name not in self._atoms_by_name
        ]
        try:
            if rules:
                self._handler.add_atoms_no_check([atom for _, atom in rules])
                self._atoms_by_name.update(rules)
        except Exception:
            self._handler = None
            self._atoms_by_name.clear()
            raise
        assumptions = [
            *(
                leak_belief_assumption(candidate.incident_id, self._belief_goals[candidate.incident_id])
                for candidate in candidates
                if candidate.incident_id in self._belief_goals
            ),
            *decision_assumptions(
                context,
                candidates,
                inspection_cost=inspection_cost,
                repair_cost=repair_cost,
                unnecessary_repair_penalty=unnecessary_repair_penalty,
            ),
        ]
        query = (
            f"(: $prf (Assuming (And {' '.join(assumptions)}) "
            f"(ActionProposal {context} $action $utility $confidence $rationale)) $tv)"
        )
        proofs = self._handler.query(query, steps=budget, timeout_sec=0)
        proposals = _action_proposals_from_proofs(proofs)
        return proposals, {
            "action_queries": 1,
            "action_proposals": len(proposals),
            "action_statements_added": len(rules),
            "action_engine_steps": None,
        }

    def observe_inspection(self, incident_id, leak):
        """An inspection settles the incident's leak belief for later decisions.

        The belief model itself learns the result from the shift's resolved
        leaks, as it does for every resolution.
        """
        name = f"inspection-{incident_id}"
        if self._handler is None or name in self._atoms_by_name:
            return
        atom = f"(: {name} (LeakBelief {incident_id}) (STV {1 if leak else 0} 1))"
        self._handler.add_atoms_no_check([atom])
        self._atoms_by_name[name] = atom


def create_backend(
    config: Config,
    name: str,
    mm2_path=None,
    pettachainer_path=None,
    *,
    sensor_models=None,
    sensor_knowledge_map=None,
    temporal_model=False,
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
            temporal_model=temporal_model,
        )
    raise ValueError(f"unknown backend: {name}")
