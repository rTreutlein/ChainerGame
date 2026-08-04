"""Reasoner adapter seam.

Future PeTTa support implements ReasonerBackend only; simulator, oracle, policy,
and scoring do not import a concrete logic engine.
"""
from __future__ import annotations

import importlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import Config
from .models import HistoryCase, Incident
from .oracle import empirical_priors, posterior


class BackendUnavailable(RuntimeError):
    pass


class ReasonerBackend(Protocol):
    name: str

    def infer(
        self, history: list[HistoryCase], incidents: list[Incident], budget: int, statements: str
    ) -> tuple[dict[str, float], dict[str, int | float | None]]: ...


@dataclass
class ReferenceBackend:
    config: Config
    name: str = "reference"

    def infer(self, history, incidents, budget, statements):
        priors = empirical_priors(history)
        beliefs = {
            x.id: posterior(priors[x.cohort], x.alarm, self.config.sensitivity, self.config.false_positive_rate)
            for x in incidents
        }
        # A zero budget intentionally supplies no proofs; positive reference budgets converge.
        return (beliefs if budget > 0 else {}), {"queries": len(incidents), "engine_steps": None}


class MM2Backend:
    name = "mm2"

    def __init__(self, config: Config, python_path: str | None = None, module=None):
        self.config = config
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

    @staticmethod
    def _beliefs_from_results(results) -> dict[str, float]:
        """Map each proven SealLeak goal to its strongest returned STV."""
        beliefs = {}
        for tag, proofs in results:
            strengths = [
                proof["truth_value"]["strength"]
                for proof in proofs
                if isinstance(proof, dict)
                and isinstance(proof.get("truth_value"), dict)
                and isinstance(proof["truth_value"].get("strength"), (int, float))
            ]
            if strengths:
                beliefs[tag] = max(strengths)
        return beliefs

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}
        engine = self.module.Engine()
        engine.add_many("stationops", statements)
        priors = empirical_priors(history)
        for cohort, prior in priors.items():
            engine.set_base_rate("stationops", f"(SealLeak {cohort} $unit)", f"(STV {prior} 1)")
        # SealLeak is the action belief. PatchPaysOff is a unit-strength wrapper
        # used by engines that compose inversion with another backward rule;
        # current MM2 coverage exposes the inverted SealLeak proof directly.
        queries = [(x.id, f"(SealLeak {x.cohort} {x.id})") for x in incidents]
        results = engine.query_many("stationops", queries, budget)
        beliefs = self._beliefs_from_results(results)
        return beliefs, {"queries": len(queries), "engine_steps": None}


class PeTTaChainerBackend:
    """Persistent StationOps adapter for PeTTaChainer's supported Python API.

    ``statements`` is a complete public-KB snapshot for the current round.  The
    adapter reconciles that snapshot by statement name, incrementally forwards
    newly added facts, and retains the resulting caches for later rounds.
    """

    name = "pettachainer"
    _forward_batch_size = 100
    _forward_steps_per_seed = 2
    _stv_re = re.compile(
        r"\((?:STV|stv)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(self, config: Config, python_path: str | None = None, module=None):
        self.config = config
        self._handler = None
        self._atoms_by_name: dict[str, str] = {}
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
    def _strongest_strength(cls, proofs) -> float | None:
        strengths = []
        for proof in proofs or ():
            match = cls._stv_re.search(str(proof))
            if match is not None:
                strength = float(match.group(1))
                if math.isfinite(strength):
                    strengths.append(strength)
        return max(strengths) if strengths else None

    @staticmethod
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

    @classmethod
    def _statement_parts(cls, atom: str) -> tuple[str, str]:
        fields = cls._fields(atom)
        if len(fields) != 4 or fields[0] != ":":
            raise ValueError(f"expected one named MeTTa statement: {atom}")
        return fields[1], fields[2]

    @classmethod
    def _is_rule(cls, type_expression: str) -> bool:
        fields = cls._fields(type_expression)
        return bool(fields) and fields[0] == "Implication"

    @classmethod
    def _fact_seed(cls, type_expression: str) -> str:
        """Return the surface term whose canonical fact seeds forward work."""
        fields = cls._fields(type_expression)
        if len(fields) == 2 and fields[0] == "Not":
            # PeTTaChainer canonicalizes negative STV observations as a
            # complemented-strength fact of the positive type.
            return fields[1]
        return type_expression

    def _new_handler(self):
        try:
            return self.module.PeTTaChainer()
        except (ImportError, OSError) as exc:
            dependency = getattr(exc, "name", None) or str(exc)
            raise BackendUnavailable(
                "PeTTaChainer could not construct its runtime handler; install its "
                f"PeTTa/Janus dependencies (missing or unavailable: {dependency}) and ensure "
                "both source roots and native libraries are available"
            ) from exc

    def _reconcile(self, statements: str) -> dict[str, int]:
        entries: list[tuple[str, str, str]] = []
        desired: dict[str, str] = {}
        for atom in (line.strip() for line in statements.splitlines() if line.strip()):
            name, type_expression = self._statement_parts(atom)
            if name in desired:
                raise ValueError(f"duplicate MeTTa statement name: {name}")
            desired[name] = atom
            entries.append((name, type_expression, atom))

        removed = 0
        changed_names = [
            name
            for name, old_atom in self._atoms_by_name.items()
            if desired.get(name) != old_atom
        ]
        for name in changed_names:
            self._handler.remove_statement(name)
            self._atoms_by_name.pop(name)
            removed += 1

        additions = [entry for entry in entries if entry[0] not in self._atoms_by_name]
        rules = [entry for entry in additions if self._is_rule(entry[1])]
        facts = [entry for entry in additions if not self._is_rule(entry[1])]

        if rules:
            self._handler.add_atoms_no_check([atom for _, _, atom in rules])
            self._atoms_by_name.update((name, atom) for name, _, atom in rules)

        forward_seeds = 0
        forward_steps = 0
        for offset in range(0, len(facts), self._forward_batch_size):
            batch = facts[offset:offset + self._forward_batch_size]
            self._handler.add_atoms_no_check([atom for _, _, atom in batch])
            self._atoms_by_name.update((name, atom) for name, _, atom in batch)
            seeds = self._handler.select_facts(
                [self._fact_seed(type_expression) for _, type_expression, _ in batch]
            )
            steps = self._forward_steps_per_seed * len(seeds)
            self._handler.forward_chain(seeds, steps=steps)
            forward_seeds += len(seeds)
            forward_steps += steps

        return {
            "statements_added": len(additions),
            "statements_removed": removed,
            "forward_seed_facts": forward_seeds,
            "forward_steps": forward_steps,
        }

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}

        if self._handler is None:
            self._handler = self._new_handler()
        try:
            counters = self._reconcile(statements)
        except Exception:
            # A failed batch may have partially mutated the external runtime.
            # Discard it so a later call can rebuild from its complete snapshot.
            self._handler = None
            self._atoms_by_name.clear()
            raise

        beliefs = {}
        for incident in incidents:
            proofs = self._handler.query(
                f"(: $prf (PatchPaysOff {incident.cohort} {incident.id}) $tv)",
                steps=budget,
                timeout_sec=0,
            )
            strength = self._strongest_strength(proofs)
            if strength is not None:
                beliefs[incident.id] = strength
        counters.update({"queries": len(incidents), "engine_steps": None})
        return beliefs, counters
