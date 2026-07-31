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
        """Map each proven PatchPaysOff goal to its strongest returned STV.

        PatchPaysOff is a unit-strength identity consequence of SealLeak in the
        generated KB. Its returned strength therefore has the benchmark's leak
        belief semantics. An absent proof means an absent belief, never zero.
        """
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
        queries = [(x.id, f"(PatchPaysOff {x.cohort} {x.id})") for x in incidents]
        results = engine.query_many("stationops", queries, budget)
        beliefs = self._beliefs_from_results(results)
        return beliefs, {"queries": len(queries), "engine_steps": None}


class PeTTaChainerBackend:
    """StationOps adapter for PeTTaChainer's supported Python API."""

    name = "pettachainer"
    _stv_re = re.compile(
        r"\((?:STV|stv)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(self, config: Config, python_path: str | None = None, module=None):
        self.config = config
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

    def infer(self, history, incidents, budget, statements):
        if budget <= 0:
            return {}, {"queries": len(incidents), "engine_steps": None}

        try:
            handler = self.module.PeTTaChainer()
        except (ImportError, OSError) as exc:
            dependency = getattr(exc, "name", None) or str(exc)
            raise BackendUnavailable(
                "PeTTaChainer could not construct its runtime handler; install its "
                f"PeTTa/Janus dependencies (missing or unavailable: {dependency}) and ensure "
                "both source roots and native libraries are available"
            ) from exc
        priors = empirical_priors(history)
        atoms = []
        for cohort, prior in priors.items():
            alarm = posterior(
                prior, True, self.config.sensitivity, self.config.false_positive_rate
            )
            no_alarm = posterior(
                prior, False, self.config.sensitivity, self.config.false_positive_rate
            )
            atoms.append(
                f"(: stationops-{cohort}-alarm-to-payoff "
                f"(Implication (Premises (PressureAlarm {cohort} $unit)) "
                f"(Conclusions (PatchPaysOff {cohort} $unit))) "
                f"(CTV (STV {alarm} 1) (STV {no_alarm} 1)))"
            )
        atoms.extend(
            f"(: stationops-observed-{incident.id} "
            f"(PressureAlarm {incident.cohort} {incident.id}) "
            f"(STV {1 if incident.alarm else 0} 1))"
            for incident in incidents
        )
        handler.add_atoms_no_check(atoms)

        beliefs = {}
        for incident in incidents:
            proofs = handler.query(
                f"(: $prf (PatchPaysOff {incident.cohort} {incident.id}) $tv)",
                steps=budget,
                timeout_sec=0,
            )
            strength = self._strongest_strength(proofs)
            if strength is not None:
                beliefs[incident.id] = strength
        return beliefs, {"queries": len(incidents), "engine_steps": None}
