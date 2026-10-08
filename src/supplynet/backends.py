"""Reasoners: the exact posterior, history base rates alone, and PeTTaChainer."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

from . import metta
from .world import Network, Observation, Period, Rates, exact_posterior

Key = tuple[str, str]


class BackendUnavailable(RuntimeError):
    pass


class ReferenceBackend:
    """The exact posterior under the true rates."""

    name = "reference"

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.network, self.rates = network, rates

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        posterior = exact_posterior(self.network, self.rates, observation)
        return {key: posterior[key] for key in keys}

    def resolve(self, period: Period, observation: Observation) -> None:
        pass


class PriorBackend:
    """Each statement's frequency in the resolved periods, ignoring the
    current observations: the floor any reasoner should beat."""

    name = "prior"

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.counts: dict[Key, list[int]] = {}
        for period, observation in history:
            self.resolve(period, observation)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        return {key: (self.counts.get(key, [0, 0])[0] + 1) / (self.counts.get(key, [0, 0])[1] + 2) for key in keys}

    def resolve(self, period: Period, observation: Observation) -> None:
        for kind, values in (("Storm", period.storms), ("Blocked", period.blocked)):
            for subject, value in values.items():
                count = self.counts.setdefault((kind, subject), [0, 0])
                count[0] += value
                count[1] += 1


class PeTTaChainerBackend:
    """PeTTaChainer given the rules with their rates; base rates come from the
    labelled periods it holds as facts."""

    name = "pettachainer"
    _stv_re = re.compile(
        r"\(STV\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(self, python_path: str | None = None, evidence_k: float = 5):
        if python_path:
            sys.path.insert(0, str(Path(python_path).expanduser().resolve()))
        try:
            module = importlib.import_module("pettachainer")
            self._handler = module.PeTTaChainer()
        except (AttributeError, ImportError, OSError) as exc:
            raise BackendUnavailable(
                "PeTTaChainer is not importable; pass --pettachainer-path or set PETTACHAINER_PYTHONPATH"
            ) from exc
        self._handler.set_evidence_confidence_k(evidence_k)

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self._handler.add_atoms_no_check(metta.rules(network, rates))
        for period, observation in history:
            self._handler.add_atoms_no_check(metta.observation_facts(observation))
            self.resolve(period, observation)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self._handler.add_atoms_no_check(metta.observation_facts(observation))
        goals = [metta.query(key, observation.period) for key in keys]
        answers = self._handler.query_many(goals, steps=budget, timeout_sec=0) if goals else []
        beliefs = {}
        for key, proofs in zip(keys, answers, strict=True):
            strength = self._strength(proofs)
            if strength is not None:
                beliefs[key] = strength
        return beliefs

    def resolve(self, period: Period, observation: Observation) -> None:
        self._handler.add_atoms_no_check(metta.resolution_facts(period, observation))

    @classmethod
    def _strength(cls, proofs) -> float | None:
        """The strength of the most confident answer; an answer's own truth
        value is its last."""
        best = None
        for proof in proofs or ():
            matches = cls._stv_re.findall(str(proof))
            if matches:
                strength, confidence = float(matches[-1][0]), float(matches[-1][1])
                if best is None or confidence > best[1]:
                    best = (strength, confidence)
        return None if best is None else min(1.0, max(0.0, best[0]))
