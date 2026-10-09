"""Reasoners: the exact posterior, history base rates alone, and PeTTaChainer.

A backend is given a stage's network, rates, labelled history and rounds of
observations. Stages 1-3 use ``world`` and ``metta``; stage "scale" passes its
own knowledge class and statements (``scale``)."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

from . import metta
from .world import Key, Knowledge, Network, Observation, Period, Rates


class BackendUnavailable(RuntimeError):
    pass


class ReferenceBackend:
    """The posterior of a stage's ``knowledge`` class, exact by default,
    under the true rates."""

    def __init__(self, knowledge=Knowledge, name: str = "reference"):
        self._knowledge, self.name = knowledge, name

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.knowledge = self._knowledge(network, rates)
        for period, observation in history:
            self.knowledge.observe(observation)
            self.knowledge.resolve(period)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self.knowledge.observe(observation)
        posterior = self.knowledge.posterior()
        return {key: posterior[key] for key in keys}

    def resolve(self, period: Period, observation: Observation) -> None:
        self.knowledge.resolve(period)


class PriorBackend:
    """Each statement's frequency in the resolved periods, ignoring the
    current observations: the floor any reasoner should beat."""

    name = "prior"

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.counts: dict[tuple[str, str], list[int]] = {}
        for period, observation in history:
            self.resolve(period, observation)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        return {key: (self.counts.get(key[:2], [0, 0])[0] + 1) / (self.counts.get(key[:2], [0, 0])[1] + 2) for key in keys}

    def resolve(self, period: Period, observation: Observation) -> None:
        for var, value in period.labels().items():
            count = self.counts.setdefault(var, [0, 0])
            count[0] += value
            count[1] += 1


class PeTTaChainerBackend:
    """PeTTaChainer given the rules with their rates; base rates come from the
    labelled periods it holds as facts. ``statements`` translates the stage's
    world into MeTTa (``metta`` for stages 1-3)."""

    name = "pettachainer"
    _stv_re = re.compile(
        r"\(STV\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(self, python_path: str | None = None, evidence_k: float = 5, statements=metta):
        self.statements = statements
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
        for head in self.statements.complete_predicates(network):
            self._handler.set_complete_predicate(head)
        self._handler.add_atoms_no_check(self.statements.rules(network, rates))
        for period, observation in history:
            self._handler.add_atoms_no_check(self.statements.observation_facts(observation))
            self.resolve(period, observation)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self._handler.add_atoms_no_check(self.statements.observation_facts(observation))
        goals = [self.statements.query(key) for key in keys]
        answers = self._handler.query_many(goals, steps=budget, timeout_sec=0) if goals else []
        beliefs = {}
        for key, proofs in zip(keys, answers, strict=True):
            strength = self._strength(proofs)
            if strength is not None:
                beliefs[key] = strength
        return beliefs

    def resolve(self, period: Period, observation: Observation) -> None:
        self._handler.add_atoms_no_check(self.statements.resolution_facts(period, observation))

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
