"""Reasoners: the exact posterior, Python combinations of the marginal rules
(accuracy ceilings without chainer cost), and PeTTaChainer with or without
combination hypotheses."""

from __future__ import annotations

import importlib
import math
import re
import sys
import time
from pathlib import Path

from . import metta
from .world import Case, Consequent, Marginals, Observed

MODES = ("revision", "odds", "noisy-or")
REFERENCES = ("base-rate", *MODES, "mode", "cells", "cell-mode")


class BackendUnavailable(RuntimeError):
    pass


def views(marginals: Marginals, consequent: str, observed: Observed) -> list[float]:
    """Each observed parent's rule applied to it: P(C|A_i) or P(C|not A_i)."""
    return [marginals.given[consequent][i][0 if value else 1] for i, value in sorted(observed.items())]


def combine(mode: str, base: float, views: list[float]) -> float:
    """``revision``: the views' mean, as revision of equally confident views.
    ``odds``: the prior odds times each view's likelihood ratio, exact for
    signs independent given C. ``noisy-or``: the prior's survival 1 - P(C)
    times each view's survival ratio, exact for independent noisy-OR causes."""
    if mode == "revision":
        return sum(views) / len(views) if views else base
    if mode == "odds":
        odds = base / (1 - base) * math.prod(v / (1 - v) / (base / (1 - base)) for v in views)
        return odds / (1 + odds)
    if mode == "noisy-or":
        return min(1.0, max(0.0, 1 - (1 - base) * math.prod((1 - v) / (1 - base) for v in views)))
    raise ValueError(mode)


def _log_likelihood(p: float, actual: bool) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p if actual else 1 - p)


def instances(history: list[Case], consequent: str, pattern) -> list[bool]:
    """The labels of the history cases where every literal of ``pattern``
    was observed to hold: a combination hypothesis's samples."""
    return [case.truth[consequent][1] for case in history if all(case.observed[consequent].get(i) == v for i, v in pattern)]


class ExactBackend:
    """The posterior under the true model."""

    name = "exact"

    def begin(self, problem: tuple[Consequent, ...], marginals: Marginals, history: list[Case]) -> None:
        self.models = {c.name: c.model for c in problem}

    def beliefs(self, cases: list[Case], keys: list[tuple[str, str]], budget: int) -> dict:
        by_name = {case.name: case for case in cases}
        return {key: self.models[key[0]].posterior(by_name[key[1]].observed[key[0]]) for key in keys}

    def resolve(self, cases: list[Case]) -> None:
        pass

    def stats(self) -> dict:
        return {}


class CombinationBackend:
    """The marginal rules combined in Python.

    - ``base-rate``: C's frequency in the labelled history, ignoring the parents;
    - ``revision``, ``odds``, ``noisy-or``: one combination for every consequent;
    - ``mode``: per consequent, the one of those three with the highest
      likelihood of the labelled history, re-chosen each round;
    - ``cells``: C's frequency among the history cases matching every observed
      parent of the case, smoothed towards revision with ``pseudo`` cases:
      what a per-combination hypothesis over exactly those parents converges to;
    - ``cell-mode``: the same, smoothed towards the learned mode."""

    def __init__(self, rule: str, pseudo: float = 2.0):
        self.name = rule
        self.pseudo = pseudo

    def begin(self, problem: tuple[Consequent, ...], marginals: Marginals, history: list[Case]) -> None:
        self.problem, self.marginals, self.history = problem, marginals, list(history)
        self.modes: dict[str, str] = {}

    def beliefs(self, cases: list[Case], keys: list[tuple[str, str]], budget: int) -> dict:
        if self.name in ("mode", "cell-mode"):
            self.modes = {c.name: self._learned_mode(c.name) for c in self.problem}
        by_name = {case.name: case for case in cases}
        return {key: self._belief(key[0], by_name[key[1]].observed[key[0]]) for key in keys}

    def _belief(self, consequent: str, observed: Observed) -> float:
        if self.name == "base-rate":
            labels = [case.truth[consequent][1] for case in self.history]
            return (sum(labels) + 1) / (len(labels) + 2)
        if self.name in ("mode", "cell-mode"):
            mode = self.modes[consequent]
        else:
            mode = "revision" if self.name == "cells" else self.name
        prior = combine(mode, self.marginals.base[consequent], views(self.marginals, consequent, observed))
        if self.name not in ("cells", "cell-mode"):
            return prior
        labels = instances(self.history, consequent, observed.items())
        return (sum(labels) + self.pseudo * prior) / (len(labels) + self.pseudo)

    def _learned_mode(self, consequent: str) -> str:
        """Ties keep revision, the first mode."""
        base = self.marginals.base[consequent]

        def likelihood(mode):
            return sum(
                _log_likelihood(combine(mode, base, views(self.marginals, consequent, case.observed[consequent])), case.truth[consequent][1])
                for case in self.history
            )

        return max(MODES, key=likelihood)

    def resolve(self, cases: list[Case]) -> None:
        self.history.extend(cases)

    def stats(self) -> dict:
        return {"modes": {c.relation + ":" + c.name: self.modes[c.name] for c in self.problem}} if self.modes else {}


class PeTTaChainerBackend:
    """PeTTaChainer given the marginal rules, the history as facts and the
    cases' observed parents.

    With ``hypotheses``, the backend also writes per-consequent combination
    hypotheses (``docs/metta/hypothesis_rules.md`` in PeTTaChainer): refined
    rules ``(Implication (And lit ...) (C $x))`` with a weak prior at the
    revision of their literals' views, learned by the instance fold. Before
    each round's queries it reviews them all, each by its own ``RuleTruth``
    query with ``review_steps``: in one shared batch the first goals take the
    budget and the later ones are not folded.
    Candidates, each kept once it has ``min_support`` instances in the
    labelled history, re-checked every round:

    - ``pairs``: every two observed parents of a history case, with their values;
    - ``cells``: the full observed pattern of a history case (two or more
      parents, at most ``max_literals``).

    ``cells-read`` is ``cells`` with the merge's part done by the backend:
    where the review gave a case's exact cell a learned truth (confidence
    above the prior's), the belief is that truth instead of the chainer's
    answer. It measures what the instance fold learns, apart from how the
    merge uses it.

    ``form`` writes a negative literal as ``(NotA $x)`` over complement facts
    (``complement``) or as ``(Not (A $x))`` (``not``)."""

    _stv_re = re.compile(
        r"\(STV\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
    )

    def __init__(
        self,
        python_path: str | None = None,
        evidence_k: float = 5,
        hypotheses: str | None = None,
        form: str = "complement",
        min_support: int = 5,
        rule_confidence: float = 1.0,
        review_steps: int = 20,
        max_literals: int | None = None,
    ):
        if python_path:
            sys.path.insert(0, str(Path(python_path).expanduser().resolve()))
        try:
            module = importlib.import_module("pettachainer")
            self._handler = module.PeTTaChainer()
        except (AttributeError, ImportError, OSError) as exc:
            raise BackendUnavailable(
                "PeTTaChainer is not importable; pass --pettachainer-path or set PETTACHAINER_PYTHONPATH"
            ) from exc
        self.name = "pettachainer" + (f"-{hypotheses}" if hypotheses else "")
        self.read = hypotheses == "cells-read"
        self.hypotheses_mode = "cells" if self.read else hypotheses
        self.form, self.min_support = form, min_support
        self.rule_confidence, self.review_steps, self.max_literals = rule_confidence, review_steps, max_literals
        self._handler.set_evidence_confidence_k(evidence_k)
        self._handler.set_query_metrics(True)
        if hypotheses:
            self._handler.set_rule_refinement(True)
        self.counters = {"rules": 0, "hypotheses": 0, "folds": 0, "expansions": 0, "skeleton_nodes": 0, "review_seconds": 0.0, "query_seconds": 0.0}

    def begin(self, problem: tuple[Consequent, ...], marginals: Marginals, history: list[Case]) -> None:
        self.problem, self.marginals, self.history = problem, marginals, list(history)
        statements = metta.rules(problem, marginals, self.rule_confidence)
        self.counters["rules"] = len(statements)
        for case in history:
            statements += metta.case_facts(problem, case, self._complement) + metta.label_facts(problem, case)
        self._handler.add_atoms_no_check(statements)
        self.patterns: dict[str, set] = {c.name: set() for c in problem}
        self.learned: dict[tuple, float] = {}
        self._add_hypotheses()

    @property
    def _complement(self) -> bool:
        return bool(self.hypotheses_mode) and self.form == "complement"

    def _add_hypotheses(self) -> None:
        if not self.hypotheses_mode:
            return
        statements = []
        for consequent in self.problem:
            candidates = set()
            for case in self.history:
                observed = tuple(sorted(case.observed[consequent.name].items()))
                if self.hypotheses_mode == "pairs":
                    candidates.update((a, b) for a in observed for b in observed if a[0] < b[0])
                elif 2 <= len(observed) <= (self.max_literals or len(observed)):
                    candidates.add(observed)
            for pattern in sorted(candidates - self.patterns[consequent.name]):
                if len(instances(self.history, consequent.name, pattern)) >= self.min_support:
                    self.patterns[consequent.name].add(pattern)
                    prior = combine("revision", self.marginals.base[consequent.name], views(self.marginals, consequent.name, dict(pattern)))
                    statements.append(metta.hypothesis(consequent, pattern, prior, self.form))
        if statements:
            self._handler.add_atoms_no_check(statements)
            self.counters["hypotheses"] += len(statements)

    def _query(self, goals: list[str], steps: int, clock: str) -> list:
        started = time.perf_counter()
        answers = self._handler.query_many(goals, steps=steps, timeout_sec=0)
        self.counters[clock] += time.perf_counter() - started
        metrics = self._handler.last_query_metrics()
        self.counters["expansions"] += metrics.get("expansions", 0)
        self.counters["skeleton_nodes"] = metrics.get("skeleton_nodes_after", 0)
        return answers

    def beliefs(self, cases: list[Case], keys: list[tuple[str, str]], budget: int) -> dict:
        self._handler.add_atoms_no_check([line for case in cases for line in metta.case_facts(self.problem, case, self._complement)])
        cells = [(c, pattern) for c in self.problem for pattern in sorted(self.patterns[c.name])]
        if cells:
            self.counters["folds"] += len(cells)
            for c, pattern in cells:
                truth = self._truth(self._query([metta.review(c, pattern, self.form)], self.review_steps, "review_seconds")[0])
                if truth and truth[1] > 0.05:
                    self.learned[(c.name, pattern)] = truth[0]
        answers = self._query([metta.query(key) for key in keys], budget * len(keys), "query_seconds")
        by_name = {case.name: case for case in cases}
        beliefs = {}
        for key, proofs in zip(keys, answers, strict=True):
            cell = (key[0], tuple(sorted(by_name[key[1]].observed[key[0]].items())))
            truth = self._truth(proofs)
            if self.read and cell in self.learned:
                beliefs[key] = self.learned[cell]
            elif truth:
                beliefs[key] = truth[0]
        return beliefs

    def resolve(self, cases: list[Case]) -> None:
        self._handler.add_atoms_no_check([line for case in cases for line in metta.label_facts(self.problem, case)])
        self.history.extend(cases)
        self._add_hypotheses()

    def stats(self) -> dict:
        return {key: round(value, 3) for key, value in self.counters.items()}

    @classmethod
    def _truth(cls, proofs) -> tuple[float, float] | None:
        """The most confident answer's (strength, confidence); an answer's own
        truth value is its last."""
        best = None
        for proof in proofs or ():
            matches = cls._stv_re.findall(str(proof))
            if matches:
                strength, confidence = float(matches[-1][0]), float(matches[-1][1])
                if best is None or confidence > best[1]:
                    best = (min(1.0, max(0.0, strength)), confidence)
        return best
