"""Reasoners: the exact posterior, history base rates alone, and PeTTaChainer.

A backend is given a stage's network, rates, labelled history and rounds of
observations. Stages 1-3 use ``world`` and ``metta``; stage "scale" passes its
own knowledge class and statements (``scale``)."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from . import metta
from .world import Key, Knowledge, Network, NodeRates, Observation, Period, Rates, learned_rates


class BackendUnavailable(RuntimeError):
    pass


def exact(network: Network, rates: Rates) -> Knowledge:
    return Knowledge(network, rates.nodes(network))


class ReferenceBackend:
    """The posterior of a stage's ``knowledge`` class, exact by default,
    under the true rates."""

    def __init__(self, knowledge=exact, name: str = "reference"):
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


class LearnedReferenceBackend(ReferenceBackend):
    """The exact posterior under rates estimated from the labelled periods by
    Laplace's rule (``world.learned_rates``), re-estimated each round as
    periods resolve: what any learner of the rules' rates can reach from the
    same history, so it separates learning error from inference error."""

    def __init__(self):
        super().__init__(lambda network, rates: Knowledge(network, self.estimate(network)), "learned-reference")

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.periods, self.linked = [period for period, _ in history], rates.persist is not None
        super().begin(network, rates, history)

    def estimate(self, network: Network) -> NodeRates:
        return learned_rates(network, self.periods, self.linked)

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self.knowledge.rates = self.estimate(self.knowledge.network)
        return super().beliefs(observation, keys, budget)

    def resolve(self, period: Period, observation: Observation) -> None:
        self.periods.append(period)
        super().resolve(period, observation)


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
    world into MeTTa (``metta`` for stages 1-3).

    With ``learned`` (stages 1-3) the rules come with a weak prior in their
    rates' place and the knowledge base refines them from their instances
    (``set-rule-refinement``, PeTTaChainer ``docs/metta/hypothesis_rules.md``).
    Before the first round's queries each learned rule is reviewed by its own
    ``RuleTruth`` query of ``review_steps``, which folds the labelled history.
    From then on the chainer takes a resolved period's instances into each
    truth at its next query, whatever the query asks ("Instance tallies"):
    later rounds only read the truths, all in one query without budget,
    before the round's queries, so the search scores the rules with them. The
    reviews and reads are part of the round's time.

    With ``mined`` (``(max_new, max_live)``, with ``learned``) the learned
    rules are not stated at all: the knowledge base proposes its own
    hypotheses from the facts (PeTTaChainer ``set-hypothesis-mining``), and
    each round's review or read is its ``review-hypotheses`` of all live
    ones, which also drops those that do not predict."""

    name = "pettachainer"

    def __init__(self, python_path: str | None = None, evidence_k: float = 5, statements=metta, learned: bool = False, review_steps: int = 200,
                 mined: tuple[int, int] | None = None):
        self.statements, self.learned, self.review_steps, self.mined = statements, learned, review_steps, mined
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
        if learned:
            self._handler.set_rule_refinement(True)
        if mined:
            self._handler.set_hypothesis_mining(*mined)
        self.rule_truths: dict[str, list[float]] = {}  # learned rule's review -> its CTV
        self.hypotheses: dict[str, str] = {}  # mined: live hypothesis name -> its implication
        self.dropped = 0
        self.review_seconds = 0.0
        # builds before PeTTaChainer.timing() report no inference/overhead split
        self._clock = self._handler.timing() if hasattr(self._handler, "timing") else None

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        for head in self.statements.complete_predicates(network):
            self._handler.set_complete_predicate(head)
        rules = self.statements.rules(network, rates, learned=True) if self.learned else self.statements.rules(network, rates)
        self.reviews = [goal for goal in map(self.statements.review, rules) if goal] if self.learned else []
        if self.mined:
            for head in self.statements.positive_only_predicates(network):
                self._handler.set_positive_only_predicate(head)
            rules = [rule for rule in rules if not self.statements.review(rule)]
            self.reviews = ["$goal"]
        if rules:
            self._handler.add_atoms_no_check(rules)
        self.unreviewed = bool(self.reviews)
        self.unread = False
        for period, observation in history:
            self._handler.add_atoms_no_check(self.statements.observation_facts(observation))
            self.resolve(period, observation)

    def _review_mined(self) -> None:
        for verdict in self._handler.review_hypotheses("$goal", self.review_steps):
            name, rest = verdict[len("(hypothesis "):].split(" ", 1)
            implication, kept = rest.rsplit(" ", 1)
            if kept == "dropped)":
                self.dropped += 1
                self.hypotheses.pop(name, None)
            else:
                self.hypotheses[name] = implication

    def _record(self, goals: list[str], answers) -> None:
        for goal, root_answers in zip(goals, answers, strict=True):
            for answer in root_answers:
                if answer.tv[0] == "CTV":
                    self.rule_truths[goal] = [round(value, 4) for branch in answer.tv[1:] for value in branch[1:]]

    def _review(self) -> None:
        for goal in self.reviews:
            self._record([goal], self._handler.query_many_refs([goal], steps=self.review_steps))

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        if self.unreviewed or self.unread:
            started = time.perf_counter()
            if self.mined:
                self._review_mined()
            elif self.unreviewed:
                self._review()
            else:
                self._record(self.reviews, self._handler.query_many_refs(self.reviews, steps=0))
            self.review_seconds += time.perf_counter() - started
            self.unreviewed = self.unread = False
        self._handler.add_atoms_no_check(self.statements.observation_facts(observation))
        goals = [self.statements.query(key) for key in keys]
        answers = self._handler.query_many_refs(goals, steps=budget) if goals else []
        beliefs = {}
        for key, root_answers in zip(keys, answers, strict=True):
            strength = self._strength(root_answers)
            if strength is not None:
                beliefs[key] = strength
        return beliefs

    def resolve(self, period: Period, observation: Observation) -> None:
        self._handler.add_atoms_no_check(self.statements.resolution_facts(period, observation))
        self.unread = bool(self.reviews)

    def summary(self) -> dict:
        """Review figures of learned or mined rules, and the seconds of every
        chainer call of the run (setup included) split into inference
        (activation, search, evaluation) and overhead (the rest of the calls)."""
        if self.mined:
            rules = {"review_seconds": round(self.review_seconds, 3), "mined": list(self.mined), "dropped": self.dropped,
                     "hypotheses": sorted(self.hypotheses.values())}
        else:
            rules = {"review_seconds": round(self.review_seconds, 3), "rule_truths": self.rule_truths} if self.learned else {}
        if self._clock is None:
            return rules
        now = self._handler.timing()
        return {**rules, **{f"chainer_{part}_seconds": round(now[part] - self._clock[part], 3)
                            for part in ("wall", "inference", "overhead")}}

    @staticmethod
    def _strength(answers) -> float | None:
        """The strength of the most confident answer."""
        best = None
        for answer in answers:
            if answer.tv[0] == "STV":
                strength, confidence = answer.tv[1:]
                if best is None or confidence > best[1]:
                    best = (strength, confidence)
        return None if best is None else min(1.0, max(0.0, best[0]))
