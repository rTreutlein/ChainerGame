"""Reasoners: the exact posterior (true or learned rates), Python baselines,
and PeTTaChainer under the raw and the sources encoding.

A backend gets ``begin(world, history)``, then per round
``beliefs(day, keys, budget)`` with the round's claims (its states hidden) and
``resolve(day)`` with the revealed round."""

from __future__ import annotations

import importlib
import re
import sys
import time
from pathlib import Path

from . import metta
from .world import Round, World, learned_rates, posterior, true_rates

REFERENCES = ("exact", "exact-learned", "prior", "vote", "last-wins", "loudest")


class BackendUnavailable(RuntimeError):
    pass


def _laplace(values) -> float:
    values = list(values)
    return (sum(values) + 1) / (len(values) + 2)


class ReferenceBackend:
    """Python reasoners over the same claims.

    - ``exact``: the posterior under the true model;
    - ``exact-learned``: the same model with the component priors and each
      source's two rates estimated from the labelled rounds (Laplace), as
      ProbLog learned is;
    - ``prior``: each node's frequency of being up in the labelled rounds;
    - ``vote``: the share of claim copies saying up, the prior when no source
      spoke: count-weighted revision, what the raw encoding gives at best;
    - ``last-wins``: the last claim to arrive, read with the pooled share of
      correct claims in the history;
    - ``loudest``: only the source with the most claim copies in the history,
      read with its own share of correct claims; the prior when it is silent."""

    def __init__(self, name: str):
        assert name in REFERENCES, name
        self.name = name

    def begin(self, world: World, history: list[Round]) -> None:
        self.world, self.history = world, list(history)

    def beliefs(self, day: Round, keys: list[tuple[str, str]], budget: int) -> dict:
        if self.name in ("exact", "exact-learned"):
            rates = true_rates(self.world) if self.name == "exact" else learned_rates(self.world, self.history)
            exact = posterior(self.world, rates, day.reports)
            return {key: exact[key[0]] for key in keys}
        prior = {node: _laplace(r.truth[node] for r in self.history) for node in self.world.nodes}
        if self.name == "prior":
            return {key: prior[key[0]] for key in keys}
        if self.name == "vote":
            beliefs = {}
            for node, name in keys:
                copies = [(r.copies, r.up) for r in day.reports if r.node == node]
                total = sum(n for n, _ in copies)
                beliefs[(node, name)] = sum(n for n, up in copies if up) / total if total else prior[node]
            return beliefs
        if self.name == "last-wins":
            accuracy = _laplace(r.up == past.truth[r.node] for past in self.history for r in past.reports)
            last = {r.node: r.up for r in day.reports}
            return {key: (accuracy if last[key[0]] else 1 - accuracy) if key[0] in last else prior[key[0]] for key in keys}
        volume: dict[str, int] = {}
        for past in self.history:
            for r in past.reports:
                volume[r.source] = volume.get(r.source, 0) + r.copies
        loudest = max(sorted(volume), key=volume.get)
        accuracy = _laplace(r.up == past.truth[r.node] for past in self.history for r in past.reports if r.source == loudest)
        claim = {r.node: r.up for r in day.reports if r.source == loudest}
        return {key: (accuracy if claim[key[0]] else 1 - accuracy) if key[0] in claim else prior[key[0]] for key in keys}

    def resolve(self, day: Round) -> None:
        self.history.append(day)

    def stats(self) -> dict:
        return {}


_number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_stv_re = re.compile(rf"\(STV\s+{_number}\s+{_number}\)")
_ctv_re = re.compile(rf"\(CTV\s+\(STV\s+{_number}\s+{_number}\)\s+\(STV\s+{_number}\s+{_number}\)\)\)\s*$")


def strongest(proofs) -> tuple[float, float] | None:
    """The most confident answer's (strength, confidence); an answer's own
    truth value is its last."""
    best = None
    for proof in proofs or ():
        matches = _stv_re.findall(str(proof))
        if matches:
            strength, confidence = float(matches[-1][0]), float(matches[-1][1])
            if best is None or confidence > best[1]:
                best = (min(1.0, max(0.0, strength)), confidence)
    return best


class PeTTaChainerBackend:
    """PeTTaChainer with the systems' rules and the labelled rounds as facts.

    - ``raw``: each source's claims on a node as one fact about the node, its
      copy count as evidence (``metta.raw_facts``);
    - ``sources``: claims as ``(Claims s (Up n t))`` facts and one refined
      trust rule per source (``metta.trust_rules``). Before a round's queries
      each trust rule is reviewed by its own ``RuleTruth`` query of
      ``review_steps`` (in one shared batch the first reviews take the budget,
      ChainerGame ``docs/combination_bench.md``), so its truth folds the
      labelled claims, including the last round's;
    - ``stated``: the same claims, but each source's CTV is learned once, from
      the initial history, by an implication query in a KB of its own, and
      stated as a given rate (``_learn_trust``)."""

    def __init__(self, encoding: str, python_path: str | None = None, evidence_k: float = 5, review_steps: int = 20):
        assert encoding in ("raw", "sources", "stated"), encoding
        if python_path:
            sys.path.insert(0, str(Path(python_path).expanduser().resolve()))
        try:
            module = importlib.import_module("pettachainer")
            self._handler = module.PeTTaChainer()
        except (AttributeError, ImportError, OSError) as exc:
            raise BackendUnavailable("PeTTaChainer is not importable; pass --pettachainer-path or set PETTACHAINER_PYTHONPATH") from exc
        self.name = f"pln-{encoding}"
        self.encoding, self.evidence_k, self.review_steps = encoding, evidence_k, review_steps
        self._handler.set_evidence_confidence_k(evidence_k)
        self._handler.set_query_metrics(True)
        if encoding == "sources":
            self._handler.set_rule_refinement(True)
        self.counters = {"statements": 0, "expansions": 0, "review_seconds": 0.0, "query_seconds": 0.0}
        self.trust: dict[str, list[float]] = {}

    def _add(self, statements: list[str]) -> None:
        if statements:  # PeTTa rejects an empty batch
            self.counters["statements"] += len(statements)
            self._handler.add_atoms_no_check(statements)

    def begin(self, world: World, history: list[Round]) -> None:
        self.world = world
        statements = metta.system_rules(world)
        if self.encoding == "sources":
            statements += metta.trust_rules(world)
        elif self.encoding == "stated":
            statements += self._learn_trust(world, history)
        for day in history:
            statements += (metta.claims(day.reports, day.name) if self.encoding != "raw" else []) + metta.labels(day)
        self._add(statements)

    def _learn_trust(self, world: World, history: list[Round]) -> list[str]:
        """Each source's CTV from an implication query over the labelled claims,
        in a KB of its own, stated as a given rate (confidence 1). Stated
        before the history, so the history's outcomes score the combination
        modes; at a confidence below 1 two applications of one rule to
        different claims share the rule's evidence, and the merge keeps only
        the more confident."""
        learner = type(self._handler)()
        statements = [line for day in history for line in metta.claims(day.reports, day.name) + metta.labels(day)]
        learner.add_atoms_no_check(statements)
        rules = []
        for source in world.sources:
            started = time.perf_counter()
            (proofs,) = learner.query_many([f"(: $prf {metta.trust_implication(source.name)} $tv)"], steps=self.review_steps, timeout_sec=0)
            self.counters["review_seconds"] += time.perf_counter() - started
            match = _ctv_re.search(proofs[0]) if proofs else None
            if match:  # a source with no labelled claims gets no rule: its claims are ignored
                tv = [float(v) for v in match.groups()]
                self.trust[source.name] = [round(v, 4) for v in tv]
                rules.append(f"(: trust-{source.name} {metta.trust_implication(source.name)} (CTV (STV {tv[0]:.6g} 1) (STV {tv[2]:.6g} 1)))")
        return rules

    def _query(self, goals: list[str], steps: int, clock: str) -> list:
        started = time.perf_counter()
        answers = self._handler.query_many(goals, steps=steps, timeout_sec=0)
        self.counters[clock] += time.perf_counter() - started
        self.counters["expansions"] += self._handler.last_query_metrics().get("expansions", 0)
        return answers

    def beliefs(self, day: Round, keys: list[tuple[str, str]], budget: int) -> dict:
        if self.encoding == "raw":
            self._add(metta.raw_facts(day.reports, day.name, self.evidence_k))
        else:
            self._add(metta.claims(day.reports, day.name))
        if self.encoding == "sources":
            for source in self.world.sources:
                (proofs,) = self._query([metta.review(source.name)], self.review_steps, "review_seconds")
                match = _ctv_re.search(proofs[0]) if proofs else None
                if match:
                    self.trust[source.name] = [round(float(v), 4) for v in match.groups()]
        answers = self._query([metta.query(key) for key in keys], budget, "query_seconds")
        beliefs = {}
        for key, proofs in zip(keys, answers, strict=True):
            truth = strongest(proofs)
            if truth:
                beliefs[key] = truth[0]
        return beliefs

    def resolve(self, day: Round) -> None:
        self._add(metta.labels(day))

    def stats(self) -> dict:
        return {**{k: round(v, 3) for k, v in self.counters.items()}, **({"trust": self.trust} if self.trust else {})}
