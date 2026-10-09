"""Consequents, their parents and cases: the generating models, the exact
posterior of a consequent given its observed parents, and the marginal rates
a reasoner is given.

Each consequent C has n parent predicates A_1..A_n and one of four relations:

- ``signs``: C from its base rate, then each A_i independently given C;
- ``causes``: the A_i independent, C a noisy-OR of a leak and the true A_i;
- ``redundant``: the A_i noisy copies of one latent L, and C depends on L;
- ``interacting``: the A_i independent, C logistic in main effects and
  antagonistic pairs (A_1 with A_2, A_3 with A_4, ...): each raises C alone,
  together they lower it.

A case is an entity with every predicate of every consequent; each parent is
observed with the observation rate, independently, else unknown."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

RELATIONS = ("signs", "causes", "redundant", "interacting")

Observed = dict[int, bool]  # parent index -> value, for the observed parents only


def _odds(p: float) -> float:
    return p / (1 - p)


@dataclass(frozen=True)
class Signs:
    base: float
    given_c: tuple[float, ...]
    given_not_c: tuple[float, ...]

    def sample(self, rng: random.Random) -> tuple[tuple[bool, ...], bool]:
        c = rng.random() < self.base
        return tuple(rng.random() < (t if c else f) for t, f in zip(self.given_c, self.given_not_c)), c

    def posterior(self, observed: Observed) -> float:
        odds = _odds(self.base)
        for i, value in observed.items():
            t, f = self.given_c[i], self.given_not_c[i]
            odds *= t / f if value else (1 - t) / (1 - f)
        return odds / (1 + odds)


@dataclass(frozen=True)
class Causes:
    leak: float
    prior: tuple[float, ...]
    weight: tuple[float, ...]

    def sample(self, rng: random.Random) -> tuple[tuple[bool, ...], bool]:
        parents = tuple(rng.random() < p for p in self.prior)
        survive = (1 - self.leak) * math.prod(1 - w for a, w in zip(parents, self.weight) if a)
        return parents, rng.random() >= survive

    def posterior(self, observed: Observed) -> float:
        survive = 1 - self.leak
        for i, (p, w) in enumerate(zip(self.prior, self.weight)):
            survive *= (1 - w if observed[i] else 1) if i in observed else 1 - p * w
        return 1 - survive


@dataclass(frozen=True)
class Redundant:
    latent: float
    fidelity: tuple[float, ...]
    given_l: float
    given_not_l: float

    def sample(self, rng: random.Random) -> tuple[tuple[bool, ...], bool]:
        latent = rng.random() < self.latent
        parents = tuple(latent if rng.random() < f else not latent for f in self.fidelity)
        return parents, rng.random() < (self.given_l if latent else self.given_not_l)

    def posterior(self, observed: Observed) -> float:
        def weight(latent: bool) -> float:
            return math.prod(self.fidelity[i] if value == latent else 1 - self.fidelity[i] for i, value in observed.items())

        on = self.latent * weight(True)
        off = (1 - self.latent) * weight(False)
        return (on * self.given_l + off * self.given_not_l) / (on + off)


@dataclass(frozen=True)
class Interacting:
    prior: tuple[float, ...]
    bias: float
    effect: tuple[float, ...]
    interaction: tuple[tuple[int, int, float], ...]

    def chance(self, parents) -> float:
        logit = self.bias + sum(e for a, e in zip(parents, self.effect) if a)
        logit += sum(g for i, j, g in self.interaction if parents[i] and parents[j])
        return 1 / (1 + math.exp(-logit))

    def sample(self, rng: random.Random) -> tuple[tuple[bool, ...], bool]:
        parents = tuple(rng.random() < p for p in self.prior)
        return parents, rng.random() < self.chance(parents)

    def posterior(self, observed: Observed) -> float:
        """Sums the unknown parents out; they are independent of the observed ones."""
        unknown = [i for i in range(len(self.prior)) if i not in observed]
        total = 0.0
        for values in itertools.product((True, False), repeat=len(unknown)):
            parents = observed | dict(zip(unknown, values))
            weight = math.prod(self.prior[i] if v else 1 - self.prior[i] for i, v in zip(unknown, values))
            total += weight * self.chance([parents[i] for i in range(len(self.prior))])
        return total


Model = Signs | Causes | Redundant | Interacting


@dataclass(frozen=True)
class Consequent:
    name: str
    relation: str
    parents: tuple[str, ...]
    model: Model


def draw_model(relation: str, n: int, rng: random.Random) -> Model:
    def draws(low, high):
        return tuple(rng.uniform(low, high) for _ in range(n))

    if relation == "signs":
        return Signs(rng.uniform(0.2, 0.4), draws(0.55, 0.85), draws(0.1, 0.35))
    if relation == "causes":
        return Causes(rng.uniform(0.05, 0.15), draws(0.2, 0.5), draws(0.4, 0.8))
    if relation == "redundant":
        return Redundant(rng.uniform(0.3, 0.5), draws(0.85, 0.95), rng.uniform(0.7, 0.9), rng.uniform(0.05, 0.2))
    if relation == "interacting":
        pairs = tuple((i, i + 1, -rng.uniform(2.5, 4.0)) for i in range(0, n - 1, 2))
        return Interacting(draws(0.3, 0.6), rng.uniform(-2.0, -1.0), draws(1.0, 2.0), pairs)
    raise ValueError(relation)


def generate_problem(rng: random.Random, parents: int, relations: tuple[str, ...], per_relation: int = 1) -> tuple[Consequent, ...]:
    """``per_relation`` consequents of each relation, each with its own
    ``parents`` parent predicates."""
    consequents = []
    for relation in relations:
        for _ in range(per_relation):
            j = len(consequents) + 1
            consequents.append(
                Consequent(f"C{j}", relation, tuple(f"A{j}_{i + 1}" for i in range(parents)), draw_model(relation, parents, rng))
            )
    return tuple(consequents)


@dataclass(frozen=True)
class Case:
    """An entity: per consequent, its parents' and its own true values, and
    the parents observed."""

    name: str
    truth: dict[str, tuple[tuple[bool, ...], bool]]
    observed: dict[str, Observed]


def sample_case(problem: tuple[Consequent, ...], rng: random.Random, name: str, observe_rate: float) -> Case:
    truth, observed = {}, {}
    for consequent in problem:
        parents, c = consequent.model.sample(rng)
        truth[consequent.name] = (parents, c)
        observed[consequent.name] = {i: a for i, a in enumerate(parents) if rng.random() < observe_rate}
    return Case(name, truth, observed)


@dataclass(frozen=True)
class Marginals:
    """What the rules state per consequent: P(C), and P(C|A_i), P(C|not A_i)
    per parent."""

    base: dict[str, float]
    given: dict[str, tuple[tuple[float, float], ...]]


def true_marginals(problem: tuple[Consequent, ...]) -> Marginals:
    return Marginals(
        {c.name: c.model.posterior({}) for c in problem},
        {c.name: tuple((c.model.posterior({i: True}), c.model.posterior({i: False})) for i in range(len(c.parents))) for c in problem},
    )


def learned_marginals(problem: tuple[Consequent, ...], history: list[Case]) -> Marginals:
    """Laplace-smoothed frequencies in the labelled history, each over the
    cases where the parent was observed."""

    def rate(cases):
        return (sum(1 for case in cases if case[1]) + 1) / (len(cases) + 2)

    base, given = {}, {}
    for c in problem:
        labelled = [(case.observed[c.name], case.truth[c.name][1]) for case in history]
        base[c.name] = rate(labelled)
        given[c.name] = tuple(
            (rate([x for x in labelled if x[0].get(i) is True]), rate([x for x in labelled if x[0].get(i) is False]))
            for i in range(len(c.parents))
        )
    return Marginals(base, given)
