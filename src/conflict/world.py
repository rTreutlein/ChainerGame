"""The hidden world, its sources and their reports, and the exact posterior.

Each round is a fresh day of one plant:

- **Components** ``c1..cK`` are up independently, each with its prior.
- **Systems** ``w1..wS`` each depend on two components: P(up | both up) is
  high, P(up | not both up) low. These rates are given to every reasoner, as
  SupplyNet's rules are.
- **Sources** report on nodes (components and systems). Source s reports on a
  node with probability ``coverage``; its claim is "up" with probability
  ``given_up`` when the node is up and ``given_down`` when it is down, so a
  source can be accurate, noisy, biased (optimist, alarmist) or inverted
  (contrarian). It repeats its claim ``repeats`` times: an echo is loud, not
  informed, so the copies carry no evidence beyond the first.

Volume differs by orders of magnitude: a crowd source reports on most nodes
every round, an echo repeats each claim ten times, a rare source reports a few
times in the whole game. After a round is scored its states are revealed and it
joins the labelled history, from which reliabilities can be learned.

The exact posterior of every node, given the round's claims (one per source
and node), sums over the 2^K component states; the systems and claims factor
given them."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

# Archetypes: (kind, P(claim up | up), P(claim up | down), coverage, repeats), each a range.
ARCHETYPES = {
    "expert": ((0.88, 0.95), (0.05, 0.12), (0.10, 0.20), (1, 1)),
    "crowd": ((0.60, 0.70), (0.30, 0.40), (0.50, 0.70), (1, 1)),
    "optimist": ((0.93, 0.99), (0.50, 0.65), (0.40, 0.60), (2, 4)),
    "contrarian": ((0.10, 0.25), (0.75, 0.90), (0.20, 0.40), (1, 1)),
    "echo": ((0.55, 0.70), (0.40, 0.50), (0.70, 0.90), (8, 15)),
    "alarmist": ((0.35, 0.50), (0.02, 0.08), (0.30, 0.50), (1, 2)),
    "rare": ((0.85, 0.95), (0.05, 0.15), (0.005, 0.02), (1, 1)),
}
# The order sources are drawn in: a size with n sources takes the first n.
ROSTER = ("expert", "crowd", "optimist", "contrarian", "echo", "alarmist", "rare", "crowd",
          "expert", "optimist", "crowd", "contrarian", "echo", "alarmist", "rare", "crowd")


@dataclass(frozen=True)
class Size:
    components: int
    systems: int
    sources: int


SIZES = {
    "s": Size(2, 2, 5),
    "m": Size(4, 4, 8),
    "l": Size(8, 8, 14),
    "xl": Size(16, 16, 16),
}


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    given_up: float
    given_down: float
    coverage: float
    repeats: int


@dataclass(frozen=True)
class System:
    name: str
    parents: tuple[str, str]
    given_both: float  # P(up | both parents up)
    given_not: float  # P(up | not both)


@dataclass(frozen=True)
class World:
    priors: dict[str, float]  # component -> P(up)
    systems: tuple[System, ...]
    sources: tuple[Source, ...]

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.priors)

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.components + tuple(s.name for s in self.systems)


def generate_world(rng: random.Random, size: Size) -> World:
    priors = {f"c{i + 1}": rng.uniform(0.55, 0.9) for i in range(size.components)}
    systems = tuple(
        System(f"w{j + 1}", tuple(sorted(rng.sample(sorted(priors), 2), key=lambda c: int(c[1:]))), rng.uniform(0.85, 0.97), rng.uniform(0.03, 0.15))
        for j in range(size.systems)
    )
    sources = []
    for index, kind in enumerate(ROSTER[: size.sources]):
        up, down, coverage, repeats = ARCHETYPES[kind]
        sources.append(Source(f"s{index + 1}", kind, rng.uniform(*up), rng.uniform(*down), rng.uniform(*coverage), rng.randint(*repeats)))
    return World(priors, systems, tuple(sources))


@dataclass(frozen=True)
class Report:
    source: str
    node: str
    up: bool  # the claim
    copies: int


@dataclass(frozen=True)
class Round:
    name: str
    truth: dict[str, bool]
    reports: tuple[Report, ...]  # in arrival order (random)


def sample_round(world: World, rng: random.Random, name: str) -> Round:
    truth = {c: rng.random() < p for c, p in world.priors.items()}
    for system in world.systems:
        both = all(truth[p] for p in system.parents)
        truth[system.name] = rng.random() < (system.given_both if both else system.given_not)
    reports = []
    for source in world.sources:
        for node in world.nodes:
            if rng.random() < source.coverage:
                claim = rng.random() < (source.given_up if truth[node] else source.given_down)
                reports.append(Report(source.name, node, claim, source.repeats))
    rng.shuffle(reports)
    return Round(name, truth, tuple(reports))


@dataclass(frozen=True)
class Rates:
    """What a model of the sources needs: component priors and each source's
    P(claim up | up), P(claim up | down). System rates are always the true ones."""

    priors: dict[str, float]
    reliability: dict[str, tuple[float, float]]


def true_rates(world: World) -> Rates:
    return Rates(dict(world.priors), {s.name: (s.given_up, s.given_down) for s in world.sources})


def learned_rates(world: World, history: list[Round]) -> Rates:
    """Laplace estimates from the labelled rounds: each component's frequency
    of being up, and each source's frequency of claiming up among the nodes it
    reported on that were up (were down). One claim per source and node:
    copies are not counted."""
    priors = {c: (sum(r.truth[c] for r in history) + 1) / (len(history) + 2) for c in world.priors}
    counts = {s.name: [0, 0, 0, 0] for s in world.sources}  # claimed up when up, up total, claimed up when down, down total
    for past in history:
        for report in past.reports:
            count = counts[report.source]
            offset = 0 if past.truth[report.node] else 2
            count[offset] += report.up
            count[offset + 1] += 1
    reliability = {name: ((c[0] + 1) / (c[1] + 2), (c[2] + 1) / (c[3] + 2)) for name, c in counts.items()}
    return Rates(priors, reliability)


def posterior(world: World, rates: Rates, reports: tuple[Report, ...]) -> dict[str, float]:
    """P(node up | claims) for every node, summing over the component states."""
    likelihood = {node: [1.0, 1.0] for node in world.nodes}  # P(claims on node | up), P(claims | down)
    for report in reports:
        up, down = rates.reliability[report.source]
        lk = likelihood[report.node]
        lk[0] *= up if report.up else 1 - up
        lk[1] *= down if report.up else 1 - down
    components = world.components
    total = 0.0
    on = dict.fromkeys(world.nodes, 0.0)
    for states in itertools.product((True, False), repeat=len(components)):
        state = dict(zip(components, states))
        weight = math.prod(
            (rates.priors[c] * likelihood[c][0]) if value else ((1 - rates.priors[c]) * likelihood[c][1]) for c, value in state.items()
        )
        if weight == 0.0:
            continue
        system_up = {}
        for system in world.systems:
            rate = system.given_both if all(state[parent] for parent in system.parents) else system.given_not
            lk = likelihood[system.name]
            up, down = rate * lk[0], (1 - rate) * lk[1]
            weight *= up + down
            system_up[system.name] = up / (up + down)
        total += weight
        for c, value in state.items():
            if value:
                on[c] += weight
        for name, share in system_up.items():
            on[name] += weight * share
    return {node: on[node] / total for node in world.nodes}


def contested(reports: tuple[Report, ...]) -> set[str]:
    """Nodes on which two sources' claims disagree."""
    claims: dict[str, set[bool]] = {}
    for report in reports:
        claims.setdefault(report.node, set()).add(report.up)
    return {node for node, values in claims.items() if len(values) == 2}
