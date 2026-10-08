"""SupplyNet world: regions with storms, routes a storm blocks, late shipments.

Regions are independent; within a region the storm is a common cause of its
routes' blocks, and a block is the common cause of its shipments' lateness.

Stage 1 has independent periods and shipments that arrive when they depart.
Stage 2 links periods: a storm persists with ``Rates.persist`` and starts with
the region kind's storm rate, and each route has a transit time, so a
shipment's lateness, decided by its route's block in its departure period, is
seen only when it arrives (docs/supply_network.md, stages 1 and 2).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

Key = tuple[str, str, str]  # (predicate, subject, period)


@dataclass(frozen=True)
class Rates:
    storm: dict[str, float] = field(default_factory=lambda: {"coastal": 0.3, "inland": 0.1})
    block_given_storm: dict[str, float] = field(default_factory=lambda: {"exposed": 0.8, "sheltered": 0.4})
    block_without_storm: float = 0.03
    late_given_blocked: float = 0.9
    late_given_open: float = 0.05
    persist: float | None = None  # None: storms are independent across periods

    def storm_rate(self, kind: str, previous: float) -> float:
        """P(storm now) given P(storm in the previous period)."""
        start = self.storm[kind]
        return start if self.persist is None else previous * self.persist + (1 - previous) * start

    def stationary(self, kind: str) -> float:
        start = self.storm[kind]
        return start if self.persist is None else start / (1 - self.persist + start)


@dataclass(frozen=True)
class Route:
    name: str
    region: str
    kind: str
    shipments: tuple[str, ...]
    transit: int = 0  # periods from departure to arrival


@dataclass(frozen=True)
class Network:
    regions: dict[str, str]  # region -> kind
    routes: tuple[Route, ...]

    def routes_of(self, region: str) -> list[Route]:
        return [route for route in self.routes if route.region == region]

    @property
    def max_transit(self) -> int:
        return max(route.transit for route in self.routes)


@dataclass(frozen=True)
class Period:
    name: str
    previous: str | None  # the linked previous period; None when periods are independent
    storms: dict[str, bool]
    blocked: dict[str, bool]
    late: dict[str, bool]  # shipments departing in this period


@dataclass(frozen=True)
class Observation:
    """What the operator sees in a period: the lateness of the shipments that
    arrive, keyed by (shipment, departure period), and the block state of the
    routes inspected in this period."""

    period: str
    previous: str | None
    late: dict[tuple[str, str], bool]
    inspected: dict[str, bool]


REGION_NAMES = ("north", "south", "east", "west", "central", "coast")


def generate_network(rng: random.Random, regions: int = 3, max_transit: int = 0) -> Network:
    region_kinds = {}
    routes = []
    for index in range(regions):
        region = REGION_NAMES[index] if index < len(REGION_NAMES) else f"region{index}"
        region_kinds[region] = rng.choice(("coastal", "inland"))
        for _ in range(rng.randint(2, 4)):
            number = len(routes) + 1
            shipments = tuple(f"s{number}-{k}" for k in range(1, rng.randint(1, 3) + 1))
            kind = rng.choice(("exposed", "sheltered"))
            transit = rng.randint(1, max_transit) if max_transit else 0
            routes.append(Route(f"r{number}", region, kind, shipments, transit))
    return Network(region_kinds, tuple(routes))


def sample_period(network: Network, rates: Rates, rng: random.Random, name: str, previous: Period | None) -> Period:
    """A period following ``previous`` (None for the first period, or when
    periods are independent)."""
    storms = {
        region: rng.random() < rates.storm_rate(kind, float(previous.storms[region]) if previous else rates.stationary(kind))
        for region, kind in network.regions.items()
    }
    blocked = {}
    late = {}
    for route in network.routes:
        p_block = rates.block_given_storm[route.kind] if storms[route.region] else rates.block_without_storm
        blocked[route.name] = rng.random() < p_block
        p_late = rates.late_given_blocked if blocked[route.name] else rates.late_given_open
        for shipment in route.shipments:
            late[shipment] = rng.random() < p_late
    return Period(name, previous.name if previous else None, storms, blocked, late)


def observe(network: Network, period: Period, rng: random.Random, inspect_rate: float, late: dict) -> Observation:
    inspected = {route.name: period.blocked[route.name] for route in network.routes if rng.random() < inspect_rate}
    return Observation(period.name, period.previous, late, inspected)


class Knowledge:
    """The exact posterior from what the operator knows: the storms of the
    last resolved period and the evidence about the unresolved periods since.

    Per region the storm is a hidden Markov chain over the unresolved window;
    the posterior enumerates the window's storm sequences (at most 2^(L+1)
    for the longest transit L) and sums each route-period's block out given
    that period's storm. The distribution carried into the window is the
    resolved period's storm, or the stationary rate before any period."""

    def __init__(self, network: Network, rates: Rates):
        self.network, self.rates = network, rates
        self.before = {region: rates.stationary(kind) for region, kind in network.regions.items()}
        self.window: list[str] = []
        self.late: dict[tuple[str, str], bool] = {}
        self.inspected: dict[tuple[str, str], bool] = {}

    def observe(self, observation: Observation) -> None:
        self.window.append(observation.period)
        self.late.update(observation.late)
        self.inspected.update({(route, observation.period): value for route, value in observation.inspected.items()})

    def resolve(self, period: Period) -> None:
        """The period, the oldest of the window, becomes labelled."""
        assert self.window[0] == period.name
        self.window.pop(0)
        self.before = {region: float(storm) for region, storm in period.storms.items()}
        self.late = {key: value for key, value in self.late.items() if key[1] != period.name}
        self.inspected = {key: value for key, value in self.inspected.items() if key[1] != period.name}

    def posterior(self) -> dict[Key, float]:
        """P(Storm region period) for every window period and P(Blocked route
        period) for every window route-period not inspected."""
        rates = self.rates
        posterior = {}
        for region, kind in self.network.regions.items():
            routes = self.network.routes_of(region)
            # Per (period, storm value): the likelihood of the period's route
            # evidence, and each route's block posterior given that storm.
            evidence = {}
            block_given = {}
            for period, storm in itertools.product(self.window, (True, False)):
                weight = 1.0
                for route in routes:
                    p_block = rates.block_given_storm[route.kind] if storm else rates.block_without_storm
                    like = {}
                    for blocked in (True, False):
                        if self.inspected.get((route.name, period), blocked) != blocked:
                            like[blocked] = 0.0
                            continue
                        p_late = rates.late_given_blocked if blocked else rates.late_given_open
                        value = p_block if blocked else 1 - p_block
                        for shipment in route.shipments:
                            late = self.late.get((shipment, period))
                            if late is not None:
                                value *= p_late if late else 1 - p_late
                        like[blocked] = value
                    total = like[True] + like[False]
                    weight *= total
                    block_given[(period, storm, route.name)] = like[True] / total if total else 0.0
                evidence[(period, storm)] = weight
            storm_mass = dict.fromkeys(self.window, 0.0)
            block_mass = dict.fromkeys(((route.name, period) for route in routes for period in self.window), 0.0)
            norm = 0.0
            for storms in itertools.product((True, False), repeat=len(self.window)):
                weight = 1.0
                previous = self.before[region]
                for period, storm in zip(self.window, storms):
                    p_storm = rates.storm_rate(kind, previous)
                    weight *= (p_storm if storm else 1 - p_storm) * evidence[(period, storm)]
                    previous = float(storm)
                norm += weight
                for period, storm in zip(self.window, storms):
                    storm_mass[period] += weight * storm
                    for route in routes:
                        block_mass[(route.name, period)] += weight * block_given[(period, storm, route.name)]
            for period in self.window:
                posterior[("Storm", region, period)] = storm_mass[period] / norm
                for route in routes:
                    if (route.name, period) not in self.inspected:
                        posterior[("Blocked", route.name, period)] = block_mass[(route.name, period)] / norm
        return posterior
