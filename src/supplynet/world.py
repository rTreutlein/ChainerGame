"""Stage 1 world: regions with storms, routes a storm blocks, late shipments.

Regions are independent; within a region the storm is a common cause of its
routes' blocks, and a block is the common cause of its shipments' lateness.
The exact posterior is computed per region by enumerating its storm and
summing each route's block out (docs/supply_network.md, stage 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rates:
    storm: dict[str, float] = field(default_factory=lambda: {"coastal": 0.3, "inland": 0.1})
    block_given_storm: dict[str, float] = field(default_factory=lambda: {"exposed": 0.8, "sheltered": 0.4})
    block_without_storm: float = 0.03
    late_given_blocked: float = 0.9
    late_given_open: float = 0.05


@dataclass(frozen=True)
class Route:
    name: str
    region: str
    kind: str
    shipments: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    regions: dict[str, str]  # region -> kind
    routes: tuple[Route, ...]

    def routes_of(self, region: str) -> list[Route]:
        return [route for route in self.routes if route.region == region]


@dataclass(frozen=True)
class Period:
    name: str
    storms: dict[str, bool]
    blocked: dict[str, bool]
    late: dict[str, bool]


@dataclass(frozen=True)
class Observation:
    """What the operator sees of a period: every shipment's lateness and the
    block state of the inspected routes."""

    period: str
    late: dict[str, bool]
    inspected: dict[str, bool]


REGION_NAMES = ("north", "south", "east", "west", "central", "coast")


def generate_network(rng: random.Random, regions: int = 3) -> Network:
    region_kinds = {}
    routes = []
    for index in range(regions):
        region = REGION_NAMES[index] if index < len(REGION_NAMES) else f"region{index}"
        region_kinds[region] = rng.choice(("coastal", "inland"))
        for _ in range(rng.randint(2, 4)):
            number = len(routes) + 1
            shipments = tuple(f"s{number}-{k}" for k in range(1, rng.randint(1, 3) + 1))
            routes.append(Route(f"r{number}", region, rng.choice(("exposed", "sheltered")), shipments))
    return Network(region_kinds, tuple(routes))


def sample_period(network: Network, rates: Rates, rng: random.Random, name: str) -> Period:
    storms = {region: rng.random() < rates.storm[kind] for region, kind in network.regions.items()}
    blocked = {}
    late = {}
    for route in network.routes:
        p_block = rates.block_given_storm[route.kind] if storms[route.region] else rates.block_without_storm
        blocked[route.name] = rng.random() < p_block
        p_late = rates.late_given_blocked if blocked[route.name] else rates.late_given_open
        for shipment in route.shipments:
            late[shipment] = rng.random() < p_late
    return Period(name, storms, blocked, late)


def observe(network: Network, period: Period, rng: random.Random, inspect_rate: float) -> Observation:
    inspected = {route.name: period.blocked[route.name] for route in network.routes if rng.random() < inspect_rate}
    return Observation(period.name, dict(period.late), inspected)


def exact_posterior(network: Network, rates: Rates, observation: Observation) -> dict[tuple[str, str], float]:
    """P(Storm region | observation) and P(Blocked route | observation) for
    every region and uninspected route."""
    posterior = {}
    for region, kind in network.regions.items():
        routes = network.routes_of(region)
        # Per storm value: the likelihood of each route's evidence, and each
        # route's block posterior given that storm value.
        weights = {}
        block_given = {}
        for storm in (True, False):
            weight = rates.storm[kind] if storm else 1 - rates.storm[kind]
            for route in routes:
                p_block = rates.block_given_storm[route.kind] if storm else rates.block_without_storm
                like = {}
                for blocked in (True, False):
                    if route.name in observation.inspected and observation.inspected[route.name] != blocked:
                        like[blocked] = 0.0
                        continue
                    p_late = rates.late_given_blocked if blocked else rates.late_given_open
                    value = p_block if blocked else 1 - p_block
                    for shipment in route.shipments:
                        value *= p_late if observation.late[shipment] else 1 - p_late
                    like[blocked] = value
                total = like[True] + like[False]
                weight *= total
                block_given[(storm, route.name)] = like[True] / total if total else 0.0
            weights[storm] = weight
        p_storm = weights[True] / (weights[True] + weights[False])
        posterior[("Storm", region)] = p_storm
        for route in routes:
            if route.name not in observation.inspected:
                posterior[("Blocked", route.name)] = (
                    p_storm * block_given[(True, route.name)] + (1 - p_storm) * block_given[(False, route.name)]
                )
    return posterior


def truth(period: Period, observation: Observation) -> dict[tuple[str, str], bool]:
    values = {("Storm", region): storm for region, storm in period.storms.items()}
    values.update({("Blocked", route): blocked for route, blocked in period.blocked.items() if route not in observation.inspected})
    return values
