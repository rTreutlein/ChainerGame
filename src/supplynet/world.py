"""SupplyNet world: regions with storms, routes a storm blocks, late shipments,
and in stage 3 a production cycle.

Regions are independent; within a region the storm is a common cause of its
routes' blocks, and a block is the common cause of its shipments' lateness.

Stage 1 has independent periods and shipments that arrive when they depart.
Stage 2 links periods: a storm persists with ``Rates.persist`` and starts with
the region kind's storm rate, and each route has a transit time, so a
shipment's lateness, decided by its route's block in its departure period, is
seen only when it arrives. Stage 3 adds a ``Cell``: a mine fuels a power plant
over the fuel route, and the power plant powers the mine and the plants; a
site may be degraded, and production is the least fixed point of the cycle's
rules (docs/supply_network.md, stages 1 to 3).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

Key = tuple[str, str, str]  # (predicate, subject, period)

MINE, POWER = "mine", "power-plant"


def onset(start: float, persist: float | None, previous: float) -> float:
    """P(on now) of a two-state chain that stays on with ``persist`` and turns
    on with ``start``, given P(on) in the previous period; with no
    persistence, periods are independent."""
    return start if persist is None else previous * persist + (1 - previous) * start


def stationary(start: float, persist: float | None) -> float:
    return start if persist is None else start / (1 - persist + start)


@dataclass(frozen=True)
class Rates:
    storm: dict[str, float] = field(default_factory=lambda: {"coastal": 0.3, "inland": 0.1})
    block_given_storm: dict[str, float] = field(default_factory=lambda: {"exposed": 0.8, "sheltered": 0.4})
    block_without_storm: float = 0.03
    late_given_blocked: float = 0.9
    late_given_open: float = 0.05
    persist: float | None = None  # None: storms are independent across periods
    degrade: dict[str, float] = field(default_factory=lambda: {"mine": 0.08, "power": 0.05, "plant": 0.1})
    degraded_persist: float = 0.75  # otherwise a degraded site is repaired
    stocked: float = 0.4  # the fuel warehouse holds stock in a period

    def storm_rate(self, kind: str, previous: float) -> float:
        """P(storm now) given P(storm in the previous period)."""
        return onset(self.storm[kind], self.persist, previous)

    def stationary(self, kind: str) -> float:
        return stationary(self.storm[kind], self.persist)

    def degraded_rate(self, kind: str, previous: float) -> float:
        return onset(self.degrade[kind], self.degraded_persist, previous)

    def degraded_stationary(self, kind: str) -> float:
        return stationary(self.degrade[kind], self.degraded_persist)

    def block_rate(self, kind: str, storm: bool) -> float:
        return self.block_given_storm[kind] if storm else self.block_without_storm


@dataclass(frozen=True)
class Route:
    name: str
    region: str
    kind: str
    shipments: tuple[str, ...]
    transit: int = 0  # periods from departure to arrival


@dataclass(frozen=True)
class Cell:
    """The production cycle. The mine's fuel travels over the fuel route to
    the power plant, which also draws on the fuel warehouse; the power plant
    powers the mine and the plants, and each plant takes its input from one
    shipment of the network. Timed: fuel produced at t arrives at t + 1
    (the fuel route's transit is 1); untimed: within t (transit 0)."""

    fuel_route: Route
    inputs: dict[str, str]  # plant -> its input shipment

    @property
    def timed(self) -> bool:
        return self.fuel_route.transit == 1

    @property
    def sites(self) -> dict[str, str]:
        """Site -> kind, the kind setting its degradation hazard."""
        return {MINE: "mine", POWER: "power", **dict.fromkeys(self.inputs, "plant")}


@dataclass(frozen=True)
class Network:
    regions: dict[str, str]  # region -> kind
    routes: tuple[Route, ...]  # the fuel route included
    cell: Cell | None = None

    def routes_of(self, region: str) -> list[Route]:
        return [route for route in self.routes if route.region == region]

    @property
    def max_transit(self) -> int:
        return max(route.transit for route in self.routes)


def production(
    cell: Cell, degraded: dict[str, bool], stocked: bool, inputs: dict[str, bool], fuel_blocked: bool, fuel_sent: bool
) -> dict[str, bool]:
    """The least fixed point of the cycle's rules in one period, iterated from
    nothing producing: the power plant runs when fuelled and not degraded, the
    mine and the plants when powered, supplied and not degraded. The power
    plant is fuelled by stock, or by the mine's fuel over the open fuel route:
    the fuel the mine sent in the previous period (``fuel_sent``, timed) or
    what it produces now (untimed). ``fuel_blocked`` is the fuel route's block
    in the period the fuel departs."""
    producing = dict.fromkeys(cell.sites, False)
    while True:
        fuelled = stocked or (fuel_sent if cell.timed else producing[MINE]) and not fuel_blocked
        power = fuelled and not degraded[POWER]
        step = {
            MINE: power and not degraded[MINE],
            POWER: power,
            **{plant: power and inputs[plant] and not degraded[plant] for plant in cell.inputs},
        }
        if step == producing:
            return producing
        producing = step


@dataclass(frozen=True)
class Period:
    name: str
    previous: str | None  # the linked previous period; None when periods are independent
    storms: dict[str, bool]
    blocked: dict[str, bool]
    late: dict[str, bool]  # shipments departing in this period
    degraded: dict[str, bool] = field(default_factory=dict)
    stocked: bool = False
    inputs: dict[str, bool] = field(default_factory=dict)  # plant -> its input arrived
    producing: dict[str, bool] = field(default_factory=dict)

    def truth(self, predicate: str) -> dict[str, bool]:
        return {"Storm": self.storms, "Blocked": self.blocked, "Degraded": self.degraded, "Producing": self.producing}[predicate]


@dataclass(frozen=True)
class Observation:
    """What the operator sees in a period: the lateness of the shipments that
    arrive, keyed by (shipment, departure period), and the block state of the
    routes inspected in this period. With a cell also the inspected sites'
    degradation, the fuel stock, the plants' inputs, and the previous
    period's production, reported at its end."""

    period: str
    previous: str | None
    late: dict[tuple[str, str], bool]
    inspected: dict[str, bool]
    degraded: dict[str, bool] = field(default_factory=dict)
    stocked: bool | None = None
    inputs: dict[str, bool] = field(default_factory=dict)
    reported: dict[str, bool] = field(default_factory=dict)  # the previous period's production


REGION_NAMES = ("north", "south", "east", "west", "central", "coast")


def generate_network(rng: random.Random, regions: int = 3, max_transit: int = 0, cycle: str | None = None) -> Network:
    """Stage 1 with no transit, stage 2 with transits up to ``max_transit``,
    stage 3 adding a cell whose cycle is "timed" or "untimed"."""
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
    if cycle is None:
        return Network(region_kinds, tuple(routes))
    fuel = Route("rf", rng.choice(list(region_kinds)), rng.choice(("exposed", "sheltered")), (), 1 if cycle == "timed" else 0)
    first, second = rng.sample([shipment for route in routes for shipment in route.shipments], 2)
    return Network(region_kinds, (*routes, fuel), Cell(fuel, {"plant-a": first, "plant-b": second}))


def sample_period(network: Network, rates: Rates, rng: random.Random, name: str, timeline: list[Period]) -> Period:
    """A period following the linked periods ``timeline`` (empty for the first
    period, or when periods are independent)."""
    previous = timeline[-1] if timeline else None
    storms = {
        region: rng.random() < rates.storm_rate(kind, float(previous.storms[region]) if previous else rates.stationary(kind))
        for region, kind in network.regions.items()
    }
    blocked = {}
    late = {}
    for route in network.routes:
        blocked[route.name] = rng.random() < rates.block_rate(route.kind, storms[route.region])
        p_late = rates.late_given_blocked if blocked[route.name] else rates.late_given_open
        for shipment in route.shipments:
            late[shipment] = rng.random() < p_late
    cell = network.cell
    if cell is None:
        return Period(name, previous.name if previous else None, storms, blocked, late)
    degraded = {
        site: rng.random() < rates.degraded_rate(kind, float(previous.degraded[site]) if previous else rates.degraded_stationary(kind))
        for site, kind in cell.sites.items()
    }
    stocked = rng.random() < rates.stocked
    inputs = {}
    for plant, shipment in cell.inputs.items():
        transit = next(route.transit for route in network.routes if shipment in route.shipments)
        inputs[plant] = transit > len(timeline) or not timeline[-transit].late[shipment]
    fuel = cell.fuel_route.name
    producing = production(
        cell,
        degraded,
        stocked,
        inputs,
        previous.blocked[fuel] if cell.timed and previous else blocked[fuel],
        previous is not None and previous.producing[MINE],
    )
    return Period(name, previous.name if previous else None, storms, blocked, late, degraded, stocked, inputs, producing)


def observe(
    network: Network, period: Period, rng: random.Random, inspect_rate: float, late: dict, reported: dict | None = None
) -> Observation:
    """The period's observation; ``reported`` is the previous period's
    production (none before the first period)."""
    inspected = {route.name: period.blocked[route.name] for route in network.routes if rng.random() < inspect_rate}
    if network.cell is None:
        return Observation(period.name, period.previous, late, inspected)
    degraded = {site: period.degraded[site] for site in network.cell.sites if rng.random() < inspect_rate}
    return Observation(period.name, period.previous, late, inspected, degraded, period.stocked, period.inputs, reported or {})


class Knowledge:
    """The exact posterior from what the operator knows: the last resolved
    period's labels and the evidence about the unresolved periods since.

    Per region the hidden state is a Markov chain over the unresolved window,
    filtered and smoothed by forward-backward. A region's state is its storm,
    and in the cell's region also the fuel route's block and every site's
    degradation (at most 2^6 states); the blocks of the other routes are
    summed out per period given the storm. A window period's reported
    production is a function of its state, the previous state's fuel route
    and the mine's previous production (reported too), so it enters as a 0/1
    factor between consecutive states. The state carried into the window is
    the resolved period's, or before any period a pseudo-state holding the
    stationary probabilities, from which the chains' onset gives the
    stationary distribution."""

    def __init__(self, network: Network, rates: Rates):
        self.network, self.rates = network, rates
        self.window: list[str] = []
        self.late: dict[tuple[str, str], bool] = {}
        self.inspected: dict[tuple[str, str], bool] = {}  # (route or site, period)
        self.stocked: dict[str, bool] = {}
        self.inputs: dict[tuple[str, str], bool] = {}  # (plant, period)
        self.reported: dict[tuple[str, str], bool] = {}  # (site, period) -> produced
        self.mine_before = False  # the mine produced in the last resolved period
        cell = network.cell
        self.variables = {
            region: [("Storm", region)]
            + (
                [("Blocked", cell.fuel_route.name), *(("Degraded", site) for site in cell.sites)]
                if cell and cell.fuel_route.region == region
                else []
            )
            for region in network.regions
        }
        self.before = {
            region: {
                (rates.stationary(kind),)
                + tuple(
                    0.0 if predicate == "Blocked" else rates.degraded_stationary(cell.sites[subject])
                    for predicate, subject in self.variables[region][1:]
                ): 1.0
            }
            for region, kind in network.regions.items()
        }

    def observe(self, observation: Observation) -> None:
        period = observation.period
        self.window.append(period)
        self.late.update(observation.late)
        self.inspected.update({(subject, period): value for subject, value in (observation.inspected | observation.degraded).items()})
        if observation.stocked is not None:
            self.stocked[period] = observation.stocked
        self.inputs.update({(plant, period): value for plant, value in observation.inputs.items()})
        if observation.previous in self.window:
            self.reported.update({(site, observation.previous): value for site, value in observation.reported.items()})

    def resolve(self, period: Period) -> None:
        """The period, the oldest of the window, becomes labelled."""
        assert self.window[0] == period.name
        self.window.pop(0)
        for region, variables in self.variables.items():
            self.before[region] = {tuple(period.truth(predicate)[subject] for predicate, subject in variables): 1.0}
        self.mine_before = period.producing.get(MINE, False)
        self.late, self.inspected, self.inputs, self.reported = (
            {key: value for key, value in store.items() if key[1] != period.name}
            for store in (self.late, self.inspected, self.inputs, self.reported)
        )
        self.stocked.pop(period.name, None)

    def posterior(self) -> dict[Key, float]:
        """P(Storm region period), P(Degraded site period) and P(Blocked route
        period) for every window period, the inspected ones excepted, and
        P(Producing site now) for the last window period."""
        rates, cell = self.rates, self.network.cell
        posterior = {}
        for region, kind in self.network.regions.items():
            variables = self.variables[region]
            in_cell = len(variables) > 1
            routes = [route for route in self.network.routes_of(region) if not (in_cell and route == cell.fuel_route)]
            states = list(itertools.product((True, False), repeat=len(variables)))
            # Per period and storm value: the likelihood of the period's route
            # evidence, and each route's block posterior given that storm.
            evidence = {}
            block_given = {}
            for period, storm in itertools.product(self.window, (True, False)):
                weight = 1.0
                for route in routes:
                    p_block = rates.block_rate(route.kind, storm)
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

            def local(period: str, state: tuple) -> float:
                """The period's evidence and the fuel route's block given the storm."""
                weight = evidence[(period, state[0])]
                for (predicate, subject), value in zip(variables[1:], state[1:]):
                    if self.inspected.get((subject, period), value) != value:
                        return 0.0
                    if predicate == "Blocked":
                        p_block = rates.block_rate(cell.fuel_route.kind, state[0])
                        weight *= p_block if value else 1 - p_block
                return weight

            def transition(previous: tuple, state: tuple) -> float:
                weight = 1.0
                for (predicate, subject), before, now in zip(variables, previous, state):
                    if predicate != "Blocked":
                        p = rates.storm_rate(kind, before) if predicate == "Storm" else rates.degraded_rate(cell.sites[subject], before)
                        weight *= p if now else 1 - p
                return weight

            def produce(index: int, previous: tuple, state: tuple) -> dict[str, bool]:
                period = self.window[index]
                mine_before = self.reported[(MINE, self.window[index - 1])] if index else self.mine_before
                inputs = {plant: self.inputs[(plant, period)] for plant in cell.inputs}
                fuel_blocked = bool((previous if cell.timed else state)[1])
                return production(cell, dict(zip(cell.sites, state[2:])), self.stocked[period], inputs, fuel_blocked, mine_before)

            # factors[i][(previous, state)]: from the state before window
            # period i to its state, with period i's evidence.
            transitions = {(previous, state): transition(previous, state) for previous in states for state in states}
            factors = []
            for index, period in enumerate(self.window):
                reported = {site: self.reported[(site, period)] for site in cell.sites} if in_cell and (MINE, period) in self.reported else None
                weights = {state: local(period, state) for state in states}
                factor = {}
                for previous in self.before[region] if index == 0 else states:
                    for state in states:
                        weight = weights[state] and weights[state] * (transitions[(previous, state)] if index else transition(previous, state))
                        if weight and reported is not None and produce(index, previous, state) != reported:
                            weight = 0.0
                        factor[(previous, state)] = weight
                factors.append(factor)
            forward = [self.before[region]]
            for factor in factors:
                forward.append({state: sum(p * factor[(previous, state)] for previous, p in forward[-1].items()) for state in states})
            backward = [dict.fromkeys(states, 1.0)]
            for factor in reversed(factors[1:]):
                backward.insert(0, {previous: sum(factor[(previous, state)] * backward[0][state] for state in states) for previous in states})
            norm = sum(forward[-1].values())
            for index, period in enumerate(self.window):
                marginal = {state: forward[index + 1][state] * backward[index][state] / norm for state in states}
                for position, (predicate, subject) in enumerate(variables):
                    if (subject, period) not in self.inspected:
                        posterior[(predicate, subject, period)] = sum(p for state, p in marginal.items() if state[position])
                for route in routes:
                    if (route.name, period) not in self.inspected:
                        posterior[("Blocked", route.name, period)] = sum(
                            p * block_given[(period, state[0], route.name)] for state, p in marginal.items()
                        )
            if in_cell and self.window:
                last = len(self.window) - 1
                producing = dict.fromkeys(cell.sites, 0.0)
                for (previous, state), weight in factors[last].items():
                    if weight:
                        for site, on in produce(last, previous, state).items():
                            producing[site] += forward[last][previous] * weight / norm * on
                posterior.update({("Producing", site, self.window[last]): p for site, p in producing.items()})
        return posterior
