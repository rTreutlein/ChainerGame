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
rules. Stage 4 adds decisions: each period an ``Action`` stocks the fuel
warehouse, sends the mine's fuel over an alternate route, and repairs sites
(docs/supply_network.md, stages 1 to 4).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace

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

    def nodes(self, network: Network) -> NodeRates:
        """The rates per node of ``network``: each region's storm, each
        route's block, each shipment's lateness and each site's degradation.
        Without persistence a storm's rate does not depend on its parent."""
        table = {
            ("Storm", region): (self.storm[kind] if self.persist is None else self.persist, self.storm[kind])
            for region, kind in network.regions.items()
        }
        for route in network.routes:
            table[("Blocked", route.name)] = (self.block_given_storm[route.kind], self.block_without_storm)
            table.update({("Late", shipment): (self.late_given_blocked, self.late_given_open) for shipment in route.shipments})
        if network.cell:
            table.update({("Degraded", site): (self.degraded_persist, self.degrade[kind]) for site, kind in network.cell.sites.items()})
        return NodeRates(table)


@dataclass(frozen=True)
class NodeRates:
    """P(node | parent) and P(node | not parent) per (predicate, subject). A
    node's parent is fixed by the network: a storm's or a degradation's is its
    own state in the previous period, a block's its region's storm, a
    lateness its route's block."""

    table: dict[tuple[str, str], tuple[float, float]]

    def rate(self, predicate: str, subject: str, parent: float) -> float:
        """P(node) given P(parent)."""
        given, without = self.table[(predicate, subject)]
        return parent * given + (1 - parent) * without

    def stationary(self, predicate: str, subject: str) -> float:
        given, without = self.table[(predicate, subject)]
        return stationary(without, given)


Count = tuple[int, int]  # (samples with the node true, samples)


def counts(network: Network, periods: list[Period], linked: bool) -> dict[tuple[str, str], tuple[Count, Count]]:
    """Per node, its labelled samples in ``periods`` (resolved, in order)
    with the parent true and with it false. With ``linked`` periods a storm's
    or degradation's parent is its state in the previous period, so the first
    period is no sample; with independent periods a storm has no parent, and
    every period counts in both branches."""
    tally: dict[tuple[str, str], list[list[int]]] = {}

    def add(node: tuple[str, str], parent: bool, value: bool) -> None:
        branch = tally.setdefault(node, [[0, 0], [0, 0]])[0 if parent else 1]
        branch[0] += value
        branch[1] += 1

    by_name = {period.name: period for period in periods}
    for period in periods:
        previous = by_name.get(period.previous)
        for region, storm in period.storms.items():
            if not linked:
                add(("Storm", region), True, storm)
                add(("Storm", region), False, storm)
            elif previous:
                add(("Storm", region), previous.storms[region], storm)
        for route in network.routes:
            add(("Blocked", route.name), period.storms[route.region], period.blocked[route.name])
            for shipment in route.shipments:
                add(("Late", shipment), period.blocked[route.name], period.late[shipment])
        if previous:
            for site, degraded in period.degraded.items():
                add(("Degraded", site), previous.degraded[site], degraded)
    return {node: (tuple(given), tuple(without)) for node, (given, without) in tally.items()}


def laplace(count: Count) -> float:
    return (count[0] + 1) / (count[1] + 2)


def learned_rates(network: Network, periods: list[Period], linked: bool) -> NodeRates:
    """Every node's rates estimated from the labelled ``periods`` by Laplace's
    rule, (k + 1) / (n + 2) per branch; a node with no samples gets 1/2."""
    tally = counts(network, periods, linked)
    return NodeRates({node: tuple(laplace(count) for count in tally.get(node, ((0, 0), (0, 0)))) for node in Rates().nodes(network).table})


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
    (the fuel route's transit is 1); untimed: within t (transit 0). With
    decisions (stage 4) the fuel may instead leave over the ``alternate``
    route, in the fuel route's region."""

    fuel_route: Route
    inputs: dict[str, str]  # plant -> its input shipment
    alternate: Route | None = None

    @property
    def fuel_routes(self) -> tuple[Route, ...]:
        return (self.fuel_route, self.alternate) if self.alternate else (self.fuel_route,)

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


@dataclass(frozen=True, order=True)
class Action:
    """A period's decision (stage 4): stock the fuel warehouse, send the
    mine's fuel over the cell's alternate route, and repair sites. A repaired
    site produces nothing in the period and is healthy at its end."""

    stock: bool = False
    reroute: bool = False
    repairs: tuple[str, ...] = ()  # sorted
    inspections: tuple[str, ...] = ()  # stage 5: results arrive before the next decision

    @property
    def units(self) -> int:
        """The capacity it takes: one unit per stockpile, reroute and repair."""
        return self.stock + self.reroute + len(self.repairs) + len(self.inspections)

    def via(self, cell: Cell) -> str:
        """The route the mine's fuel leaves on."""
        return (cell.alternate if self.reroute else cell.fuel_route).name


def production(
    cell: Cell,
    degraded: dict[str, bool],
    stocked: bool,
    inputs: dict[str, bool],
    fuel_blocked: bool,
    fuel_sent: bool,
    repairing: tuple[str, ...] = (),
) -> dict[str, bool]:
    """The least fixed point of the cycle's rules in one period, iterated from
    nothing producing: the power plant runs when fuelled, not degraded and
    not under repair, the mine and the plants when powered, supplied, not
    degraded and not under repair. The power plant is fuelled by stock, or by
    the mine's fuel over the open route it took: the fuel the mine sent in the
    previous period (``fuel_sent``, timed) or what it produces now (untimed).
    ``fuel_blocked`` is that route's block in the period the fuel departs."""
    producing = dict.fromkeys(cell.sites, False)
    while True:
        fuelled = stocked or (fuel_sent if cell.timed else producing[MINE]) and not fuel_blocked
        power = fuelled and not degraded[POWER] and POWER not in repairing
        step = {
            MINE: power and not degraded[MINE] and MINE not in repairing,
            POWER: power,
            **{plant: power and inputs[plant] and not degraded[plant] and plant not in repairing for plant in cell.inputs},
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
    repairing: tuple[str, ...] = ()
    via: str | None = None  # the route the mine's fuel leaves on

    def action(self, cell: Cell) -> Action:
        """The action the period was run under."""
        return Action(self.stocked, self.via == cell.alternate.name, self.repairing)

    def truth(self, predicate: str) -> dict[str, bool]:
        return {"Storm": self.storms, "Blocked": self.blocked, "Degraded": self.degraded, "Producing": self.producing}[predicate]

    def labels(self) -> dict[tuple[str, str], bool]:
        """(predicate, subject) -> value of every labelled statement."""
        return {(predicate, subject): value for predicate in ("Storm", "Blocked", "Degraded", "Producing") for subject, value in self.truth(predicate).items()}


@dataclass(frozen=True)
class Observation:
    """What the operator sees in a period: the lateness of the shipments that
    arrive, keyed by (shipment, departure period), and the block state of the
    routes inspected in this period. With a cell also the inspected sites'
    degradation, the fuel stock, the plants' inputs, and the previous
    period's production, reported at its end. With decisions the stock is
    part of the period's action, not observed (None): the action is told
    apart (``Knowledge.act``).

    Reports are true. ``report_confidence`` is the evidence weight they are
    stated with (PeTTaChainer's ``(STV v c)``: strength the observed value,
    confidence the weight of its evidence, not a chance that it is wrong),
    so it leaves the exact posterior unchanged; below 1 the chainer counts a
    report as shared evidence and conditions on it."""

    period: str
    previous: str | None
    late: dict[tuple[str, str], bool]
    inspected: dict[str, bool]
    degraded: dict[str, bool] = field(default_factory=dict)
    stocked: bool | None = None
    inputs: dict[str, bool] = field(default_factory=dict)
    reported: dict[str, bool] = field(default_factory=dict)  # the previous period's production
    report_confidence: float = 1.0


REGION_NAMES = ("north", "south", "east", "west", "central", "coast")


def decides(network) -> bool:
    """Whether a stage's network takes decisions (stage 4); stage scale's
    networks have no cell."""
    cell = getattr(network, "cell", None)
    return bool(cell and cell.alternate)


PLANT_NAMES = tuple(f"plant-{letter}" for letter in "abcdefgh")


def generate_network(rng: random.Random, regions: int = 3, max_transit: int = 0, cycle: str | None = None, plants: int = 2,
                     decisions: bool = False) -> Network:
    """Stage 1 with no transit, stage 2 with transits up to ``max_transit``,
    stage 3 adding a cell whose cycle is "timed" or "untimed" with
    ``plants`` plants. With ``decisions`` (stage 4) the cell's fuel route is
    exposed and a sheltered alternate route ``ra`` joins it in its region; the
    draws are stage 3's, so a seed keeps its stage 3 network otherwise."""
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
    inputs = dict(zip(PLANT_NAMES, rng.sample([shipment for route in routes for shipment in route.shipments], plants)))
    if not decisions:
        return Network(region_kinds, (*routes, fuel), Cell(fuel, inputs))
    fuel = replace(fuel, kind="exposed")
    alternate = Route("ra", fuel.region, "sheltered", (), fuel.transit)
    return Network(region_kinds, (*routes, fuel, alternate), Cell(fuel, inputs, alternate))


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
        site: rng.random()
        < rates.degraded_rate(kind, float(previous.degraded[site] and site not in previous.repairing) if previous else rates.degraded_stationary(kind))
        for site, kind in cell.sites.items()
    }
    stocked = rng.random() < rates.stocked
    inputs = {}
    for plant, shipment in cell.inputs.items():
        transit = next(route.transit for route in network.routes if shipment in route.shipments)
        inputs[plant] = transit > len(timeline) or not timeline[-transit].late[shipment]
    period = Period(name, previous.name if previous else None, storms, blocked, late, degraded, inputs=inputs)
    return enact(network, period, previous, Action(stock=stocked))


def enact(network: Network, period: Period, previous: Period | None, action: Action) -> Period:
    """The period under ``action``: its stock, repairs and fuel route, and the
    production they give with the period's hidden state. Timed, the fuel
    arriving is what the mine produced in the previous period, over the
    route it took then."""
    cell = network.cell
    via = action.via(cell)
    producing = production(
        cell,
        period.degraded,
        action.stock,
        period.inputs,
        (previous.blocked[previous.via] if previous else False) if cell.timed else period.blocked[via],
        previous is not None and previous.producing[MINE],
        action.repairs,
    )
    return replace(period, stocked=action.stock, repairing=action.repairs, via=via, producing=producing)


def observe(
    network: Network, period: Period, rng: random.Random, inspect_rate: float, late: dict, reported: dict | None = None,
    report_confidence: float = 1.0,
) -> Observation:
    """The period's observation; ``reported`` is the previous period's
    production (none before the first period), stated at
    ``report_confidence``."""
    inspected = {route.name: period.blocked[route.name] for route in network.routes if rng.random() < inspect_rate}
    if network.cell is None:
        return Observation(period.name, period.previous, late, inspected)
    degraded = {site: period.degraded[site] for site in network.cell.sites if rng.random() < inspect_rate}
    stocked = None if network.cell.alternate else period.stocked
    return Observation(period.name, period.previous, late, inspected, degraded, stocked, period.inputs, reported or {}, report_confidence)


class Knowledge:
    """The exact posterior from what the operator knows: the last resolved
    period's labels, the evidence about the unresolved periods since, and the
    actions taken in them.

    Per region the hidden state is a Markov chain over the unresolved window,
    filtered and smoothed by forward-backward. A region's state is its storm,
    and in the cell's region also the fuel routes' blocks and every site's
    degradation; the blocks of the other routes are summed out per period
    given the storm. A state is a bit mask over the region's variables, and
    the transition factorizes over them (a storm and a degradation follow
    their own previous value, a block only its period's storm), so a
    forward or backward step is one contraction per variable. A window
    period's reported production is a function of its state, the block of the
    route the arriving fuel took (in the previous state, timed) and the mine's
    previous production (reported too), so it enters as a 0/1 factor that
    reads the previous state through one bit. The state carried into the
    window is the resolved period's, or before any period the stationary
    probabilities, independent per variable, from which the chains' onset
    gives the stationary distribution. A report is an observation at any
    ``Observation.report_confidence``: its confidence weighs its evidence and
    does not make it noisy, as PeTTaChainer reads a fact ``(STV 0|1 c)``
    (its test_deep_conditioning ``exact_t8``). ``rates`` are read when the posterior
    is computed, so they may be replaced between rounds."""

    def __init__(self, network: Network, rates: NodeRates):
        self.network, self.rates = network, rates
        self.window: list[str] = []
        self.late: dict[tuple[str, str], bool] = {}
        self.inspected: dict[tuple[str, str], bool] = {}  # (route or site, period)
        self.stocked: dict[str, bool] = {}
        self.inputs: dict[tuple[str, str], bool] = {}  # (plant, period)
        self.reported: dict[tuple[str, str], bool] = {}  # (site, period) -> produced
        self.repairing: dict[str, tuple[str, ...]] = {}  # period -> sites repaired
        self.via: dict[str, str] = {}  # period -> the route the mine's fuel left on
        self.mine_before = False  # the mine produced in the last resolved period
        self.repairs_before: tuple[str, ...] = ()
        cell = network.cell
        self.via_before = cell.fuel_route.name if cell else None
        self.variables = {
            region: [("Storm", region)]
            + (
                [*(("Blocked", route.name) for route in cell.fuel_routes), *(("Degraded", site) for site in cell.sites)]
                if cell and cell.fuel_route.region == region
                else []
            )
            for region in network.regions
        }
        self.before = {
            region: tuple(0.0 if predicate == "Blocked" else rates.stationary(predicate, subject) for predicate, subject in variables)
            for region, variables in self.variables.items()
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

    def act(self, period: str, action: Action) -> None:
        """The action taken in the window's last period."""
        self.stocked[period], self.repairing[period], self.via[period] = action.stock, action.repairs, action.via(self.network.cell)

    def resolve(self, period: Period) -> None:
        """The period, the oldest of the window, becomes labelled."""
        assert self.window[0] == period.name
        self.window.pop(0)
        for region, variables in self.variables.items():
            self.before[region] = tuple(period.truth(predicate)[subject] for predicate, subject in variables)
        self.mine_before = period.producing.get(MINE, False)
        self.repairs_before, self.via_before = period.repairing, period.via
        self.late, self.inspected, self.inputs, self.reported = (
            {key: value for key, value in store.items() if key[1] != period.name}
            for store in (self.late, self.inspected, self.inputs, self.reported)
        )
        for store in (self.stocked, self.repairing, self.via):
            store.pop(period.name, None)

    def _fuel_route(self, index: int) -> str:
        """The route the fuel took in window period ``index`` (-1: the resolved period)."""
        cell = self.network.cell
        return self.via_before if index < 0 else self.via.get(self.window[index], cell.fuel_route.name)

    def _filter(self, region: str):
        """Forward-backward over the window for one region. Returns each
        window period's state marginals (lists over the bit-mask states), each
        other route's block posterior per (period, storm, route), and for the
        cell's region the last period's joint weights of the previous state's
        fuel-route block (as the arriving fuel saw it) and the state,
        normalized, as {(block, state): weight}."""
        rates, cell = self.rates, self.network.cell
        variables = self.variables[region]
        in_cell = len(variables) > 1
        size = 1 << len(variables)
        position = {var: i for i, var in enumerate(variables)}
        routes = [route for route in self.network.routes_of(region) if not (in_cell and route in cell.fuel_routes)]

        # Per period and storm value: the likelihood of the period's route
        # evidence, and each route's block posterior given that storm.
        evidence, block_given = {}, {}
        for period, storm in itertools.product(self.window, (True, False)):
            weight = 1.0
            for route in routes:
                p_block = rates.rate("Blocked", route.name, storm)
                like = {}
                for blocked in (True, False):
                    if self.inspected.get((route.name, period), blocked) != blocked:
                        like[blocked] = 0.0
                        continue
                    value = p_block if blocked else 1 - p_block
                    for shipment in route.shipments:
                        late = self.late.get((shipment, period))
                        if late is not None:
                            p_late = rates.rate("Late", shipment, blocked)
                            value *= p_late if late else 1 - p_late
                    like[blocked] = value
                total = like[True] + like[False]
                weight *= total
                block_given[(period, storm, route.name)] = like[True] / total if total else 0.0
            evidence[(period, storm)] = weight

        def local(period: str) -> list[float]:
            """The period's evidence, its inspections of the region's
            variables, and the fuel routes' blocks given the storm."""
            weights = []
            for state in range(size):
                storm = bool(state & 1)
                weight = evidence[(period, storm)]
                for i, (predicate, subject) in enumerate(variables[1:], 1):
                    value = bool(state >> i & 1)
                    if self.inspected.get((subject, period), value) != value:
                        weight = 0.0
                        break
                    if predicate == "Blocked":
                        p_block = rates.rate(predicate, subject, storm)
                        weight *= p_block if value else 1 - p_block
                weights.append(weight)
            return weights

        def matrices(repaired: tuple[str, ...]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
            """Per variable, P(value now | previous value): a storm and a
            degradation follow their chain (a repaired site restarts
            healthy), a block does not depend on its previous value."""
            result = []
            for predicate, subject in variables:
                if predicate == "Blocked":
                    result.append(((1.0, 1.0), (1.0, 1.0)))
                    continue
                rows = []
                for before in (False, True):
                    p = rates.rate(predicate, subject, before and subject not in repaired)
                    rows.append((1 - p, p))
                result.append(tuple(rows))
            return result

        def forward(values: list[float], mats) -> list[float]:
            """Sum over previous states of values × P(state | previous)."""
            for i, matrix in enumerate(mats):
                bit = 1 << i
                values = [values[state & ~bit] * matrix[0][state >> i & 1] + values[state | bit] * matrix[1][state >> i & 1] for state in range(size)]
            return values

        def backward(values: list[float], mats) -> list[float]:
            """Sum over next states of P(state | previous) × values."""
            for i, matrix in enumerate(mats):
                bit = 1 << i
                values = [values[state & ~bit] * matrix[state >> i & 1][0] + values[state | bit] * matrix[state >> i & 1][1] for state in range(size)]
            return values

        def produce(index: int, block: bool, state: int) -> dict[str, bool]:
            period = self.window[index]
            mine_before = self.reported[(MINE, self.window[index - 1])] if index else self.mine_before
            inputs = {plant: self.inputs[(plant, period)] for plant in cell.inputs}
            fuel_blocked = block if cell.timed else bool(state >> position[("Blocked", self._fuel_route(index))] & 1)
            degraded = {site: bool(state >> position[("Degraded", site)] & 1) for site in cell.sites}
            return production(cell, degraded, self.stocked[period], inputs, fuel_blocked, mine_before, self.repairing.get(period, ()))

        before = self.before[region]
        start = [1.0] * size  # the carried-in state, independent per variable
        for i, value in enumerate(before):
            start = [weight * (value if state >> i & 1 else 1 - value) for state, weight in enumerate(start)]
        # steps[i]: period i's local weights, transition matrices, the bit of
        # the previous state that the report factor reads, and that factor
        # per (bit value, state), or None without a report.
        steps, alphas, splits = [], [], []
        previous = start
        for index, period in enumerate(self.window):
            mats = matrices(self.repairs_before if index == 0 else self.repairing.get(self.window[index - 1], ()))
            weights = local(period)
            bit = position[("Blocked", self._fuel_route(index - 1))] if in_cell else None
            reported = {site: self.reported[(site, period)] for site in cell.sites} if in_cell and (MINE, period) in self.reported else None
            factor = None if reported is None else {
                (block, state): float(produce(index, block, state) == reported) for block in (False, True) for state in range(size) if weights[state]
            }
            split = {}
            for block in (False, True) if in_cell else (False,):
                masked = previous if bit is None else [w if bool(state >> bit & 1) == block else 0.0 for state, w in enumerate(previous)]
                split[block] = [w * m for w, m in zip(weights, forward(masked, mats))]
            alpha = [sum(split[block][state] * (factor.get((block, state), 0.0) if factor else 1.0) for block in split) for state in range(size)]
            steps.append((weights, mats, bit, factor))
            alphas.append(alpha)
            splits.append(split)
            previous = alpha
        betas = [[1.0] * size]
        for weights, mats, bit, factor in reversed(steps[1:]):
            beta = betas[0]
            by_block = {
                block: backward([w * b * (factor.get((block, state), 0.0) if factor else 1.0) for state, (w, b) in enumerate(zip(weights, beta))], mats)
                for block in ((False, True) if bit is not None else (False,))
            }
            betas.insert(0, [by_block[bool(state >> bit & 1)][state] if bit is not None else by_block[False][state] for state in range(size)])
        norm = sum(alphas[-1]) if alphas else 1.0
        marginals = [[a * b / norm for a, b in zip(alpha, beta)] for alpha, beta in zip(alphas, betas)]
        last = {(block, state): w / norm for block, weights in splits[-1].items() for state, w in enumerate(weights) if w} if in_cell and splits else {}
        return marginals, block_given, last

    def posterior(self) -> dict[Key, float]:
        """P(Storm region period), P(Degraded site period) and P(Blocked route
        period) for every window period, the inspected ones excepted, and,
        once the last window period's stock is known, P(Producing site now)."""
        posterior = {}
        cell = self.network.cell
        for region, variables in self.variables.items():
            marginals, block_given, last = self._filter(region)
            routes = [route for route in self.network.routes_of(region) if not (len(variables) > 1 and route in cell.fuel_routes)]
            for period, marginal in zip(self.window, marginals):
                for position, (predicate, subject) in enumerate(variables):
                    if (subject, period) not in self.inspected:
                        posterior[(predicate, subject, period)] = sum(p for state, p in enumerate(marginal) if state >> position & 1)
                for route in routes:
                    if (route.name, period) not in self.inspected:
                        posterior[("Blocked", route.name, period)] = sum(
                            p * block_given[(period, bool(state & 1), route.name)] for state, p in enumerate(marginal)
                        )
            if last and self.window[-1] in self.stocked:
                producing = dict.fromkeys(cell.sites, 0.0)
                for (values, fuel_blocked), weight in self.now(last).items():
                    for site, on in self.production(values, fuel_blocked, self.stocked[self.window[-1]], self.repairing.get(self.window[-1], ())).items():
                        producing[site] += weight * on
                posterior.update({("Producing", site, self.window[-1]): p for site, p in producing.items()})
        return posterior

    def now(self, last: dict | None = None) -> dict[tuple[tuple, bool], float]:
        """The cell region's state in the last window period: the weight of
        each (its variables' values, as ((predicate, subject), value) pairs,
        and the block of the route the arriving fuel took)."""
        if last is None:
            last = self._filter(self.network.cell.fuel_route.region)[2]
        variables = self.variables[self.network.cell.fuel_route.region]
        return {(tuple((var, bool(state >> i & 1)) for i, var in enumerate(variables)), block): w for (block, state), w in last.items()}

    def production(self, values: tuple, fuel_blocked: bool, stocked: bool, repairing: tuple[str, ...]) -> dict[str, bool]:
        """The last window period's production in the cell region's state
        ``values`` (from ``now``) under the given stock and repairs."""
        cell, period = self.network.cell, self.window[-1]
        state = dict(values)
        mine_before = self.reported[(MINE, self.window[-2])] if len(self.window) > 1 else self.mine_before
        inputs = {plant: self.inputs[(plant, period)] for plant in cell.inputs}
        if not cell.timed:
            fuel_blocked = state[("Blocked", self.via.get(period, cell.fuel_route.name))]
        return production(cell, {site: state[("Degraded", site)] for site in cell.sites}, stocked, inputs, fuel_blocked, mine_before, repairing)
