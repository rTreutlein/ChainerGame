"""The MeTTa a reasoner receives: the network's rules with their known rates,
labelled periods as facts, a period's observations, and belief queries."""

from __future__ import annotations

from .world import MINE, POWER, Key, Network, Observation, Period, Rates


def _number(value: float) -> str:
    return f"{value:.6g}"


def _tv(value: bool) -> str:
    return f"(STV {1 if value else 0} 1)"


def _certain(name: str, antecedent: str, consequent: str) -> str:
    return f"(: {name} (Implication {antecedent} {consequent}) (CTV (STV 1 1) (STV 0 1)))"


def rules(network: Network, rates: Rates) -> list[str]:
    """A storm blocks each route of its region with the route's rate, and a
    blocked route delays each shipment departing on it; when storms persist, a
    storm continues into the next period or starts with the region's rate.
    With a cell, a degradation persists or starts likewise, and the cycle's
    production rules hold with certainty. All stated as certain CTVs, so the
    rates are given, not learned."""
    lines = []
    if rates.persist is not None:
        for region, kind in network.regions.items():
            lines.append(
                f"(: persist-{region} (Implication (And (NextPeriod $p $t) (Storm {region} $p)) (Storm {region} $t)) "
                f"(CTV (STV {_number(rates.persist)} 1) (STV {_number(rates.storm[kind])} 1)))"
            )
    for route in network.routes:
        lines.append(
            f"(: block-{route.name} (Implication (Storm {route.region} $p) (Blocked {route.name} $p)) "
            f"(CTV (STV {_number(rates.block_given_storm[route.kind])} 1) "
            f"(STV {_number(rates.block_without_storm)} 1)))"
        )
        for shipment in route.shipments:
            lines.append(
                f"(: late-{shipment} (Implication (Blocked {route.name} $p) (Late {shipment} $p)) "
                f"(CTV (STV {_number(rates.late_given_blocked)} 1) "
                f"(STV {_number(rates.late_given_open)} 1)))"
            )
    return lines + (_cycle_rules(network, rates) if network.cell else [])


def _cycle_rules(network: Network, rates: Rates) -> list[str]:
    """The power plant runs when fuelled, by stock or by the mine's fuel
    arriving over the open fuel route, and is not degraded; the mine and the
    plants run when the power plant does, their input arrived, and they are
    not degraded. Timed, the fuel arriving now left in the previous period;
    untimed, it is the mine's production now, and the rules form a loop."""
    cell = network.cell
    fuel = cell.fuel_route.name
    lines = [
        f"(: persist-degraded-{site} (Implication (And (NextPeriod $p $t) (Degraded {site} $p)) (Degraded {site} $t)) "
        f"(CTV (STV {_number(rates.degraded_persist)} 1) (STV {_number(rates.degrade[kind])} 1)))"
        for site, kind in cell.sites.items()
    ]
    arriving = (
        f"(And (NextPeriod $p $t) (Producing {MINE} $p) (Not (Blocked {fuel} $p)))"
        if cell.timed
        else f"(And (Producing {MINE} $t) (Not (Blocked {fuel} $t)))"
    )
    lines += [
        _certain("fuel-arrives", arriving, "(FuelArrived $t)"),
        _certain("fuelled", "(Or (StockedFuel $t) (FuelArrived $t))", f"(Fuelled {POWER} $t)"),
        _certain(f"runs-{POWER}", f"(And (Fuelled {POWER} $t) (Not (Degraded {POWER} $t)))", f"(Producing {POWER} $t)"),
        _certain(f"runs-{MINE}", f"(And (Producing {POWER} $t) (Not (Degraded {MINE} $t)))", f"(Producing {MINE} $t)"),
    ]
    lines += [
        _certain(
            f"runs-{plant}",
            f"(And (Producing {POWER} $t) (InputArrived {plant} $t) (Not (Degraded {plant} $t)))",
            f"(Producing {plant} $t)",
        )
        for plant in cell.inputs
    ]
    return lines


def complete_predicates(network: Network) -> list[str]:
    """The predicates whose rules are every way they become true: with a cell,
    production, fuelling and the fuel's arrival. The untimed loop's least
    fixed point is then what the chainer reads (nothing runs without stock);
    the timed cell's rules are as complete, and acyclic within a period."""
    return ["Producing", "Fuelled", "FuelArrived"] if network.cell else []


def observation_facts(observation: Observation) -> list[str]:
    """The period's link to its predecessor, the lateness of the shipments
    arriving now (keyed by departure period), and the inspected routes; with a
    cell also the inspected sites, the fuel stock, the plants' inputs, and the
    previous period's production."""
    period, previous = observation.period, observation.previous
    lines = [f"(: next-{period} (NextPeriod {previous} {period}) (STV 1 1))"] if previous else []
    lines += [f"(: late-{shipment}-{departure} (Late {shipment} {departure}) {_tv(value)})" for (shipment, departure), value in observation.late.items()]
    lines += [f"(: blocked-{route}-{period} (Blocked {route} {period}) {_tv(value)})" for route, value in observation.inspected.items()]
    lines += [f"(: degraded-{site}-{period} (Degraded {site} {period}) {_tv(value)})" for site, value in observation.degraded.items()]
    if observation.stocked is not None:
        lines.append(f"(: stocked-{period} (StockedFuel {period}) {_tv(observation.stocked)})")
    lines += [f"(: input-{plant}-{period} (InputArrived {plant} {period}) {_tv(value)})" for plant, value in observation.inputs.items()]
    lines += [f"(: producing-{site}-{previous} (Producing {site} {previous}) {_tv(value)})" for site, value in observation.reported.items()]
    return lines


def resolution_facts(period: Period, observation: Observation) -> list[str]:
    """What a resolved period adds beyond its observations: every storm, and
    every block and degradation not inspected."""
    name = period.name
    lines = [f"(: storm-{region}-{name} (Storm {region} {name}) {_tv(value)})" for region, value in period.storms.items()]
    lines += [
        f"(: blocked-{route}-{name} (Blocked {route} {name}) {_tv(value)})"
        for route, value in period.blocked.items()
        if route not in observation.inspected
    ]
    lines += [
        f"(: degraded-{site}-{name} (Degraded {site} {name}) {_tv(value)})"
        for site, value in period.degraded.items()
        if site not in observation.degraded
    ]
    return lines


def query(key: Key) -> str:
    predicate, subject, period = key
    return f"(: $prf ({predicate} {subject} {period}) $tv)"
