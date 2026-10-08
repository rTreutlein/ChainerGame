"""The MeTTa a reasoner receives: the network's rules with their known rates,
labelled periods as facts, a period's observations, and belief queries."""

from __future__ import annotations

from .world import Network, Observation, Period, Rates


def _number(value: float) -> str:
    return f"{value:.6g}"


def _tv(value: bool) -> str:
    return f"(STV {1 if value else 0} 1)"


def rules(network: Network, rates: Rates) -> list[str]:
    """A storm blocks each route of its region with the route's rate, and a
    blocked route delays each of its shipments; both stated as certain CTVs, so
    the rates are given, not learned."""
    lines = []
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
    return lines


def observation_facts(observation: Observation) -> list[str]:
    period = observation.period
    lines = [f"(: late-{shipment}-{period} (Late {shipment} {period}) {_tv(value)})" for shipment, value in observation.late.items()]
    lines += [f"(: blocked-{route}-{period} (Blocked {route} {period}) {_tv(value)})" for route, value in observation.inspected.items()]
    return lines


def resolution_facts(period: Period, observation: Observation) -> list[str]:
    """What a resolved period adds beyond its observations: every storm and
    every block not inspected."""
    name = period.name
    lines = [f"(: storm-{region}-{name} (Storm {region} {name}) {_tv(value)})" for region, value in period.storms.items()]
    lines += [
        f"(: blocked-{route}-{name} (Blocked {route} {name}) {_tv(value)})"
        for route, value in period.blocked.items()
        if route not in observation.inspected
    ]
    return lines


def query(key: tuple[str, str], period: str) -> str:
    predicate, subject = key
    return f"(: $prf ({predicate} {subject} {period}) $tv)"
