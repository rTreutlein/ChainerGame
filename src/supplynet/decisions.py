"""Stage 4 decisions: each period the operator stocks the fuel warehouse,
reroutes the mine's fuel over the sheltered alternate route, and repairs
sites, within a shared capacity (docs/supply_network.md, "Stage 4 as built").

A decision is scored by its expected utility under the exact posterior: the
period's production, the fuel it sends on for the next period, and the
degradations its repairs clear, less the actions' costs. The actions interact
through the cycle: a repaired power plant stops everything, so a stockpile
or a reroute with it is wasted; a repaired mine sends no fuel, so a reroute
with it is wasted; a stockpile that restarts the mine makes the reroute
worth more; and repairs taken together lose less production than one by one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .world import MINE, Action, Cell, Knowledge, production


@dataclass(frozen=True)
class Values:
    """What a period's outcomes are worth, per site kind, and what the
    actions cost; each action takes one unit of ``capacity``."""

    output: dict[str, float] = field(default_factory=lambda: {"mine": 2.0, "power": 1.0, "plant": 4.0})
    fuel: float = 8.0  # the mine's fuel sent over an open route, for the next period
    repair: dict[str, float] = field(default_factory=lambda: {"mine": 10.0, "power": 20.0, "plant": 8.0})  # a cleared degradation
    cost: dict[str, float] = field(default_factory=lambda: {"stock": 2.0, "reroute": 1.0, "repair": 1.0, "inspect": 0.5})
    capacity: int = 2

    def cost_of(self, action: Action) -> float:
        return (self.cost["stock"] * action.stock + self.cost["reroute"] * action.reroute
                + self.cost["repair"] * len(action.repairs) + self.cost["inspect"] * len(action.inspections))


def options(cell: Cell, capacity: int, inspect: bool = False) -> tuple[Action, ...]:
    """Every joint action within ``capacity``, fewer units first."""
    singles = [Action(stock=True), Action(reroute=True), *(Action(repairs=(site,)) for site in sorted(cell.sites))]
    if inspect:
        singles += [Action(inspections=(site,)) for site in sorted(cell.sites)]
    result = []
    for size in range(capacity + 1):
        for chosen in itertools.combinations(singles, size):
            result.append(combine(chosen))
    return tuple(result)


def combine(actions) -> Action:
    actions = list(actions)
    return Action(any(a.stock for a in actions), any(a.reroute for a in actions),
                  tuple(sorted(site for a in actions for site in a.repairs)),
                  tuple(sorted(site for a in actions for site in a.inspections)))


def singles(action: Action) -> list[Action]:
    """The single actions ``action`` joins."""
    return [*([Action(stock=True)] if action.stock else []), *([Action(reroute=True)] if action.reroute else []),
            *(Action(repairs=(site,)) for site in action.repairs), *(Action(inspections=(site,)) for site in action.inspections)]


@dataclass(frozen=True)
class Problem:
    """A period's decision: what production in it depends on besides the
    hidden state (the plants' inputs, observed, and the mine's previous
    production, reported), the feasible joint actions and the values."""

    period: str
    previous: str
    inputs: dict[str, bool]
    mine_before: bool
    fuel_before: str  # the route the fuel arriving now took
    options: tuple[Action, ...]
    values: Values


def utility(cell: Cell, problem: Problem, action: Action, degraded: dict[str, bool], fuel_blocked: bool, blocked_now: dict[str, bool]) -> float:
    """The period's outcome under ``action`` in one hidden state: its
    production, the fuel sent on over the route taken if it is open now, and
    the degradations its repairs clear, less the actions' cost."""
    values = problem.values
    producing = production(cell, degraded, action.stock, problem.inputs, fuel_blocked, problem.mine_before, action.repairs)
    gross = sum(values.output[cell.sites[site]] for site, on in producing.items() if on)
    gross += values.fuel * (producing[MINE] and not blocked_now[action.via(cell)])
    gross += sum(values.repair[cell.sites[site]] for site in action.repairs if degraded[site])
    return gross - values.cost_of(action)


def expected_utilities(knowledge: Knowledge, problem: Problem) -> dict[Action, float]:
    """Every option's expected utility under ``knowledge``'s posterior of the
    cell region's state now (with the block the arriving fuel met)."""
    cell = knowledge.network.cell
    states = []
    for (values, fuel_blocked), weight in knowledge.now().items():
        state = dict(values)
        states.append((weight, {site: state[("Degraded", site)] for site in cell.sites}, fuel_blocked,
                       {route.name: state[("Blocked", route.name)] for route in cell.fuel_routes}))
    return {action: sum(w * utility(cell, problem, action, d, f, b) for w, d, f, b in states) for action in problem.options}


def factored_utilities(cell: Cell, problem: Problem, marginals: dict[tuple[str, str, str], float]) -> dict[Action, float]:
    """Every option's expected utility with the hidden variables independent
    at their ``marginals``: each site's degradation and each fuel route's
    block now, and the block the arriving fuel met (the fuel's route in the
    previous period)."""
    period, previous = problem.period, problem.previous
    variables = [("Degraded", site, period) for site in cell.sites] + [("Blocked", route.name, period) for route in cell.fuel_routes]
    variables.append(("Blocked", problem.fuel_before, previous))
    states = []
    for values in itertools.product((True, False), repeat=len(variables)):
        weight = 1.0
        for var, value in zip(variables, values):
            p = marginals[var]
            weight *= p if value else 1 - p
        if weight:
            state = dict(zip(variables, values))
            states.append((weight, {site: state[("Degraded", site, period)] for site in cell.sites}, values[-1],
                           {route.name: state[("Blocked", route.name, period)] for route in cell.fuel_routes}))
    return {action: sum(w * utility(cell, problem, action, d, f, b) for w, d, f, b in states) for action in problem.options}


def best(utilities: dict[Action, float]) -> Action:
    """The option of highest expected utility; ties go to the first, which
    takes fewer units."""
    return max(utilities, key=utilities.get)


def propose(problem: Problem, evaluate, beam: int | None) -> dict[Action, float | None]:
    """The values of the options a decision asks for, ``evaluate(actions)``
    giving a list's values (None for one it cannot value). With no ``beam``
    every option; with one, doing nothing and every single action, then the
    ``beam`` best of the last layer each extended by one more single action,
    until the capacity is used: 1 + n queries for n single actions, and at
    most beam·(n − 1) per extra unit."""
    if beam is None:
        return evaluate(list(problem.options))
    feasible = set(problem.options)
    values = evaluate([action for action in problem.options if action.units <= 1])
    layer = single = [action for action in values if action.units == 1]
    for _ in range(problem.values.capacity - 1):
        top = sorted((action for action in layer if values[action] is not None), key=values.get, reverse=True)[:beam]
        joints = dict.fromkeys(combine([action, extra]) for action in top for extra in single if extra not in singles(action))
        layer = [joint for joint in joints if joint in feasible and joint not in values]
        values |= evaluate(layer)
    return values


def greedy(utilities: dict[Action, float], capacity: int) -> Action:
    """The single actions ranked by their own gain over doing nothing, taken
    while they gain and capacity lasts: blind to how they interact."""
    gains = {action: utilities[action] - utilities[Action()] for action in utilities if action.units == 1}
    chosen = [action for action in sorted(gains, key=gains.get, reverse=True) if gains[action] > 0][:capacity]
    return combine(chosen)
