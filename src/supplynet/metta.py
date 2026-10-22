"""The MeTTa a reasoner receives: the network's rules with their known rates
(or, with learned rules, with a weak prior in their rates' place), labelled
periods as facts, a period's observations, and belief queries."""

from __future__ import annotations

from .world import MINE, POWER, Action, Key, Network, Observation, Period, Rates, decides


def _number(value: float) -> str:
    return f"{value:.6g}"


def _tv(value: bool, confidence: float = 1) -> str:
    return f"(STV {1 if value else 0} {_number(confidence)})"


PRIOR = "(CTV (STV 0.5 0.02) (STV 0.5 0.02))"  # a learned rule's weak prior on both branches


def _certain(name: str, antecedent: str, consequent: str) -> str:
    return f"(: {name} (Implication {antecedent} {consequent}) (CTV (STV 1 1) (STV 0 1)))"


def rules(network: Network, rates: Rates, learned: bool = False, positive_actions: bool = False) -> list[str]:
    """A storm blocks each route of its region with the route's rate, and a
    blocked route delays each shipment departing on it; when storms persist, a
    storm continues into the next period or starts with the region's rate.
    With a cell, a degradation persists or starts likewise (with decisions,
    unless the site was repaired), and the cycle's production rules hold with
    certainty. All stated as certain CTVs, so the rates are given; with
    ``learned`` every rule but the certain production rules states its
    structure only, with the weak ``PRIOR`` in place of its rates, which are
    left for the labelled history. ``positive_actions``: the chainer's
    encoding of stage 4's actions (``_cycle_rules``)."""
    nodes = rates.nodes(network).table

    def ctv(node: tuple[str, str]) -> str:
        return PRIOR if learned else "(CTV (STV {} 1) (STV {} 1))".format(*map(_number, nodes[node]))

    lines = []
    if rates.persist is not None:
        lines += [
            f"(: persist-{region} (Implication (And (NextPeriod $p $t) (Storm {region} $p)) (Storm {region} $t)) {ctv(('Storm', region))})"
            for region in network.regions
        ]
    for route in network.routes:
        lines.append(f"(: block-{route.name} (Implication (Storm {route.region} $p) (Blocked {route.name} $p)) {ctv(('Blocked', route.name))})")
        lines += [
            f"(: late-{shipment} (Implication (Blocked {route.name} $p) (Late {shipment} $p)) {ctv(('Late', shipment))})"
            for shipment in route.shipments
        ]
    if network.cell:
        repaired = (lambda site: f" (Not (Repairing {site} $p))") if network.cell.alternate else (lambda site: "")
        lines += [
            f"(: persist-degraded-{site} (Implication (And (NextPeriod $p $t) (Degraded {site} $p){repaired(site)}) (Degraded {site} $t)) "
            f"{ctv(('Degraded', site))})"
            for site in network.cell.sites
        ]
    return lines + (_cycle_rules(network, positive_actions) if network.cell else [])


def _cycle_rules(network: Network, positive_actions: bool = False) -> list[str]:
    """The power plant runs when fuelled, by stock or by the mine's fuel
    arriving over the open fuel route, and is not degraded; the mine and the
    plants run when the power plant does, their input arrived, and they are
    not degraded. Timed, the fuel arriving now left in the previous period;
    untimed, it is the mine's production now, and the rules form a loop. With
    decisions the fuel leaves over the route the period's ``FuelVia`` names,
    and a site under repair does not run.

    WORKAROUND (PeTTaChainer master 2307ca50, docs/supply_network.md, "Stage
    4 as built", gap 1) with ``positive_actions``: in an ``Assuming``
    context the chainer never reads an unstated complete statement as false,
    so a decision that does not assume a repair or a stock leaves every rule
    reading it without an answer. The chainer's rules therefore read only
    positive action facts, one for each branch: a site runs while
    ``(InService site t)``, and the power plant is fuelled by one rule per
    stock state, ``StockedFuel`` or ``Unstocked``, exactly one of which is
    stated or assumed in a period, so the two never give a view together.
    The semantic rules (ProbLog's) keep the ``Or`` and ``Not Repairing``.

    Production remains invertible so failed output reports diagnose degradation.
    Decision and choice rules retain their own no_inverse declarations.
    """
    cell = network.cell
    if cell.alternate:
        arriving = f"(And (NextPeriod $p $t) (Producing {MINE} $p) (FuelVia $r $p) (Not (Blocked $r $p)))"
    elif cell.timed:
        arriving = f"(And (NextPeriod $p $t) (Producing {MINE} $p) (Not (Blocked {cell.fuel_route.name} $p)))"
    else:
        arriving = f"(And (Producing {MINE} $t) (Not (Blocked {cell.fuel_route.name} $t)))"
    if positive_actions:
        repairing = lambda site: f" (InService {site} $t)"  # noqa: E731
        fuelled = [f"(: fuelled-by-stock (Implication (StockedFuel $t) (Fuelled {POWER} $t)) (STV 1 1))",
                   f"(: fuelled-by-mine (Implication (And (Unstocked $t) (FuelArrived $t)) (Fuelled {POWER} $t)) (STV 1 1))"]
    else:
        repairing = (lambda site: f" (Not (Repairing {site} $t))") if cell.alternate else (lambda site: "")
        fuelled = [_certain("fuelled", "(Or (StockedFuel $t) (FuelArrived $t))", f"(Fuelled {POWER} $t)")]
    lines = [
        _certain("fuel-arrives", arriving, "(FuelArrived $t)"),
        *fuelled,
        _certain(f"runs-{POWER}", f"(And (Fuelled {POWER} $t) (Not (Degraded {POWER} $t)){repairing(POWER)})", f"(Producing {POWER} $t)"),
        _certain(f"runs-{MINE}", f"(And (Producing {POWER} $t) (Not (Degraded {MINE} $t)){repairing(MINE)})", f"(Producing {MINE} $t)"),
    ]
    lines += [
        _certain(
            f"runs-{plant}",
            f"(And (Producing {POWER} $t) (InputArrived {plant} $t) (Not (Degraded {plant} $t)){repairing(plant)})",
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


def positive_only_predicates(network: Network) -> list[str]:
    """With decisions, the predicates ``action_facts`` states only when true:
    the route the fuel left on and the stock state. Undeclared, their stored
    instances would be the chainer's sample of their population, a rate near
    1, so an absent instance, such as ``(FuelVia r1 t)`` for a route the fuel
    did not take, would read as almost certainly true; an antecedent
    completion of the fuel rule would then read the period's predicted fuel
    arrival as evidence about that route's block (PeTTaChainer #119). Stage
    5 declares the same three (``lookahead_worker``)."""
    return ["FuelVia", "StockedFuel", "Unstocked"] if decides(network) else []


def predicate_times(network: Network) -> dict[str, tuple[int, int]]:
    """Where the predicates with learned rules sit in time, for a knowledge
    base mining its own hypotheses (PeTTaChainer set-predicate-time): each
    fact refers to the period in its last argument (a shipment's lateness to
    its departure period), whenever it arrives, and within a period a storm
    or a degradation comes before the blocks it causes, and a block before
    the lateness. The period links are NextPeriod facts (set-time-link). The
    production predicates, whose rules are given, are not mined."""
    return {"Storm": (2, 0), "Degraded": (2, 0), "Blocked": (2, 1), "Late": (2, 2)}


def observation_facts(observation: Observation) -> list[str]:
    """The period's link to its predecessor, the lateness of the shipments
    arriving now (keyed by departure period), and the inspected routes; with a
    cell also the inspected sites, the fuel stock, the plants' inputs, and the
    previous period's production, at the observation's report confidence."""
    period, previous = observation.period, observation.previous
    lines = [f"(: next-{period} (NextPeriod {previous} {period}) (STV 1 1))"] if previous else []
    lines += [f"(: late-{shipment}-{departure} (Late {shipment} {departure}) {_tv(value)})" for (shipment, departure), value in observation.late.items()]
    lines += [f"(: blocked-{route}-{period} (Blocked {route} {period}) {_tv(value)})" for route, value in observation.inspected.items()]
    lines += [f"(: degraded-{site}-{period} (Degraded {site} {period}) {_tv(value)})" for site, value in observation.degraded.items()]
    if observation.stocked is not None:
        lines.append(f"(: stocked-{period} (StockedFuel {period}) {_tv(observation.stocked)})")
    lines += [f"(: input-{plant}-{period} (InputArrived {plant} {period}) {_tv(value)})" for plant, value in observation.inputs.items()]
    lines += [f"(: producing-{site}-{previous} (Producing {site} {previous}) {_tv(value, observation.report_confidence)})" for site, value in observation.reported.items()]
    return lines


def action_facts(network: Network, period: str, action: Action) -> list[str]:
    """A taken action as the chainer's facts (``rules`` with
    ``positive_actions``): the route the fuel left on, the stock state as
    ``StockedFuel`` or ``Unstocked``, every site's repair, true or false, and
    ``InService`` for each site not repaired."""
    return [
        f"(: via-{period} (FuelVia {action.via(network.cell)} {period}) (STV 1 1))",
        f"(: stocked-{period} (StockedFuel {period}) (STV 1 1))" if action.stock else f"(: unstocked-{period} (Unstocked {period}) (STV 1 1))",
        *(f"(: repairing-{site}-{period} (Repairing {site} {period}) {_tv(site in action.repairs)})" for site in network.cell.sites),
        *(f"(: in-service-{site}-{period} (InService {site} {period}) (STV 1 1))" for site in network.cell.sites if site not in action.repairs),
    ]


def decision_rules(network: Network, values) -> list[str]:
    """A decision's outcomes, read from the beliefs (PeTTaChainer ``Assuming``
    queries, as in StationOps' single-KB actions): the sites' production, the
    mine's fuel sent over the route taken if it is open, and the
    degradations the repairs clear. Each outcome carries its amount, which a
    stored value fact gives, so a decision's expected value is the sum of
    its outcomes' amounts weighted by their truth. The sum is the client's:
    an in-KB ``FoldAllTruth`` over the outcomes misses members at some
    budgets (docs/supply_network.md, "Stage 4 as built"). Every rule is
    (no_inverse ...): a decision is never evidence for the beliefs it reads."""
    cell = network.cell
    certain = "(CTV (STV 1 1) (STV 0 1))"
    lines = [
        f"(: (no_inverse output-outcome) (Implication (And (OutputValue $site $v) (Producing $site $t)) (Outcome $t (Output $site) $v)) {certain})",
        f"(: (no_inverse fuel-outcome) (Implication (And (FuelValue $v) (FuelVia $r $t) (Producing {MINE} $t) (Not (Blocked $r $t))) "
        f"(Outcome $t FuelSent $v)) {certain})",
        f"(: (no_inverse repair-outcome) (Implication (And (RepairValue $site $v) (Repairing $site $t) (Degraded $site $t)) "
        f"(Outcome $t (Cleared $site) $v)) {certain})",
        f"(: fuel-value (FuelValue {_number(values.fuel)}) (STV 1 1))",
    ]
    for site, kind in cell.sites.items():
        lines.append(f"(: output-value-{site} (OutputValue {site} {_number(values.output[kind])}) (STV 1 1))")
        lines.append(f"(: repair-value-{site} (RepairValue {site} {_number(values.repair[kind])}) (STV 1 1))")
    return lines


def decision_query(network: Network, period: str, action: Action) -> str:
    """The ``Assuming`` query of one joint action's outcomes, assuming what
    ``action_facts`` would state, its true facts only."""
    facts = [f"(FuelVia {action.via(network.cell)} {period})", f"({'StockedFuel' if action.stock else 'Unstocked'} {period})"]
    facts += [f"(Repairing {site} {period})" for site in action.repairs]
    facts += [f"(InService {site} {period})" for site in network.cell.sites if site not in action.repairs]
    return f"(: $prf (Assuming (And {' '.join(facts)}) (Outcome {period} $o $v)) $tv)"


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


def review(rule: str) -> str | None:
    """The ``RuleTruth`` query of a learned rule, one stated with ``PRIOR``,
    which folds the rule's labelled instances; None for a rule with rates."""
    if not rule.endswith(f" {PRIOR})"):
        return None
    implication = rule[: -len(PRIOR) - 2].split(" ", 2)[2]
    return f"(: $prf (RuleTruth {implication}) $tv)"


def query(key: Key) -> str:
    predicate, subject, period = key
    return f"(: $prf ({predicate} {subject} {period}) $tv)"
