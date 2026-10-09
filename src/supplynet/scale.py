"""SupplyNet stage "scale": more relevant evidence per query than a search
budget can reach, with an exact posterior that stays cheap at any size.

Each zone has a weather front, a two-state chain over periods. Per period the
zone is a tree hanging from its front: front -> region storms -> route blocks
-> tiers of delayed depots -> late shipments, and storm -> weather alarms.
Zones are independent. Within a zone, every storm, block and delay of a
period depends on every alarm and late shipment of the zone over the whole
unresolved window, through the front's chain; the exact posterior is
forward-backward over the front with each period's likelihood summed up its
tree (docs/supply_network.md, stage "scale").

Every non-root node is one CTV implication from its parent, so the module
also states the network as the MeTTa a reasoner receives; it serves as the
``statements`` of ``backends.PeTTaChainerBackend``.
"""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass, field, replace

from .game import _mean, score
from .metta import query  # noqa: F401  (part of the statements interface)
from .world import onset, stationary

Var = tuple[str, str]  # (predicate, subject)
Key = tuple[str, str, str]  # (predicate, subject, period)


@dataclass(frozen=True)
class Rates:
    front_start: float = 0.05
    front_persist: float = 0.9
    storm: dict[str, tuple[float, float]] = field(default_factory=lambda: {"coastal": (0.7, 0.05), "inland": (0.5, 0.02)})
    block: dict[str, tuple[float, float]] = field(default_factory=lambda: {"exposed": (0.8, 0.03), "sheltered": (0.4, 0.03)})
    delay: tuple[float, float] = (0.85, 0.05)  # P(delayed | upstream blocked or delayed), P(delayed | not)
    late: tuple[float, float] = (0.9, 0.05)
    alarm: tuple[float, float] = (0.6, 0.25)

    def front_rate(self, previous: float) -> float:
        return onset(self.front_start, self.front_persist, previous)

    @property
    def front_stationary(self) -> float:
        return stationary(self.front_start, self.front_persist)


@dataclass(frozen=True)
class Node:
    """A node below a front: ``given`` and ``without`` are P(true) when its
    parent is true and false; ``seen`` is the delay after which an observed
    node (a shipment's lateness, an alarm) is seen, None for a hidden one."""

    parent: Var
    given: float
    without: float
    seen: int | None = None

    def p(self, parent: bool) -> float:
        return self.given if parent else self.without


@dataclass(frozen=True)
class Size:
    zones: int = 2
    regions: int = 3  # per zone
    routes: int = 2  # per region
    tiers: int = 2  # hops from a route's block to a shipment's lateness
    fanout: int = 2  # children of a block or a depot
    sensors: int = 2  # alarms per region


# Statements over the default 20 history periods and 10 rounds, against
# stage 2's 1.6k (3 regions, 30 history periods, 20 rounds): s 1.4k (1x),
# m 14k (9x), l 32k (20x), xl 122k (75x).
SIZES = {
    "s": Size(1, 3, 2, 2, 2, 2),
    "m": Size(3, 4, 3, 2, 3, 3),
    "l": Size(4, 6, 3, 3, 2, 3),
    "xl": Size(6, 6, 3, 3, 3, 3),
}


@dataclass(frozen=True)
class Network:
    fronts: tuple[Var, ...]
    nodes: dict[Var, Node]  # parents before children

    @property
    def max_seen(self) -> int:
        return max(node.seen or 0 for node in self.nodes.values())

    def hidden(self) -> list[Var]:
        return [*self.fronts, *(var for var, node in self.nodes.items() if node.seen is None)]


def generate_network(rng: random.Random, size: Size, rates: Rates) -> Network:
    """Region and route kinds and route transits (1-3) are drawn; the shape is
    fixed by ``size``. Names: zone z1, region g1, alarm a1-1, route k1, its
    depots d1-1.., its shipments s1-1.."""
    fronts = []
    nodes: dict[Var, Node] = {}
    region = route = 0
    for zone in range(1, size.zones + 1):
        front = ("Front", f"z{zone}")
        fronts.append(front)
        for _ in range(size.regions):
            region += 1
            storm = ("Storm", f"g{region}")
            nodes[storm] = Node(front, *rates.storm[rng.choice(("coastal", "inland"))])
            for sensor in range(1, size.sensors + 1):
                nodes[("Alarm", f"a{region}-{sensor}")] = Node(storm, *rates.alarm, seen=0)
            for _ in range(size.routes):
                route += 1
                transit = rng.randint(1, 3)
                level = [("Blocked", f"k{route}")]
                nodes[level[0]] = Node(storm, *rates.block[rng.choice(("exposed", "sheltered"))])
                depots = 0
                for tier in range(1, size.tiers + 1):
                    last = tier == size.tiers
                    parents = [parent for parent in level for _ in range(size.fanout)]
                    if last:
                        level = [("Late", f"s{route}-{index}") for index in range(1, len(parents) + 1)]
                    else:
                        level = [("Delayed", f"d{route}-{depots + index}") for index in range(1, len(parents) + 1)]
                        depots += len(parents)
                    for child, parent in zip(level, parents):
                        nodes[child] = Node(parent, *(rates.late if last else rates.delay), seen=transit if last else None)
    return Network(tuple(fronts), nodes)


@dataclass(frozen=True)
class Period:
    name: str
    previous: str | None
    values: dict[Var, bool]  # every front and node

    def labels(self) -> dict[Var, bool]:
        return {var: value for var, value in self.values.items() if var[0] not in ("Late", "Alarm")}


@dataclass(frozen=True)
class Observation:
    period: str
    previous: str | None
    seen: dict[Key, bool]  # observed nodes, keyed by the period they belong to


def sample_period(network: Network, rates: Rates, rng: random.Random, name: str, previous: Period | None) -> Period:
    values = {
        front: rng.random() < (rates.front_rate(float(previous.values[front])) if previous else rates.front_stationary)
        for front in network.fronts
    }
    for var, node in network.nodes.items():
        values[var] = rng.random() < node.p(values[node.parent])
    return Period(name, previous.name if previous else None, values)


class Knowledge:
    """The posterior from the last resolved period's labels and the
    observations of the unresolved window since.

    Exact: per zone and window period, upward messages sum each node's subtree
    evidence into its parent, giving the front's likelihood; forward-backward
    over the front's chain gives its marginals, carried in from the resolved
    front (a point mass) or the stationary rate; a downward pass gives every
    hidden node's. Local: the cheap neighbourhood reasoner, each query from
    its own period alone and the stationary front, a front from its zone and
    a storm or block from its region."""

    def __init__(self, network: Network, rates: Rates, local: bool = False):
        self.network, self.rates, self.local = network, rates, local
        self.window: list[str] = []
        self.seen: dict[Key, bool] = {}
        self.before = dict.fromkeys(network.fronts, rates.front_stationary)
        self.children: dict[Var, list[Var]] = {var: [] for var in [*network.fronts, *network.nodes]}
        for var, node in network.nodes.items():
            self.children[node.parent].append(var)

    def observe(self, observation: Observation) -> None:
        self.window.append(observation.period)
        self.seen.update(observation.seen)

    def resolve(self, period: Period) -> None:
        assert self.window[0] == period.name
        self.window.pop(0)
        self.before = {front: float(period.values[front]) for front in self.network.fronts}
        self.seen = {key: value for key, value in self.seen.items() if key[2] != period.name}

    def _up(self, period: str) -> dict[Var, list[float]]:
        """up[v][x]: the likelihood of v's subtree evidence given v = x, for
        every hidden node, normalized (a node's scale cancels in its posterior)."""
        up = {}
        nodes = self.network.nodes
        for var in reversed([*self.network.fronts, *nodes]):
            node = nodes.get(var)
            if node is not None and node.seen is not None:
                continue
            u = [1.0, 1.0]
            for child in self.children[var]:
                c = nodes[child]
                if c.seen is None:
                    m = [sum((c.p(y) if x else 1 - c.p(y)) * up[child][x] for x in (0, 1)) for y in (0, 1)]
                else:
                    value = self.seen.get((*child, period))
                    if value is None:
                        continue
                    m = [c.p(y) if value else 1 - c.p(y) for y in (0, 1)]
                u = [u[0] * m[0], u[1] * m[1]]
                total = u[0] + u[1]
                u = [u[0] / total, u[1] / total]
            up[var] = u
        return up

    def _fronts(self, ups: list[dict]) -> dict[Var, list[float]]:
        """P(front true) per window period, by forward-backward."""
        rates = self.rates
        marginals = {}
        for front in self.network.fronts:
            forward = []
            belief = self.before[front]
            for up in ups:
                prior = rates.front_rate(belief)
                a = [(1 - prior) * up[front][0], prior * up[front][1]]
                belief = a[1] / (a[0] + a[1])
                forward.append(belief)
            beta = [1.0, 1.0]
            smoothed = [0.0] * len(ups)
            for index in range(len(ups) - 1, -1, -1):
                f = forward[index]
                post = [(1 - f) * beta[0], f * beta[1]]
                smoothed[index] = post[1] / (post[0] + post[1])
                # beta for the previous period: sum over this period's front.
                like = [up_x * b for up_x, b in zip(ups[index][front], beta)]
                beta = [
                    (1 - rates.front_rate(float(y))) * like[0] + rates.front_rate(float(y)) * like[1] for y in (0, 1)
                ]
                total = beta[0] + beta[1]
                beta = [beta[0] / total, beta[1] / total]
            marginals[front] = smoothed
        return marginals

    def posterior(self) -> dict[Key, float]:
        """P(var true) for every hidden var in every window period."""
        network, nodes = self.network, self.network.nodes
        ups = [self._up(period) for period in self.window]
        pi = self.rates.front_stationary
        fronts = {front: [pi] * len(ups) for front in network.fronts} if self.local else self._fronts(ups)
        posterior = {}
        for index, (period, up) in enumerate(zip(self.window, ups)):
            marginal: dict[Var, float] = {}
            for front in network.fronts:
                p = fronts[front][index]
                if self.local:
                    a = [(1 - p) * up[front][0], p * up[front][1]]
                    marginal[front] = a[1] / (a[0] + a[1])
                else:
                    marginal[front] = p
            for var, node in nodes.items():
                if node.seen is not None:
                    continue
                parent = node.parent
                if self.local and parent in network.fronts:
                    # A storm from its region alone, under the stationary front.
                    q = (1 - pi) * node.without + pi * node.given
                    a = [(1 - q) * up[var][0], q * up[var][1]]
                    marginal[var] = a[1] / (a[0] + a[1])
                    continue
                p_parent = marginal[parent]
                total = 0.0
                for y, weight in ((0, 1 - p_parent), (1, p_parent)):
                    py = node.p(y)
                    a = [(1 - py) * up[var][0], py * up[var][1]]
                    total += weight * a[1] / (a[0] + a[1])
                marginal[var] = total
            posterior.update({(*var, period): p for var, p in marginal.items()})
        return posterior


# The statements a reasoner receives (the backends' ``statements`` interface).


def _number(value: float) -> str:
    return f"{value:.6g}"


def _fact(key: Key, value: bool) -> str:
    predicate, subject, period = key
    return f"(: {predicate.lower()}-{subject}-{period} ({predicate} {subject} {period}) (STV {int(value)} 1))"


def rules(network: Network, rates: Rates) -> list[str]:
    """A front persists or starts; every node follows its parent with its two
    rates. All certain CTVs: the rates are given."""
    lines = [
        f"(: persist-{zone} (Implication (And (NextPeriod $p $t) (Front {zone} $p)) (Front {zone} $t)) "
        f"(CTV (STV {_number(rates.front_persist)} 1) (STV {_number(rates.front_start)} 1)))"
        for _, zone in network.fronts
    ]
    lines += [
        f"(: {predicate.lower()}-{subject} (Implication ({node.parent[0]} {node.parent[1]} $t) ({predicate} {subject} $t)) "
        f"(CTV (STV {_number(node.given)} 1) (STV {_number(node.without)} 1)))"
        for (predicate, subject), node in network.nodes.items()
    ]
    return lines


def complete_predicates(network: Network) -> list[str]:
    return []


def observation_facts(observation: Observation) -> list[str]:
    period, previous = observation.period, observation.previous
    lines = [f"(: next-{period} (NextPeriod {previous} {period}) (STV 1 1))"] if previous else []
    return lines + [_fact(key, value) for key, value in observation.seen.items()]


def resolution_facts(period: Period, observation: Observation) -> list[str]:
    return [_fact((*var, period.name), value) for var, value in period.labels().items()]


# The game.


@dataclass(frozen=True)
class GameConfig:
    seed: int = 7
    size: Size = SIZES["s"]
    window: int = 8  # periods resolve this many rounds after they are observed
    history: int = 20
    rounds: int = 10
    budget: int | None = None  # one expansion budget per round, shared by its queries
    steps_per_query: float = 4.0  # the budget when ``budget`` is None


def run_game(config: GameConfig, backend, rates: Rates | None = None, on_round=None) -> dict:
    """Each round observes a new period: the alarms of that period and the
    shipments arriving in it. The queries are every front, storm and block
    now, and every front and storm of the oldest window period, which the
    window's later evidence smooths."""
    rates = rates or Rates()
    rng = random.Random(config.seed)
    network = generate_network(rng, config.size, rates)
    assert config.window > network.max_seen, "a period must resolve after its last shipment arrives"
    observed = [var for var, node in network.nodes.items() if node.seen is not None]
    knowledge = Knowledge(network, rates)
    timeline: list[Period] = []

    def next_period(name):
        timeline.append(sample_period(network, rates, rng, name, timeline[-1] if timeline else None))
        return timeline[-1]

    history = []
    statements = len(rules(network, rates))
    for index in range(config.history):
        period = next_period(f"h{index + 1}")
        observation = Observation(period.name, period.previous, {(*var, period.name): period.values[var] for var in observed})
        knowledge.observe(observation)
        knowledge.resolve(period)
        history.append((period, observation))
        statements += len(observation_facts(observation)) + len(resolution_facts(period, observation))
    started = time.perf_counter()
    backend.begin(network, rates, history)
    setup_seconds = time.perf_counter() - started
    setup_statements = statements

    first = len(timeline)
    unresolved = []
    rounds = []
    for index in range(config.rounds):
        period = next_period(f"t{index + 1}")
        now = len(timeline) - 1
        seen = {}
        for var in observed:
            departed = now - network.nodes[var].seen
            if departed >= first:
                seen[(*var, timeline[departed].name)] = timeline[departed].values[var]
        observation = Observation(period.name, period.previous, seen)
        knowledge.observe(observation)
        oldest = knowledge.window[0]
        hidden = network.hidden()
        keys = [(*var, period.name) for var in hidden if var[0] != "Delayed"]
        if oldest != period.name:
            keys += [(*var, oldest) for var in hidden if var[0] in ("Front", "Storm")]
        full = knowledge.posterior()
        posterior = {key: full[key] for key in keys}
        by_name = {p.name: p for p in timeline[first:]}
        actual = {key: by_name[key[2]].values[key[:2]] for key in keys}

        def kind_of(key):
            return key[0].lower() if key[2] == period.name else f"past_{key[0].lower()}"

        budget = config.budget if config.budget is not None else max(1, round(config.steps_per_query * len(keys)))
        started = time.perf_counter()
        beliefs = backend.beliefs(observation, sorted(keys), budget)
        seconds = time.perf_counter() - started
        statements += len(observation_facts(observation))
        unresolved.append((period, observation))
        if len(unresolved) == config.window:
            done, done_seen = unresolved.pop(0)
            knowledge.resolve(done)
            backend.resolve(done, done_seen)
            statements += len(resolution_facts(done, done_seen))
        record = {
            "type": "round",
            "round": index + 1,
            "budget": budget,
            "seconds": round(seconds, 3),
            **score(beliefs, actual, posterior, kind_of),
        }
        rounds.append(record)
        if on_round:
            on_round(record)

    kinds = sorted({kind for r in rounds for kind in r["kinds"]})
    keys = sum(r["keys"] for r in rounds)
    seconds = sum(r["seconds"] for r in rounds)
    return {
        "type": "run-summary",
        "backend": backend.name,
        "stage": "scale",
        "seed": config.seed,
        **vars(config.size),
        "window": config.window,
        "history": config.history,
        "rounds": config.rounds,
        "budget": config.budget,
        "steps_per_query": config.steps_per_query if config.budget is None else None,
        "queries_per_round": keys / len(rounds),
        "hidden_per_period": len(network.hidden()),
        "observed_per_period": len(observed),
        "setup_statements": setup_statements,
        "statements": statements,
        "setup_seconds": round(setup_seconds, 3),
        "seconds": round(seconds, 3),
        "ms_per_query": round(1000 * seconds / keys, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **_mean(rounds),
        "kinds": {kind: _mean([r["kinds"][kind] for r in rounds if kind in r["kinds"]]) for kind in kinds},
        **(backend.summary() if hasattr(backend, "summary") else {}),
    }


def sized(name: str | None = None, **knobs) -> Size:
    """A named size with any knob given (not None) replaced."""
    return replace(SIZES[name] if name else Size(), **{k: v for k, v in knobs.items() if v is not None})
