"""ProbLog over the conflicting-sources stage: one program per round, ground,
compiled and evaluated in a forked child (``supplynet.problog_backend.solve``).

- ``oracle``: the generating model written out. Each component is up with its
  prior; each system is up with P(up | both parents up) or P(up | not both)
  (two clauses with exclusive bodies, so the CTV is exact); each claim (one
  per source and node) is a probabilistic atom ``claim_i`` that holds with the
  source's P(claim up | up) when the node is up and P(claim up | down) when it
  is down, and is conditioned on with ``evidence(claim_i, true|false)``. With
  the true rates this is the exact posterior.
- ``learned``: the same program with the component priors and the sources'
  rates estimated from the labelled rounds (Laplace counts, ``learned_rates``),
  re-estimated every round.
- ``naive``: every claim copy is a probabilistic fact about its node with the
  pooled share of correct claims t in the history. A node is up when something
  supports it (its prior, or any up-claim copy: a noisy-OR) and no down-claim
  copy denies it (a noisy-OR of denials); a system is also supported through
  its rule from its parents. No source identity, copies counted."""

from __future__ import annotations

from supplynet.problog_backend import load, solve, summary

from .world import Report, Round, World, learned_rates, true_rates


def _p(value: float) -> str:
    return f"{min(1.0, max(0.0, value)):.10g}"


def model_program(world: World, rates, reports: tuple[Report, ...]) -> str:
    """The oracle/learned program: components, systems, claims as evidence."""
    lines = [f"{_p(rates.priors[c])}::up({c})." for c in world.components]
    for s in world.systems:
        a, b = s.parents
        lines += [
            f"both({s.name}) :- up({a}), up({b}).",
            f"{_p(s.given_both)}::up({s.name}) :- both({s.name}).",
            f"{_p(s.given_not)}::up({s.name}) :- \\+both({s.name}).",
        ]
    seen = set()
    for index, r in enumerate(reports):
        if (r.source, r.node) in seen:
            continue
        seen.add((r.source, r.node))
        given_up, given_down = rates.reliability[r.source]
        lines += [
            f"{_p(given_up)}::claim({index}) :- up({r.node}).",
            f"{_p(given_down)}::claim({index}) :- \\+up({r.node}).",
            f"evidence(claim({index}), {'true' if r.up else 'false'}).",
        ]
    return "\n".join([*lines, *(f"query(up({n}))." for n in world.nodes), ""])


def naive_program(world: World, priors: dict[str, float], trust: float, reports: tuple[Report, ...]) -> str:
    lines = [f"{_p(priors[c])}::support({c})." for c in world.components]
    for s in world.systems:
        a, b = s.parents
        lines += [
            f"both({s.name}) :- up({a}), up({b}).",
            f"{_p(s.given_both)}::support({s.name}) :- both({s.name}).",
            f"{_p(s.given_not)}::support({s.name}) :- \\+both({s.name}).",
        ]
    for r in reports:
        lines += [f"{_p(trust)}::{'support' if r.up else 'denied'}({r.node})."] * r.copies
    lines += [f"up({n}) :- support({n}), \\+denied({n})." for n in world.nodes]
    return "\n".join([*lines, *(f"query(up({n}))." for n in world.nodes), ""])


class ProblogBackend:
    def __init__(self, model: str, engine: str = "ddnnf", timeout: float = 60):
        assert model in ("naive", "oracle", "learned"), model
        load(engine)
        self.model, self.engine, self.timeout = model, engine, timeout
        self.name = f"problog-{model}"
        self.rounds: list[dict] = []

    def begin(self, world: World, history: list[Round]) -> None:
        self.world, self.history = world, list(history)

    def program(self, day: Round) -> str:
        if self.model == "oracle":
            return model_program(self.world, true_rates(self.world), day.reports)
        rates = learned_rates(self.world, self.history)
        if self.model == "learned":
            return model_program(self.world, rates, day.reports)
        claims = [(r.up == past.truth[r.node], r.copies) for past in self.history for r in past.reports]
        trust = (sum(n for right, n in claims if right) + 1) / (sum(n for _, n in claims) + 2)
        return naive_program(self.world, rates.priors, trust, day.reports)

    def beliefs(self, day: Round, keys: list[tuple[str, str]], budget: int) -> dict:
        probabilities, record = solve(self.program(day), [f"up({node})" for node, _ in keys], self.engine, self.timeout)
        self.rounds.append(record)
        return dict(zip(keys, probabilities, strict=True)) if probabilities is not None else {}

    def resolve(self, day: Round) -> None:
        self.history.append(day)

    def stats(self) -> dict:
        return {"problog_engine": self.engine, **summary(self.rounds)}
