"""ProbLog as an exact probabilistic-logic reasoner for SupplyNet.

The backend translates the MeTTa a reasoner receives (``metta`` for stages
1-3, ``scale`` for stage "scale") into a ProbLog program and asks ProbLog for
every belief key in one compilation per round. The mapping, and why it is
exact, is in docs/problog_backend.md:

- a CTV implication ``A -> B`` with P(B|A) = p and P(B|not A) = q becomes an
  antecedent predicate ``ant_i`` holding A, and ``p::b :- live(T), ant_i.``
  ``q::b :- live(T), \\+ant_i.``; the two bodies exclude each other, so the
  rule is exact, not a noisy-OR (every head has one rule);
- ``And`` is a conjunction (negated parts last, once their variables are
  bound), ``Or`` one clause of ``ant_i`` per part, ``Not`` is ``\\+``;
  variables an antecedent has beyond its head's are existential inside
  ``ant_i``;
- a period is ``live`` while it is unresolved; the rules hold only for live
  periods. The last resolved period is stated by its labels as plain facts,
  which d-separate the window from older history (as in the exact
  reference), so older periods are left out;
- a live period's observation of a statement some rule derives is
  ``evidence/2``; a statement no rule derives (``NextPeriod``,
  ``StockedFuel``, ``InputArrived``) is a fact when true and absent when
  false (ProbLog's closed world);
- a statement no rule derives but the round asks about (stage 1's storms)
  takes its given rate: ``r::storm(R, T) :- live(T).``;
- every key is a ``query/1``; the cyclic untimed production rules are read
  under ProbLog's least-fixpoint semantics, the same as the chainer's
  complete predicates.

Each round's program is ground, compiled and evaluated in a forked child, so a
round that exceeds its timeout (or its memory) is killed and its keys stay
unanswered."""

from __future__ import annotations

import multiprocessing
import os
import re
import resource
import signal
import tempfile
import time

from . import metta
from .world import Key, Network, Observation, Period, Rates


def parse(text: str):
    """One MeTTa expression as nested lists of atoms."""
    stack = [[]]
    for token in re.findall(r"\(|\)|[^\s()]+", text):
        if token == "(":
            stack.append([])
        elif token == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    (expr,) = stack[0]
    return expr


def argument(atom: str) -> str:
    return "V" + atom[1:].replace("-", "_") if atom.startswith("$") else f"'{atom}'"


def term(expr) -> str:
    """A MeTTa statement as a ProbLog term: ``(Storm north t5)`` is ``storm('north', 't5')``."""
    head, *args = expr
    return f"{head.lower()}({', '.join(map(argument, args))})"


def _variables(expr) -> list[str]:
    if isinstance(expr, str):
        return [argument(expr)] if expr.startswith("$") else []
    return list(dict.fromkeys(variable for part in expr for variable in _variables(part)))


def _literal(expr) -> str:
    assert expr[0] not in ("And", "Or") and (expr[0] != "Not" or isinstance(expr[1][0], str) and expr[1][0] not in ("And", "Or", "Not")), expr
    return f"\\+{term(expr[1])}" if expr[0] == "Not" else term(expr)


def _conjunction(expr) -> str:
    """An ``And`` of atoms and negated atoms, negations last so their variables are bound."""
    parts = expr[1:] if expr[0] == "And" else [expr]
    return ", ".join(sorted(map(_literal, parts), key=lambda literal: literal.startswith("\\+")))


def _probability(p: float) -> str:
    return "" if p == 1 else f"{p:.10g}::"


def rule(text: str, index: int) -> tuple[list[str], tuple[str, tuple[str, ...]]]:
    """A CTV implication as ProbLog clauses, and its head as (predicate, constant arguments)."""
    _, _, (implication, antecedent, consequent), (ctv, (_, p, _), (_, q, _)) = parse(text)
    assert implication == "Implication" and ctv == "CTV", text
    head, period = term(consequent), argument(consequent[-1])
    variables = ", ".join(_variables(consequent))
    name = f"ant_{index}({variables})"
    clauses = [f"{name} :- {_conjunction(part)}." for part in (antecedent[1:] if antecedent[0] == "Or" else [antecedent])]
    p, q = float(p), float(q)
    if p > 0:
        clauses.append(f"{_probability(p)}{head} :- live({period}), {name}.")
    if q > 0:
        clauses.append(f"{_probability(q)}{head} :- live({period}), \\+{name}.")
    return clauses, (consequent[0], tuple(consequent[1:-1]))


def given_priors(network: Network, rates: Rates) -> dict[tuple[str, tuple[str, ...]], float]:
    """Rates of statements no rule derives that a round asks about: with
    independent periods (stage 1), each region's storm."""
    return {} if rates.persist is not None else {("Storm", (region,)): rates.storm[kind] for region, kind in network.regions.items()}


def _infer(text: str, queries: list[str], engine: str, scratch: str, connection) -> None:
    """In a child process: ground, compile, evaluate, and send the query
    probabilities in order with the phases' seconds and the ground program's
    size, or the error that stopped it."""
    os.setpgrp()  # the parent kills the group on timeout, external compilers (dsharp) included
    tempfile.tempdir = scratch  # dsharp's CNF and NNF files, removed by the parent
    from problog import get_evaluatable
    from problog.engine import DefaultEngine, ground
    from problog.logic import Term
    from problog.program import PrologString

    try:
        started = time.perf_counter()
        # A statement no rule derives and no fact states is false (closed world), not an error.
        formula = ground(PrologString(text), unknown=DefaultEngine.UNKNOWN_FAIL)
        grounded = time.perf_counter()
        circuit = get_evaluatable(engine).create_from(formula)
        compiled = time.perf_counter()
        result = circuit.evaluate()
        evaluated = time.perf_counter()
        peak = max(resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))  # dsharp is a child
        phases = {
            "ground": grounded - started,
            "compile": compiled - grounded,
            "evaluate": evaluated - compiled,
            "ground_nodes": len(formula),
            "peak_mb": peak / 1024,
        }
        connection.send(("answered", [result[Term.from_string(query)] for query in queries], phases))
    except Exception as error:  # dsharp out of memory under the cap, a MemoryError, ...
        connection.send(("failed", None, {"error": f"{type(error).__name__}: {str(error)[:300]}"}))


def solve(text: str, queries: list[str], engine: str, timeout: float) -> tuple[list[float] | None, dict]:
    """The probabilities of ``queries`` under the program ``text`` (None
    unless answered) and the run's record: its outcome ("answered",
    "timeout" or "failed"), wall seconds, statements, and the child's phases
    or error."""
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    with tempfile.TemporaryDirectory(prefix="problog-") as scratch:
        child = context.Process(target=_infer, args=(text, queries, engine, scratch, send), daemon=True)
        started = time.perf_counter()
        child.start()
        send.close()
        outcome, probabilities, details = "timeout", None, {}
        if receive.poll(timeout):
            try:
                outcome, probabilities, details = receive.recv()
            except EOFError:
                outcome, details = "failed", {"error": f"exit {child.exitcode}"}  # killed, e.g. out of memory
        seconds = time.perf_counter() - started
        if child.is_alive():
            os.killpg(child.pid, signal.SIGKILL)
        child.join()
        receive.close()
    return probabilities, {"outcome": outcome, "seconds": seconds, "statements": text.count("\n"), **details}


def load(engine: str) -> None:
    """Import ProbLog and the engine once in the parent, so a forked round
    does not; an unknown or uninstalled engine fails here."""
    from problog import get_evaluatable
    from problog.engine import ground  # noqa: F401
    from problog.program import PrologString  # noqa: F401

    get_evaluatable(engine)


def summary(records: list[dict]) -> dict:
    """ProbLog's cost per round: outcomes, and the answered rounds' phases."""
    answered = [r for r in records if r["outcome"] == "answered"]
    mean = lambda field: round(sum(r[field] for r in answered) / len(answered), 4) if answered else None  # noqa: E731
    return {
        "problog_answered": len(answered),
        "problog_timeouts": sum(r["outcome"] == "timeout" for r in records),
        "problog_failed": sum(r["outcome"] == "failed" for r in records),
        "problog_ground_seconds": mean("ground"),
        "problog_compile_seconds": mean("compile"),
        "problog_evaluate_seconds": mean("evaluate"),
        "problog_ground_nodes": mean("ground_nodes"),
        "problog_statements": mean("statements"),
        "problog_max_round_seconds": round(max((r["seconds"] for r in records), default=0), 3),
        "problog_peak_mb": round(max((r["peak_mb"] for r in answered), default=0), 1),
        "problog_rounds": [[r["outcome"], round(r["seconds"], 3)] for r in records],
        "problog_errors": sorted({r["error"] for r in records if "error" in r}),
    }


class ProblogBackend:
    """ProbLog given the translated rules and, per round, the live periods'
    observations and the last resolved period's labels. ``engine`` is a
    ProbLog knowledge compiler ("sdd", "ddnnf", ...); a round taking longer
    than ``timeout`` seconds is killed and left unanswered."""

    name = "problog"

    def __init__(self, statements=metta, engine: str = "ddnnf", timeout: float = 60, priors=given_priors):
        load(engine)
        self.statements, self.engine, self.timeout, self.priors = statements, engine, timeout, priors
        self.rounds: list[dict] = []

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.clauses, self.derived = [], set()
        for index, text in enumerate(self.statements.rules(network, rates)):
            clauses, head = rule(text, index)
            assert head not in self.derived, f"one rule per head keeps a CTV exact: {head}"
            self.clauses += clauses
            self.derived.add(head)
        for (predicate, constants), p in self.priors(network, rates).items():
            self.clauses.append(f"{_probability(p)}{term([predicate, *constants, '$t'])} :- live(Vt).")
            self.derived.add((predicate, constants))
        self.facts: dict[str, list[tuple[list, bool]]] = {}  # period -> its statements and values
        self.resolved: set[str] = set()
        self.last: str | None = None
        for period, observation in history:
            self._observe(observation)
            self.resolve(period, observation)

    def _add(self, texts: list[str]) -> None:
        for text in texts:
            _, _, expr, (_, value, _) = parse(text)
            self.facts.setdefault(expr[-1], []).append((expr, float(value) >= 0.5))

    def _observe(self, observation) -> None:
        self._add(self.statements.observation_facts(observation))

    def resolve(self, period, observation) -> None:
        self._add(self.statements.resolution_facts(period, observation))
        self.resolved.add(period.name)
        self.last = period.name

    def program(self, keys: list[Key]) -> str:
        """The round's program: rules, the live periods, their observations
        as evidence (facts for statements no rule derives), and the last
        resolved period's labels as facts."""
        live = sorted({name for name in self.facts if name not in self.resolved} | {key[2] for key in keys})
        lines = [*self.clauses, *(f"live('{name}')." for name in live)]
        for name in [*live, *([self.last] if self.last else [])]:
            for expr, value in self.facts.get(name, ()):
                if name != self.last and (expr[0], tuple(expr[1:-1])) in self.derived:
                    lines.append(f"evidence({term(expr)}, {'true' if value else 'false'}).")
                elif value:
                    lines.append(f"{term(expr)}.")
        return "\n".join([*lines, *(f"query({term(key)})." for key in keys), ""])

    def beliefs(self, observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self._observe(observation)
        text = self.program(keys)
        probabilities, record = solve(text, [term(key) for key in keys], self.engine, self.timeout)
        self.rounds.append(record)
        return dict(zip(keys, probabilities, strict=True)) if probabilities is not None else {}

    def summary(self) -> dict:
        return {"problog_engine": self.engine, "problog_timeout": self.timeout, **summary(self.rounds)}
