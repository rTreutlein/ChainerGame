"""A NARS reasoner (OpenNARS for Applications, ONA) for SupplyNet stages 1-3.

The backend translates the MeTTa a reasoner receives (``metta``) into
Narsese, feeds it to a fresh ``NAR shell`` each round, runs a number of
inference cycles, and asks one question per belief key. The translation and
its losses are described in docs/nars_backend.md:

- a statement ``(Pred a b)`` is the inheritance ``<(a * b) --> pred>``, and
  ``(Pred a)`` is ``<a --> pred>``; ``(Not x)`` is ``(! x)``;
- a fact is stated in its true polarity: ``x`` with its strength when that is
  at least 0.5, else ``(! x)`` with one minus it;
- a CTV implication ``A -> B`` with P(B|A) = p and P(B|not A) = q is
  ``<A ==> B> %p%``, ``<not-A ==> B> %q%``, and the same two with ``(! B)``
  and 1 - p, 1 - q (ONA derives ``B`` from ``(! B)`` but never ``(! B)`` from
  ``B``, and a deduction from a rule of frequency 0 has confidence 0). ``not-A``
  keeps ``(NextPeriod $p $t)`` and negates the rest; an ``Or`` antecedent
  and a negated ``And`` of several parts become one rule per part, which is
  exact only for strengths 0 and 1, the only ones they have here;
- a base rate is NARS's induction over the labelled periods, as revision would
  sum it: ``<<$t --> period> ==> x($t)>`` with frequency k/n and confidence
  n/(n+1), and the same for ``(! x($t))``;
- the KB of a round holds the rules, the base rates, and the facts of the
  unresolved periods and of the last resolved one; older periods enter only
  through the base rates.

ONA keeps at most 255 atoms and 4096 concepts, so the whole history cannot be
stated as facts; a fresh process per round keeps rounds independent of what
earlier rounds left in memory."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from . import metta
from .world import Key, Network, Observation, Period, Rates

DEFAULT_NAR = os.environ.get("NARS_PATH", "/nexus/Dev/OpenCog/ONA/NAR")
MAX_CONFIDENCE = 0.99  # ONA's MAX_CONFIDENCE: a certain MeTTa truth value
STRUCTURAL = {"NextPeriod", "NextState"}  # given links, kept positive in a rule's negative branch
LABELS = ("Storm", "Blocked", "Degraded", "Producing")  # the hidden state a period resolves

_answer_re = re.compile(r"^Answer: (?:None\.|.*Truth: frequency=([0-9.]+), confidence=([0-9.]+))")


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


def _atom(name: str) -> str:
    return name if name.startswith("$") else name.replace("-", "_")


def statement(expr) -> str:
    """A MeTTa statement as a Narsese term."""
    head, *args = expr
    if head == "Not":
        return f"(! {statement(args[0])})"
    if head == "And":
        return conjunction([statement(arg) for arg in args])
    subject = _atom(args[0]) if len(args) == 1 else "(" + " * ".join(_atom(arg) for arg in args) + ")"
    return f"<{subject} --> {head.lower()}>"


def conjunction(parts: list[str]) -> str:
    """ONA reads ``&&`` as binary and drops a third part silently: nest left."""
    term = parts[0]
    for part in parts[1:]:
        term = f"({term} && {part})"
    return term


def negate(expr) -> list:
    """The negation of an antecedent as a list of alternatives, each a MeTTa
    expression; a negated ``And`` keeps its structural parts and negates
    each other part in its own alternative (De Morgan)."""
    if expr[0] == "Or":
        return [["And", *(negate(arg)[0] for arg in expr[1:])]]
    if expr[0] == "And":
        given = [arg for arg in expr[1:] if arg[0] in STRUCTURAL]
        return [["And", *given, *negate(arg)] if given else negate(arg)[0] for arg in expr[1:] if arg[0] not in STRUCTURAL]
    return [expr[1]] if expr[0] == "Not" else [["Not", expr]]


def alternatives(expr) -> list:
    """An antecedent as a list of conjunctions: an ``Or`` splits into its parts."""
    return list(expr[1:]) if expr[0] == "Or" else [expr]


def _truth(frequency: float, confidence: float) -> str:
    return f"%{frequency:.6g};{min(confidence, MAX_CONFIDENCE):.6g}%"


def _stv(expr) -> tuple[float, float]:
    assert expr[0] == "STV", expr
    return float(expr[1]), float(expr[2])


def rule(text: str) -> list[str]:
    """A CTV implication as Narsese (``implication``)."""
    _, _, (_, antecedent, consequent), (_, positive, negative) = parse(text)
    return implication(antecedent, consequent, _stv(positive), _stv(negative))


def implication(antecedent, consequent, positive: tuple[float, float], negative: tuple[float, float] | None) -> list[str]:
    """``antecedent -> consequent`` with P(consequent | antecedent) and, when
    known, P(consequent | not antecedent), as Narsese implications for both
    polarities of the consequent; implications of frequency 0 are left out,
    since every ONA truth function gives their conclusions confidence 0."""
    branches = []
    for truth, cases in ((positive, alternatives(antecedent)), (negative, negate(antecedent))):
        if truth is None:
            continue
        strength, confidence = truth
        if len(cases) > 1:
            assert strength in (0.0, 1.0), f"splitting is exact only for certain branches: {antecedent}"
        branches += [(statement(case), strength, confidence) for case in cases]
    target = statement(consequent)
    lines = []
    for condition, strength, confidence in branches:
        for head, frequency in ((target, strength), (f"(! {target})", 1 - strength)):
            if frequency > 0:
                lines.append(f"<{condition} ==> {head}>. {_truth(frequency, confidence)}")
    return lines


def fact(text: str) -> str:
    """A fact in its true polarity."""
    _, _, expr, tv = parse(text)
    strength, confidence = _stv(tv)
    term = statement(expr)
    return f"{term}. {_truth(strength, confidence)}" if strength >= 0.5 else f"(! {term}). {_truth(1 - strength, confidence)}"


def base_rate(predicate: str, subject: str, positive: int, total: int) -> list[str]:
    term = statement([predicate, subject, "$t"])
    truth = lambda f: _truth(f, total / (total + 1))  # noqa: E731
    return [
        f"<<$t --> period> ==> {term}>. {truth(positive / total)}",
        f"<<$t --> period> ==> (! {term})>. {truth(1 - positive / total)}",
    ]


def question(key: Key) -> str:
    predicate, subject, period = key
    return f"{statement([predicate, subject, period])}?"


def answers(output: str) -> list[tuple[float, float] | None]:
    """The truth of each ``Answer:`` line, None for ``Answer: None.``"""
    found = []
    for line in output.splitlines():
        match = _answer_re.match(line)
        if match:
            found.append(None if match.group(1) is None else (float(match.group(1)), float(match.group(2))))
    return found


def run(nar: Path, text: str, cache: Path | None = None, timeout: float = 3600) -> str:
    """ONA's output for a shell input. ONA is deterministic on these inputs, so
    with ``cache`` an output is stored under its input's hash and another
    reading of the same run costs nothing."""
    stored = cache / f"{hashlib.sha256(text.encode()).hexdigest()}.out" if cache else None
    if stored and stored.exists():
        return stored.read_text()
    output = subprocess.run([str(nar), "shell"], input=text, capture_output=True, text=True, timeout=timeout).stdout
    if stored:
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text(output)
    return output


def executable(nar: str | None) -> Path:
    path = Path(nar or DEFAULT_NAR)
    if not os.access(path, os.X_OK):
        raise FileNotFoundError(f"ONA is not built at {path}; build it or pass --nars-path / set NARS_PATH")
    return path


class NarsBackend:
    """ONA given the translated rules, base rates and the round's facts; a
    round's budget is scaled to inference cycles (``cycles_per_step``). A
    belief is the answer's frequency, or its expectation c (f - 0.5) + 0.5
    with ``reading="expectation"``; ``cache`` as for ``run``."""

    def __init__(
        self, nar: str | None = None, cycles_per_step: float = 10, reading: str = "frequency", cache: str | None = None, timeout: float = 3600
    ):
        self.nar = executable(nar)
        assert reading in ("frequency", "expectation")
        self.cycles_per_step, self.reading, self.timeout = cycles_per_step, reading, timeout
        self.cache = Path(cache) if cache else None
        self.name = "nars" if reading == "frequency" else "nars-expectation"

    def begin(self, network: Network, rates: Rates, history: list[tuple[Period, Observation]]) -> None:
        self.rules = [line for text in metta.rules(network, rates) for line in rule(text)]
        self.counts: Counter = Counter()  # (predicate, subject) -> [positives, total]
        self.totals: Counter = Counter()
        self.facts: dict[str, list[str]] = {}  # period -> its facts
        self.resolved: list[str] = []
        for period, observation in history:
            self._observe(observation)
            self.resolve(period, observation)

    def _observe(self, observation: Observation) -> None:
        for text in metta.observation_facts(observation):
            self.facts.setdefault(parse(text)[2][-1], []).append(fact(text))

    def resolve(self, period: Period, observation: Observation) -> None:
        for text in metta.resolution_facts(period, observation):
            self.facts.setdefault(period.name, []).append(fact(text))
        for (predicate, subject), value in period.labels().items():
            self.counts[(predicate, subject)] += value
            self.totals[(predicate, subject)] += 1
        self.resolved.append(period.name)

    def kb(self, keys: list[Key]) -> list[str]:
        """The round's Narsese: rules, base rates, and the facts of the last
        resolved period and every later one."""
        last = self.resolved[-1] if self.resolved else None
        live = [name for name in self.facts if name not in self.resolved or name == last]
        lines = list(self.rules)
        for (predicate, subject), total in sorted(self.totals.items()):
            lines += base_rate(predicate, subject, self.counts[(predicate, subject)], total)
        periods = sorted(set(live) | {key[2] for key in keys})
        lines += [f"<{_atom(name)} --> period>. {_truth(1, 1)}" for name in periods]
        for name in live:
            lines += self.facts[name]
        return lines

    def beliefs(self, observation: Observation, keys: list[Key], budget: int) -> dict[Key, float]:
        self._observe(observation)
        cycles = max(0, round(budget * self.cycles_per_step))
        text = "\n".join(["*volume=0", *self.kb(keys), str(cycles), *(question(key) for key in keys), ""])
        found = answers(run(self.nar, text, self.cache, self.timeout))
        assert len(found) == len(keys), (len(found), len(keys), output[-2000:])
        beliefs = {}
        for key, truth in zip(keys, found, strict=True):
            if truth is not None:
                frequency, confidence = truth
                beliefs[key] = frequency if self.reading == "frequency" else confidence * (frequency - 0.5) + 0.5
        return beliefs
