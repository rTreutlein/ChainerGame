"""A NARS reasoner (ONA) for StationOps' temporal model, beliefs only.

Each shift's public view is merged into the statements seen so far (views
are append-only, as for PeTTaChainer), translated into Narsese with
``supplynet.nars`` and given to a fresh ``NAR shell``, which is asked
``(SealLeak cohort type unit)`` for every incident. Beyond SupplyNet's
translation (docs/nars_backend.md):

- ``(ForAll ($cohort) (WithPrior P (Implication A B)))``: the prior P stands
  for A, stated as ``P -> A`` and ``not P -> not A`` with certainty, and the
  sensor rule as a CTV. A sensor rate the view leaves open (``(STV s 1)``
  gives P(B|A) only, ``(STV 0.5 0.5)`` neither) is NARS's induction over the
  resolved units holding both statements: frequency k/n, confidence n/(n+1);
- the shortfall rule (``FoldAllTruth`` and ``Compute``) has no Narsese
  counterpart and is left out, with its facts; so are the decision rules
  (``patchGoal``), which replay does not ask;
- ONA holds at most 255 atoms and every unit is one, so a shift's KB keeps the
  units of the last ``window`` shifts (and history units while they are in
  it); a persistence chain older than that starts unsupported.

Decisions (``propose_actions``) need expected values over truth values, which
Narsese cannot state: the backend plays replays only."""

from __future__ import annotations

import os
import re
from pathlib import Path

from supplynet import nars as narsese

from .models import Belief

_unit_re = re.compile(r"^(?:h-\d+|shift-(\d+))-M\d+$")


def _shift(atom: str) -> int | None:
    """The shift a unit belongs to (history units: 0); None for other atoms."""
    match = _unit_re.match(atom)
    return None if match is None else int(match.group(1) or 0)


def _atoms(expr) -> list[str]:
    return [expr] if isinstance(expr, str) else [atom for part in expr for atom in _atoms(part)]


class NarsBackend:
    name = "nars"

    def __init__(self, config=None):
        self.nar = narsese.executable(os.environ.get("NARS_PATH"))
        self.cycles_per_step = float(os.environ.get("NARS_CYCLES_PER_STEP", "0"))
        self.window = int(os.environ.get("NARS_WINDOW", "15"))
        self.cache = Path(os.environ["NARS_CACHE"]) if os.environ.get("NARS_CACHE") else None
        self.statements: dict[str, list] = {}  # name -> parsed (: name expr truth)

    def infer(self, history, incidents, budget: int, statements: str):
        for line in statements.splitlines():
            if line.startswith("(:"):
                parsed = narsese.parse(line)
                self.statements[parsed[1]] = parsed
        now = max((_shift(item.id) or 0) for item in incidents) if incidents else 0
        lines, unsupported = self._kb(now)
        goals = [["SealLeak", item.cohort, item.equipment_type, item.id] for item in incidents]
        cycles = max(0, round(budget * self.cycles_per_step))
        text = "\n".join(["*volume=0", *lines, str(cycles), *(f"{narsese.statement(goal)}?" for goal in goals), ""])
        found = narsese.answers(narsese.run(self.nar, text, self.cache))
        assert len(found) == len(goals), (len(found), len(goals))
        beliefs = {item.id: Belief(truth[0], truth[1]) for item, truth in zip(incidents, found, strict=True) if truth is not None}
        return beliefs, {"nars_statements": len(lines), "nars_cycles": cycles, "nars_unsupported": unsupported, "nars_answered": len(beliefs)}

    def _kb(self, now: int) -> tuple[list[str], list[str]]:
        facts = [parsed for parsed in self.statements.values() if parsed[3][0] == "STV" and parsed[2][0] not in ("Implication", "ForAll")]
        known = {tuple(parsed[2]): float(parsed[3][1]) for parsed in facts}
        rules, unsupported = [], []
        for _, name, expr, truth in self.statements.values():
            head = expr[0]
            if head == "Implication" and truth[0] == "CTV" and not any(atom in ("FoldAllTruth", "Compute") for atom in _atoms(expr)):
                if not name.startswith("patchGoal"):
                    rules += narsese.implication(expr[1], expr[2], narsese._stv(truth[1]), narsese._stv(truth[2]))
            elif head == "ForAll":
                _, prior, (_, cause, effect) = expr[2]
                rules += narsese.implication(prior, cause, (1.0, 1.0), (0.0, 1.0))
                rules += narsese.implication(cause, effect, *self._sensor(truth, cause, effect, known))
            elif head in ("Implication", "ForAll"):
                unsupported.append(name)
        oldest = now - self.window
        kept = [
            narsese.fact(f"(: f {self._text(expr)} {self._text(truth)})")
            for _, _, expr, truth in facts
            if expr[0] not in ("ShortfallLoss", "ShortfallCandidate")
            and all(shift >= oldest for shift in (_shift(atom) for atom in _atoms(expr)) if shift is not None)
        ]
        return rules + kept, unsupported

    @staticmethod
    def _sensor(truth, cause, effect, known) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """The sensor rule's two branches: given when the view states them,
        else counted over the resolved units of the rule's equipment type."""
        if truth[0] == "CTV":
            return narsese._stv(truth[1]), narsese._stv(truth[2])
        kind = cause[2]
        counts = {True: [0, 0], False: [0, 0]}
        for statement, value in known.items():
            if statement[0] == cause[0] and statement[2] == kind:
                alarm = known.get((effect[0], *statement[1:]))
                if alarm is not None:
                    counts[value >= 0.5][0] += alarm >= 0.5
                    counts[value >= 0.5][1] += 1
        learned = {leak: (k / n, n / (n + 1)) if n else None for leak, (k, n) in counts.items()}
        strength, confidence = narsese._stv(truth)
        return ((strength, confidence) if confidence >= 1 else learned[True]), learned[False]

    @staticmethod
    def _text(expr) -> str:
        return expr if isinstance(expr, str) else "(" + " ".join(NarsBackend._text(part) for part in expr) + ")"

    def propose_actions(self, *args, **kwargs):
        raise NotImplementedError("NARS cannot state expected-value decision queries; run StationOps with --replay")
