"""ProbLog for StationOps' temporal model, beliefs only.

Each shift's public view is merged into the statements seen so far (views
are append-only), translated into one ProbLog program and solved exactly
with ``supplynet.problog_backend.solve``; every incident asks
``(SealLeak cohort type unit)``. The translation (docs/problog_backend.md):

- a unit is ``live`` until it is labelled: a labelled unit's ``SealLeak`` is
  a fact (true) or absent (false), which d-separates its past; a live unit's
  leak is its ``LeakPredicted`` (the ``WithPrior``), and its alarm and
  noise readings are ``evidence/2`` through their CTV rules;
- ``persist-<group>`` is exact: ``LeakPredicted`` holds when the previous
  unit leaked at shift end, else with the group's hazard. ``fresh-<group>``
  is left out: a serviced unit has no ``NextState`` link, so the persistence
  rule already gives it the hazard, which is what ``fresh`` states (both
  together would be a noisy-OR, 1 - (1 - h)^2);
- ``SealLeakAtShiftEnd`` is ``SealLeak``; the shortfall rule (``FoldAllTruth``
  and ``Compute``, a per-shift DP over the reasoner's own beliefs) becomes
  exact evidence instead: the impacts of an event's leaking candidates sum to
  its reported loss, conditioned jointly with every other observation;
- a sensor rate the view leaves open (``(STV s 1)`` gives only P(alarm |
  leak), ``(STV 0.5 0.5)`` neither) is the Laplace estimate (k + 1)/(n + 2)
  over the labelled units of that equipment type;
- ``patchGoal`` (decisions) and the ``Consequence`` chains (never asked, never
  observed) are left out; neither changes a ``SealLeak`` posterior.

Options from the environment: PROBLOG_ENGINE (default ddnnf), PROBLOG_TIMEOUT
(seconds per shift, default 300). Decisions need expected values over
beliefs, which a replay does not ask: the backend plays replays only."""

from __future__ import annotations

import os

from supplynet.problog_backend import argument, load, parse, solve, summary

from .models import Belief


def _term(head: str, *args: str) -> str:
    return f"{head.lower()}({', '.join(map(argument, args))})"


class ProblogBackend:
    name = "problog"

    def __init__(self, config=None):
        self.engine = os.environ.get("PROBLOG_ENGINE", "ddnnf")
        self.timeout = float(os.environ.get("PROBLOG_TIMEOUT", "300"))
        load(self.engine)
        self.statements: dict[str, list] = {}  # name -> parsed (: name expr truth)
        self.records: list[dict] = []

    def infer(self, history, incidents, budget: int, statements: str):
        for line in statements.splitlines():
            if line.startswith("(:"):
                parsed = parse(line)
                self.statements[parsed[1]] = parsed
        goals = [("SealLeak", item.cohort, item.equipment_type, item.id) for item in incidents]
        text = self.program(goals)
        probabilities, record = solve(text, [_term(*goal) for goal in goals], self.engine, self.timeout)
        self.records.append(record)
        beliefs = {} if probabilities is None else {item.id: Belief(p, 1.0) for item, p in zip(incidents, probabilities, strict=True)}
        return beliefs, {f"problog_{key}": value for key, value in record.items()}

    def program(self, goals: list[tuple]) -> str:
        facts = {tuple(expr): float(truth[1]) >= 0.5 for _, _, expr, truth in self.statements.values() if truth[0] == "STV" and expr[0] not in ("Implication", "ForAll")}
        labels = {key[3]: value for key, value in facts.items() if key[0] == "SealLeak"}
        units = {key[1:4] for key in facts if key[0] == "PressureAlarm"} | {goal[1:] for goal in goals}
        live = {unit for unit in units if unit[2] not in labels}
        lines = [
            "sealleak(C, T, U) :- live(C, T, U), leakpredicted(C, T, U).",
            "sealleakatshiftend(C, T, U) :- sealleak(C, T, U).",
            "leaked(C, T, U) :- nextstate(P, U), sealleakatshiftend(C, T, P).",
            # The impacts of the leaking candidates sum to the loss.
            "leaksum([], 0).",
            "leaksum([u(C, T, U, I) | Rest], S) :- sealleakatshiftend(C, T, U), R is S - I, R >= 0, leaksum(Rest, R).",
            "leaksum([u(C, T, U, _) | Rest], S) :- \\+sealleakatshiftend(C, T, U), leaksum(Rest, S).",
        ]
        for _, name, expr, truth in self.statements.values():
            if name.startswith("persist-"):
                _, (_, _, (_, cohort, kind, _)), _ = expr
                _, (_, p, _), (_, hazard, _) = truth
                assert float(p) == 1, name
                group = f"'{cohort}', '{kind}'"
                lines += [
                    f"leakpredicted({group}, U) :- live({group}, U), leaked({group}, U).",
                    f"{float(hazard):.10g}::leakpredicted({group}, U) :- live({group}, U), \\+leaked({group}, U).",
                ]
            elif expr[0] == "ForAll":
                _, _, (_, _, (_, (_, _, kind, _), (sensor, *_))) = expr
                given, without = self._rates(truth, kind, sensor, facts, labels)
                lines += [
                    f"{given:.10g}::{sensor.lower()}(C, '{kind}', U) :- live(C, '{kind}', U), sealleak(C, '{kind}', U).",
                    f"{without:.10g}::{sensor.lower()}(C, '{kind}', U) :- live(C, '{kind}', U), \\+sealleak(C, '{kind}', U).",
                ]
        lines += [f"live({', '.join(map(argument, unit))})." for unit in sorted(live)]
        lines += [f"{_term(*key)}." for key, value in facts.items() if value and key[0] in ("SealLeak", "NextState")]
        lines += [
            f"evidence({_term(*key)}, {'true' if value else 'false'})."
            for key, value in facts.items()
            if (key[0] == "PressureAlarm" or key[0].startswith("Noise")) and key[1:4] in live
        ]
        losses = {key[1]: key[2] for key in facts if key[0] == "ShortfallLoss"}
        for event, loss in losses.items():
            candidates = ", ".join(f"u({', '.join(map(argument, key[2:5]))}, {key[5]})" for key in facts if key[0] == "ShortfallCandidate" and key[1] == event)
            lines.append(f"evidence(leaksum([{candidates}], {loss}), true).")
        lines += [f"query({_term(*goal)})." for goal in goals]
        return "\n".join([*lines, ""])

    @staticmethod
    def _rates(truth, kind: str, sensor: str, facts: dict, labels: dict) -> tuple[float, float]:
        """A sensor's P(reading | leak) and P(reading | no leak): as stated,
        or the Laplace estimate over the labelled units of its equipment type."""
        if truth[0] == "CTV":
            return float(truth[1][1]), float(truth[2][1])
        counts = {True: [0, 0], False: [0, 0]}
        for key, value in facts.items():
            if key[0] == sensor and key[2] == kind and key[3] in labels:
                counts[labels[key[3]]][0] += value
                counts[labels[key[3]]][1] += 1
        estimate = {leak: (k + 1) / (n + 2) for leak, (k, n) in counts.items()}
        stated = float(truth[2]) >= 1
        return (float(truth[1]) if stated else estimate[True]), estimate[False]

    def summary(self) -> dict:
        return {"problog_engine": self.engine, "problog_timeout": self.timeout, **summary(self.records)}

    def propose_actions(self, *args, **kwargs):
        raise NotImplementedError("ProbLog answers beliefs only here; run StationOps with --replay")
