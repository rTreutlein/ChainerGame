"""ONA (OpenNARS for Applications) over the conflicting-sources stage, through
the SupplyNet translation (``supplynet.nars``, docs/nars_backend.md): a fresh
``NAR shell`` per round with the systems' rules, each node's base rate as the
induction over the labelled rounds, the round's claims, a number of cycles,
and one question per node. A system's negative branch, P(up | not both
parents up), is stated from each parent's negation: exact only for strengths
0 and 1, NARS's own reading of a negated conjunction.

- ``raw``: a source's claims on a node as one judgement about the node,
  ``<(c1 * t5) --> up>`` (or its negation for "down") with confidence
  n/(n+k) for n copies (``metta.raw_facts`` read by ``supplynet.nars.fact``).
- ``sources``: claims as ``<(s1 * c1 * t5) --> claims>`` (negated for "down"),
  one per source and node, and per source the implications
  ``<<(s1 * $n * $t) --> claims> ==> <($n * $t) --> up>>`` and the same from
  the negated claim, with frequency P(up | claim) and confidence n/(n+1) over
  the n labelled claims of that polarity: what NARS induction plus revision
  over the labelled rounds would sum to, computed by the adapter as for the
  base rates."""

from __future__ import annotations

from supplynet.nars import _atom, _truth, answers, base_rate, executable, fact, implication, question, run

from . import metta
from .world import Round, World


class NarsBackend:
    def __init__(self, encoding: str, nar: str | None = None, cycles_per_step: float = 1, evidence_k: float = 5, timeout: float = 3600):
        assert encoding in ("raw", "sources"), encoding
        self.nar = executable(nar)
        self.encoding, self.cycles_per_step, self.evidence_k, self.timeout = encoding, cycles_per_step, evidence_k, timeout
        self.name = f"nars-{encoding}"
        self.unanswered = 0

    def begin(self, world: World, history: list[Round]) -> None:
        self.world, self.history = world, list(history)
        self.rules = []
        for s in world.systems:
            parents, head = [["Up", p, "$t"] for p in s.parents], ["Up", s.name, "$t"]
            self.rules += implication(["And", *parents], head, (s.given_both, 1.0), None)
            for parent in parents:  # not both: either parent down, each with P(up | not both)
                self.rules += implication(["Not", parent], head, (s.given_not, 1.0), None)

    def _trust(self) -> list[str]:
        lines = []
        for source in self.world.sources:
            samples = {True: [], False: []}  # claim -> node states
            for past in self.history:
                for r in past.reports:
                    if r.source == source.name:
                        samples[r.up].append(past.truth[r.node])
            claim = ["Claims", source.name, "$n", "$t"]
            positive, negative = (
                (sum(samples[v]) / len(samples[v]), len(samples[v]) / (len(samples[v]) + 1)) if samples[v] else None for v in (True, False)
            )
            if positive is None:
                positive, negative = negative, None
                claim = ["Not", claim]
            if positive is not None:
                lines += implication(claim, ["Up", "$n", "$t"], positive, negative)
        return lines

    def kb(self, day: Round) -> list[str]:
        lines = list(self.rules)
        for node in self.world.nodes:
            lines += base_rate("Up", node, sum(past.truth[node] for past in self.history), len(self.history))
        lines.append(f"<{_atom(day.name)} --> period>. {_truth(1, 1)}")
        if self.encoding == "raw":
            lines += [fact(text) for text in metta.raw_facts(day.reports, day.name, self.evidence_k)]
        else:
            lines += self._trust()
            seen = set()
            for r in day.reports:
                if (r.source, r.node) not in seen:
                    seen.add((r.source, r.node))
                    term = f"<({r.source} * {r.node} * {_atom(day.name)}) --> claims>"
                    lines.append(f"{term if r.up else f'(! {term})'}. {_truth(1, 1)}")
        return lines

    def beliefs(self, day: Round, keys: list[tuple[str, str]], budget: int) -> dict:
        cycles = max(0, round(budget * self.cycles_per_step))
        text = "\n".join(["*volume=0", *self.kb(day), str(cycles), *(question(("Up", node, name)) for node, name in keys), ""])
        found = answers(run(self.nar, text, None, self.timeout))
        assert len(found) == len(keys), (len(found), len(keys))
        beliefs = {}
        for key, truth in zip(keys, found, strict=True):
            if truth is None:
                self.unanswered += 1
            else:
                beliefs[key] = truth[0]
        return beliefs

    def resolve(self, day: Round) -> None:
        self.history.append(day)

    def stats(self) -> dict:
        return {"nars_unanswered": self.unanswered}
