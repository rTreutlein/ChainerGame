"""The MeTTa a PLN reasoner receives.

Every node of every round is the statement ``(Up <node> <round>)``. Shared by
both encodings: the systems' rules with their given rates, and the labelled
rounds' states as certain facts.

- **raw:** a source's claims on a node become one fact about the node itself,
  ``(Up c1 t5)`` with the claim as strength and its copy count n as
  confidence n/(n+k). No source identity reaches the KB.
- **sources:** a claim is ``(Claims s1 (Up c1 t5))``, strength 1 for "up" and 0
  for "down", one fact per source and node however often it was repeated, and
  each source has one hypothesis rule
  ``(Implication (Claims s1 (Up $n $t)) (Up $n $t))`` with a weak CTV prior,
  so P(up | s1 claims up) and P(up | s1 claims down) are learned from the
  labelled rounds by rule refinement.
- **reliable:** each source has two latent statements, ``(Reliable s1 up)``
  and ``(Reliable s1 down)``: its up-claims, its down-claims, are correct.
  Every claim is one certain rule from the source's reliability to the
  claimed state, ``(Implication (Reliable s1 up) (Up c1 t5))`` with
  ``(CTV (STV 1 1) (STV 0 1))`` for "up" and ``(CTV (STV 0 1) (STV 1 1))`` for
  "down", so P(up) is the reliability (or its complement). Named
  ``no_inverse``: an outcome is no observation of a source-wide statement.
  The reliabilities are evidence: each resolved round adds, per source and
  polarity, the share of its claims that were correct at the confidence of
  their count, which revision pools across rounds, over a prior of one
  correct and one wrong claim."""

from __future__ import annotations

from .world import Report, Round, World


def _number(value: float) -> str:
    return f"{value:.6g}"


def system_rules(world: World) -> list[str]:
    return [
        f"(: sys-{s.name} (Implication (And (Up {s.parents[0]} $t) (Up {s.parents[1]} $t)) (Up {s.name} $t)) "
        f"(CTV (STV {_number(s.given_both)} 1) (STV {_number(s.given_not)} 1)))"
        for s in world.systems
    ]


def labels(day: Round) -> list[str]:
    return [f"(: up-{node}-{day.name} (Up {node} {day.name}) (STV {int(up)} 1))" for node, up in day.truth.items()]


def raw_facts(reports: tuple[Report, ...], name: str, k: float) -> list[str]:
    return [
        f"(: rep-{r.source}-{r.node}-{name} (Up {r.node} {name}) (STV {int(r.up)} {_number(r.copies / (r.copies + k))}))"
        for r in reports
    ]


def trust_implication(source: str) -> str:
    return f"(Implication (Claims {source} (Up $n $t)) (Up $n $t))"


def trust_rules(world: World, prior: float = 0.5, confidence: float = 0.02) -> list[str]:
    tv = f"(STV {_number(prior)} {_number(confidence)})"
    return [f"(: trust-{s.name} {trust_implication(s.name)} (CTV {tv} {tv}))" for s in world.sources]


def claims(reports: tuple[Report, ...], name: str) -> list[str]:
    return [f"(: claim-{r.source}-{r.node}-{name} (Claims {r.source} (Up {r.node} {name})) (STV {int(r.up)} 1))" for r in reports]


def reliability_rules(reports: tuple[Report, ...], name: str) -> list[str]:
    """One certain rule per claim from its source's reliability to the claimed
    state; copies carry nothing beyond the first."""
    up, down = "(STV 1 1)", "(STV 0 1)"
    return [
        f"(: (no_inverse claim-{r.source}-{r.node}-{name}) (Implication (Reliable {r.source} {_polarity(r.up)}) (Up {r.node} {name})) "
        f"(CTV {up if r.up else down} {down if r.up else up}))"
        for r in reports
    ]


def _polarity(up: bool) -> str:
    return "up" if up else "down"


def reliability_evidence(day: Round, k: float) -> list[str]:
    """Per source and polarity, the share of the round's claims that were
    correct, at the confidence n/(n + k) of their count n: revision of the
    rounds' facts is the pooled share."""
    counts: dict[tuple[str, str], list[int]] = {}
    for r in day.reports:
        count = counts.setdefault((r.source, _polarity(r.up)), [0, 0])
        count[0] += r.up == day.truth[r.node]
        count[1] += 1
    return [
        f"(: reliability-{source}-{polarity}-{day.name} (Reliable {source} {polarity}) (STV {_number(correct / n)} {_number(n / (n + k))}))"
        for (source, polarity), (correct, n) in sorted(counts.items())
    ]


def reliability_priors(world: World, k: float) -> list[str]:
    """One correct and one wrong claim per source and polarity."""
    return [
        f"(: reliability-{s.name}-{polarity}-prior (Reliable {s.name} {polarity}) (STV 0.5 {_number(2 / (2 + k))}))"
        for s in world.sources
        for polarity in ("up", "down")
    ]


def review(source: str) -> str:
    return f"(: $prf (RuleTruth {trust_implication(source)}) $tv)"


def query(key: tuple[str, str]) -> str:
    node, name = key
    return f"(: $prf (Up {node} {name}) $tv)"
