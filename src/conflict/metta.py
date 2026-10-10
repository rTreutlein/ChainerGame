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
  labelled rounds by rule refinement."""

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


def review(source: str) -> str:
    return f"(: $prf (RuleTruth {trust_implication(source)}) $tv)"


def query(key: tuple[str, str]) -> str:
    node, name = key
    return f"(: $prf (Up {node} {name}) $tv)"
