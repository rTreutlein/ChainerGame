"""The MeTTa a reasoner receives: one marginal rule per parent, the observed
parents and revealed consequents of cases as facts, combination hypotheses,
and belief and review queries."""

from __future__ import annotations

from .world import Case, Consequent, Marginals

Pattern = tuple[tuple[int, bool], ...]  # (parent index, value), sorted by index


def _number(value: float) -> str:
    return f"{value:.6g}"


def rules(problem: tuple[Consequent, ...], marginals: Marginals, confidence: float = 1.0) -> list[str]:
    """``(Implication (A_i $x) (C $x))`` with CTV (P(C|A_i), P(C|not A_i)).
    At confidence 1 the rates are given: their views share no rule evidence,
    and refinement leaves them alone."""
    c = _number(confidence)
    return [
        f"(: r-{parent} (Implication ({parent} $x) ({consequent.name} $x)) "
        f"(CTV (STV {_number(t)} {c}) (STV {_number(f)} {c})))"
        for consequent in problem
        for parent, (t, f) in zip(consequent.parents, marginals.given[consequent.name])
    ]


def case_facts(problem: tuple[Consequent, ...], case: Case, complement: bool = False) -> list[str]:
    """The observed parents. With ``complement``, each observation is also
    stated as its complement ``(NotA x)`` under the same proof name: one
    observation, two statements, so a hypothesis over complements carries
    the very evidence the marginal rule's view does."""
    lines = []
    for consequent in problem:
        for i, value in case.observed[consequent.name].items():
            parent = consequent.parents[i]
            name = f"o-{parent}-{case.name}"
            lines.append(f"(: {name} ({parent} {case.name}) (STV {int(value)} 1))")
            if complement:
                lines.append(f"(: {name} (Not{parent} {case.name}) (STV {int(not value)} 1))")
    return lines


def label_facts(problem: tuple[Consequent, ...], case: Case) -> list[str]:
    return [f"(: l-{c.name}-{case.name} ({c.name} {case.name}) (STV {int(case.truth[c.name][1])} 1))" for c in problem]


def _literal(consequent: Consequent, index: int, value: bool, form: str) -> str:
    parent = consequent.parents[index]
    if value:
        return f"({parent} $x)"
    return f"(Not{parent} $x)" if form == "complement" else f"(Not ({parent} $x))"


def hypothesis_implication(consequent: Consequent, pattern: Pattern, form: str) -> str:
    literals = " ".join(_literal(consequent, i, v, form) for i, v in pattern)
    return f"(Implication (And {literals}) ({consequent.name} $x))"


def hypothesis(consequent: Consequent, pattern: Pattern, prior: float, form: str, confidence: float = 0.02) -> str:
    """A refined rule for one combination of parent polarities, with a weak
    prior: the data decides its strength."""
    name = "h-" + consequent.name + "".join(f"-{i + 1}{'T' if v else 'F'}" for i, v in pattern)
    return f"(: {name} {hypothesis_implication(consequent, pattern, form)} (STV {_number(prior)} {_number(confidence)}))"


def review(consequent: Consequent, pattern: Pattern, form: str) -> str:
    """Reading a hypothesis's truth folds its instances."""
    return f"(: $prf (RuleTruth {hypothesis_implication(consequent, pattern, form)}) $tv)"


def query(key: tuple[str, str]) -> str:
    consequent, entity = key
    return f"(: $prf ({consequent} {entity}) $tv)"
