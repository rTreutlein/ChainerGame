"""Exact aggregate-loss conditioning shared by StationOps reasoner backends."""
from __future__ import annotations

import hashlib
import math
from importlib.resources import as_file, files


_POSTERIOR_MODULE = files("stationops").joinpath(
    "metta/weighted_subset_posterior.metta"
)


def load_weighted_subset_posterior(handler) -> None:
    """Load StationOps' weighted-subset Compute operators into a chainer."""
    with as_file(_POSTERIOR_MODULE) as path:
        handler.load_metta_file(path)


# Keep the compact prefix/postfix posterior table in one value. Projecting it
# to one module at a time is safe and avoids enumerating the mutually exclusive
# fault configurations that produce the observed loss.
SHORTFALL_RULES = (
    "(: diagnoseShortfall "
    "(Implication "
    "(And (ShortfallCandidates $event $candidates) "
    "(ObservedProductionLoss $event $loss) "
    "(Compute WeightedSubsetPosteriorDP ($candidates $loss) -> $posterior)) "
    "(ShortfallPosterior $event $posterior)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: projectShortfallMarginal "
    "(Implication "
    "(And (ShortfallPosterior $event $posterior) "
    "(ShortfallCandidateUnit $event $unit) "
    "(Compute WeightedSubsetPosteriorMarginal "
    "($posterior $unit) -> $probability)) "
    "(ShortfallMarginal $event $unit $probability)) "
    "(CTV (STV 1 1) (STV 0 1)))",
)


def event_atom(event: dict) -> str:
    """Name an immutable revision of one still-ambiguous shortfall."""
    state = (
        int(event["remaining_loss"]),
        tuple(
            (unit, int(row["impact"]), float(row["prior"]))
            for unit, row in sorted(event["candidates"].items())
        ),
    )
    revision = hashlib.sha1(repr(state).encode("utf-8")).hexdigest()[:12]
    return f"shortfall-s{int(event['shift'])}-v{revision}"


def generate_shortfall_statements(events: list[dict]) -> str:
    """Encode aggregate observations for the compact weighted posterior DP."""
    lines = list(SHORTFALL_RULES)
    for event in events:
        event_name = event_atom(event)
        candidates = " ".join(
            f"(WeightedCandidate {unit} {int(row['impact'])} "
            f"{float(row['prior']):.17g})"
            for unit, row in sorted(event["candidates"].items())
        )
        lines.append(
            f"(: candidates-{event_name} "
            f"(ShortfallCandidates {event_name} ({candidates})) (STV 1 1))"
        )
        lines.append(
            f"(: observed-{event_name} "
            f"(ObservedProductionLoss {event_name} "
            f"{int(event['remaining_loss'])}) (STV 1 1))"
        )
        for unit in sorted(event["candidates"]):
            lines.append(
                f"(: candidate-unit-{event_name}-{unit} "
                f"(ShortfallCandidateUnit {event_name} {unit}) (STV 1 1))"
            )
    return "\n".join(lines)


def marginal_queries(events: list[dict]) -> list[tuple[str, str]]:
    return [
        (
            f"{event_atom(event)}|{unit}",
            f"(ShortfallMarginal {event_atom(event)} {unit} $probability)",
        )
        for event in events
        for unit in sorted(event["candidates"])
    ]


def oracle_shortfall_marginals(event: dict) -> dict[str, float]:
    """Exact prefix/suffix DP retained only as a parity oracle."""
    rows = sorted(event["candidates"].items())
    target = event["remaining_loss"]

    def extend(distribution, row):
        result = {}
        for impact, weight in distribution.items():
            result[impact] = result.get(impact, 0.0) + weight * (1.0 - row["prior"])
            included = impact + row["impact"]
            if included <= target:
                result[included] = result.get(included, 0.0) + weight * row["prior"]
        return result

    prefix = [{0: 1.0}]
    for _, row in rows:
        prefix.append(extend(prefix[-1], row))
    suffix = [None] * (len(rows) + 1)
    suffix[-1] = {0: 1.0}
    for index in range(len(rows) - 1, -1, -1):
        suffix[index] = extend(suffix[index + 1], rows[index][1])

    total = prefix[-1].get(target, 0.0)
    if total <= 0:
        return {}
    marginals = {}
    for index, (unit, row) in enumerate(rows):
        remaining = target - row["impact"]
        included_weight = 0.0
        if remaining >= 0:
            for left_impact, left_weight in prefix[index].items():
                right_weight = suffix[index + 1].get(remaining - left_impact, 0.0)
                included_weight += left_weight * right_weight * row["prior"]
        marginals[unit] = min(included_weight / total, 1.0)
    return marginals


def oracle_event_marginals(events: list[dict]) -> dict[int, dict[str, float]]:
    return {
        int(event["shift"]): oracle_shortfall_marginals(event)
        for event in events
    }


def finite_probability(value) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if math.isfinite(probability) and 0.0 <= probability <= 1.0 else None
