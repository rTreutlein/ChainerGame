"""Declarative action semantics shared by StationOps reasoner backends.

Diagnosis produces uncertain truth values, while ordinary chainer premises do
not expose a proof's strength as a numeric variable.  The adapter therefore
asserts the current, context-scoped probability as an explicit term before
asking one open ``ActionProposal`` query.  All valuation after that boundary is
performed by these rules.
"""
from __future__ import annotations

from dataclasses import dataclass


ACTION_RULES = (
    "(: calculateRepairValue "
    "(Implication "
    "(And (LeakProbability $context $incident $probability) "
    "(ProductionAtRisk $context $incident $risk) "
    "(ActionCosts $context $_inspectionCost $repairCost $unnecessaryPenalty) "
    "(Compute - (1 $probability) -> $healthyProbability) "
    "(Compute * ($probability $risk) -> $protectedProduction) "
    "(Compute * ($healthyProbability $unnecessaryPenalty) -> $unnecessaryLoss) "
    "(Compute - ($protectedProduction $unnecessaryLoss) -> $afterUnnecessaryLoss) "
    "(Compute - ($afterUnnecessaryLoss $repairCost) -> $utility)) "
    "(RepairValue $context $incident $utility)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: calculateInspectionValueWhenRepairPays "
    "(Implication "
    "(And (InspectionEligible $context $incident) "
    "(LeakProbability $context $incident $probability) "
    "(ProductionAtRisk $context $incident $risk) "
    "(ActionCosts $context $inspectionCost $repairCost $_unnecessaryPenalty) "
    "(RepairValue $context $incident $repairUtility) "
    "(Compute - ($risk $repairCost) -> $leakRepairValue) "
    "(Compute > ($leakRepairValue 0) -> True) "
    "(Compute > ($repairUtility 0) -> True) "
    "(Compute * ($probability $leakRepairValue) -> $postInspectionValue) "
    "(Compute - ($postInspectionValue $repairUtility) -> $informationGain) "
    "(Compute - ($informationGain $inspectionCost) -> $utility)) "
    "(InspectionValue $context $incident $utility)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: calculateInspectionValueWhenRepairWaits "
    "(Implication "
    "(And (InspectionEligible $context $incident) "
    "(LeakProbability $context $incident $probability) "
    "(ProductionAtRisk $context $incident $risk) "
    "(ActionCosts $context $inspectionCost $repairCost $_unnecessaryPenalty) "
    "(RepairValue $context $incident $repairUtility) "
    "(Compute - ($risk $repairCost) -> $leakRepairValue) "
    "(Compute > ($leakRepairValue 0) -> True) "
    "(Compute <= ($repairUtility 0) -> True) "
    "(Compute * ($probability $leakRepairValue) -> $postInspectionValue) "
    "(Compute - ($postInspectionValue $inspectionCost) -> $utility)) "
    "(InspectionValue $context $incident $utility)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: proposeRepair "
    "(Implication "
    "(And (DecisionCandidate $context $incident) "
    "(LeakProbability $context $incident $_probability) "
    "(BeliefConfidence $context $incident $confidence) "
    "(RepairValue $context $incident $utility)) "
    "(ActionProposal $context (Repair $incident) $utility $confidence)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: proposeInspection "
    "(Implication "
    "(And (DecisionCandidate $context $incident) "
    "(LeakProbability $context $incident $_probability) "
    "(BeliefConfidence $context $incident $confidence) "
    "(InspectionValue $context $incident $utility)) "
    "(ActionProposal $context (Inspect $incident) $utility $confidence)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: proposeLearningProbe "
    "(Implication "
    "(And (LearningProbeCandidate $context $incident $sampleCount $risk) "
    "(Compute * ($sampleCount 1000) -> $samplePenalty) "
    "(Compute - ($risk $samplePenalty) -> $priority)) "
    "(ActionProposal $context (InspectForLearning $incident) $priority 1)) "
    "(CTV (STV 1 1) (STV 0 1)))",
)


@dataclass(frozen=True)
class ActionCandidate:
    incident_id: str
    probability: float | None
    production_at_risk: float
    confidence: float = 1.0
    inspection_eligible: bool = True
    learning_samples: int | None = None


@dataclass(frozen=True)
class ActionProposal:
    context: str
    action: str
    incident_id: str
    utility: float
    confidence: float


def _number(value: float) -> str:
    return format(float(value), ".17g")


def generate_action_statements(
    context: str,
    candidates: list[ActionCandidate],
    *,
    inspection_cost: float,
    repair_cost: float,
    unnecessary_repair_penalty: float,
) -> str:
    """Encode one immutable decision context and the generic action rules."""
    lines = list(ACTION_RULES)
    lines.append(
        f"(: action-costs-{context} "
        f"(ActionCosts {context} {_number(inspection_cost)} {_number(repair_cost)} "
        f"{_number(unnecessary_repair_penalty)}) (STV 1 1))"
    )
    for candidate in candidates:
        incident = candidate.incident_id
        prefix = f"action-{context}-{incident}"
        lines.extend((
            f"(: {prefix}-candidate "
            f"(DecisionCandidate {context} {incident}) (STV 1 1))",
            f"(: {prefix}-risk "
            f"(ProductionAtRisk {context} {incident} "
            f"{_number(candidate.production_at_risk)}) (STV 1 1))",
        ))
        if candidate.probability is not None:
            lines.extend((
                f"(: {prefix}-probability "
                f"(LeakProbability {context} {incident} "
                f"{_number(candidate.probability)}) (STV 1 1))",
                f"(: {prefix}-confidence "
                f"(BeliefConfidence {context} {incident} "
                f"{_number(candidate.confidence)}) (STV 1 1))",
            ))
        if candidate.inspection_eligible:
            lines.append(
                f"(: {prefix}-inspection-eligible "
                f"(InspectionEligible {context} {incident}) (STV 1 1))"
            )
        if candidate.learning_samples is not None:
            lines.append(
                f"(: {prefix}-learning-probe "
                f"(LearningProbeCandidate {context} {incident} "
                f"{int(candidate.learning_samples)} "
                f"{_number(candidate.production_at_risk)}) (STV 1 1))"
            )
    return "\n".join(lines)


def reference_action_proposals(
    context: str,
    candidates: list[ActionCandidate],
    *,
    inspection_cost: float,
    repair_cost: float,
    unnecessary_repair_penalty: float,
) -> list[ActionProposal]:
    """Direct oracle for the declarative rules, used by the reference backend."""
    proposals = []
    for candidate in candidates:
        probability = candidate.probability
        if probability is not None:
            repair_value = (
                probability * candidate.production_at_risk
                - (1.0 - probability) * unnecessary_repair_penalty
                - repair_cost
            )
            proposals.append(ActionProposal(
                context,
                "Repair",
                candidate.incident_id,
                repair_value,
                candidate.confidence,
            ))
            leak_repair_value = max(0.0, candidate.production_at_risk - repair_cost)
            if candidate.inspection_eligible and leak_repair_value > 0:
                inspection_value = (
                    probability * leak_repair_value
                    - max(0.0, repair_value)
                    - inspection_cost
                )
                proposals.append(ActionProposal(
                    context,
                    "Inspect",
                    candidate.incident_id,
                    inspection_value,
                    candidate.confidence,
                ))
        if candidate.learning_samples is not None:
            proposals.append(ActionProposal(
                context,
                "InspectForLearning",
                candidate.incident_id,
                candidate.production_at_risk - 1000 * candidate.learning_samples,
                1.0,
            ))
    return proposals
