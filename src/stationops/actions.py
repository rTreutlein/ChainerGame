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
    "(ActionProposal $context (Repair $incident) $utility $confidence Intervention)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: proposeInspection "
    "(Implication "
    "(And (DecisionCandidate $context $incident) "
    "(LeakProbability $context $incident $_probability) "
    "(BeliefConfidence $context $incident $confidence) "
    "(InspectionValue $context $incident $utility)) "
    "(ActionProposal $context (Inspect $incident) $utility $confidence Diagnostic)) "
    "(CTV (STV 1 1) (STV 0 1)))",
    "(: proposeLearningProbe "
    "(Implication "
    "(And (LearningProbeCandidate $context $incident $sampleCount $risk) "
    "(Compute * ($sampleCount 1000) -> $samplePenalty) "
    "(Compute - ($risk $samplePenalty) -> $priority)) "
    "(ActionProposal $context (Inspect $incident) $priority 1 Learning)) "
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
    rationale: str


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


# PeTTaChainer decides in the knowledge base that holds the beliefs
# (docs/single_kb_actions.md): each decision step is one Assuming query over
# the step's facts. Outcomes follow from an incident's leak belief
# through ordinary connectives, an action's expected utility folds its
# outcomes' amounts weighted by their truth values, and no number is copied
# out of a belief. Every rule is named (no_inverse ...): a decision is never
# evidence for the beliefs it reads.
_CERTAIN_RULE = "(CTV (STV 1 1) (STV 0 1))"
_EXPECTED_VALUE = (
    "(FoldAllTruth (Outcome $step {action} $outcome $value) "
    "(STV $s $_c) (Weighted $s $value) 0.0 "
    "(|-> ($acc (Weighted $p $v)) (+ $acc (* $p $v))) -> $gross)"
)
_BELIEF_CONFIDENCE = (
    "(FoldAllTruth (LeakBelief $incident) (STV $_s $c) $c 0.0 "
    "(|-> ($acc $elem) $elem) -> $confidence)"
)
BELIEF_ACTION_RULES = tuple(f"(: (no_inverse {name}) (Implication {body}) {_CERTAIN_RULE})" for name, body in (
    # An outcome carries its amount, so an action's expected value folds one
    # statement per outcome.
    ("repairProtects",
     "(And (DecisionCandidate $step $incident) (LeakBelief $incident) "
     "(OutcomeValue $step (Repair $incident) protected $value)) "
     "(Outcome $step (Repair $incident) protected $value)"),
    ("repairUnneeded",
     "(And (DecisionCandidate $step $incident) (Not (LeakBelief $incident)) "
     "(OutcomeValue $step (Repair $incident) unnecessary $value)) "
     "(Outcome $step (Repair $incident) unnecessary $value)"),
    ("inspectionFindsLeak",
     "(And (InspectionEligible $step $incident) (LeakBelief $incident) "
     "(OutcomeValue $step (Inspect $incident) found $value)) "
     "(Outcome $step (Inspect $incident) found $value)"),
    ("expectedRepairValue",
     "(And (DecisionCandidate $step $incident) "
     "(ActionCost $step (Repair $incident) $cost) "
     + _EXPECTED_VALUE.format(action="(Repair $incident)") + " "
     "(Compute - ($gross $cost) -> $utility)) "
     "(RepairValue $step $incident $utility)"),
    # Inspecting is worth what a found leak's repair gains, less the best
    # value without inspecting: the repair when it pays, nothing otherwise.
    ("inspectionValueWhenRepairPays",
     "(And (InspectionEligible $step $incident) "
     "(RepairValue $step $incident $repairUtility) "
     "(Compute > ($repairUtility 0) -> True) "
     "(ActionCost $step (Inspect $incident) $cost) "
     + _EXPECTED_VALUE.format(action="(Inspect $incident)") + " "
     "(Compute - ($gross $repairUtility) -> $gain) "
     "(Compute - ($gain $cost) -> $utility)) "
     "(InspectionValue $step $incident $utility)"),
    ("inspectionValueWhenRepairWaits",
     "(And (InspectionEligible $step $incident) "
     "(RepairValue $step $incident $repairUtility) "
     "(Compute <= ($repairUtility 0) -> True) "
     "(ActionCost $step (Inspect $incident) $cost) "
     + _EXPECTED_VALUE.format(action="(Inspect $incident)") + " "
     "(Compute - ($gross $cost) -> $utility)) "
     "(InspectionValue $step $incident $utility)"),
    ("proposeRepair",
     "(And (DecisionCandidate $step $incident) "
     "(RepairValue $step $incident $utility) " + _BELIEF_CONFIDENCE + ") "
     "(ActionProposal $step (Repair $incident) $utility $confidence Intervention)"),
    ("proposeInspection",
     "(And (DecisionCandidate $step $incident) "
     "(InspectionValue $step $incident $utility) " + _BELIEF_CONFIDENCE + ") "
     "(ActionProposal $step (Inspect $incident) $utility $confidence Diagnostic)"),
    ("proposeLearningProbe",
     "(And (LearningProbeCandidate $step $incident $sampleCount $risk) "
     "(Compute * ($sampleCount 1000) -> $samplePenalty) "
     "(Compute - ($risk $samplePenalty) -> $priority)) "
     "(ActionProposal $step (Inspect $incident) $priority 1 Learning)"),
))


def leak_belief_assumption(incident_id: str, belief: str) -> str:
    """Name, for one decision, the statement whose truth is the incident's leak
    belief: assumed as an implication, it adds no rule to the knowledge base."""
    return f"(Implication {belief} (LeakBelief {incident_id}))"


def decision_assumptions(
    step: str,
    candidates: list[ActionCandidate],
    *,
    inspection_cost: float,
    repair_cost: float,
    unnecessary_repair_penalty: float,
) -> list[str]:
    """The facts one decision step assumes: its candidates, outcome amounts
    and costs.

    A candidate is decided on only when it has a leak belief. Inspection is
    eligible only when a found leak's repair would pay.
    """
    facts = []
    for candidate in candidates:
        incident = candidate.incident_id
        risk = candidate.production_at_risk
        if candidate.probability is not None:
            facts.extend((
                f"(DecisionCandidate {step} {incident})",
                f"(ActionCost {step} (Repair {incident}) {_number(repair_cost)})",
                f"(OutcomeValue {step} (Repair {incident}) protected {_number(risk)})",
                f"(OutcomeValue {step} (Repair {incident}) unnecessary "
                f"{_number(-unnecessary_repair_penalty)})",
            ))
            if candidate.inspection_eligible and risk - repair_cost > 0:
                facts.extend((
                    f"(InspectionEligible {step} {incident})",
                    f"(ActionCost {step} (Inspect {incident}) {_number(inspection_cost)})",
                    f"(OutcomeValue {step} (Inspect {incident}) found {_number(risk - repair_cost)})",
                ))
        if candidate.learning_samples is not None:
            facts.append(
                f"(LearningProbeCandidate {step} {incident} "
                f"{int(candidate.learning_samples)} {_number(risk)})"
            )
    return facts


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
                "Intervention",
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
                    "Diagnostic",
                ))
        if candidate.learning_samples is not None:
            proposals.append(ActionProposal(
                context,
                "Inspect",
                candidate.incident_id,
                candidate.production_at_risk - 1000 * candidate.learning_samples,
                1.0,
                "Learning",
            ))
    return proposals
