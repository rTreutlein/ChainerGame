from .config import Config
from .models import HistoryCase, Incident


def generate_statements(
    history: list[HistoryCase], incidents: list[Incident], config: Config
) -> str:
    """Restricted common-subset MeTTa; negative labels are explicit Not facts."""
    lines = [
        "(: alarmGivenLeak (Implication (Premises (SealLeak $cohort $unit)) "
        "(Conclusions (PressureAlarm $cohort $unit))) "
        f"(CTV (STV {config.sensitivity} 1) (STV {config.false_positive_rate} 1)))",
        "(: patchGoal (Implication (Premises (SealLeak $cohort $unit)) "
        "(Conclusions (PatchPaysOff $cohort $unit))) "
        "(CTV (STV 1 1) (STV 0 1)))",
    ]
    for case in history:
        leak = f"(SealLeak {case.cohort} {case.id})"
        alarm = f"(PressureAlarm {case.cohort} {case.id})"
        lines.append(f"(: leak-{case.id} {' ' if case.leak else '(Not '}{leak}{'' if case.leak else ')'} (STV 1 1))")
        lines.append(f"(: alarm-{case.id} {' ' if case.alarm else '(Not '}{alarm}{'' if case.alarm else ')'} (STV 1 1))")
    for item in incidents:
        alarm = f"(PressureAlarm {item.cohort} {item.id})"
        lines.append(f"(: observed-{item.id} {' ' if item.alarm else '(Not '}{alarm}{'' if item.alarm else ')'} (STV 1 1))")
    for i in range(config.irrelevant_statements):
        lines.append(f"(: irrelevant-fact-{i} (TelemetryNoise noise-{i}) (STV 1 1))")
        lines.append(
            f"(: irrelevant-rule-{i} (Implication (Premises (TelemetryNoise $x)) "
            f"(Conclusions (ArchivedNoise{i} $x))) (CTV (STV 0.5 1) (STV 0 1)))"
        )
    return "\n".join(lines)

