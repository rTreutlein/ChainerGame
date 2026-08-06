from .config import Config
from .models import HistoryCase, Incident


def _arguments(item: HistoryCase | Incident, include_id: bool = True) -> str:
    fields = [item.cohort]
    if item.equipment_type is not None:
        fields.append(item.equipment_type)
    if include_id:
        fields.append(item.id)
    return " ".join(fields)


def _group(item: HistoryCase | Incident) -> tuple[str, str | None]:
    return item.cohort, item.equipment_type


def _group_suffix(group: tuple[str, str | None]) -> str:
    return "-".join(field for field in group if field is not None)


def _concept(name: str, group: tuple[str, str | None]) -> str:
    return f"({name} {' '.join(field for field in group if field is not None)})"


def _number(value: float) -> str:
    return format(value, ".15g")


def generate_statements(
    history: list[HistoryCase],
    incidents: list[Incident],
    config: Config,
    *,
    sensor_knowledge: dict[str, str] | None = None,
    sensor_models: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Emit public facts plus full, positive-only, or induced sensor knowledge.

    Typed StationOps cases additionally share a ``State`` subject through
    ``Inheritance`` observations.  Only resolved history enters those pairs;
    an unresolved current alarm must not become an unlabeled induction sample.
    """
    groups = sorted(
        {_group(item) for item in history} | {_group(item) for item in incidents},
        key=lambda group: (group[0], group[1] or ""),
    )
    lines = []
    for group in groups:
        cohort, equipment_type = group
        suffix = _group_suffix(group)
        mode = (
            sensor_knowledge.get(equipment_type, "full")
            if sensor_knowledge is not None and equipment_type is not None
            else "full"
        )
        sensitivity, false_positive = (
            sensor_models[equipment_type]
            if sensor_models is not None and equipment_type is not None
            else (config.sensitivity, config.false_positive_rate)
        )
        variables = (
            f"{cohort} "
            + (f"{equipment_type} " if equipment_type else "")
            + "$unit"
        )
        if mode == "full":
            lines.append(
                f"(: alarmGivenLeak-{suffix} "
                f"(Implication (SealLeak {variables}) (PressureAlarm {variables})) "
                f"(CTV (STV {_number(sensitivity)} 1) "
                f"(STV {_number(false_positive)} 1)))"
            )
        elif mode == "positive":
            lines.append(
                f"(: alarmGivenLeak-{suffix} "
                f"(Implication (SealLeak {variables}) (PressureAlarm {variables})) "
                f"(STV {_number(sensitivity)} 1))"
            )
        elif mode != "induced":
            raise ValueError(f"unknown sensor knowledge mode: {mode}")

        lines.append(
            f"(: patchGoal-{suffix} "
            f"(Implication (SealLeak {variables}) (PatchPaysOff {variables})) "
            f"(CTV (STV 1 1) (STV 0 1)))"
        )

    for case in history:
        arguments = _arguments(case)
        lines.append(
            f"(: leak-{case.id} (SealLeak {arguments}) "
            f"(STV {1 if case.leak else 0} 1))"
        )
        lines.append(
            f"(: alarm-{case.id} (PressureAlarm {arguments}) "
            f"(STV {1 if case.alarm else 0} 1))"
        )
        if case.equipment_type is not None:
            group = _group(case)
            state = f"(State {case.id})"
            leak_concept = _concept("SealLeak", group)
            alarm_concept = _concept("PressureAlarm", group)
            normal_concept = _concept("PressureNormal", group)
            lines.extend((
                f"(: state-leak-{case.id} (Inheritance {state} {leak_concept}) "
                f"(STV {1 if case.leak else 0} 1))",
                f"(: state-alarm-{case.id} (Inheritance {state} {alarm_concept}) "
                f"(STV {1 if case.alarm else 0} 1))",
                f"(: state-normal-{case.id} (Inheritance {state} {normal_concept}) "
                f"(STV {0 if case.alarm else 1} 1))",
            ))

    for item in incidents:
        # Keep the same statement identity when this incident later becomes a
        # history case.  State-pair facts are intentionally delayed until the
        # leak outcome is resolved.
        lines.append(
            f"(: alarm-{item.id} (PressureAlarm {_arguments(item)}) "
            f"(STV {1 if item.alarm else 0} 1))"
        )

    for i in range(config.irrelevant_statements):
        lines.append(f"(: irrelevant-fact-{i} (TelemetryNoise noise-{i}) (STV 1 1))")
        lines.append(
            f"(: irrelevant-rule-{i} (Implication (TelemetryNoise $x) "
            f"(ArchivedNoise{i} $x)) (CTV (STV 0.5 1) (STV 0 1)))"
        )
    return "\n".join(lines)
