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


def generate_dependency_statements(
    incidents: list[Incident],
    *,
    shift: int,
    sensor_knowledge: dict[str, str],
    sensor_models: dict[str, tuple[float, float]],
) -> str:
    """Emit the public topology and the causal availability chain for one shift.

    Rules are concrete at the shift boundary so both chainer implementations
    exercise the same small common language.  A diagnosis can travel backwards
    from a downstream alarm through any number of ``Unavailable`` edges to the
    local seal fault that could have initiated the outage.
    """
    by_module = {
        item.module_id: item for item in incidents if item.module_id is not None
    }
    context = f"shift-{shift:02d}"
    lines = []
    for item in incidents:
        if item.module_id is None:
            continue
        arguments = _arguments(item)
        lines.append(
            f"(: local-outage-{item.id} "
            f"(Implication (SealLeak {arguments}) "
            f"(Unavailable {context} {item.module_id})) "
            f"(CTV (STV 1 1) (STV 0 1)))"
        )
        mode = (
            sensor_knowledge.get(item.equipment_type, "full")
            if item.equipment_type is not None
            else "full"
        )
        sensitivity, false_positive = sensor_models.get(
            item.equipment_type, (1.0, 0.0)
        )
        if mode == "full":
            truth_value = (
                f"(CTV (STV {_number(sensitivity)} 1) "
                f"(STV {_number(false_positive)} 1))"
            )
        elif mode in {"positive", "induced"}:
            # There are no public availability labels from which to learn this
            # relation yet.  A positive-only STV also cannot be composed with
            # the deterministic CTV availability edges by both backends: PeTTa
            # correctly surfaces the missing inverse base rate as ``no-tv``.
            # The topology remains usable once a fully characterized sensor on
            # the path supplies an endpoint.
            truth_value = None
        else:
            raise ValueError(f"unknown sensor knowledge mode: {mode}")
        if truth_value is not None:
            lines.append(
                f"(: outage-alarm-{item.id} "
                f"(Implication (Unavailable {context} {item.module_id}) "
                f"(PressureAlarm {arguments})) {truth_value})"
            )

    for downstream in incidents:
        if downstream.module_id is None:
            continue
        for upstream_id in downstream.upstream_module_ids:
            if upstream_id not in by_module:
                continue
            lines.extend((
                f"(: topology-{context}-{upstream_id}-{downstream.module_id} "
                f"(DependsOn {downstream.module_id} {upstream_id}) (STV 1 1))",
                f"(: propagate-{context}-{upstream_id}-{downstream.module_id} "
                f"(Implication (Unavailable {context} {upstream_id}) "
                f"(Unavailable {context} {downstream.module_id})) "
                f"(CTV (STV 1 1) (STV 0 1)))",
            ))
    return "\n".join(lines)
