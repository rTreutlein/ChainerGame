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
    direct_sensor_rules: bool = True,
    problem_history: bool = False,
    temporal: dict | None = None,
) -> str:
    """Emit public facts plus full, positive-only, or induced sensor knowledge.

    Each typed resolved case is additionally one individual, a ``Member`` of
    its group's concepts.  Only resolved history enters those memberships;
    an unresolved current alarm must not become an unlabeled induction sample.

    With ``temporal`` (hazards per group, each current incident's link to
    the module's previous shift, and the shortfall reports) the knowledge
    instead describes a filter: each incident's seal state is predicted from
    the module's state at the end of the previous shift and updated by the
    incident's alarm, and a shift's state is conditioned on its shortfall.
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
        if temporal is not None and equipment_type is not None:
            # A leak persists until repaired; a sound or freshly serviced seal
            # starts leaking with the group's per-shift hazard.
            hazard = _number(temporal["hazards"][group])
            lines.append(
                f"(: persist-{suffix} "
                f"(Implication (And (NextState $previous $unit) "
                f"(SealLeakAtShiftEnd {cohort} {equipment_type} $previous)) "
                f"(LeakPredicted {variables})) "
                f"(CTV (STV 1 1) (STV {hazard} 1)))"
            )
            lines.append(
                f"(: fresh-{suffix} "
                f"(Implication (Serviced {variables}) (LeakPredicted {variables})) "
                f"(CTV (STV {hazard} 1) (STV {hazard} 1)))"
            )
        elif direct_sensor_rules and mode == "full":
            lines.append(
                f"(: alarmGivenLeak-{suffix} "
                f"(Implication (SealLeak {variables}) (PressureAlarm {variables})) "
                f"(CTV (STV {_number(sensitivity)} 1) "
                f"(STV {_number(false_positive)} 1)))"
            )
        elif direct_sensor_rules and mode == "positive":
            lines.append(
                f"(: alarmGivenLeak-{suffix} "
                f"(Implication (SealLeak {variables}) (PressureAlarm {variables})) "
                f"(STV {_number(sensitivity)} 1))"
            )
        elif mode not in {"full", "positive", "induced"}:
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
        if (
            problem_history
            and case.equipment_type is not None
            and case.problem is not None
        ):
            lines.append(
                f"(: problem-history-{case.id} "
                f"(Problem {arguments}) "
                f"(STV {1 if case.problem else 0} 1))"
            )
        if temporal is None and case.equipment_type is not None:
            # A resolved case is one individual, so it is a Member of each
            # group concept and counts once in the concepts' member folds.
            group = _group(case)
            equipment_concept = _concept("EquipmentState", group)
            leak_concept = _concept("SealLeak", group)
            alarm_concept = _concept("PressureAlarm", group)
            normal_concept = _concept("PressureNormal", group)
            lines.extend((
                f"(: state-equipment-{case.id} "
                f"(Member {case.id} {equipment_concept}) (STV 1 1))",
                f"(: state-leak-{case.id} (Member {case.id} {leak_concept}) "
                f"(STV {1 if case.leak else 0} 1))",
                f"(: state-alarm-{case.id} (Member {case.id} {alarm_concept}) "
                f"(STV {1 if case.alarm else 0} 1))",
                f"(: state-normal-{case.id} (Member {case.id} {normal_concept}) "
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

    if temporal is not None:
        # A sensor behaves the same in every cohort: its rule is stated once
        # per equipment type, refined from the resolved cases, and each alarm
        # updates the incident's own predicted state.
        for equipment_type in sorted({group[1] for group in groups if group[1] is not None}):
            mode = (
                sensor_knowledge.get(equipment_type, "full")
                if sensor_knowledge is not None
                else "full"
            )
            sensitivity, false_positive = (
                sensor_models[equipment_type]
                if sensor_models is not None
                else (config.sensitivity, config.false_positive_rate)
            )
            truth = {
                "full": f"(CTV (STV {_number(sensitivity)} 1) (STV {_number(false_positive)} 1))",
                "positive": f"(STV {_number(sensitivity)} 1)",
                "induced": "(STV 0.5 0.5)",
            }[mode]
            variables = f"$cohort {equipment_type} $unit"
            lines.append(
                f"(: alarmGivenLeak-{equipment_type} "
                f"(ForAll ($cohort) (WithPrior (LeakPredicted {variables}) "
                f"(Implication (SealLeak {variables}) (PressureAlarm {variables})))) "
                f"{truth})"
            )
        # A shift ends in the state its incidents were in, unless the shift
        # reported a shortfall: then its candidates' states are conditioned on
        # the loss by the exact weighted-subset posterior over their beliefs.
        lines.append(
            "(: shiftEnd (Implication (SealLeak $cohort $type $unit) "
            "(SealLeakAtShiftEnd $cohort $type $unit)) "
            "(CTV (STV 1 1) (STV 0 1)))"
        )
        lines.append(
            "(: shortfallPosterior (Implication (And "
            "(ShortfallCandidate $event $cohort $type $unit $impact) "
            "(ShortfallLoss $event $loss) "
            "(FoldAllTruth (And (ShortfallCandidate $event $c $t $v $i) (SealLeak $c $t $v)) "
            "(STV $s $_c) (WeightedCandidate $v $i $s) () "
            "(|-> ($acc $elem) (cons $elem $acc)) -> $candidates) "
            "(Compute WeightedSubsetPosteriorDP ($candidates $loss) -> $posterior) "
            "(SealLeak $cohort $type $unit) "
            "(Compute WeightedSubsetPosteriorMarginal ($posterior $unit) -> $p)) "
            "(SealLeakAtShiftEnd $cohort $type $unit)) "
            "(CTV (STV $p 1) (STV $p 1)))"
        )
        for report in temporal["shortfalls"]:
            event = f"shortfall-s{report['shift']:02d}"
            lines.append(f"(: {event} (ShortfallLoss {event} {report['loss']}) (STV 1 1))")
            lines.extend(
                f"(: {event}-{unit} (ShortfallCandidate {event} "
                f"{cohort} {equipment_type} {unit} {impact}) (STV 1 1))"
                for cohort, equipment_type, unit, impact in report["candidates"]
            )
        for item in incidents:
            previous, serviced = temporal["links"].get(item.id, (None, False))
            if serviced:
                lines.append(
                    f"(: serviced-{item.id} (Serviced {_arguments(item)}) (STV 1 1))"
                )
            elif previous is not None:
                lines.append(
                    f"(: next-{item.id} (NextState {previous} {item.id}) (STV 1 1))"
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

    A local problem and an upstream dependency are the two possible causes of
    the same module-level problem.  They remain the same propositions in prior
    and diagnostic use: FoldAll can run their complete queries to obtain the
    base rate of the literal OR, and backward OR projection can then condition
    the local cause on current alarms.
    """
    by_module = {
        item.module_id: item for item in incidents if item.module_id is not None
    }
    context = f"shift-{shift:02d}"

    def graph_atom(head: str, item: Incident) -> str:
        return (
            f"({head} {context} {item.module_id} {item.cohort} "
            f"{item.equipment_type} {item.id})"
        )

    def local_cause_atom(item: Incident) -> str:
        return (
            f"(Member (ModuleState {context} {item.module_id}) "
            f"(SealLeak {item.cohort} {item.equipment_type}))"
        )

    def problem_atom(item: Incident) -> str:
        return f"(Problem {item.cohort} {item.equipment_type} {item.id})"

    lines = []
    for item in incidents:
        if item.module_id is None:
            continue
        local_source = local_cause_atom(item)
        local_problem = graph_atom("LocalProblem", item)
        dependency = graph_atom("ProblemDependency", item)
        problem = problem_atom(item)
        upstream_ids = tuple(
            upstream_id
            for upstream_id in item.upstream_module_ids
            if upstream_id in by_module
        )
        lines.append(
            f"(: current-equipment-state-{item.id} "
            f"(Member (ModuleState {context} {item.module_id}) "
            f"{_concept('EquipmentState', _group(item))}) (STV 1 1))"
        )
        lines.append(
            f"(: local-problem-{item.id} "
            f"(BiImplication {local_source} {local_problem}) "
            f"(CTV (STV 1 1) (STV 0 1)))"
        )
        if upstream_ids:
            lines.append(
                f"(: problem-from-disjunction-{item.id} "
                f"(BiImplication (Or {local_problem} {dependency}) {problem}) "
                f"(CTV (STV 1 1) (STV 0 1)))"
            )
        else:
            lines.append(
                f"(: root-problem-{item.id} "
                f"(BiImplication {local_problem} {problem}) "
                f"(CTV (STV 1 1) (STV 0 1)))"
            )

    for cohort, equipment_type in sorted(
        {_group(item) for item in incidents},
        key=lambda group: (group[0], group[1] or ""),
    ):
        mode = sensor_knowledge.get(equipment_type, "full")
        if mode == "full":
            sensitivity, false_positive = sensor_models.get(
                equipment_type, (1.0, 0.0)
            )
            suffix = _group_suffix((cohort, equipment_type))
            lines.append(
                f"(: problem-alarm-{suffix} "
                f"(Implication "
                f"(Problem {cohort} {equipment_type} $unit) "
                f"(PressureAlarm {cohort} {equipment_type} $unit)) "
                f"(CTV (STV {_number(sensitivity)} 1) "
                f"(STV {_number(false_positive)} 1)))"
            )
        elif mode == "positive":
            sensitivity, _ = sensor_models.get(equipment_type, (1.0, 0.0))
            suffix = _group_suffix((cohort, equipment_type))
            lines.append(
                f"(: problem-alarm-{suffix} "
                f"(Implication "
                f"(Problem {cohort} {equipment_type} $unit) "
                f"(PressureAlarm {cohort} {equipment_type} $unit)) "
                f"(STV {_number(sensitivity)} 1))"
            )
        elif mode != "induced":
            raise ValueError(f"unknown sensor knowledge mode: {mode}")

    for downstream in incidents:
        if downstream.module_id is None:
            continue
        for upstream_id in downstream.upstream_module_ids:
            if upstream_id not in by_module:
                continue
            upstream = by_module[upstream_id]
            lines.extend((
                f"(: topology-{context}-{upstream_id}-{downstream.module_id} "
                f"(DependsOn {downstream.module_id} {upstream_id}) (STV 1 1))",
                f"(: dependency-problem-{context}-{upstream_id}-{downstream.module_id} "
                f"(BiImplication {problem_atom(upstream)} "
                f"{graph_atom('ProblemDependency', downstream)}) "
                f"(CTV (STV 1 1) (STV 0 1)))",
            ))
    return "\n".join(lines)
