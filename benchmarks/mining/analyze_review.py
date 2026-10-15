"""Build a review report from immutable raw runs, including behavioural probes.

Run through capped, with the shared Python and PYTHONPATH=src. This script never
changes existing run evidence. Supply a fresh --out directory.
"""

from dataclasses import asdict, replace
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import statistics

from miningbench.report import report, interval, formatted, separate_runtime_cohorts
from miningbench.runner import source_hash, revision, is_new_entity_context
from miningbench.scoring import (
    losses,
    CostMeter,
    RunningAverage,
    association_scores,
    recovery,
)
from miningbench.world import Config, World, Rule, Literal
from miningbench.petta import parse, decode, export_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results, failures = [], []
    paths = {}
    current_source = source_hash()
    for path in sorted(args.root.rglob("result.json")):
        # Replaced reference-arm runs retain evidence but are not double counted.
        if path.parent.parent.name in (
            "directed",
            "directed-bounded",
            "directed-entity-final",
            "directed-fair",
        ) and json.loads(path.read_text())["metadata"]["system"].startswith("budgeted"):
            continue
        result = json.loads(path.read_text())
        if result["metadata"].get(
            "level"
        ) == "entity_holdout" and path.parent.parent.name not in (
            "entity-final",
            "directed-entity-final",
            "petta-entity-final",
            "directed-fair",
            "directed-audited",
        ):
            continue
        if path.parent.parent.name == "pure" and result["metadata"]["config"][
            "kinds"
        ] == ["numeric"]:
            continue
        status_path = path.parent / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["returncode"]:
            continue
        if path.parent.parent.name in (
            "directed",
            "directed-bounded",
            "directed-entity-final",
            "directed-fair",
            "directed-audited",
        ):
            # These are a distinct six-target cohort, not extra independent
            # seeds for the single-type base experiment.
            result["metadata"]["level"] = "directed_" + result["metadata"].get(
                "level", "run"
            )
        # Recompute the common cost definition from immutable observations.
        meters = {
            name: CostMeter(result["metadata"]["config"]["samples"])
            for name in ("cost", "mining_cost", "feedback_cost")
        }
        config = result["metadata"]["config"]
        world = World(Config(**config))
        work_totals = {}
        held = {}
        held_all = RunningAverage()
        live, negative_rates = {}, {}
        domains = result["metadata"]["domains"]
        label_counts = {target: [0] * len(domain) for target, domain in domains.items()}
        checkpoints = {point["samples"]: point for point in result["learning_curve"]}
        is_petta = result["metadata"]["system"].startswith("petta")
        for point in checkpoints.values():
            point["recorded_recovery"] = point["recovery"]
            point["recovery_scoring"] = "recorded_at_run"
            if not is_petta and result["metadata"]["source_sha256"] != current_source:
                # Reference runs did not save intermediate rule snapshots.
                # Preserve their historical values without presenting them as
                # current-matcher results. The final rules are saved below.
                point["recovery"] = None
                point["recovery_scoring"] = "historical_only_no_rule_snapshot"
        for line, sample_line in zip(
            (path.parent / "steps.jsonl").open(),
            (path.parent / "samples.jsonl").open(),
            strict=True,
        ):
            step = json.loads(line)
            work = step.get("work_units", 0)
            if isinstance(work, dict):
                work = dict(work)
                if step.get("holdout"):
                    # Older artifacts retained the previous learn() counters.
                    # The recorded holdout protocol performs prediction only.
                    assert not step["events"]
                    if result["metadata"]["system"].startswith("petta"):
                        work.update(
                            input_atoms=len(world.atoms), label_atoms=0, review_rows=0
                        )
                    else:
                        work.update(
                            table_updates=0,
                            candidate_evaluations=0,
                            base_rate_updates=0,
                            by_lag={},
                        )
                for key, value in work.items():
                    if isinstance(value, (int, float)):
                        work_totals[key] = work_totals.get(key, 0) + value
                    elif key == "by_lag":
                        for lag, count in value.items():
                            name = "lag_" + lag
                            work_totals[name] = work_totals.get(name, 0) + count
            if not step.get("holdout"):
                for target, value in step["label"].items():
                    label_counts[target][value] += 1
            for event in step.get("events", []):
                if "raw" not in event:
                    continue
                _, name, implication, tv, verdict = parse(event["raw"])
                rule = decode(implication, domains)
                positive = tv[1] if tv[0] == "CTV" else tv
                rule = replace(rule, strength=float(positive[1]))
                if verdict == "dropped":
                    live.pop(name, None)
                    negative_rates.pop(name, None)
                else:
                    live[name] = rule
                    if tv[0] == "CTV":
                        negative_rates[name] = float(tv[2][1])
            point = checkpoints.get(step["step"] + 1)
            if is_petta and point is not None:
                checkpoint_rules = export_rules(
                    live, negative_rates, label_counts, domains
                )
                phase = point["phase"]
                point["recovery"] = recovery(
                    checkpoint_rules, world.rules[phase], world.types, world, phase
                )
                point["recovery_scoring"] = "raw_verdict_replay"
            if step.get("holdout"):
                observed = {
                    (head, tuple(args)): value
                    for head, args, value in json.loads(sample_line)["history"][-1]
                }
                for target, score in step["scores"].items():
                    kind = result["metadata"]["types"][target]
                    if kind == "binding" and not is_new_entity_context(
                        target, observed
                    ):
                        continue
                    held.setdefault(kind, RunningAverage()).add(score)
                    held_all.add(score)
            for name, field in (
                ("cost", "total_seconds"),
                ("mining_cost", "mining_path_seconds"),
                ("feedback_cost", "learn_seconds"),
            ):
                meters[name].add(step[field])
        result.update({name: meter.result() for name, meter in meters.items()})
        result["holdout_by_type"] = {
            kind: score.result() for kind, score in held.items()
        }
        result["holdout_scores"] = held_all.result()
        result["holdout_scored_predictions_by_type"] = {
            kind: score.count for kind, score in held.items()
        }
        result["work_totals"] = work_totals
        rules = [
            Rule(
                **dict(
                    rule,
                    body=tuple(
                        Literal(**dict(literal, args=tuple(literal["args"])))
                        for literal in rule["body"]
                    ),
                )
            )
            for rule in result["learned_rules"]
        ]
        if result["metadata"]["system"].startswith("petta"):
            expected_live = [
                Rule(
                    **dict(
                        rule,
                        body=tuple(
                            Literal(**dict(literal, args=tuple(literal["args"])))
                            for literal in rule["body"]
                        ),
                    )
                )
                for rule in result["petta"]["live_rules"]
            ]
            if set(live.values()) != set(expected_live):
                raise ValueError(
                    f"raw verdict replay disagrees with saved live rules: {path}"
                )
            rules = export_rules(live, negative_rates, label_counts, domains)
            result["learned_rules"] = [asdict(rule) for rule in rules]
            result["rule_state_replay_verified"] = True
        phase = int(config["drift"])
        result["associations"] = association_scores(
            rules, world.rules[phase], world, phase, path.parent / "samples.jsonl"
        )
        result["recovery"] = recovery(
            rules, world.rules[phase], world.types, world, phase
        )
        final_point = checkpoints[config["samples"]]
        if is_petta and final_point["recovery"] != result["recovery"]:
            raise ValueError(f"checkpoint replay disagrees with final recovery: {path}")
        final_point["recovery"] = result["recovery"]
        final_point["recovery_scoring"] = "final_rules_recomputed"

        results.append(result)
        paths[id(result)] = path
    for path in sorted(args.root.rglob("status.json")):
        status = json.loads(path.read_text())
        if status["returncode"]:
            raw = path.parent / "steps.jsonl"
            status["completed_steps"] = (
                sum(1 for _ in raw.open()) if raw.exists() else 0
            )
            world_path = path.parent / "world.json"
            metadata = json.loads(world_path.read_text()) if world_path.exists() else {}
            if metadata:
                status["petta_commit"] = metadata.get("petta_commit")
                status["input_order"] = metadata.get("input_order")
            if raw.exists():
                meter = CostMeter(metadata["config"]["samples"])
                peak = 0
                total_seconds = 0
                for line in raw.open():
                    step = json.loads(line)
                    meter.add(step["mining_path_seconds"])
                    total_seconds += step["total_seconds"]
                    peak = max(peak, step["rss_peak_kib"])
                status["partial_mining_cost"] = meter.result() if meter.count else None
                status["partial_total_seconds"] = total_seconds
                status["logged_peak_rss_mib"] = peak / 1024
            failures.append(status)
    cohorts, failures = separate_runtime_cohorts(results, failures)
    paths = {id(new): paths[id(old)] for old, new in zip(results, cohorts)}
    results = cohorts
    with (args.out / "rescored_results.jsonl").open("x") as output:
        for row in results:
            output.write(
                json.dumps({"source": str(paths[id(row)]), "result": row}) + "\n"
            )
    text, checks = report(results, failures)
    (args.out / "report.md").write_text(text)
    (args.out / "acceptance.json").write_text(json.dumps(checks, indent=2) + "\n")
    groups = defaultdict(dict)
    numeric_memory = defaultdict(list)
    curves = defaultdict(list)
    holdout = defaultdict(list)
    for result in results:
        metadata = result["metadata"]
        config = metadata["config"]
        level, system = metadata.get("level", "run"), metadata["system"]
        groups[(level, config["seed"], tuple(config["kinds"]), config["targets"])][
            system
        ] = result
        for point in result["learning_curve"]:
            for kind, score in point["cumulative"].items():
                curves[(system, level, kind, point["samples"])].append(score["brier"])
        for kind, score in result["holdout_by_type"].items():
            holdout[(system, (kind,), config["targets"])].append(score["brier"])
        if (
            system == "oracle"
            and config["kinds"] == ["numeric"]
            and level in ("base", "long")
        ):
            memory, errors = {}, []
            path = paths[id(result)].parent / "samples.jsonl"
            for line in path.open():
                sample = json.loads(line)
                key = tuple((h, tuple(a)) for h, a, v in sample["history"][-1] if v)
                q = sample["q"]["T0_numeric"]
                predicted = memory.get(key, [1 / len(q)] * len(q))
                errors.append(losses(predicted, q)["brier"])
                memory[key] = q
            numeric_memory[level].append(statistics.fmean(errors))
    directed_by_seed = defaultdict(lambda: defaultdict(list))
    work = defaultdict(list)
    for (level, seed, kinds, targets), arms in groups.items():
        if "budgeted" in arms and "budgeted-flat" in arms:
            directed_by_seed[level][seed].append(
                arms["budgeted-flat"]["relevance_weighted_brier"]
                - arms["budgeted"]["relevance_weighted_brier"]
            )
        for system, result in arms.items():
            if system.startswith("budgeted"):
                counts = result["work_totals"]
                updates = counts.get("table_updates", 0)
                deep = sum(
                    v
                    for k, v in counts.items()
                    if k.startswith("lag_") and int(k[4:]) > 3
                )
                work[(level, system)].append(
                    {
                        "updates": updates,
                        "deep": deep,
                        "reads": counts.get("feature_reads"),
                        "weighted_brier": result["relevance_weighted_brier"],
                    }
                )
    # Ten kinds evaluated with the same ten seeds are ten independent seed
    # clusters, not one hundred independent replications.
    directed = {
        level: [statistics.fmean(values[seed]) for seed in sorted(values)]
        for level, values in directed_by_seed.items()
    }
    behaviour = {
        "relevance_priority_paired_gain": {k: interval(v) for k, v in directed.items()},
        "numeric_exact_context_memorizer_brier": {
            k: interval(v) for k, v in numeric_memory.items()
        },
        "holdout_brier": {str(k): interval(v) for k, v in holdout.items()},
        "learning_curves": {str(k): interval(v) for k, v in curves.items()},
        "work": {str(k): v for k, v in work.items()},
    }
    (args.out / "behaviour.json").write_text(json.dumps(behaviour, indent=2) + "\n")
    sections = [
        "",
        "## Behavioural probes",
        "",
        "| Cohort | Priority gain in relevance-weighted Brier [95% paired CI] |",
        "|---|---|",
    ]
    for level, values in sorted(directed.items()):
        sections.append(f"| {level} | {formatted(values)} |")
    sections += [
        "",
        "| System | Types | Target copies | New-entity Brier [95% CI] |",
        "|---|---|---:|---|",
    ]
    for (system, kinds, targets), values in sorted(holdout.items()):
        sections.append(
            f"| {system} | {', '.join(kinds)} | {targets} | {formatted(values)} |"
        )
    sections += [
        "",
        "| Numeric stream | Context memorizer Brier [95% CI] |",
        "|---|---|",
    ]
    for level, values in sorted(numeric_memory.items()):
        sections.append(f"| {level} | {formatted(values)} |")
    sections += [
        "",
        "| Cohort | System | Mean table updates per run | Mean deep-lag updates |",
        "|---|---|---:|---:|",
    ]
    for (level, system), rows in sorted(work.items()):
        sections.append(
            f"| {level} | {system} | {statistics.fmean(r['updates'] for r in rows):.0f} | "
            f"{statistics.fmean(r['deep'] for r in rows):.0f} |"
        )
    sections += [
        "",
        "## Censored cost measurements",
        "",
        "Only completed steps are measured; an in-flight stalled step can make these lower bounds. "
        "These are not full-stream cost fits or completed prediction scores.",
        "",
        "| Difficulty | System | Type | Seed | Exit | Steps | Mean mining seconds | Logged peak MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for failure in failures:
        cost = failure.get("partial_mining_cost")
        if cost and failure["returncode"] == 124:
            sections.append(
                f"| {failure['level']} | {failure['system']} | {', '.join(failure['kinds'])} | "
                f"{failure['seed']} | {failure['returncode']} | {failure['completed_steps']} | "
                f"{cost['mean_seconds']:.4f} | {failure['logged_peak_rss_mib']:.1f} |"
            )
    sections += [
        "",
        "## Learning curves",
        "",
        "| System | Difficulty | Type | Samples | Brier [95% CI] |",
        "|---|---|---|---:|---|",
    ]
    for (system, level, kind, samples), values in sorted(curves.items()):
        sections.append(
            f"| {system} | {level} | {kind} | {samples} | {formatted(values)} |"
        )
    with (args.out / "report.md").open("a") as output:
        output.write("\n".join(sections) + "\n")
    provenance = {
        "analysis_source_sha256": source_hash(),
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "analysis_commit": revision(Path(__file__).resolve().parents[2]),
        "recomputed": [
            "cost slopes after 32 steps",
            "holdout work counts from the prediction-only protocol",
            "binding holdout scores restricted to observed new-entity contexts",
            "numeric finite-history association calibration",
            "recovery threshold in the observed outcome channel",
            "CTV exports replayed from raw verdicts, preserving the pooled complement",
        ],
        "runs": [str(paths[id(r)]) for r in results],
        "run_systems": {str(paths[id(r)]): r["metadata"]["system"] for r in results},
        "failures": failures,
        "petta_commits": sorted(
            {
                r["metadata"]["petta_commit"]
                for r in results
                if "petta_commit" in r["metadata"]
            }
        ),
        "source_hashes": sorted({r["metadata"]["source_sha256"] for r in results}),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {"completed": len(results), "failed": len(failures), "out": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
