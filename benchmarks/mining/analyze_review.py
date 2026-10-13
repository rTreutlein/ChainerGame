"""Build a review report from immutable raw runs, including behavioural probes.

Run through capped, with the shared Python and PYTHONPATH=src. This script never
changes existing run evidence. Supply a fresh --out directory.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from miningbench.report import report, interval, formatted
from miningbench.runner import source_hash, revision
from miningbench.scoring import losses, CostMeter, association_scores
from miningbench.world import Config, World, Rule, Literal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results, failures = [], []
    paths = {}
    for path in sorted(args.root.rglob("result.json")):
        # Replaced reference-arm runs retain evidence but are not double counted.
        if path.parent.parent.name == "directed" and json.loads(path.read_text())[
            "metadata"
        ]["system"].startswith("budgeted"):
            continue
        result = json.loads(path.read_text())
        if path.parent.parent.name == "pure" and result["metadata"]["config"][
            "kinds"
        ] == ["numeric"]:
            continue
        status_path = path.parent / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["returncode"]:
            continue
        if path.parent.parent.name in ("directed", "directed-bounded"):
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
        for line in (path.parent / "steps.jsonl").open():
            step = json.loads(line)
            for name, field in (
                ("cost", "total_seconds"),
                ("mining_cost", "mining_path_seconds"),
                ("feedback_cost", "learn_seconds"),
            ):
                meters[name].add(step[field])
        result.update({name: meter.result() for name, meter in meters.items()})
        config = result["metadata"]["config"]
        if "numeric" in config["kinds"]:
            world = World(Config(**config))
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
            phase = int(config["drift"])
            result["associations"] = association_scores(
                rules, world.rules[phase], world, phase, path.parent / "samples.jsonl"
            )

        results.append(result)
        paths[id(result)] = path
    for path in sorted(args.root.rglob("status.json")):
        status = json.loads(path.read_text())
        if status["returncode"]:
            raw = path.parent / "steps.jsonl"
            status["completed_steps"] = (
                sum(1 for _ in raw.open()) if raw.exists() else 0
            )
            failures.append(status)
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
        if result["holdout_scores"]:
            holdout[(system, tuple(config["kinds"]), config["targets"])].append(
                result["holdout_scores"]["brier"]
            )
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
    directed = defaultdict(list)
    work = defaultdict(list)
    for (level, seed, kinds, targets), arms in groups.items():
        if "budgeted" in arms and "budgeted-flat" in arms:
            directed[level].append(
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
        "analysis_commit": revision(Path(__file__).resolve().parents[2]),
        "recomputed": [
            "cost slopes after 32 steps",
            "numeric finite-history association calibration",
        ],
        "runs": [str(paths[id(r)]) for r in results],
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
