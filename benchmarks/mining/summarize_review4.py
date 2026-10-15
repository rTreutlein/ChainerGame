"""Summarize the narrowly scoped final-review experiments without pooling seeds."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from miningbench.report import interval, formatted
from miningbench.world import Config, World, rng_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    paired = []
    for path in sorted(
        (args.root / "review-v2/petta-review4-shuffled").rglob("result.json")
    ):
        result = json.loads(path.read_text())
        meta = result["metadata"]
        old_path = (
            args.root
            / "review-v2/petta-171/distractors"
            / path.parent.name
            / "result.json"
        )
        old = json.loads(old_path.read_text())
        assert old["data_sha256"] == result["data_sha256"]
        assert meta["input_order"] == "seeded-per-period-v1"
        row = dict(
            system=meta["system"],
            seed=meta["config"]["seed"],
            brier=result["scores"]["brier"],
            old_brier=old["scores"]["brier"],
            recall=result["recovery"]["all"]["recall"],
            precision=result["recovery"]["all"]["precision"],
            old_peak_mib=old["peak_rss_kib"] / 1024,
            peak_mib=result["peak_rss_kib"] / 1024,
            brier_improvement=old["scores"]["brier"] - result["scores"]["brier"],
            data_identical=True,
            source=str(path),
        )
        paired.append(row)
        groups[meta["system"]].append(row)
    summary = {
        system: {
            field: interval([row[field] for row in rows])
            for field in ("brier", "recall", "precision", "brier_improvement")
        }
        for system, rows in groups.items()
    }
    timing = []
    for path in sorted(args.timing.glob("paired-*/result.json")):
        result = json.loads(path.read_text())
        load = json.loads((path.parent / "load.json").read_text())
        timing.append(
            dict(
                case=path.parent.name,
                commit=result["metadata"]["petta_commit"],
                samples=result["metadata"]["config"]["samples"],
                mining_seconds_per_step=result["mining_cost"]["mean_seconds"],
                total_seconds_per_step=result["cost"]["mean_seconds"],
                cpu_to_wall=load.get("active_cpu_seconds", load["cpu_seconds"])
                / load.get("active_wall_seconds", load["wall_seconds"]),
                active_cpu_seconds_per_step=load.get(
                    "active_cpu_seconds", load["cpu_seconds"]
                )
                / result["metadata"]["config"]["samples"],
                peak_mib=result["peak_rss_kib"] / 1024,
                data_sha256=result["data_sha256"],
                input_order=result["metadata"]["input_order"],
                config=result["metadata"]["config"],
                load=load,
            )
        )
    if timing:
        assert len({r["data_sha256"] for r in timing}) == 1
        assert len({r["load"]["petta_runtime_commit"] for r in timing}) == 1
        assert all(r["input_order"] == "historical-fixed-order-repro" for r in timing)
        assert len({r["load"]["protocol"] for r in timing}) == 1
        assert all(r["load"]["admission_cpu_slots"] in (2, 24) for r in timing)
        assert all(r["config"] == timing[0]["config"] for r in timing)
    timing_groups = defaultdict(list)
    for row in timing:
        timing_groups[row["commit"][:8]].append(row["mining_seconds_per_step"])
    ratio = (
        statistics.fmean(timing_groups["171b2820"])
        / statistics.fmean(timing_groups["7925f8d5"])
        if all(timing_groups.get(commit) for commit in ("171b2820", "7925f8d5"))
        else None
    )
    census = []
    census_paths = list(
        (args.root / "review-v2/petta-review4-shuffled").rglob("store_census.jsonl")
    )
    census_paths += list((args.root / "review4-retention").rglob("store_census.jsonl"))
    for path in sorted(census_paths):
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            stores = sample["stores"]
            selected = {
                name: {
                    "rows": sum(r["rows"] for r in stores if r["name"] == name),
                    "bytes": sum(r["bytes"] for r in stores if r["name"] == name),
                }
                for name in ("&hm_seen", "&hm_period", "&hm_outcomes", "&hm_tried")
            }
            skeleton = [r for r in stores if r["name"].startswith(("&sk_", "sk_"))]
            census.append(
                dict(
                    source=str(path),
                    step=sample["step"],
                    rss_mib=sample["rss_kib"] / 1024,
                    stores=selected,
                    skeleton_rows=sum(r["rows"] for r in skeleton),
                    skeleton_mib=sum(r["bytes"] for r in skeleton) / 2**20,
                    dynamic_mib=sum(r["bytes"] for r in stores) / 2**20,
                    top_stores=stores[:12],
                    census_seconds=sample["census_seconds"],
                )
            )
    opportunities = []
    for seed in sorted({row["seed"] for row in paired}):
        true_count = visible_count = 0
        world = World(
            Config(seed=seed, samples=256, kinds=("single",), distractors=100)
        )
        for sample in world.stream():
            inputs = sorted(sample["history"][-1].items())
            period = sample["step"] + 2  # Three initial history periods.
            rng_for(seed, f"petta-inputs:{period}").shuffle(inputs)
            positives = [atom for atom, value in inputs if value > 0.5]
            signal = ("T0_single_F0", ())
            if signal in positives:
                true_count += 1
                visible_count += signal in positives[-24:]
        opportunities.append(
            dict(
                seed=seed,
                true_inputs=true_count,
                in_newest24_true_inputs=visible_count,
                fraction=visible_count / true_count,
                interpretation="Input-only opportunity, not internal miner work",
            )
        )
    retention_checks = []
    for case, original in (
        ("fixed-d100", "review-v2/petta-171/distractors/distractors-petta-9-single"),
        ("fixed-deep", "review-v2/petta-171/deep/deep_lags-petta-1-single"),
    ):
        current = args.root / "review4-retention" / case
        if not (current / "steps.jsonl").exists():
            continue
        old = args.root / original
        check = {
            "case": case,
            "original": str(old),
            "complete": (current / "result.json").exists(),
        }
        for filename, keys in (
            ("steps.jsonl", ("predicted", "exact", "label")),
            ("samples.jsonl", ()),
        ):
            before = [
                json.loads(line) for line in (old / filename).read_text().splitlines()
            ]
            after = [
                json.loads(line)
                for line in (current / filename).read_text().splitlines()
            ]
            for a, b in zip(before, after):
                assert all(a[key] == b[key] for key in keys) if keys else a == b
            check[filename + "_verified_rows"] = min(len(before), len(after))
        retention_checks.append(check)
    result = dict(
        analysis_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        retention_prefix_checks=retention_checks,
        input_tail_opportunities=opportunities,
        distractor_cells=paired,
        distractor_summary=summary,
        numeric_timing=timing,
        numeric_revision_ratio=ratio,
        numeric_cpu_revision_ratio=(
            timing[1]["active_cpu_seconds_per_step"]
            / timing[0]["active_cpu_seconds_per_step"]
            if len(timing) == 2
            else None
        ),
        synchronization=(
            json.loads((args.timing / "synchronization.json").read_text())
            if (args.timing / "synchronization.json").exists()
            else None
        ),
        census=census,
    )
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Final-review experiments",
        "",
        "| Shuffled arm | Seeds | Brier [95% CI] | Recall [95% CI] | Precision [95% CI] | Paired Brier improvement [95% CI] |",
        "|---|---:|---|---|---|---|",
    ]
    for system, rows in groups.items():
        lines.append(
            f"| {system} | {len(rows)} | {formatted([r['brier'] for r in rows])} | "
            f"{formatted([r['recall'] for r in rows])} | {formatted([r['precision'] for r in rows])} | {formatted([r['brier_improvement'] for r in rows])} |"
        )
    lines += [
        "",
        "Fewer than ten seeds is underpowered. Positive improvement means the shuffle lowered Brier.",
        "",
        "| Timing case | Steps | Mining s/step | Total s/step | Active CPU s/step | CPU/wall | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing:
        lines.append(
            f"| {row['case']} | {row['samples']} | {row['mining_seconds_per_step']:.3f} | "
            f"{row['total_seconds_per_step']:.3f} | {row['active_cpu_seconds_per_step']:.3f} | {row['cpu_to_wall']:.3f} | {row['peak_mib']:.1f} |"
        )
    lines += [
        "",
        f"Mean revision timing ratio (171 / 792): {ratio}. One seed; repeated timing runs are not independent seeds.",
        "",
        "| Census case | Step | RSS MiB | Dynamic predicate MiB | Skeleton MiB | hm_seen / period / outcomes / tried rows |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in census:
        counts = " / ".join(
            str(row["stores"][name]["rows"])
            for name in ("&hm_seen", "&hm_period", "&hm_outcomes", "&hm_tried")
        )
        lines.append(
            f"| {Path(row['source']).parent.name} | {row['step']} | {row['rss_mib']:.1f} | "
            f"{row['dynamic_mib']:.1f} | {row['skeleton_mib']:.1f} | {counts} |"
        )
    lines += [
        "",
        "Predicate sizes are allocation estimates, not an exact partition of RSS. No GC or store mutation was performed.",
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
