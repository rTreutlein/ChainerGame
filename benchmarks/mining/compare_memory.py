"""Compare measured RSS at equal history lengths in same-config seeded runs.

Coverage manifests select immutable prior attempts. Full-run peaks are reported
separately from matched-history peaks: a short timeout is not a low-memory
completion. The generated tables are descriptive per-cell measurements, not
powered estimates of an asymptotic memory bound.
"""

import argparse
from itertools import islice
import json
from pathlib import Path


def read_run(directory):
    metadata = json.loads((directory / "world.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    steps = []
    with (directory / "steps.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            steps.append(
                {
                    key: row[key]
                    for key in (
                        "rss_peak_kib",
                        "predicted",
                        "exact",
                        "label",
                        "mining_path_seconds",
                    )
                }
            )
    complete = status["returncode"] == 0
    peak = max((row["rss_peak_kib"] for row in steps), default=0) / 1024
    if complete:
        peak = (
            json.loads((directory / "result.json").read_text())["peak_rss_kib"] / 1024
        )
    return {
        "path": str(directory),
        "runtime": metadata["petta_commit"],
        "config": metadata["config"],
        "returncode": status["returncode"],
        "completed_steps": len(steps),
        "complete": complete,
        "peak_mib": peak,
        "peak_is_lower_bound": not complete,
    }, steps


def verified_input_prefix_length(left, right, count):
    # JSON spelling (for example 1 versus 1.0) is not a changed observation.
    # Killed runs can lose buffered sample lines; report the verified length.
    verified = 0
    with (
        (left / "samples.jsonl").open() as first,
        (right / "samples.jsonl").open() as second,
    ):
        for a, b in zip(islice(first, count), islice(second, count)):
            if json.loads(a) != json.loads(b):
                raise ValueError(f"unequal saved input: {left} vs {right}")
            verified += 1
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("after", type=Path)
    parser.add_argument("--before-coverage", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = []
    for path in args.before_coverage:
        coverage = json.loads(path.read_text())
        selected = {
            (row["level"], row["kind"], row["seed"], row["system"]): row["selected"]
            for row in coverage["conditions"]
            if row["selected"]
        }
        previous.append((coverage["runtime"], selected))
    rows = []
    for status_path in sorted(args.after.rglob("status.json")):
        status = json.loads(status_path.read_text())
        key = (status["level"], status["kinds"][0], status["seed"], status["system"])
        directory = status_path.parent
        after, after_steps = read_run(directory)
        comparisons = []
        for runtime, selected in previous:
            if key not in selected:
                continue
            prior = Path(selected[key]["path"])
            before, before_steps = read_run(prior)
            assert before["runtime"] == runtime, prior
            if before["config"] != after["config"]:
                raise ValueError(
                    f"unequal generator configurations: {prior} vs {directory}"
                )
            count = min(len(before_steps), len(after_steps))
            if not count:
                continue
            verified_inputs = verified_input_prefix_length(prior, directory, count)
            same_truth = all(
                left["exact"] == right["exact"] and left["label"] == right["label"]
                for left, right in zip(before_steps[:count], after_steps[:count])
            )
            if not same_truth:
                raise ValueError(f"unequal generated prefix: {prior} vs {directory}")
            before_peak = before_steps[count - 1]["rss_peak_kib"] / 1024
            after_peak = after_steps[count - 1]["rss_peak_kib"] / 1024
            comparisons.append(
                {
                    "before": before,
                    "prefix_steps": count,
                    "generator_config_identical": True,
                    "input_log_verified_steps": verified_inputs,
                    "full_input_log_verified": verified_inputs == count,
                    "prefix_truth_identical": same_truth,
                    "changed_prediction_steps": sum(
                        left["predicted"] != right["predicted"]
                        for left, right in zip(
                            before_steps[:count], after_steps[:count]
                        )
                    ),
                    "before_prefix_peak_mib": before_peak,
                    "max_abs_prediction_difference": max(
                        abs(a - b)
                        for left, right in zip(
                            before_steps[:count], after_steps[:count]
                        )
                        for target in left["predicted"]
                        for a, b in zip(
                            left["predicted"][target], right["predicted"][target]
                        )
                    ),
                    "after_prefix_peak_mib": after_peak,
                    "prefix_peak_reduction": 1 - after_peak / before_peak,
                    "before_prefix_mining_ms_per_step": 1000
                    * sum(step["mining_path_seconds"] for step in before_steps[:count])
                    / count,
                    "after_prefix_mining_ms_per_step": 1000
                    * sum(step["mining_path_seconds"] for step in after_steps[:count])
                    / count,
                }
            )
        rows.append(
            {
                "level": key[0],
                "kind": key[1],
                "seed": key[2],
                "system": key[3],
                "after": after,
                "comparisons": comparisons,
            }
        )
    with args.out.open("x") as output:
        json.dump(
            {
                "rows": rows,
                "method": "Recorded process high-water RSS at the same completed step. Censored full-run peaks are lower bounds. Revisions are not pooled; per-cell measurements are underpowered.",
            },
            output,
            indent=2,
        )
        output.write("\n")
    lines = [
        "# Targeted memory follow-up",
        "",
        "Per-cell measurements; no population confidence interval is claimed. Failed runs retain only lower bounds on full-run peak RSS.",
        "",
        "| Level / kind | Seed | Arm | Runtime | Completed steps | Complete | Peak MiB |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    for row in rows:
        run = row["after"]
        lines.append(
            f"| {row['level']} / {row['kind']} | {row['seed']} | {row['system']} | {run['runtime'][:8]} | {run['completed_steps']} | {run['complete']} | {run['peak_mib']:.1f}{'+' if run['peak_is_lower_bound'] else ''} |"
        )
    lines += [
        "",
        "Comparisons pair the same generator configuration and seed at equal completed-step counts; exact probabilities and outcomes match. Saved input prefixes are also checked where available. Killed runs can lose buffered sample logs, so the JSON states how many input rows were independently verified. A positive reduction means lower RSS. The 4e6b059b comparison also includes the withdrawal fix; 7925f8d5 is the direct pre-memory-fix comparison.",
        "",
        "| Level / kind | Seed | Arm | Before | Shared steps | Before MiB | After MiB | Reduction | Changed predictions |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for comparison in row["comparisons"]:
            lines.append(
                f"| {row['level']} / {row['kind']} | {row['seed']} | {row['system']} | {comparison['before']['runtime'][:8]} | {comparison['prefix_steps']} | {comparison['before_prefix_peak_mib']:.1f} | {comparison['after_prefix_peak_mib']:.1f} | {100 * comparison['prefix_peak_reduction']:.1f}% | {comparison['changed_prediction_steps']} |"
            )
    with args.out.with_suffix(".md").open("x") as output:
        output.write("\n".join(lines) + "\n")
    print(f"Compared {len(rows)} new conditions without pooling runtime revisions")


if __name__ == "__main__":
    main()
