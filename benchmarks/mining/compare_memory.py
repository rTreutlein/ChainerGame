"""Compare measured RSS on identical prefixes of completed or censored runs.

Coverage manifests select immutable prior attempts. Full-run peaks are reported
separately from matched-history peaks: a short timeout is not a low-memory
completion. The generated tables are descriptive per-cell measurements, not
powered estimates of an asymptotic memory bound.
"""

import argparse
import hashlib
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
                    for key in ("rss_peak_kib", "predicted", "exact", "label")
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
        "returncode": status["returncode"],
        "completed_steps": len(steps),
        "complete": complete,
        "peak_mib": peak,
        "peak_is_lower_bound": not complete,
    }, steps


def prefix_digest(directory, count):
    digest = hashlib.sha256()
    with (directory / "samples.jsonl").open("rb") as source:
        for line in islice(source, count):
            digest.update(line)
    return digest.hexdigest()


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
            count = min(len(before_steps), len(after_steps))
            if not count:
                continue
            same_samples = prefix_digest(prior, count) == prefix_digest(
                directory, count
            )
            same_truth = all(
                left["exact"] == right["exact"] and left["label"] == right["label"]
                for left, right in zip(before_steps[:count], after_steps[:count])
            )
            if not same_samples or not same_truth:
                raise ValueError(f"unequal generated prefix: {prior} vs {directory}")
            before_peak = before_steps[count - 1]["rss_peak_kib"] / 1024
            after_peak = after_steps[count - 1]["rss_peak_kib"] / 1024
            comparisons.append(
                {
                    "before": before,
                    "prefix_steps": count,
                    "generated_prefix_identical": same_samples,
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
        "Matched-history comparisons include identical generated inputs and outcomes. A positive reduction means lower RSS in the new runtime. The 4e6b059b comparison also includes the withdrawal fix; 7925f8d5 is the direct pre-memory-fix comparison.",
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
