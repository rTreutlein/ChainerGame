"""Re-render immutable scored evidence with current gates and selected new cells.

Recovery/scoring definitions are unchanged in this review. Reusing the previous
rescored snapshot avoids rerunning or relabeling any powered base experiment.
Numeric repeated timing probes are deliberately excluded from seed statistics.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from miningbench.report import report
from miningbench.runner import source_hash, revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(line) for line in (args.snapshot / "rescored_results.jsonl").open()
    ]
    original_base = [
        row for row in rows if row["result"]["metadata"].get("level") == "base"
    ]
    original_base_hash = hashlib.sha256(
        json.dumps(original_base, sort_keys=True).encode()
    ).hexdigest()
    provenance = json.loads((args.snapshot / "provenance.json").read_text())
    failures = provenance["failures"]
    for path in sorted(args.extra.rglob("result.json")):
        status = path.parent / "status.json"
        if status.exists() and json.loads(status.read_text())["returncode"]:
            continue
        rows.append({"source": str(path), "result": json.loads(path.read_text())})
    for path in sorted(args.extra.rglob("status.json")):
        status = json.loads(path.read_text())
        if not status["returncode"]:
            continue
        world = path.parent / "world.json"
        if world.exists():
            metadata = json.loads(world.read_text())
            status["petta_commit"] = metadata["petta_commit"]
            status["input_order"] = metadata.get("input_order")
        steps = path.parent / "steps.jsonl"
        status["completed_steps"] = (
            len(steps.read_text().splitlines()) if steps.exists() else 0
        )
        failures.append(status)
    text, checks = report([row["result"] for row in rows], failures)
    (args.out / "report.md").write_text(text)
    (args.out / "acceptance.json").write_text(json.dumps(checks, indent=2) + "\n")
    (args.out / "rescored_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    base = [row for row in rows if row["result"]["metadata"].get("level") == "base"]
    base_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    assert base_hash == original_base_hash, (
        "this refresh must not change powered base cells"
    )
    sources = [row["source"] for row in rows]
    assert len(sources) == len(set(sources)), "duplicate experiment source"
    provenance.update(
        snapshot_analysis_commit=provenance.get("analysis_commit"),
        analysis_commit=revision(Path(__file__).resolve().parents[2]),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=sources,
        run_systems={
            row["source"]: row["result"]["metadata"]["system"] for row in rows
        },
        source_hashes=sorted(
            {row["result"]["metadata"]["source_sha256"] for row in rows}
        ),
        snapshot=str(args.snapshot),
        source_sha256=source_hash(),
        added_root=str(args.extra),
        failures=failures,
        completed=len(rows),
        frozen_base_rows=len(base),
        frozen_base_sha256=base_hash,
        controls={
            "families": sorted({c["control_family"] for c in checks}),
            "negative_rejections": sum(c["discriminates"] for c in checks),
            "positive_passes": sum(c["positive_passes"] for c in checks),
            "screens": len(checks),
        },
        acceptance_counts=dict(Counter(c["status"] for c in checks)),
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in (
                    "completed",
                    "frozen_base_rows",
                    "controls",
                    "acceptance_counts",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
