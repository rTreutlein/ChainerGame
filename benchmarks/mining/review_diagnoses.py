"""Extract review/lag diagnoses from immutable verdicts, without rerunning PeTTa."""

import argparse
import hashlib
import json
from pathlib import Path
from miningbench.scoring import CostMeter


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pair_path = args.root / "petta-792/base/base-petta-2-and2/steps.jsonl"
    pair_rows = rows(pair_path)
    verdicts = [
        dict(step=row["step"], **event)
        for row in pair_rows
        for event in row["events"]
        if event["candidate"] in ("hypothesis-3", "hypothesis-4")
    ]
    hashes = {}
    for arm in ("petta", "petta-link-first"):
        path = args.root / f"petta-792/base/base-{arm}-1-lag/steps.jsonl"
        values = rows(path)
        hashes[arm] = hashlib.sha256(
            json.dumps([row["predicted"] for row in values], sort_keys=True).encode()
        ).hexdigest()
    long = []
    for path in args.root.rglob("result.json"):
        result = json.loads(path.read_text())
        meta = result["metadata"]
        if (
            meta.get("level") == "long"
            and meta["system"].startswith("petta")
            and meta.get("petta_commit", "").startswith("7925f8d5")
        ):
            meter = CostMeter(meta["config"]["samples"])
            for row in rows(path.parent / "steps.jsonl"):
                meter.add(row["mining_path_seconds"])
            long.append(
                dict(
                    source=str(path),
                    system=meta["system"],
                    seed=meta["config"]["seed"],
                    **meter.result(),
                )
            )
    args.out.write_text(
        json.dumps(
            dict(
                and2_source=str(pair_path),
                and2_verdicts=verdicts,
                lag_prediction_hashes=hashes,
                plain_long_fits=long,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
