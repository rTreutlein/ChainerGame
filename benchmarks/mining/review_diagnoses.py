"""Extract review/lag diagnoses from immutable verdicts, without rerunning PeTTa."""

import argparse
import hashlib
import json
from pathlib import Path
from miningbench.scoring import CostMeter
from miningbench.petta import parse, decode


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
    and2_lifecycle = []
    for arm in ("petta", "petta-link-first"):
        path = args.root / f"petta-792/base/base-{arm}-1-lag/steps.jsonl"
        values = rows(path)
        hashes[arm] = hashlib.sha256(
            json.dumps([row["predicted"] for row in values], sort_keys=True).encode()
        ).hexdigest()
        for seed in range(1, 11):
            path = args.root / f"petta-792/base/base-{arm}-{seed}-and2/steps.jsonl"
            observations = []
            for row in rows(path):
                for event in row["events"]:
                    term = parse(event["raw"])
                    try:
                        rule = decode(term[2], {"T0_and2": ("0", "1")})
                    except ValueError:
                        continue  # Other outcomes or unsupported rule shapes.
                    if (
                        rule
                        and rule.value == "1"
                        and len(rule.body) == 2
                        and {
                            (literal.head, literal.lag, literal.negative)
                            for literal in rule.body
                        }
                        == {("T0_and2_F0", 1, False), ("T0_and2_F1", 1, False)}
                    ):
                        observations.append(
                            dict(
                                step=row["step"],
                                candidate=event["candidate"],
                                status=term[-1],
                            )
                        )
            and2_lifecycle.append(
                dict(
                    system=arm,
                    seed=seed,
                    source=str(path),
                    first_seen=observations[0] if observations else None,
                    last_seen=observations[-1] if observations else None,
                    retirements=[v for v in observations if v["status"] == "dropped"],
                )
            )
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
                and2_lifecycle=and2_lifecycle,
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
