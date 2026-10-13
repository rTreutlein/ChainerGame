"""Resume powered live sweeps with per-run capped admission and immutable attempts.

The coordinator itself must be launched through capped --slots 1 --peak-mb 128.
Child waiting for admission does not consume its execution timeout.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from miningbench.world import KINDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument(
        "--levels",
        nargs="+",
        default=[
            "samples",
            "strength",
            "observation_noise",
            "drift",
            "targets",
            "distractors",
        ],
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--stress-seconds", type=int, default=1800)
    parser.add_argument("--required-petta-commit")
    parser.add_argument("--keep-timeouts", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("MININGBENCH_CAPPED"):
        parser.error("launch the coordinator through capped with MININGBENCH_CAPPED=1")
    args.out.mkdir(parents=True, exist_ok=False)
    done = set()
    for path in args.evidence.rglob("result.json"):
        row = json.loads(path.read_text())
        meta = row["metadata"]
        status = path.parent / "status.json"
        if status.exists() and json.loads(status.read_text())["returncode"]:
            continue
        if (
            args.required_petta_commit
            and meta.get("petta_commit") != args.required_petta_commit
        ):
            continue
        config = meta["config"]
        if len(config["kinds"]) == 1:
            done.add(
                (meta.get("level"), config["kinds"][0], config["seed"], meta["system"])
            )
    if args.keep_timeouts:
        for path in args.evidence.rglob("status.json"):
            status = json.loads(path.read_text())
            world_path = path.parent / "world.json"
            if status["returncode"] != 124 or not world_path.exists():
                continue
            metadata = json.loads(world_path.read_text())
            if (
                args.required_petta_commit
                and metadata.get("petta_commit") != args.required_petta_commit
            ):
                continue
            if len(status["kinds"]) == 1:
                done.add(
                    (
                        status["level"],
                        status["kinds"][0],
                        status["seed"],
                        status["system"],
                    )
                )
    tasks = [
        (level, kind, seed, system)
        for level in args.levels
        for kind in args.kinds
        for seed in range(1, 11)
        for system in ("petta", "petta-link-first")
        if (level, kind, seed, system) not in done
    ]
    (args.out / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")

    def execute(task):
        level, kind, seed, system = task
        out = args.out / f"{level}-{system}-{seed}-{kind}"
        out.mkdir()
        stress = (
            level in ("targets", "distractors", "long", "deep_lags")
            or kind == "numeric"
        )
        command = [
            sys.executable,
            "-m",
            "miningbench",
            "run",
            "--out",
            str(out),
            "--system",
            system,
            "--seed",
            str(seed),
            "--kinds",
            kind,
            "--level",
            level,
            "--samples",
            str(
                64
                if level == "samples"
                else 1024
                if level in ("long", "deep_lags")
                else 256
            ),
            "--targets",
            str(3 if level == "targets" else 1),
            "--distractors",
            str(100 if level in ("distractors", "deep_lags") else 0),
            "--deep-lags",
            str(12 if level == "deep_lags" else 3),
            "--strength",
            str(0.7 if level == "strength" else 1),
            "--observation-noise",
            str(0.2 if level == "observation_noise" else 0),
            "--budget",
            "40",
        ]
        if level == "drift":
            command.append("--drift")
        if level == "entity_holdout":
            command.append("--entity-holdout")
        command = [
            "/nexus/Dev/OpenCog/bench/capped",
            "--slots",
            "1",
            "--peak-mb",
            "15000" if kind == "numeric" else "6000" if stress else "1500",
            "timeout",
            "--kill-after=10",
            str(args.stress_seconds if stress else args.seconds),
            *command,
        ]
        with (out / "process.log").open("w") as log:
            status = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        record = {
            "level": level,
            "system": system,
            "seed": seed,
            "kinds": [kind],
            "returncode": status,
            "result": str(out / "result.json"),
        }
        (out / "status.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        statuses = list(pool.map(execute, tasks))
    (args.out / "failures.json").write_text(
        json.dumps([s for s in statuses if s["returncode"]], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
