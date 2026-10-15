"""Narrow numeric feasibility and paired numeric timing under full-pool admission.

No backend changes. Every engine lives in a fresh capped child. Timing children
reserve all admission CPU slots so no other pool benchmark overlaps either
revision. An optional ABBA repeat checks order effects. Host load and CPU time are
recorded as checks, not silently assumed equal. Run the coordinator via capped.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAPPED = "/nexus/Dev/OpenCog/bench/capped"


def worker(args):
    from miningbench.runner import run, revision
    import petta
    from miningbench.world import Config

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    load_start = os.getloadavg()
    pool_start = subprocess.check_output(
        ["python3", "/nexus/Dev/OpenCog/bench/admission.py", "status"], text=True
    )
    result = run(
        Config(
            seed=1, samples=args.samples, kinds=("numeric",), numeric_width=args.width
        ),
        "petta",
        args.out,
        budget=40,
    )
    result["metadata"]["level"] = f"numeric_probe_width{args.width}"
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "load.json").write_text(
        json.dumps(
            {
                "cpu_seconds": time.process_time() - cpu_start,
                "wall_seconds": time.perf_counter() - wall_start,
                "load_start": load_start,
                "load_end": os.getloadavg(),
                "admission_cpu_slots": args.slots,
                "pool_start": pool_start,
                "pool_end": subprocess.check_output(
                    ["python3", "/nexus/Dev/OpenCog/bench/admission.py", "status"],
                    text=True,
                ),
                "petta_runtime_path": petta.__file__,
                "petta_runtime_commit": revision(Path(petta.__file__).parent),
                "protocol": (
                    "full-pool CPU reservation; fresh process"
                    if args.slots == 24
                    else "narrow feasibility probe; shared pool"
                ),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--skip-narrow", action="store_true")
    parser.add_argument("--repeat-pair", action="store_true")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    # Measured numeric peaks from the immutable 64-step prefixes.
    evidence = Path("/nexus/Dev/OpenCog/bench/results/mining-bench")
    comparisons = json.loads((evidence / "review-v3-171-memory.json").read_text())
    (args.out / "memory_profile_source.json").write_text(
        json.dumps(comparisons, indent=2)
    )
    numeric = next(
        row
        for row in comparisons["rows"]
        if row["kind"] == "numeric" and row["system"] == "petta"
    )
    matched = next(
        row
        for row in numeric["comparisons"]
        if row["before"]["runtime"].startswith("7925f8d5")
    )
    peaks = {
        "7925f8d5": matched["before_prefix_peak_mib"],
        "171b2820": matched["after_prefix_peak_mib"],
    }
    cases = [("narrow", "171b2820", 8, 1)] + [
        (f"paired-{index}", revision, 32, 24)
        for index, revision in enumerate(
            (
                ("7925f8d5", "171b2820", "171b2820", "7925f8d5")
                if args.repeat_pair
                else ("7925f8d5", "171b2820")
            )
        )
    ]
    if args.skip_narrow:
        cases = [case for case in cases if case[0] != "narrow"]
    for label, revision, width, slots in cases:
        out = args.out / f"{label}-{revision}"
        environment = dict(
            os.environ,
            BENCH_PRIORITY="sweep",
            PYTHONPATH=f"{ROOT}/.scratch/petta-{revision}:{ROOT}/src",
        )
        environment.pop("MININGBENCH_STORE_CENSUS", None)
        command = [
            CAPPED,
            "--slots",
            str(slots),
            "--peak-mb",
            str(math.ceil(peaks[revision] * 1.2)),
            "--label",
            f"mining-petta-numeric-width{width}-{revision}",
            "--priority",
            "sweep",
            "timeout",
            "--kill-after=10",
            "3600" if slots == 24 else "1800",
            sys.executable,
            __file__,
            "--worker",
            "--width",
            str(width),
            "--slots",
            str(slots),
            "--samples",
            str(args.samples),
            "--out",
            str(out),
        ]
        with (args.out / f"{label}-{revision}.log").open("w") as log:
            status = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        record = {
            "case": label,
            "revision": revision,
            "width": width,
            "slots": slots,
            "returncode": status,
            "command": command,
        }
        with (args.out / "status.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
