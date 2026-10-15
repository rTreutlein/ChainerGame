"""Narrow numeric feasibility and paired numeric timing under full-pool admission.

No backend changes. Every engine lives in a fresh capped child. Timing children
reserve all admission CPU slots so no other pool benchmark overlaps either
revision. An optional ABBA repeat checks order effects. Host load and CPU time are
recorded as checks, not silently assumed equal. Timing reproduces the original
fixed input order to avoid confounding the revision comparison with the shuffle
fix. The narrow feasibility probe uses the corrected shuffle. Run via capped.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAPPED = "/nexus/Dev/OpenCog/bench/capped"


def runtime_identity(wrapper_path, runtime_path):
    """Hash the installed VM; never mistake an enclosing venv repo for its source."""
    root = Path(runtime_path).resolve()
    files = {}
    digest = hashlib.sha256()
    paths = [
        path
        for folder in ("src", "lib", "python")
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    native = root / "mork_ffi/target/release/libmork_ffi.so"
    if native.is_file():
        paths.append(native)
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode() + b"\0" + content)
        files[relative] = hashlib.sha256(content).hexdigest()
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = None
    if top and Path(top).resolve() == root:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    return {
        "path": str(root),
        "commit": commit,
        "sha256": digest.hexdigest(),
        "files": files,
        "wrapper_path": str(Path(wrapper_path).resolve()),
        "wrapper_sha256": hashlib.sha256(Path(wrapper_path).read_bytes()).hexdigest(),
    }


def worker(args):
    from miningbench.runner import run
    import petta
    from miningbench.world import Config

    if args.historical_order:
        from miningbench.petta import Petta
        from repros.store_retention import legacy_inputs

        Petta.add_inputs = legacy_inputs
        Petta.input_order = "historical-fixed-order-repro"
    runtime_before = runtime_identity(petta.__file__, petta._resolve_petta_path())
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
    runtime_after = runtime_identity(petta.__file__, petta._resolve_petta_path())
    assert runtime_before == runtime_after
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
                "runtime_identity": runtime_before,
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
    parser.add_argument("--historical-order", action="store_true")
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
        if label.startswith("paired-"):
            command.append("--historical-order")
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
