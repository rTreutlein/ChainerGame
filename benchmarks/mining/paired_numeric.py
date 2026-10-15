"""Step-synchronized numeric timing on two revisions under shared host load.

Run the coordinator under capped --slots 2 with a reservation for BOTH measured
peaks. Both fresh children inherit that scope. A barrier before each generated
sample aligns history lengths and excludes waiting from runner step timers.
Active process CPU time is recorded independently of scheduler wall time. This
is a simultaneous shared-load comparison, not an idle-machine measurement.
"""

import argparse
import json
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

from numeric_review import worker as numeric_worker

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "step-synchronized two-worker shared-load pair"


class BarrierManager(BaseManager):
    pass


def worker(args):
    from miningbench.world import World

    BarrierManager.register("step_barrier")
    manager = BarrierManager(
        address=("127.0.0.1", args.port), authkey=bytes.fromhex(args.auth)
    )
    manager.connect()
    barrier = manager.step_barrier()
    original_stream = World.stream
    timings = []

    def synchronized_stream(world):
        for sample in original_stream(world):
            barrier.wait(600)
            started = time.perf_counter()
            cpu_started = time.process_time()
            load = os.getloadavg()
            yield sample
            timings.append(
                {
                    "step": sample["step"],
                    "started_monotonic": started,
                    "active_cpu_seconds": time.process_time() - cpu_started,
                    "active_wall_seconds": time.perf_counter() - started,
                    "host_load_start": load,
                }
            )

    World.stream = synchronized_stream
    args.historical_order = True
    args.width = 32
    args.slots = 2
    numeric_worker(args)
    assert len(timings) == args.samples
    (args.out / "active_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    load_path = args.out / "load.json"
    load = json.loads(load_path.read_text())
    load.update(
        protocol=PROTOCOL,
        active_cpu_seconds=sum(row["active_cpu_seconds"] for row in timings),
        active_wall_seconds=sum(row["active_wall_seconds"] for row in timings),
    )
    load_path.write_text(json.dumps(load, indent=2) + "\n")


def coordinator(args):
    args.out.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(2)
    BarrierManager.register("step_barrier", callable=lambda: barrier)
    auth = secrets.token_bytes(24)
    manager = BarrierManager(address=("127.0.0.1", 0), authkey=auth)
    manager.start()
    children = []
    logs = []
    records = []
    try:
        for index, revision in enumerate(("7925f8d5", "171b2820")):
            name = f"paired-{index}-{revision}"
            command = [
                "timeout",
                "--kill-after=10",
                "3600",
                sys.executable,
                __file__,
                "--worker",
                "--samples",
                str(args.samples),
                "--port",
                str(manager.address[1]),
                "--auth",
                auth.hex(),
                "--out",
                str(args.out / name),
            ]
            environment = dict(os.environ, BENCH_PRIORITY="sweep")
            environment["PYTHONPATH"] = f"{ROOT}/.scratch/petta-{revision}:{ROOT}/src"
            environment.pop("MININGBENCH_STORE_CENSUS", None)
            log = (args.out / f"{name}.log").open("w")
            logs.append(log)
            child = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            children.append(child)
            record = {
                "case": name,
                "revision": revision,
                "pid": child.pid,
                "samples": args.samples,
                "protocol": PROTOCOL,
                "shared_admission_slots": 2,
            }
            records.append(record)
        (args.out / "started.json").write_text(json.dumps(records, indent=2) + "\n")
        for child, record in zip(children, records):
            record["returncode"] = child.wait(timeout=3700)
        (args.out / "status.json").write_text(json.dumps(records, indent=2) + "\n")
        if any(record["returncode"] for record in records):
            raise RuntimeError("paired worker failed; retain logs and partial evidence")
        timings = [
            json.loads((args.out / r["case"] / "active_timing.json").read_text())
            for r in records
        ]
        skews = [
            abs(a["started_monotonic"] - b["started_monotonic"])
            for a, b in zip(*timings)
        ]
        (args.out / "synchronization.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "steps": len(skews),
                    "max_start_skew_seconds": max(skews),
                    "mean_start_skew_seconds": sum(skews) / len(skews),
                    "interpretation": "Both revisions start each decision together; faster worker waits outside measured work. External load is shared, not excluded.",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        manager.shutdown()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--auth")
    args = parser.parse_args()
    (worker if args.worker else coordinator)(args)


if __name__ == "__main__":
    main()
