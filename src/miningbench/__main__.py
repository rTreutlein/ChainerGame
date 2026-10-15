from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from .world import Config, KINDS
from .runner import run
from .report import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--samples", type=int, default=256)
        p.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
        p.add_argument("--targets", type=int, default=1)
        p.add_argument("--distractors", type=int, default=0)
        p.add_argument("--strength", type=float, default=1)
        p.add_argument("--observation-noise", type=float, default=0)
        p.add_argument("--deep-lags", type=int, default=3)
        p.add_argument("--entity-holdout", action="store_true")
        p.add_argument("--drift", action="store_true")
        p.add_argument("--budget", type=int, default=80)
        p.add_argument("--max-new", type=int, default=4)
        p.add_argument("--max-live", type=int, default=8)
        if name == "run":
            p.add_argument(
                "--system",
                choices=(
                    "oracle",
                    "marginal",
                    "petta",
                    "petta-link-first",
                    "budgeted",
                    "budgeted-flat",
                ),
                default="marginal",
            )
            p.add_argument("--level", default="run")
        else:
            p.add_argument(
                "--systems",
                nargs="+",
                choices=(
                    "oracle",
                    "marginal",
                    "petta",
                    "petta-link-first",
                    "budgeted",
                    "budgeted-flat",
                ),
                default=["oracle", "marginal"],
            )
            p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
            p.add_argument("--timeout", type=int, default=600)
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--separate-kinds", action="store_true")
            p.add_argument(
                "--levels",
                nargs="+",
                choices=(
                    "base",
                    "distractors",
                    "strength",
                    "observation_noise",
                    "samples",
                    "drift",
                    "targets",
                    "long",
                    "deep_lags",
                    "entity_holdout",
                ),
                default=[
                    "base",
                    "distractors",
                    "strength",
                    "observation_noise",
                    "samples",
                    "drift",
                    "targets",
                ],
            )
    p = commands.add_parser("report")
    p.add_argument("path", type=Path)
    a = parser.parse_args()
    if a.command == "report":
        results, failures = [], []
        for p in sorted(a.path.rglob("result.json")):
            status = p.parent / "status.json"
            if not status.exists() or json.loads(status.read_text())["returncode"] == 0:
                results.append(json.loads(p.read_text()))
        for p in sorted(a.path.rglob("status.json")):
            item = json.loads(p.read_text())
            if item["returncode"]:
                trace = p.parent / "steps.jsonl"
                item["completed_steps"] = (
                    sum(1 for _ in trace.open()) if trace.exists() else 0
                )
                world_path = p.parent / "world.json"
                if world_path.exists():
                    metadata = json.loads(world_path.read_text())
                    item["petta_commit"] = metadata.get("petta_commit")
                    item["input_order"] = metadata.get("input_order")
                failures.append(item)
        text, checks = report(results, failures)
        (a.path / "report.md").write_text(text)
        (a.path / "acceptance.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(text)
        return
    if a.command == "run":
        if a.system.startswith("petta") and not os.environ.get("MININGBENCH_CAPPED"):
            parser.error(
                "PeTTa runs require capped; see docs/mining_bench.md (MININGBENCH_CAPPED=1)"
            )
        c = Config(
            a.seed,
            a.samples,
            tuple(a.kinds),
            a.targets,
            a.distractors,
            a.strength,
            a.observation_noise,
            a.drift,
            a.deep_lags,
            a.entity_holdout,
        )
        result = run(c, a.system, a.out, a.budget, a.max_new, a.max_live)
        result["metadata"]["level"] = a.level
        (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "out": str(a.out),
                    "scores": result["scores"],
                    "recovery": result["recovery"]["all"],
                }
            )
        )
        return
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        parser.error("sweep output directory must be empty; preserve prior evidence")
    tasks = []
    kind_groups = [[k] for k in a.kinds] if a.separate_kinds else [a.kinds]
    for kinds in kind_groups:
        for level in a.levels:
            for seed in a.seeds:
                for system in a.systems:
                    tasks.append((kinds, level, seed, system))

    def execute(task):
        kinds, level, seed, system = task
        suffix = "-" + kinds[0] if a.separate_kinds else ""
        out = a.out / f"{level}-{system}-{seed}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "miningbench",
            "run",
            "--out",
            str(out),
            "--seed",
            str(seed),
            "--samples",
            str(
                64
                if level == "samples"
                else 1024
                if level in ("long", "deep_lags")
                else a.samples
            ),
            "--targets",
            str(3 if level == "targets" else a.targets),
            "--distractors",
            str(100 if level in ("distractors", "deep_lags") else a.distractors),
            "--strength",
            str(0.7 if level == "strength" else a.strength),
            "--observation-noise",
            str(0.2 if level == "observation_noise" else a.observation_noise),
            "--budget",
            str(a.budget),
            "--max-new",
            str(a.max_new),
            "--max-live",
            str(a.max_live),
            "--system",
            system,
            "--level",
            level,
            "--kinds",
            *kinds,
        ]
        cmd += ["--deep-lags", str(12 if level == "deep_lags" else a.deep_lags)]
        if a.entity_holdout or level == "entity_holdout":
            cmd.append("--entity-holdout")
        if a.drift or level == "drift":
            cmd.append("--drift")
        with (out / "process.log").open("w") as log:
            status = subprocess.run(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=10",
                    str(
                        a.timeout
                        * (
                            6
                            if level in ("distractors", "targets", "deep_lags", "long")
                            else 1
                        )
                    ),
                    *cmd,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        item = {
            "level": level,
            "seed": seed,
            "system": system,
            "kinds": kinds,
            "returncode": status,
            "result": str(out / "result.json"),
        }
        (out / "status.json").write_text(json.dumps(item, indent=2) + "\n")
        print(json.dumps(item), flush=True)
        return item

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        statuses = list(pool.map(execute, tasks))
    failures = [s for s in statuses if s["returncode"]]
    (a.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "-m", "miningbench", "report", str(a.out)], check=True
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
