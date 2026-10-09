"""The stage "scale" grid: error against the exact posterior versus steps per
query, one curve per network size, with the exact, local and base-rate
reasoners as baselines.

    python -m supplynet.sweep run --pettachainer-path BUILD --out grid.json
    python -m supplynet.sweep report grid.json

Each run is a ``supplynet.cli run --stage scale`` subprocess under a 16 GB
address-space limit; the JSON is rewritten as runs finish, so a partial grid
can be reported.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .scale import SIZES

BASELINES = ("reference", "local", "prior")
SRC = str(Path(__file__).resolve().parents[1])


def _limit(kilobytes: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (kilobytes * 1024, kilobytes * 1024))

    return apply


def _run(job: dict, args) -> dict:
    command = [
        sys.executable, "-m", "supplynet.cli", "run", "--stage", "scale", "--backend", job["backend"],
        "--size", job["size"], "--seed", str(job["seed"]), "--window", str(args.window),
        "--history", str(args.history), "--rounds", str(args.rounds),
    ]
    if job["steps_per_query"] is not None:
        command += ["--steps-per-query", str(job["steps_per_query"])]
    if job["backend"] == "pettachainer":
        command += ["--pettachainer-path", args.pettachainer_path, "--evidence-k", str(args.evidence_k)]
    env = {key: value for key, value in os.environ.items() if key != "DISPLAY"} | {"PYTHONPATH": SRC}
    started = time.perf_counter()
    try:
        done = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=args.timeout, preexec_fn=_limit(args.memory_kb)
        )
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1]) if done.returncode == 0 and lines else {"failed": done.returncode, "stderr": done.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        result = {"failed": "timeout"}
    return {**result, **job, "wall_seconds": round(time.perf_counter() - started, 1)}


def run(args) -> None:
    sizes, seeds = args.sizes.split(","), [int(seed) for seed in args.seeds.split(",")]
    steps = [float(value) for value in args.steps_per_query.split(",")]
    jobs = [{"backend": b, "size": s, "seed": n, "steps_per_query": None} for s in sizes for n in seeds for b in BASELINES]
    jobs += [{"backend": "pettachainer", "size": s, "seed": n, "steps_per_query": q} for s in sizes for n in seeds for q in steps]
    # Largest first, so the longest runs do not start last.
    jobs.sort(key=lambda job: (list(SIZES).index(job["size"]), job["steps_per_query"] or 0), reverse=True)
    out = Path(args.out)
    records = []
    started = time.perf_counter()
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(_run, job, args) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "FAILED " + str(record["failed"]) if "failed" in record else f"error {record['posterior_error']:.3f} coverage {record['coverage']:.2f}"
            print(f"[{len(records)}/{len(jobs)} {time.perf_counter() - started:.0f}s] {record['backend']} {record['size']} seed {record['seed']} "
                  f"spq {record['steps_per_query']}: {status} ({record['wall_seconds']}s)", flush=True)
            out.write_text(json.dumps({"pettachainer_path": args.pettachainer_path, "records": records}, indent=1))
    print(report(records))


def _mean(records: list[dict], field: str) -> float:
    return sum(r[field] for r in records) / len(records)


def _number(value: float) -> str:
    return f"{value:g}"


def report(records: list[dict]) -> str:
    """Markdown tables, mean over seeds: error and coverage per size and steps
    per query with the baselines, cost per run, and an ASCII plot of error
    against log2 steps per query."""
    ok = [r for r in records if "failed" not in r]
    sizes = [size for size in SIZES if any(r["size"] == size for r in ok)]
    sizes += sorted({r["size"] for r in ok} - set(sizes))
    steps = sorted({r["steps_per_query"] for r in ok if r["backend"] == "pettachainer"})

    def group(size, backend, spq=None):
        return [r for r in ok if r["size"] == size and r["backend"] == backend and (backend != "pettachainer" or r["steps_per_query"] == spq)]

    lines = ["Error to the exact posterior (coverage in brackets), mean over seeds:", ""]
    lines += ["| steps/query | " + " | ".join(sizes) + " |", "|---" * (len(sizes) + 1) + "|"]
    for spq in steps:
        cells = []
        for size in sizes:
            runs = group(size, "pettachainer", spq)
            cells.append(f"{_mean(runs, 'posterior_error'):.3f} ({_mean(runs, 'coverage'):.2f})" if runs else "")
        lines.append(f"| {_number(spq)} | " + " | ".join(cells) + " |")
    for backend in ("local", "prior"):
        cells = [f"{_mean(runs, 'posterior_error'):.3f}" if (runs := group(size, backend)) else "" for size in sizes]
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    lines += ["", "Per size: every run, mean over seeds; Brier against the exact posterior's.", ""]
    lines += ["| size | steps/query | error | coverage | Brier | exact Brier | ms/query | setup s | peak MB |", "|---" * 9 + "|"]
    for size in sizes:
        for label, runs in [(_number(spq), group(size, "pettachainer", spq)) for spq in steps] + [(b, group(size, b)) for b in ("local", "prior")]:
            if runs:
                lines.append(
                    f"| {size} | {label} | {_mean(runs, 'posterior_error'):.3f} | {_mean(runs, 'coverage'):.2f} | {_mean(runs, 'brier'):.3f} | "
                    f"{_mean(runs, 'posterior_brier'):.3f} | {_mean(runs, 'ms_per_query'):.1f} | {_mean(runs, 'setup_seconds'):.1f} | "
                    f"{_mean(runs, 'peak_rss_mb'):.0f} |"
                )
    lines += ["", "Sizes:", ""]
    for size in sizes:
        sample = next(r for r in ok if r["size"] == size)
        knobs = ", ".join(f"{knob} {sample[knob]}" for knob in ("zones", "regions", "routes", "tiers", "fanout", "sensors"))
        lines.append(
            f"- {size}: {knobs}; {sample['hidden_per_period']} hidden and {sample['observed_per_period']} observed nodes per period, "
            f"{sample['queries_per_round']:.0f} queries per round, {sample['setup_statements']} statements at setup, {sample['statements']} at the end"
        )
    failed = [r for r in records if "failed" in r]
    if failed:
        lines += ["", "Failed: " + ", ".join(f"{r['backend']} {r['size']} seed {r['seed']} spq {r['steps_per_query']} ({r['failed']})" for r in failed)]
    if steps:
        lines += ["", "```", *_plot(sizes, steps, group), "```"]
    return "\n".join(lines)


def _plot(sizes, steps, group, height=12) -> list[str]:
    """Error (rows, 0 at the bottom) against log2 steps per query (columns);
    each size draws with its initial, local as 'l' and prior as 'p' at the
    right edge."""
    top = max(
        [_mean(runs, "posterior_error") for size in sizes for spq in steps if (runs := group(size, "pettachainer", spq))]
        + [_mean(runs, "posterior_error") for size in sizes for b in ("local", "prior") if (runs := group(size, b))]
    ) or 1.0
    width = 6
    columns = len(steps) * width + 2 + 2 * len(sizes)
    grid = [[" "] * columns for _ in range(height + 1)]

    def put(value, column, mark):
        row = height - round(value / top * height)
        grid[row][column] = mark if grid[row][column] == " " else "*"

    for size in sizes:
        mark = size[0] if len(size) == 1 else size[-1].upper()
        for index, spq in enumerate(steps):
            if runs := group(size, "pettachainer", spq):
                put(_mean(runs, "posterior_error"), index * width + width // 2, mark)
    for offset, size in enumerate(sizes):
        for backend in ("local", "prior"):
            if runs := group(size, backend):
                put(_mean(runs, "posterior_error"), len(steps) * width + 1 + 2 * offset, backend[0])
    rows = [f"{top * (height - i) / height:5.3f} |" + "".join(row) for i, row in enumerate(grid)]
    axis = "      +" + "-" * columns
    labels = "       " + "".join(f"{_number(spq):^{width}}" for spq in steps) + " " + " ".join(s[:1] for s in sizes)
    legend = "       steps per query (log scale); marks: " + ", ".join(f"{s}={s[0] if len(s) == 1 else s[-1].upper()}" for s in sizes)
    return rows + [axis, labels, legend, "       right edge: local (l) and prior (p) per size, in size order; * = overlap"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="supplynet.sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("run", help="run the grid in parallel and write JSON")
    grid.add_argument("--pettachainer-path", required=True)
    grid.add_argument("--out", required=True)
    grid.add_argument("--sizes", default="s,m,l,xl")
    grid.add_argument("--seeds", default="1,2")
    grid.add_argument("--steps-per-query", default="0.25,0.5,1,2,4,8,16")
    grid.add_argument("--window", type=int, default=8)
    grid.add_argument("--history", type=int, default=20)
    grid.add_argument("--rounds", type=int, default=10)
    grid.add_argument("--evidence-k", type=float, default=5)
    grid.add_argument("--jobs", type=int, default=16)
    grid.add_argument("--memory-kb", type=int, default=16_000_000, help="address-space limit per run (ulimit -v)")
    grid.add_argument("--timeout", type=float, default=1200, help="seconds per run")
    show = sub.add_parser("report", help="markdown tables and an ASCII plot from a grid's JSON")
    show.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        print(report(json.loads(Path(args.path).read_text())["records"]))


if __name__ == "__main__":
    main()
