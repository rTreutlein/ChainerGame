"""Tables for the ProbLog comparison: SupplyNet stages 1-3 (one JSON summary
per run, from run_grid.sh), stage scale (ProbLog summaries from run_scale.sh
and a PeTTaChainer sweep JSON from ``supplynet.sweep run``) and StationOps
temporal replay (--stream files, from run_stationops.sh). Means over seeds.

    python benchmarks/problog/report.py RESULTS_DIR

RESULTS_DIR holds stages/, scale/, scale_pettachainer.json and stationops/;
missing parts are skipped."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONFIGS = (("s1", "timed", "stage 1"), ("s2", "timed", "stage 2"), ("s3", "timed", "stage 3 timed"), ("s3", "untimed", "stage 3 untimed"))
ORDER = ("reference", "prior", "pettachainer", "problog", "problog-sdd")
SIZES = ("s", "m", "l", "xl")


def mean(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def stages(directory: Path) -> None:
    runs = defaultdict(list)
    for path in sorted(directory.glob("*-s*-*-seed*.json")):
        backend, stage, cycle, _ = path.stem.rsplit("-", 3)
        runs[(stage, cycle, backend)].append(json.loads(path.read_text()))
    if not runs:
        return
    print("\n### SupplyNet stages 1-3 (seeds 1-4, 30 rounds, budget 100)\n")
    print("Error is the mean absolute difference from the exact posterior; seconds are inference per round (PeTTaChainer: query_many; ProbLog: ground + compile + evaluate in a forked child).\n")
    print("| stage | reasoner | error to exact | Brier | log loss | coverage | s per round | max s per round |")
    print("|---|---|---|---|---|---|---|---|")
    for stage, cycle, label in CONFIGS:
        for backend in ORDER:
            rows = runs.get((stage, cycle, backend))
            if rows:
                rounds = [seconds for row in rows for _, seconds in row.get("problog_rounds", [])]
                print(
                    f"| {label} | {backend} | {mean(rows, 'posterior_error'):.4f} | {mean(rows, 'brier'):.4f} | {mean(rows, 'log_loss'):.4f} | "
                    f"{mean(rows, 'coverage'):.2f} | {mean(rows, 'seconds') / 30:.3f} | {f'{max(rounds):.3f}' if rounds else ''} |"
                )


def scale(directory: Path, sweep: Path) -> None:
    problog = defaultdict(list)
    for path in sorted(directory.glob("problog-*-seed*.json")):
        _, engine, size, _ = path.stem.split("-")
        problog[(engine, size)].append(json.loads(path.read_text()))
    chainer = defaultdict(list)
    if sweep.exists():
        for record in json.loads(sweep.read_text())["records"]:
            if "failed" not in record:
                chainer[(record["backend"], record["size"], record["steps_per_query"])].append(record)
    if not problog and not chainer:
        return
    print("\n### SupplyNet stage scale (10 rounds, window 8, 20 history periods)\n")
    print("ProbLog per round (all seeds' rounds pooled): rounds answered / timed out (120 s) / failed (out of memory under an 8 GB address-space cap), the median and the largest answered round, peak memory of an answered round, and the error to the exact posterior over all queries (an unanswered query counts as 0.5).\n")
    print("| size | engine | hidden / observed per period | queries per round | answered / timeout / failed | median s per round | max s per round | peak MB | error to exact | coverage |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for engine in ("ddnnf", "sdd"):
        for size in SIZES:
            rows = problog.get((engine, size))
            if not rows:
                continue
            rounds = [record for row in rows for record in row["problog_rounds"]]
            answered = [seconds for outcome, seconds in rounds if outcome == "answered"]
            counts = [sum(outcome == kind for outcome, _ in rounds) for kind in ("answered", "timeout", "failed")]
            print(
                f"| {size} | {engine} | {rows[0]['hidden_per_period']} / {rows[0]['observed_per_period']} | {mean(rows, 'queries_per_round'):.0f} | "
                f"{' / '.join(map(str, counts))} | {f'{statistics.median(answered):.2f}' if answered else '–'} | "
                f"{f'{max(answered):.2f}' if answered else '–'} | {max(row.get('problog_peak_mb', 0) for row in rows):.0f} | "
                f"{mean(rows, 'posterior_error'):.3f} | {mean(rows, 'coverage'):.2f} |"
            )
    print("\nProbLog (d-DNNF) seconds per round by round, mean over seeds (t = timeout, f = failed; the window fills over rounds 1-8):\n")
    print("| size | " + " | ".join(f"r{index}" for index in range(1, 11)) + " |")
    print("|---|" + "---|" * 10)
    for size in SIZES:
        rows = problog.get(("ddnnf", size))
        if rows:
            cells = []
            for index in range(10):
                outcomes = [row["problog_rounds"][index] for row in rows if len(row["problog_rounds"]) > index]
                done = [seconds for outcome, seconds in outcomes if outcome == "answered"]
                marks = "".join(outcome[0] for outcome, _ in outcomes if outcome != "answered")
                cells.append((f"{sum(done) / len(done):.1f}" if done else "") + (f" ({marks})" if marks else ""))
            print(f"| {size} | " + " | ".join(cells) + " |")
    if chainer:
        steps = sorted({key[2] for key in chainer if key[0] == "pettachainer"})
        print("\nPeTTaChainer against ProbLog per size: error to exact (coverage) and seconds per round, mean over seeds.\n")
        print("| size | " + " | ".join(f"PeTTaChainer {s:g} steps/query" for s in steps) + " | ProbLog d-DNNF | local | base rates |")
        print("|---|" + "---|" * (len(steps) + 3))
        for size in SIZES:
            cells = []
            for s in steps:
                rows = chainer.get(("pettachainer", size, s))
                cells.append(
                    f"{mean(rows, 'posterior_error'):.3f} ({mean(rows, 'coverage'):.2f}), {mean(rows, 'seconds') / 10:.2f} s, {max(r['peak_rss_mb'] for r in rows) / 1024:.1f} GB"
                    if rows
                    else ""
                )
            rows = problog.get(("ddnnf", size))
            if rows:
                answered = [seconds for row in rows for outcome, seconds in row["problog_rounds"] if outcome == "answered"]
                cells.append(f"{mean(rows, 'posterior_error'):.3f} ({mean(rows, 'coverage'):.2f}), {mean(rows, 'seconds') / 10:.2f} s")
            else:
                cells.append("")
            for baseline in ("local", "prior"):
                rows = chainer.get((baseline, size, None))
                cells.append(f"{mean(rows, 'posterior_error'):.3f}" if rows else "")
            print(f"| {size} | " + " | ".join(cells) + " |")


def stationops(directory: Path) -> None:
    found = {backend: sorted(directory.glob(f"{backend}-seed*.jsonl")) for backend in ("reference", "pettachainer", "problog")}
    if not any(found.values()):
        return
    print("\n### StationOps temporal replay (seeds 1-4, 20 shifts, budget 50)\n")
    print("| reasoner | seeds | error to oracle | Brier | log loss | coverage | reasoner-only regret | s per shift | max s per shift |")
    print("|---|---|---|---|---|---|---|---|---|")
    for backend, paths in found.items():
        rows = []
        for path in paths:
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            shifts = [line["result"] for line in lines if line["type"] == "shift"]
            summary = next(line["result"] for line in lines if line["type"] == "run-summary")
            weight = sum(s["belief_metrics"]["evaluated_count"] for s in shifts)
            weighted = lambda field: sum((s["belief_metrics"][field] or 0) * s["belief_metrics"]["evaluated_count"] for s in shifts) / weight  # noqa: E731,B023
            rows.append(
                {
                    "error": weighted("mean_absolute_error"),
                    "brier": weighted("brier_score"),
                    "log_loss": weighted("log_loss"),
                    "coverage": weight / sum(s["belief_metrics"]["expected_count"] for s in shifts),
                    "regret": summary["aggregate"]["reasoner_only_regret"],
                    "seconds": mean(shifts, "wall_time_seconds"),
                    "max": max(s["wall_time_seconds"] for s in shifts),
                }
            )
        if rows:
            print(
                f"| {backend} | {len(rows)} | {mean(rows, 'error'):.3f} | {mean(rows, 'brier'):.3f} | {mean(rows, 'log_loss'):.3f} | "
                f"{mean(rows, 'coverage'):.2f} | {mean(rows, 'regret'):.1f} | {mean(rows, 'seconds'):.2f} | {max(r['max'] for r in rows):.2f} |"
            )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    stages(root / "stages")
    scale(root / "scale", root / "scale_pettachainer.json")
    stationops(root / "stationops")
