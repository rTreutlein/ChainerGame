"""Tables for the NARS comparison: SupplyNet stages 1-3 (one JSON summary per
run, from run_grid.sh) and StationOps temporal replay (--stream files, from
run_stationops.sh). Means over seeds; StationOps belief metrics are weighted
by evaluated beliefs as in bench/compare.py.

    python benchmarks/nars/report.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a directory of runs; its name labels the NARS variant."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

CONFIGS = (("1", "timed", "stage 1"), ("2", "timed", "stage 2"), ("3", "timed", "stage 3 timed"), ("3", "untimed", "stage 3 untimed"))
ORDER = ("reference", "prior", "pettachainer")


def supplynet(dirs: list[Path]) -> None:
    runs = defaultdict(list)  # (config label, backend label) -> summaries
    kinds = defaultdict(lambda: defaultdict(list))
    for directory in dirs:
        for path in sorted(directory.glob("*-s*-*-seed*.json")):
            backend, stage, cycle, _ = path.stem.rsplit("-", 3)
            label = backend if backend in ORDER else f"{backend} ({directory.name})"
            config = next(name for s, c, name in CONFIGS if f"s{s}" == stage and (c == cycle or s != "3"))
            summary = json.loads(path.read_text())
            runs[(config, label)].append(summary)
            for kind, record in summary["kinds"].items():
                kinds[(config, label)][kind].append(record)
    labels = sorted({label for _, label in runs}, key=lambda l: (ORDER.index(l) if l in ORDER else len(ORDER), l))
    mean = lambda rows, field: sum(r[field] for r in rows) / len(rows)  # noqa: E731
    for _, _, config in CONFIGS:
        print(f"\n### SupplyNet {config} (seeds 1-4, 30 rounds, budget 100)\n")
        print("| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |")
        print("|---|---|---|---|---|---|---|")
        for label in labels:
            rows = runs.get((config, label))
            if rows:
                print(
                    f"| {label} | {len(rows)} | {mean(rows, 'posterior_error'):.3f} | {mean(rows, 'brier'):.3f} | "
                    f"{mean(rows, 'log_loss'):.3f} | {mean(rows, 'coverage'):.2f} | {mean(rows, 'seconds') + mean(rows, 'setup_seconds'):.1f} |"
                )
        exact = runs.get((config, "reference"))
        if exact:
            print(f"\nExact posterior's own Brier {mean(exact, 'posterior_brier'):.3f}, log loss {mean(exact, 'posterior_log_loss'):.3f}.")
        names = sorted({kind for label in labels for kind in kinds[(config, label)]})
        print("\nError to exact by query kind (coverage in brackets):\n")
        print("| reasoner | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for label in labels:
            if (config, label) in runs:
                cells = []
                for kind in names:
                    rows = kinds[(config, label)].get(kind)
                    cells.append(f"{mean(rows, 'posterior_error'):.3f} [{mean(rows, 'coverage'):.2f}]" if rows else "")
                print(f"| {label} | " + " | ".join(cells) + " |")


def stationops(directory: Path) -> None:
    print("\n### StationOps temporal replay (seeds 1-4, 20 shifts, budget 50)\n")
    print("| reasoner | seeds | error to oracle | Brier | log loss | coverage | reasoner-only regret | seconds per run |")
    print("|---|---|---|---|---|---|---|---|")
    for backend in ("reference", "pettachainer", "nars"):
        rows = []
        for path in sorted(directory.glob(f"{backend}-seed*.jsonl")):
            shifts = [json.loads(line)["result"] for line in path.read_text().splitlines() if json.loads(line)["type"] == "shift"]
            summary = next(json.loads(line)["result"] for line in path.read_text().splitlines() if json.loads(line)["type"] == "run-summary")
            weight = sum(s["belief_metrics"]["evaluated_count"] for s in shifts)
            weighted = lambda field: sum((s["belief_metrics"][field] or 0) * s["belief_metrics"]["evaluated_count"] for s in shifts) / weight  # noqa: E731
            rows.append(
                {
                    "error": weighted("mean_absolute_error"),
                    "brier": weighted("brier_score"),
                    "log_loss": weighted("log_loss"),
                    "coverage": sum(s["belief_metrics"]["evaluated_count"] for s in shifts) / sum(s["belief_metrics"]["expected_count"] for s in shifts),
                    "regret": summary["aggregate"]["reasoner_only_regret"],
                    "seconds": sum(s["wall_time_seconds"] for s in shifts),
                }
            )
        if rows:
            mean = lambda field: sum(r[field] for r in rows) / len(rows)  # noqa: E731
            print(
                f"| {backend} | {len(rows)} | {mean('error'):.3f} | {mean('brier'):.3f} | {mean('log_loss'):.3f} | "
                f"{mean('coverage'):.2f} | {mean('regret'):.1f} | {mean('seconds'):.1f} |"
            )


if __name__ == "__main__":
    paths = [Path(arg) for arg in sys.argv[1:]]
    supplynet([path for path in paths if path.name != "stationops"])
    for path in paths:
        if path.name == "stationops":
            stationops(path)
