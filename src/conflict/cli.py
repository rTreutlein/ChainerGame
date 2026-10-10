"""``python -m conflict.cli run|table|summary``: play the conflicting-sources
stage with one reasoner, tabulate a directory of run summaries
(``benchmarks/conflict/run_grid.sh``), and write their machine-readable
summary."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from .backends import REFERENCES, PeTTaChainerBackend, ReferenceBackend
from .game import GameConfig, run_game
from .world import SIZES

PROBLOG = ("problog-naive", "problog-oracle", "problog-learned")
NARS = ("nars-raw", "nars-sources")
BACKENDS = (*REFERENCES, "pln-raw", "pln-sources", "pln-stated", "pln-given", *PROBLOG, *NARS)


def _backend(args):
    if args.backend in REFERENCES:
        return ReferenceBackend(args.backend)
    if args.backend.startswith("pln-"):
        return PeTTaChainerBackend(args.backend.removeprefix("pln-"), args.pettachainer_path, args.evidence_k, args.review_steps)
    if args.backend in PROBLOG:
        from .problog_backend import ProblogBackend

        return ProblogBackend(args.backend.removeprefix("problog-"), args.problog_engine, args.problog_timeout)
    from .nars import NarsBackend

    return NarsBackend(args.backend.removeprefix("nars-"), args.nars_path, args.nars_cycles_per_step, args.evidence_k)


def _run(args) -> None:
    config = GameConfig(args.seed, args.size, args.history, args.rounds, args.steps_per_query)
    on_round = (lambda record: print(json.dumps(record), flush=True)) if args.stream else None
    print(json.dumps(run_game(config, _backend(args), on_round=on_round)), flush=True)


BUDGETED = ("pln-", "nars-")


def _load(directory: str) -> tuple[list[dict], list[dict]]:
    runs, failures = [], []
    for path in sorted(Path(directory).glob("*.json")):
        lines = [line for line in path.read_text().splitlines() if line.startswith("{")]
        if lines:
            runs.append(json.loads(lines[-1]))
    for path in sorted(Path(directory).glob("*.failed")):
        failures.append(json.loads(path.read_text()))
    return runs, failures


def _budgeted(backend: str) -> bool:
    return backend.startswith(BUDGETED)


def _cells(runs: list[dict], failures: list[dict]) -> dict:
    """(backend, size, steps per query) -> runs and failures; budget-free
    backends under steps per query None."""
    cells: dict[tuple, dict] = {}
    for record in runs + failures:
        spq = record["steps_per_query"] if _budgeted(record["backend"]) else None
        cell = cells.setdefault((record["backend"], record["size"], spq), {"runs": [], "failures": []})
        cell["failures" if record["type"] == "run-failed" else "runs"].append(record)
    return cells


def _stats(cell: dict) -> dict:
    runs = cell["runs"]

    def mean(read):
        values = [v for v in map(read, runs) if v is not None]
        return sum(values) / len(values) if values else None

    errors = [r["posterior_error"] for r in runs]
    return {
        "seeds": sorted(r["seed"] for r in runs),
        "failures": len(cell["failures"]),
        "mean_error": mean(lambda r: r["posterior_error"]),
        "std_error": statistics.stdev(errors) if len(errors) > 1 else 0.0 if errors else None,
        "coverage": mean(lambda r: r["coverage"]),
        "brier": mean(lambda r: r["brier"]),
        "exact_brier": mean(lambda r: r["posterior_brier"]),
        "mean_seconds_per_round": mean(lambda r: r["seconds_per_round"]),
        "setup_seconds": mean(lambda r: r["setup_seconds"]),
        "max_rss_mb": mean(lambda r: r["max_rss_mb"]),
        "error_component": mean(lambda r: r["kinds"].get("component", {}).get("posterior_error")),
        "error_system": mean(lambda r: r["kinds"].get("system", {}).get("posterior_error")),
        "error_contested": mean(lambda r: r["groups"].get("contested", {}).get("posterior_error")),
        "error_agreed": mean(lambda r: r["groups"].get("agreed", {}).get("posterior_error")),
    }


def _order(backend: str) -> int:
    return BACKENDS.index(backend) if backend in BACKENDS else len(BACKENDS)


def _table(args) -> None:
    """Markdown: per size, every backend (and budget) with its error to the
    exact posterior (mean ± std over seeds), its parts, and its cost."""
    cells = _cells(*_load(args.results))
    sizes = sorted({size for _, size, _ in cells}, key=list(SIZES).index)

    def number(value, digits=3):
        return "" if value is None else f"{value:.{digits}f}"

    for size in sizes:
        print(f"### Size {size}\n")
        print("| contender | steps/query | seeds | error to exact | ± std | component | system | contested | agreed | coverage | Brier | s/round | failed |")
        print("|---" * 13 + "|")
        rows = sorted(((b, q) for b, s, q in cells if s == size), key=lambda row: (_order(row[0]), row[1] or 0))
        for backend, spq in rows:
            stat = _stats(cells[(backend, size, spq)])
            print(
                f"| {backend} | {'any' if spq is None else f'{spq:g}'} | {len(stat['seeds'])} | {number(stat['mean_error'])} | {number(stat['std_error'])} | "
                f"{number(stat['error_component'])} | {number(stat['error_system'])} | {number(stat['error_contested'])} | {number(stat['error_agreed'])} | "
                f"{number(stat['coverage'], 2)} | {number(stat['brier'])} | {number(stat['mean_seconds_per_round'], 3)} | {stat['failures']} |"
            )
        print()


def _summary(args) -> None:
    """contender -> size -> steps per query -> statistics; a budget-free
    contender's entry is repeated under every budget of the grid."""
    runs, failures = _load(args.results)
    cells = _cells(runs, failures)
    budgets = sorted({q for _, _, q in cells if q is not None}) or [None]
    results: dict = {}
    for (backend, size, spq), cell in sorted(cells.items(), key=lambda item: (_order(item[0][0]), item[0][1], item[0][2] or 0)):
        stat = _stats(cell) | {"budget_independent": spq is None}
        for budget in [spq] if spq is not None else budgets:
            results.setdefault(backend, {}).setdefault(size, {})[f"{budget:g}" if budget is not None else "any"] = stat
    summary = {
        "stage": "conflicting-sources",
        "metric": "mean absolute error to the exact posterior under the true model, per query; mean over rounds, then over seeds",
        "time": "seconds per round: the backend's beliefs and resolve calls",
        "budget": "steps per query (PLN backward steps, NARS inference cycles); exact, Python and ProbLog contenders are budget-free",
        "sizes": {name: vars(SIZES[name]) for name in sorted({s for _, s, _ in cells}, key=list(SIZES).index)},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="conflict")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play the rounds with one reasoner")
    run.add_argument("--backend", choices=BACKENDS, default="exact")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--size", choices=sorted(SIZES), default="s")
    run.add_argument("--history", type=int, default=30, help="labelled rounds before the first")
    run.add_argument("--rounds", type=int, default=10)
    run.add_argument("--steps-per-query", type=float, default=4.0, help="the round's budget per query")
    run.add_argument("--evidence-k", type=float, default=5)
    run.add_argument("--review-steps", type=int, default=20, help="pln-sources: backward steps per trust-rule review")
    run.add_argument("--pettachainer-path", default=os.environ.get("PETTACHAINER_PYTHONPATH"))
    run.add_argument("--problog-engine", default="ddnnf")
    run.add_argument("--problog-timeout", type=float, default=60)
    run.add_argument("--nars-path", help="the ONA NAR executable (default $NARS_PATH or /nexus/Dev/OpenCog/ONA/NAR)")
    run.add_argument("--nars-cycles-per-step", type=float, default=1, help="ONA inference cycles per budget step")
    run.add_argument("--stream", action="store_true", help="print each round as it completes")
    table = sub.add_parser("table", help="markdown tables from a directory of run summaries")
    table.add_argument("results")
    summary = sub.add_parser("summary", help="machine-readable summary of a directory of run summaries")
    summary.add_argument("results")
    summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"run": _run, "table": _table, "summary": _summary}[args.command](args)


if __name__ == "__main__":
    main()
