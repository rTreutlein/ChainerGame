import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from .backends import REFERENCES, CombinationBackend, ExactBackend, PeTTaChainerBackend
from .game import GameConfig, run_game
from .world import RELATIONS

BACKENDS = ("exact", *REFERENCES, "pettachainer", "pettachainer-pairs", "pettachainer-cells", "pettachainer-cells-read")


def _game_options(parser):
    parser.add_argument("--relations", default=",".join(RELATIONS), help="comma-separated relations, one consequent per relation each")
    parser.add_argument("--per-relation", type=int, default=1)
    parser.add_argument("--history", type=int, default=50, help="labelled cases before the first round")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--cases", type=int, default=10, help="new cases per round")
    parser.add_argument("--observe-rate", type=float, default=0.8)
    parser.add_argument("--budget", type=int, default=20, help="backward steps per query")
    parser.add_argument("--rates", choices=("true", "learned"), default="true", help="the marginal rules' rates")
    parser.add_argument("--evidence-k", type=float, default=5)
    parser.add_argument("--rule-confidence", type=float, default=1.0)
    parser.add_argument("--form", choices=("complement", "not"), default="complement", help="negative literals of hypotheses")
    parser.add_argument("--min-support", type=int, default=5, help="instances before a hypothesis is written")
    parser.add_argument("--max-literals", type=int, help="largest cell hypothesis")
    parser.add_argument("--review-steps", type=int, default=20, help="backward steps per hypothesis review")
    parser.add_argument("--pettachainer-path", default=os.environ.get("PETTACHAINER_PYTHONPATH"))


def _forwarded(args) -> list[str]:
    """The game options of ``args`` as command-line flags for ``run``."""
    flags = []
    for name in ("relations", "per_relation", "history", "rounds", "cases", "observe_rate", "budget", "rates", "evidence_k",
                 "rule_confidence", "form", "min_support", "max_literals", "review_steps", "pettachainer_path"):
        value = getattr(args, name)
        if value is not None:
            flags += ["--" + name.replace("_", "-"), str(value)]
    return flags


def _backend(args):
    if args.backend == "exact":
        return ExactBackend()
    if args.backend in REFERENCES:
        return CombinationBackend(args.backend)
    hypotheses = args.backend.removeprefix("pettachainer").removeprefix("-") or None
    return PeTTaChainerBackend(
        args.pettachainer_path, args.evidence_k, hypotheses, args.form, args.min_support, args.rule_confidence, args.review_steps, args.max_literals
    )


def _run(args):
    config = GameConfig(
        args.seed, args.parents, tuple(args.relations.split(",")), args.per_relation, args.history, args.rounds, args.cases,
        args.observe_rate, args.budget, args.rates,
    )
    on_round = (lambda record: print(json.dumps(record), flush=True)) if args.stream else None
    print(json.dumps(run_game(config, _backend(args), on_round=on_round)), flush=True)


def _sweep(args):
    """Each (backend, parents, seed) in its own process, so a chainer run's
    time and memory are its own."""
    runs = list(itertools.product(args.backends.split(","), [int(n) for n in args.parents.split(",")], [int(s) for s in args.seeds.split(",")]))

    def play(run):
        backend, parents, seed = run
        command = [sys.executable, "-m", "combinet.cli", "run", "--backend", backend, "--parents", str(parents), "--seed", str(seed)]
        failed = {"type": "run-failed", "backend": backend, "parents": parents, "seed": seed}
        try:
            done = subprocess.run(command + _forwarded(args), capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return failed | {"error": f"timeout after {args.timeout} s"}
        if done.returncode:
            lines = [line for line in done.stderr.splitlines() if "Error" in line] or done.stderr.splitlines()[-1:]
            return failed | {"error": lines[-1] if lines else f"exit {done.returncode}"}
        return json.loads(done.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(args.jobs) as pool, open(args.out, "a") as out:
        for summary in pool.map(play, runs):
            out.write(json.dumps(summary) + "\n")
            out.flush()
    _table(argparse.Namespace(results=args.out))


def _table(args):
    """Means over seeds per (backend, parents): accuracy overall and per
    relation (error against the exact posterior), then cost."""
    groups: dict[tuple, list[dict]] = {}
    failures = []
    with open(args.results) as results:
        for line in results:
            summary = json.loads(line)
            if summary["type"] == "run-failed":
                failures.append(summary)
            else:
                groups.setdefault((summary["parents"], summary["backend"]), []).append(summary)
    order = {name: index for index, name in enumerate(BACKENDS)}
    rows = sorted(groups.items(), key=lambda item: (item[0][0], order.get(item[0][1], len(order))))
    relations = sorted({kind for runs in groups.values() for run in runs for kind in run["kinds"]})

    def mean(runs, read):
        values = [read(run) for run in runs]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def cell(value, digits=3):
        return "" if value is None else f"{value:.{digits}f}"

    print("| n | backend | seeds | coverage | Brier | log loss | error | " + " | ".join(f"error {r}" for r in relations) + " |")
    print("|---" * (7 + len(relations)) + "|")
    for (parents, backend), runs in rows:
        values = [mean(runs, lambda r, f=f: r[f]) for f in ("coverage", "brier", "log_loss", "posterior_error")]
        values += [mean(runs, lambda r, k=k: r["kinds"].get(k, {}).get("posterior_error")) for k in relations]
        print(f"| {parents} | {backend} | {len(runs)} | " + " | ".join(cell(v) for v in values) + " |")
    print()
    print("Chainer cost:\n")
    print("| n | backend | setup s | s/round | review s/round | rules | hypotheses | folds/round | expansions/round | RSS MB |")
    print("|---" * 10 + "|")
    for (parents, backend), runs in rows:
        if "rules" not in runs[0]:
            continue
        values = [
            cell(mean(runs, lambda r: r["setup_seconds"]), 2),
            *(cell(mean(runs, lambda r, f=f: r.get(f) and r[f] / r["rounds"]), 2) for f in ("seconds", "review_seconds")),
            *(cell(mean(runs, lambda r, f=f: r.get(f)), 0) for f in ("rules", "hypotheses")),
            *(cell(mean(runs, lambda r, f=f: r.get(f) and r[f] / r["rounds"]), 0) for f in ("folds", "expansions")),
            cell(mean(runs, lambda r: r["max_rss_mb"]), 0),
        ]
        print(f"| {parents} | {backend} | " + " | ".join(values) + " |")
    print()
    for failure in failures:
        print(f"Failed: n={failure['parents']} {failure['backend']} seed {failure['seed']}: {failure['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="combinet")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play the rounds with one reasoner")
    run.add_argument("--backend", choices=BACKENDS, default="exact")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--parents", type=int, default=3, help="parent rules per consequent")
    run.add_argument("--stream", action="store_true", help="print each round as it completes")
    _game_options(run)
    sweep = sub.add_parser("sweep", help="run backends x parents x seeds, one process each, and tabulate")
    sweep.add_argument("--backends", default=",".join(BACKENDS[:-4]))
    sweep.add_argument("--parents", default="1,2,3,5,10")
    sweep.add_argument("--seeds", default="1,2,3")
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument("--timeout", type=float, default=1800, help="seconds per run")
    sweep.add_argument("--out", required=True, help="JSONL file the summaries are appended to")
    _game_options(sweep)
    table = sub.add_parser("table", help="tabulate a sweep's JSONL results")
    table.add_argument("results")
    args = parser.parse_args(argv)
    {"run": _run, "sweep": _sweep, "table": _table}[args.command](args)


if __name__ == "__main__":
    main()
