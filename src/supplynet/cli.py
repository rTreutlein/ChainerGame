import argparse
import json
import os

from . import scale
from .backends import LearnedReferenceBackend, PeTTaChainerBackend, PriorBackend, ReferenceBackend
from .game import GameConfig, run_game
from .nars import NarsBackend
from .problog_backend import ProblogBackend


def main(argv=None):
    parser = argparse.ArgumentParser(prog="supplynet")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play rounds of a stage with one reasoner")
    run.add_argument("--stage", choices=("1", "2", "3", "scale"), default="1")
    run.add_argument("--cycle", choices=("timed", "untimed"), default="timed", help="stage 3: the fuel loop through time or within a period")
    run.add_argument(
        "--backend",
        choices=("reference", "learned-reference", "local", "prior", "pettachainer", "nars", "problog"),
        default="reference",
        help="local: stage scale only; nars, learned-reference: stages 1-3",
    )
    run.add_argument(
        "--learned-rules",
        action="store_true",
        help="stages 1-3: rules without their rates, which the reasoners learn from the labelled periods (docs/supplynet_learned_rules.md)",
    )
    run.add_argument("--review-steps", type=int, default=200, help="pettachainer with --learned-rules: budget of each learned rule's review")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--regions", type=int, help="stages 1-3: regions (default 3); scale: regions per zone")
    run.add_argument("--history", type=int, help="labelled periods before the first round (default 30; scale 20)")
    run.add_argument("--rounds", type=int, help="default 20; scale 10")
    run.add_argument("--inspect-rate", type=float, default=0.25, help="stages 1-3")
    run.add_argument("--budget", type=int, help="one expansion budget per round (default 100; scale: --steps-per-query)")
    run.add_argument("--evidence-k", type=float, default=5)
    run.add_argument("--pettachainer-path", default=os.environ.get("PETTACHAINER_PYTHONPATH"))
    run.add_argument("--nars-path", help="the ONA NAR executable (default $NARS_PATH or /nexus/Dev/OpenCog/ONA/NAR)")
    run.add_argument("--nars-cycles-per-step", type=float, default=10, help="ONA inference cycles per budget step")
    run.add_argument("--nars-reading", choices=("frequency", "expectation"), default="frequency", help="the answer's value read as a probability")
    run.add_argument("--nars-cache", help="a directory storing ONA's output per round input, shared by readings")
    run.add_argument("--problog-engine", default="ddnnf", help="ProbLog's knowledge compiler: ddnnf (dsharp), sdd, sddx or fsdd")
    run.add_argument("--problog-timeout", type=float, default=60, help="seconds per round before ProbLog's inference is killed")
    run.add_argument("--stream", action="store_true", help="print each round as it completes")
    knobs = run.add_argument_group("stage scale")
    knobs.add_argument("--size", choices=sorted(scale.SIZES), default="s", help="a named size; the knobs below override it")
    knobs.add_argument("--zones", type=int)
    knobs.add_argument("--routes", type=int, help="per region")
    knobs.add_argument("--tiers", type=int, help="hops from a route's block to a shipment's lateness")
    knobs.add_argument("--fanout", type=int, help="children of a block or a depot")
    knobs.add_argument("--sensors", type=int, help="weather alarms per region")
    knobs.add_argument("--window", type=int, default=8, help="rounds before a period resolves")
    knobs.add_argument("--steps-per-query", type=float, default=4.0, help="the round's budget per query, when --budget is not given")
    args = parser.parse_args(argv)

    if args.backend == "local" and args.stage != "scale":
        parser.error("--backend local needs --stage scale")
    if args.stage == "scale" and (args.backend in ("nars", "learned-reference") or args.learned_rules):
        parser.error("--backend nars, --backend learned-reference and --learned-rules run stages 1-3")
    on_round = (lambda record: print(json.dumps(record), flush=True)) if args.stream else None
    if args.stage == "scale":
        backend = {
            "reference": lambda: ReferenceBackend(scale.Knowledge),
            "local": lambda: ReferenceBackend(lambda network, rates: scale.Knowledge(network, rates, local=True), "local"),
            "prior": PriorBackend,
            "pettachainer": lambda: PeTTaChainerBackend(args.pettachainer_path, args.evidence_k, scale),
            "problog": lambda: ProblogBackend(scale, args.problog_engine, args.problog_timeout, priors=lambda network, rates: {}),
        }[args.backend]()
        size = scale.sized(args.size, zones=args.zones, regions=args.regions, routes=args.routes, tiers=args.tiers, fanout=args.fanout, sensors=args.sensors)
        config = scale.GameConfig(
            args.seed,
            size,
            args.window,
            20 if args.history is None else args.history,
            10 if args.rounds is None else args.rounds,
            args.budget,
            args.steps_per_query,
        )
        print(json.dumps(scale.run_game(config, backend, on_round=on_round)), flush=True)
        return
    learned = args.learned_rules
    backend = {
        "reference": ReferenceBackend,
        "learned-reference": LearnedReferenceBackend,
        "prior": PriorBackend,
        "pettachainer": lambda: PeTTaChainerBackend(args.pettachainer_path, args.evidence_k, learned=learned, review_steps=args.review_steps),
        "nars": lambda: NarsBackend(args.nars_path, args.nars_cycles_per_step, args.nars_reading, args.nars_cache, learned=learned),
        "problog": lambda: ProblogBackend(engine=args.problog_engine, timeout=args.problog_timeout, learned=learned),
    }[args.backend]()
    config = GameConfig(
        args.seed,
        3 if args.regions is None else args.regions,
        30 if args.history is None else args.history,
        20 if args.rounds is None else args.rounds,
        args.inspect_rate,
        100 if args.budget is None else args.budget,
        int(args.stage),
        args.cycle,
    )
    print(json.dumps({**run_game(config, backend, on_round=on_round), "learned_rules": learned}), flush=True)


if __name__ == "__main__":
    main()
