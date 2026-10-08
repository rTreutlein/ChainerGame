import argparse
import json
import os

from .backends import PeTTaChainerBackend, PriorBackend, ReferenceBackend
from .game import GameConfig, run_game


def main(argv=None):
    parser = argparse.ArgumentParser(prog="supplynet")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play rounds of a stage with one reasoner")
    run.add_argument("--stage", type=int, choices=(1, 2), default=1)
    run.add_argument("--backend", choices=("reference", "prior", "pettachainer"), default="reference")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--regions", type=int, default=3)
    run.add_argument("--history", type=int, default=30, help="labelled periods before the first round")
    run.add_argument("--rounds", type=int, default=20)
    run.add_argument("--inspect-rate", type=float, default=0.25)
    run.add_argument("--budget", type=int, default=100)
    run.add_argument("--evidence-k", type=float, default=5)
    run.add_argument("--pettachainer-path", default=os.environ.get("PETTACHAINER_PYTHONPATH"))
    run.add_argument("--stream", action="store_true", help="print each round as it completes")
    args = parser.parse_args(argv)

    if args.backend == "reference":
        backend = ReferenceBackend()
    elif args.backend == "prior":
        backend = PriorBackend()
    else:
        backend = PeTTaChainerBackend(args.pettachainer_path, args.evidence_k)
    config = GameConfig(args.seed, args.regions, args.history, args.rounds, args.inspect_rate, args.budget, args.stage)
    on_round = (lambda record: print(json.dumps(record), flush=True)) if args.stream else None
    print(json.dumps(run_game(config, backend, on_round=on_round)), flush=True)


if __name__ == "__main__":
    main()
