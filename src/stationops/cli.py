import argparse
import json
import os

from .config import Config
from .episode import run_episode
from .metta import generate_statements
from .simulator import generate_history, generate_incidents
from .v1 import play_episode_v1, prior_shift_fixture, run_episode_v1


def _config(args):
    return Config(
        seed=args.seed,
        history_size=args.history_size,
        incidents=args.incidents,
        repair_slots=args.repair_slots,
        irrelevant_statements=args.irrelevant,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stationops")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "sweep", "play"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--history-size", type=int, default=1000)
        p.add_argument("--incidents", type=int, default=100)
        p.add_argument("--repair-slots", type=int, default=10)
        p.add_argument("--irrelevant", type=int, default=0)
        p.add_argument("--backend", choices=("reference", "mm2", "pettachainer"), default="reference")
        p.add_argument("--mm2-path", default=os.environ.get("MM2_CHAINER_PYTHONPATH"))
        p.add_argument(
            "--pettachainer-path",
            default=os.environ.get("PETTACHAINER_PYTHONPATH"),
        )
        p.add_argument("--budget", type=int, default=100)
        p.add_argument("--budgets", default="1,10,100")
        p.add_argument("--benchmark", choices=("v0", "v1"), default="v0")
    args = parser.parse_args(argv)
    config = _config(args)
    if args.command == "play":
        if args.benchmark != "v1":
            parser.error("play requires --benchmark v1")
        print(json.dumps(play_episode_v1(config), sort_keys=True))
        return
    if args.command == "generate":
        if args.benchmark == "v0":
            print(generate_statements(generate_history(config), generate_incidents(config), config))
        else:
            fixture = prior_shift_fixture(config)
            history = list(fixture.history)
            for round_ in fixture.rounds:
                print(generate_statements(history, list(round_.incidents), config))
                history.extend(round_.resolutions)
        return
    budgets = [args.budget] if args.command == "run" else [int(x) for x in args.budgets.split(",")]
    for budget in budgets:
        print(
            json.dumps(
                (run_episode if args.benchmark == "v0" else run_episode_v1)(
                    config,
                    args.backend,
                    budget,
                    args.mm2_path,
                    args.pettachainer_path,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
