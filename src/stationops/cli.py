import argparse
import json
import os

from .config import Config
from .episode import run_episode
from .metta import generate_statements
from .simulator import generate_history, generate_incidents


def _config(args):
    return Config(
        seed=args.seed,
        incidents=args.incidents,
        repair_slots=args.repair_slots,
        irrelevant_statements=args.irrelevant,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stationops")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=7)
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
    args = parser.parse_args(argv)
    config = _config(args)
    if args.command == "generate":
        print(generate_statements(generate_history(config), generate_incidents(config), config))
        return
    budgets = [args.budget] if args.command == "run" else [int(x) for x in args.budgets.split(",")]
    for budget in budgets:
        print(
            json.dumps(
                run_episode(
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
