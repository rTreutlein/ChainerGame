import argparse
import json
import os

from .config import Config
from .episode import run_episode
from .game import GameConfig, run_game_episode
from .metta import generate_statements
from .simulator import generate_history, generate_incidents
from .stress import run_stress_sweep
from .v1 import play_episode_v1, prior_shift_fixture, run_episode_v1
from .web import serve_game


def _config(args):
    return Config(
        seed=args.seed,
        history_size=args.history_size,
        incidents=args.incidents,
        repair_slots=10 if args.repair_slots is None else args.repair_slots,
        irrelevant_statements=args.irrelevant,
    )


def _game_config(args):
    return GameConfig(
        seed=args.seed,
        shifts=args.shifts,
        modules=args.modules,
        initial_history_per_cohort=args.initial_history_per_cohort,
        diagnostic_slots=args.diagnostic_slots,
        repair_slots=2 if args.repair_slots is None else args.repair_slots,
        initial_credits=args.credits,
        sensor_knowledge=args.sensor_knowledge,
        learning_window=args.learning_window,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stationops")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "sweep", "stress", "play"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--history-size", type=int, default=1000)
        p.add_argument("--incidents", type=int, default=100)
        p.add_argument("--repair-slots", type=int)
        p.add_argument("--irrelevant", type=int, default=0)
        p.add_argument("--backend", choices=("reference", "mm2", "pettachainer"), default="reference")
        p.add_argument("--mm2-path", default=os.environ.get("MM2_CHAINER_PYTHONPATH"))
        p.add_argument(
            "--pettachainer-path",
            default=os.environ.get("PETTACHAINER_PYTHONPATH"),
        )
        p.add_argument("--budget", type=int, default=100)
        p.add_argument(
            "--shortfall-budget",
            type=int,
            help="v2 aggregate-loss query budget; defaults to --budget",
        )
        p.add_argument("--budgets", default="1,10,100")
        if name == "stress":
            p.add_argument("--shortfall-budgets", default="10,20,50,100")
            p.add_argument("--seeds", default="7")
            p.add_argument(
                "--stream",
                action="store_true",
                help="emit each completed stress point as one JSON line",
            )
        p.add_argument("--benchmark", choices=("v0", "v1", "v2"), default="v0")
        p.add_argument("--shifts", type=int, default=5)
        p.add_argument("--modules", type=int, default=10)
        p.add_argument("--initial-history-per-cohort", type=int, default=40)
        p.add_argument("--diagnostic-slots", type=int, default=2)
        p.add_argument("--credits", type=int, default=30)
        p.add_argument(
            "--sensor-knowledge",
            choices=("full", "positive", "induced", "mixed"),
            default="mixed",
        )
        p.add_argument("--learning-window", type=int, default=10)
    game = sub.add_parser("game")
    game.add_argument("--seed", type=int, default=7)
    game.add_argument("--shifts", type=int, default=5)
    game.add_argument("--modules", type=int, default=10)
    game.add_argument("--initial-history-per-cohort", type=int, default=40)
    game.add_argument("--diagnostic-slots", type=int, default=2)
    game.add_argument("--repair-slots", type=int, default=2)
    game.add_argument("--credits", type=int, default=30)
    game.add_argument(
        "--sensor-knowledge",
        choices=("full", "positive", "induced", "mixed"),
        default="mixed",
    )
    game.add_argument("--learning-window", type=int, default=10)
    game.add_argument("--host", default="127.0.0.1")
    game.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "game":
        serve_game(_game_config(args), args.host, args.port)
        return
    config = _config(args)
    if args.command == "play":
        if args.benchmark != "v1":
            parser.error("play requires --benchmark v1; use `stationops game` for StationOps-v2")
        print(json.dumps(play_episode_v1(config), sort_keys=True))
        return
    if args.command == "generate":
        if args.benchmark == "v0":
            print(generate_statements(generate_history(config), generate_incidents(config), config))
        elif args.benchmark == "v1":
            fixture = prior_shift_fixture(config)
            history = list(fixture.history)
            for round_ in fixture.rounds:
                print(generate_statements(history, list(round_.incidents), config))
                history.extend(round_.resolutions)
        else:
            parser.error("v2 generation is action-dependent; use `run --benchmark v2`")
        return
    if args.command == "stress":
        emit = (
            lambda row: print(
                json.dumps({"type": "stress-run", "result": row}, sort_keys=True),
                flush=True,
            )
        ) if args.stream else None
        result = run_stress_sweep(
            _game_config(args),
            args.backend,
            [int(value) for value in args.budgets.split(",")],
            [int(value) for value in args.shortfall_budgets.split(",")],
            [int(value) for value in args.seeds.split(",")],
            args.mm2_path,
            args.pettachainer_path,
            on_run=emit,
        )
        print(json.dumps(
            {"type": "stress-summary", "result": result}
            if args.stream else result,
            sort_keys=True,
        ))
        return
    budgets = [args.budget] if args.command == "run" else [int(x) for x in args.budgets.split(",")]
    for budget in budgets:
        if args.benchmark == "v0":
            result = run_episode(
                config, args.backend, budget, args.mm2_path, args.pettachainer_path
            )
        elif args.benchmark == "v1":
            result = run_episode_v1(
                config, args.backend, budget, args.mm2_path, args.pettachainer_path
            )
        else:
            result = run_game_episode(
                _game_config(args),
                args.backend,
                budget,
                args.mm2_path,
                args.pettachainer_path,
                shortfall_budget=args.shortfall_budget,
            )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
