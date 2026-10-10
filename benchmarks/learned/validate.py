"""ProbLog with learned rules against the exact posterior under counted rates.

Plays each stage and seed once with both reasoners in lockstep (same
observations, same keys) and records the largest absolute difference of any
belief, and the learned reference's rates against the true ones. Run with
ProbLog's interpreter:

    PYTHONPATH=src .venv-problog/bin/python benchmarks/learned/validate.py OUT.json [rounds] [seeds...]
"""

import json
import sys

from supplynet.backends import LearnedReferenceBackend
from supplynet.game import GameConfig, run_game
from supplynet.problog_backend import ProblogBackend

CONFIGS = ((1, "timed"), (2, "timed"), (3, "timed"), (3, "untimed"))


class Lockstep:
    """The learned reference's beliefs, with ProbLog's compared to them."""

    name = "lockstep"

    def __init__(self):
        self.reference, self.problog = LearnedReferenceBackend(), ProblogBackend(learned=True)
        self.largest, self.keys = 0.0, 0

    def begin(self, network, rates, history):
        self.reference.begin(network, rates, history)
        self.problog.begin(network, rates, history)

    def beliefs(self, observation, keys, budget):
        exact = self.reference.beliefs(observation, keys, budget)
        answer = self.problog.beliefs(observation, keys, budget)
        assert answer.keys() == exact.keys(), "every ProbLog round answered"
        self.largest = max([self.largest, *(abs(answer[key] - p) for key, p in exact.items())])
        self.keys += len(keys)
        return exact

    def resolve(self, period, observation):
        self.reference.resolve(period, observation)
        self.problog.resolve(period, observation)


def main(out, rounds=30, seeds=(1, 2, 3, 4)):
    rows = []
    for stage, cycle in CONFIGS:
        for seed in seeds:
            pair = Lockstep()
            run_game(GameConfig(seed=seed, rounds=rounds, stage=stage, cycle=cycle), pair)
            rows.append({"stage": stage, "cycle": cycle, "seed": seed, "keys": pair.keys, "largest_difference": pair.largest})
            print(json.dumps(rows[-1]), flush=True)
    with open(out, "w") as handle:
        json.dump({"rounds": rounds, "rows": rows, "largest_difference": max(r["largest_difference"] for r in rows)}, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30, tuple(map(int, sys.argv[3:])) or (1, 2, 3, 4))
