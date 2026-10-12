"""Gap 3 (docs/supply_network.md, "Stage 4 as built"), run from the
ChainerGame root with PYTHONPATH=src: SupplyNet stage 4, seed 1, with the
production rules invertible (workaround 3 undone). Round 1 asks all 22
decision contexts; round 2's stockpile contexts then miss their production
outcomes at STEPS steps per context. Usage: round2_contexts.py STEPS"""
import re, sys
from supplynet import metta
from supplynet.backends import PeTTaChainerBackend
from supplynet.game import GameConfig, describe, run_game
import supplynet.game as game

steps = int(sys.argv[1])
exact = {}
reference_utilities = game.expected_utilities
game.expected_utilities = lambda knowledge, problem: exact.update(reference_utilities(knowledge, problem)) or dict(exact)
rules = metta.rules
metta.rules = lambda *args, **kwargs: [re.sub(r"^\(: \(no_inverse ((?:fuel|runs)[a-z-]*)\) ", r"(: \1 ", line) for line in rules(*args, **kwargs)]


class Contexts(PeTTaChainerBackend):
    def decide(self, problem, budget):
        if not self.valued:
            self._handler.add_atoms_no_check(metta.decision_rules(self.network, problem.values))
            self.valued = True
        for action in problem.options:
            answers = self._handler.query_many_refs([metta.decision_query(self.network, problem.period, action)], steps=steps)[0]
            value = sum(a.statement[2][3] * a.tv[1] for a in answers) - problem.values.cost_of(action)
            if action.stock:
                print(problem.period, f"{describe(action):26s} chainer {value:7.3f} exact {exact[action]:7.3f}", flush=True)
        return super().decide(problem, budget)


run_game(GameConfig(seed=1, rounds=2, stage=4, budget=steps), Contexts())
