"""PLN's learned rule truths against the counted rates.

For each PLN learned-rules run, replays the same game with the learned
reference to the last round and compares each rule's last reviewed CTV with
the Laplace counts of the same labelled periods, and with the true rates:

    PYTHONPATH=src python benchmarks/learned/rule_rates.py PLN_DIR OUT.json
"""

import json
import re
import sys
from pathlib import Path

from supplynet.backends import LearnedReferenceBackend
from supplynet.game import GameConfig, run_game, stage_rates
from supplynet.world import counts, laplace

_node = re.compile(r"\((\w+) ([\w-]+) \$\w+\)\)\) \$tv\)$")


class Final(LearnedReferenceBackend):
    """The learned reference, keeping the counts its last round reads."""

    def beliefs(self, observation, keys, budget):
        self.tally = counts(self.knowledge.network, self.periods, self.linked)
        self.true = self.network_rates.nodes(self.knowledge.network).table
        return super().beliefs(observation, keys, budget)

    def begin(self, network, rates, history):
        self.network_rates = rates
        super().begin(network, rates, history)


def main(directory: Path, out: Path) -> None:
    rows = []
    for path in sorted(directory.glob("pettachainer-s*-seed*.json")):
        run = json.loads(path.read_text())
        final = Final()
        run_game(GameConfig(seed=run["seed"], rounds=30, stage=run["stage"], cycle=run.get("cycle", "timed")), final)
        for goal, (given, given_confidence, without, without_confidence) in run["rule_truths"].items():
            node = tuple(_node.search(goal).groups())
            (k1, n1), (k0, n0) = final.tally.get(node, ((0, 0), (0, 0)))
            rows.append({
                "run": path.stem, "node": node, "pln": [given, given_confidence, without, without_confidence],
                "samples": [[k1, n1], [k0, n0]], "laplace": [laplace((k1, n1)), laplace((k0, n0))], "true": list(final.true[node]),
            })
    summary = {}
    for branch, index in (("given", 0), ("without", 1)):
        learned = [r for r in rows if r["pln"][2 * index + 1] > 0.02]
        summary[branch] = {
            "rules": len(rows),
            "kept_prior": len(rows) - len(learned),
            "pln_vs_laplace": sum(abs(r["pln"][2 * index] - r["laplace"][index]) for r in learned) / len(learned),
            "pln_vs_true": sum(abs(r["pln"][2 * index] - r["true"][index]) for r in learned) / len(learned),
            "laplace_vs_true": sum(abs(r["laplace"][index] - r["true"][index]) for r in rows) / len(rows),
        }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
