"""Tables of a stage 4 sweep (sweep.sh): python3 report.py OUT"""
import json
import statistics
import sys
from pathlib import Path

runs = {path.stem: json.loads(path.read_text().strip().splitlines()[-1]) for path in Path(sys.argv[1]).glob("*.json") if path.read_text().strip()}


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else float("nan")


print("| reasoner | regret | agreement | utility | Brier | error to exact | s/round beliefs | s/round decision | queries/round | seeds |")
print("|---|---|---|---|---|---|---|---|---|---|")
groups = {}
for name, run in runs.items():
    if not name.startswith("growth") and not name.startswith("probe"):
        groups.setdefault(name.rsplit("-s", 1)[0], []).append(run)
for group, members in sorted(groups.items()):
    n = len(members)
    print(f"| {group} | {mean(r['regret'] for r in members):.3f} | {mean(r['agreement'] for r in members):.2f} | {mean(r['utility'] for r in members):.2f} | "
          f"{mean(r['brier'] for r in members):.3f} | {mean(r['posterior_error'] for r in members):.3f} | "
          f"{mean(r['seconds'] for r in members) / 30:.2f} | {mean(r['decision_seconds'] for r in members) / 30:.2f} | "
          f"{mean(r.get('decision_queries', float('nan')) for r in members):.1f} | {n} |")
print()
print("| plants | singles | capacity | options | proposer | queries/round | answered/round | s/round decision | regret |")
print("|---|---|---|---|---|---|---|---|---|")
for name, run in sorted(runs.items()):
    if name.startswith("growth"):
        _, proposer, plants, capacity = name.split("-")
        print(f"| {plants[1:]} | {int(plants[1:]) + 4} | {capacity[1:]} | {run['options']} | {proposer} | {run.get('decision_queries', 0):.1f} | "
              f"{run.get('decision_answered', 0):.1f} | {run['decision_seconds'] / 10:.2f} | {run['regret']:.3f} |")
