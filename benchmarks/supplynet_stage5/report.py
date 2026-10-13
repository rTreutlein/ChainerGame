"""Print stage 5 summary and candidate-cost tables from a sweep."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/nexus/Dev/OpenCog/bench/results/supplynet-stage5/partial-network")
runs = {}
cost_profiles = {}
cost_path = root/"baseline-costs.jsonl"
if cost_path.exists():
    for line in cost_path.read_text().splitlines():
        profile = json.loads(line)
        assert profile['actions_match']
        cost_profiles[f"{profile['backend']}-s{profile['seed']}"] = profile['amortized_seconds']
for path in root.glob("*.json"):
    text = path.read_text().strip()
    if not text:
        continue  # A probe or a summary writer is still running.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        print(f"Skipping incomplete JSON: {path.name}", file=sys.stderr)
        continue
    if value.get("type") == "run-summary":
        if path.stem in cost_profiles:
            value['amortized_seconds'] = cost_profiles[path.stem]
        runs[path.stem] = value


def mean(rows, field):
    return statistics.mean(r[field] for r in rows if r.get(field) is not None) if any(r.get(field) is not None for r in rows) else float("nan")


groups = {}
for name, run in sorted(runs.items()):
    if not name.startswith("growth"):
        groups.setdefault(name.rsplit("-s", 1)[0], []).append(run)
print("| policy | seeds | regret/round | gain over myopic | myopic suboptimal % | realized reward | s/round decision | s/round reference | pairs/round | pair mean error |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, rows in groups.items():
    label = name+" (unavailable; no-action fallback)" if any(r.get('backend_unavailable') for r in rows) else name
    print(f"| {label} | {len(rows)} | {mean(rows, 'regret'):.3f} | {mean(rows, 'lookahead_gain'):.3f} | {100*mean(rows, 'myopic_suboptimal_fraction'):.1f} | {mean(rows, 'utility'):.3f} | {mean(rows, 'amortized_seconds'):.3f} | {mean(rows, 'reference_seconds'):.3f} | {mean(rows, 'candidate_pairs'):.1f} | {mean(rows, 'pair_mean_error'):.3f} |".replace('nan','—'))
if cost_profiles:
    print("\nMyopic/greedy seconds include separately measured exact belief construction; all 240 replayed actions match the full runs.")
print("\n| case | options | possible pairs | candidate pair branches | context bound | queried roots | s/round | regret | open answers |")
print("|---|---|---|---|---|---|---|---|---|")
for name, r in sorted(runs.items()):
    if name.startswith("growth"):
        print(f"| {name} | {r['options']} | {r['all_pairs']} | {r.get('candidate_pairs', 0):.1f} | {r.get('context_bound', 0):.1f} | {r.get('decision_queries', 0):.1f} | {r['amortized_seconds']:.3f} | {r['regret']:.3f} | {r.get('open_answered', 0):.1f} |")
print("\nProbLog outcomes:")
for name, r in sorted(runs.items()):
    if r['backend'] == 'problog':
        print(name, "answered", r["problog_answered"], "timeouts", r["problog_timeouts"], "failed", r["problog_failed"], "max Q error", r.get("problog_reference_max_error"))
print("\n| policy | first 10 rounds s/round | last 10 rounds s/round | worker setup s/round | inference s/round | overhead s/round | answered pairs | open second answers |")
print("|---|---|---|---|---|---|---|---|")
for name, rows in groups.items():
    if rows[0]['backend'] in ('pettachainer', 'mm2'):
        print(f"| {name} | {mean(rows, 'early_decision_seconds'):.3f} | {mean(rows, 'late_decision_seconds'):.3f} | {mean(rows, 'worker_setup_seconds'):.3f} | {mean(rows, 'chainer_inference_seconds'):.3f} | {mean(rows, 'chainer_overhead_seconds'):.3f} | {mean(rows, 'decision_answered'):.1f} | {mean(rows, 'open_answered'):.1f} |".replace('nan','—'))
print("\nNARS: skipped; no comparable joint expected-utility planner.")
for name, rows in groups.items():
    if rows[0]['backend'] == 'mm2':
        print(name, "unsupported:", sorted({setting for r in rows for setting in r.get('unsupported', [])}))
print("\nVisited belief coverage (absolute period identifiers excluded):")
coverage = root/"visited-beliefs.jsonl"
if coverage.exists():
    for line in coverage.read_text().splitlines():
        r = json.loads(line)
        print(r['policy'], f"{r['suboptimal_states']}/{r['unique_states']} states;",
              f"{r['suboptimal_rounds']}/{r['rounds']} rounds;",
              f"myopic regret {r['myopic_regret']:.6f}")
ablation = root/"myopic-ablation.jsonl"
if ablation.exists():
    rows = [json.loads(line) for line in ablation.read_text().splitlines()]
    print("Strict immediate-reward myopic ablation:",
          f"regret {mean(rows, 'regret'):.6f};",
          f"suboptimal rounds {100*mean(rows, 'myopic_suboptimal_fraction'):.1f}%")
print("\nFailed runs:")
for path in sorted(root.glob("*.status.json")):
    text = path.read_text().strip()
    if not text:
        continue
    r = json.loads(text)
    if r["returncode"]:
        print(path.stem, "exit", r["returncode"], "see", path.with_name(path.name.replace(".status.json", ".err")))
