"""Summarize run_counterfactual.sh outputs: mean scores per phase, ms per trial and trials to criterion.

    python3 summarize_counterfactual.py RAW_DIR [--json OUT]

Trials to criterion: within a feedback phase (train; reversal's reverse), the
number of trials up to and including the end of the first run of 8
consecutive correct trials; a run that never reaches it counts as the phase
length and is reported as not reached."""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

RUN = 8


def to_criterion(trials):
    streak = 0
    for i, t in enumerate(trials, 1):
        streak = streak + 1 if t["op"] == t["expected"] else 0
        if streak == RUN:
            return i, True
    return len(trials), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--json")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.raw, "*.json"))):
        m = re.match(r"(\w+?)_(mined-)?(\w+)-(\w+)_cd(\w+)_s(\d+)\.json", os.path.basename(f))
        task, mined, ev, de, cd, seed = m.groups()
        d = json.load(open(f))
        T = d["timing"]
        row = {"seed": int(seed), "score": d["score"], "phases": d["phase_scores"],
               "ms": 1000 * statistics.mean(t["add"] + t["review"] + t["decide"] + t.get("feedback", 0) for t in T),
               "inference_ms": 1000 * statistics.mean(t["inference"] for t in T),
               "prior": sum(len(t["prior"]) for t in T)}
        for phase in ("train", "reverse"):
            trials = [t for b in d["blocks"] if b["phase"] == phase for t in b["trials"]]
            if trials:
                row[f"ttc_{phase}"] = to_criterion(trials)
        groups[(task, bool(mined), cd, f"{ev}/{de}")].append(row)
    out = []
    for (task, mined, cd, mode), rows in sorted(groups.items()):
        s = {"task": task + (" mined" if mined else ""), "cd": cd, "mode": mode, "n": len(rows),
             "score": statistics.mean(r["score"] for r in rows),
             "phases": {p: statistics.mean(r["phases"][p] for r in rows) for p in rows[0]["phases"]},
             "ms": statistics.mean(r["ms"] for r in rows), "inference_ms": statistics.mean(r["inference_ms"] for r in rows),
             "prior_answers": statistics.mean(r["prior"] for r in rows),
             "seeds": {r["seed"]: r["score"] for r in rows}}
        for phase in ("train", "reverse"):
            if f"ttc_{phase}" in rows[0]:
                s[f"ttc_{phase}"] = statistics.mean(r[f"ttc_{phase}"][0] for r in rows)
                s[f"reached_{phase}"] = sum(r[f"ttc_{phase}"][1] for r in rows)
        out.append(s)
        ph = " ".join(f"{p} {v:.3f}" for p, v in s["phases"].items() if p != "baseline")
        ttc = " ".join(f"ttc-{p} {s['ttc_' + p]:.1f} ({s['reached_' + p]}/{s['n']})" for p in ("train", "reverse") if f"ttc_{p}" in s)
        print(f"{s['task']:<22} cd {cd:<6} {mode:<14} n {s['n']:>2} score {s['score']:.3f} | {ph} | {ttc} | {s['ms']:.0f} ms/trial | prior {s['prior_answers']:.0f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
