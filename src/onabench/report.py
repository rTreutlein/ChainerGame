"""Summarise matching-to-sample runs (onabench.mts JSON files).

    python -m onabench.report RAW_DIR

Per task and agent: the scored phases (mean and range over seeds), every
phase's mean block score, trials to criterion (first block of a feedback
phase from which every later block of it scores 1), and time per trial
(PLN: adding facts, review, decision and feedback; ONA: the harness's wall
per trial, which includes its cycles between trials).
"""
import glob
import json
import statistics
import sys
from collections import defaultdict


def criterion(blocks, phase):
    scores = [b["correct"] for b in blocks if b["phase"] == phase]
    for i in range(len(scores)):
        if all(s == 1.0 for s in scores[i:]):
            return i + 1
    return None


def main(raw):
    runs = defaultdict(list)
    for path in sorted(glob.glob(f"{raw}/*.json")):
        r = json.load(open(path))
        label = r["agent"] if r["agent"] == "ona" else f"pln-{r.get('space', '?')}"
        runs[(r["task"], label)].append(r)
    for (task, label), rs in sorted(runs.items()):
        phases = list(rs[0]["phase_scores"])
        scores = [r["score"] for r in rs]
        print(f"## {task} / {label}  (seeds {sorted(r['seed'] for r in rs)})")
        print(f"score (scored phases) mean {statistics.mean(scores):.3f}  min {min(scores):.3f}  max {max(scores):.3f}")
        print("phase means: " + ", ".join(f"{p} {statistics.mean(r['phase_scores'][p] for r in rs):.3f}" for p in phases))
        fb_phases = [p for p in phases if p in ("train", "reverse")]
        for p in fb_phases:
            crit = [criterion(r["blocks"], p) for r in rs]
            reached = [c for c in crit if c is not None]
            print(f"{p}: criterion block (all later blocks 1.0) reached in {len(reached)}/{len(crit)} seeds"
                  + (f", median block {statistics.median(reached)}" if reached else ""))
        trials = [t for r in rs for b in r["blocks"] for t in b["trials"]]
        if label == "ona":
            print(f"time per trial (wall incl. cycles): mean {1000 * statistics.mean(t['wall'] for t in trials):.1f} ms")
        else:
            tim = [x for r in rs for x in r["timing"]]
            parts = {k: statistics.mean(x.get(k, 0.0) for x in tim) for k in ("add", "review", "decide", "feedback")}
            tot = [x["add"] + x["review"] + x["decide"] + x.get("feedback", 0.0) for x in tim]
            print("time per trial: mean {:.1f} ms, median {:.1f} ms, max {:.0f} ms (add {:.1f}, review {:.1f}, decide {:.1f}, feedback {:.1f}); whole run incl. load mean {:.1f} s; hypotheses {}".format(
                1000 * statistics.mean(tot), 1000 * statistics.median(tot), 1000 * max(tot),
                *(1000 * parts[k] for k in ("add", "review", "decide", "feedback")),
                statistics.mean(r["wall"] for r in rs), sorted({r["hypotheses"] for r in rs})))
        print()


if __name__ == "__main__":
    main(sys.argv[1])
