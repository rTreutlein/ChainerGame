"""Tables of the Pong runs: hit ratio, time per decision and learning curves
for ONA's C game (bench/results/pong/ona_c/*.curve, cumulative counts every
100 steps), ONA on the port and PeTTaChainer (raw/*.json from onabench.pong).

    python -m onabench.pong_report [RESULTS_DIR]
"""
import glob
import json
import os
import re
import statistics
import sys

WINDOWS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10001)]


def window_ratios(curve):
    """Hit ratio of the balls scored in each window of steps."""
    at = {0: (0, 0)}
    at.update({s: (h, m) for s, h, m in curve})
    out = []
    for a, b in WINDOWS:
        a = max(s for s in at if s <= a)
        b = max(s for s in at if s <= b)
        dh, dm = at[b][0] - at[a][0], at[b][1] - at[a][1]
        out.append(dh / (dh + dm) if dh + dm else float("nan"))
    return out


def load(results):
    runs = []
    for f in sorted(glob.glob(os.path.join(results, "ona_c", "*.curve"))):
        variant, seed = re.match(r"(pong2?)_(\w+)\.curve", os.path.basename(f)).groups()
        curve = [tuple(map(int, line.split())) for line in open(f)]
        if seed == "default":
            continue
        s, h, m = curve[-1]
        runs.append({"group": f"ONA C (./NAR {variant} 10000)", "variant": variant, "seed": seed,
                     "ratio": h / (h + m), "curve": curve})
    for f in sorted(glob.glob(os.path.join(results, "raw", "*.json"))):
        r = json.load(open(f))
        if r["agent"] == "ona":
            group = "ONA on the port (NAR.py)" + (", generator seeded" if r["ona_seed"] is not None else "")
        else:
            group = "PLN" + (" mined" if r["mined"] else "") + (", ONA's 0.501 threshold" if r["policy"] == "ona" else "") + (
                f", {r['deadline_ms']:g} ms deadline" if r["deadline_ms"] else ", unconstrained")
        runs.append({"group": group, "variant": r["variant"], "seed": r["seed"], "ratio": r["ratio"],
                     "curve": r["curve"], "timing": r.get("timing_summary", {}), "late": r.get("late_steps", 0),
                     "decisions": r.get("decisions")})
    return runs


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "/nexus/Dev/OpenCog/bench/results/pong"
    runs = load(results)
    groups = sorted({(r["variant"], r["group"]) for r in runs})
    print("## Hit ratio over 10001 steps\n")
    print("| game | agent | seeds | mean | sd | min | max | ms per decision (mean, p95) | inference / overhead ms | late steps | decisions |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for variant, group in groups:
        rs = [r for r in runs if (r["variant"], r["group"]) == (variant, group)]
        ratios = [r["ratio"] for r in rs]
        t = [r.get("timing", {}) for r in rs]
        ms = (f"{statistics.fmean(x['mean_ms'] for x in t):.1f}, {statistics.fmean(x['p95_ms'] for x in t):.1f}"
              if t[0] else "")
        split = (f"{statistics.fmean(x['inference_ms'] for x in t):.1f} / {statistics.fmean(x['overhead_ms'] for x in t):.1f}"
                 if t[0] and "inference_ms" in t[0] else "")
        late = f"{statistics.fmean(r['late'] for r in rs):.0f}" if "late" in rs[0] else ""
        dec = f"{statistics.fmean(r['decisions'] for r in rs):.0f}" if rs[0].get("decisions") else ""
        sd = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        print(f"| {variant} | {group} | {len(rs)} | {statistics.fmean(ratios):.3f} | {sd:.3f} | {min(ratios):.3f} |"
              f" {max(ratios):.3f} | {ms} | {split} | {late} | {dec} |")
    print("\n## Learning curves: hit ratio of the balls scored in each window of steps (mean over seeds)\n")
    print("| game | agent | " + " | ".join(f"{a}-{min(b, 10000)}" for a, b in WINDOWS) + " |")
    print("|---|---|" + "---|" * len(WINDOWS))
    for variant, group in groups:
        ws = [window_ratios(r["curve"]) for r in runs if (r["variant"], r["group"]) == (variant, group)]
        cols = []
        for i in range(len(WINDOWS)):
            vals = [w[i] for w in ws if w[i] == w[i]]
            cols.append(f"{statistics.fmean(vals):.3f}" if vals else "")
        print(f"| {variant} | {group} | " + " | ".join(cols) + " |")
    print("\n## Per seed\n")
    for variant, group in groups:
        rs = [r for r in runs if (r["variant"], r["group"]) == (variant, group)]
        print(f"- {variant}, {group}: " + ", ".join(f"{r['seed']}: {r['ratio']:.3f}" for r in rs))


if __name__ == "__main__":
    main()
