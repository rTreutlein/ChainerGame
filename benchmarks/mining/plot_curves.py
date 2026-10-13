"""Export powered learning curves with seed bootstrap intervals (matplotlib)."""

import json
from pathlib import Path
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from miningbench.report import interval

root = Path(sys.argv[1])
provenance = json.loads((root / "provenance.json").read_text())
selected = {}
for path in provenance["runs"]:
    result = json.loads(Path(path).read_text())
    metadata = result["metadata"]
    config = metadata["config"]
    if (
        metadata.get("level") not in ("base", "long")
        or len(config["kinds"]) != 1
        or config["targets"] != 1
    ):
        continue
    key = (
        provenance.get("run_systems", {}).get(path, metadata["system"]),
        config["kinds"][0],
        config["seed"],
    )
    if (
        key not in selected
        or config["samples"] > selected[key]["metadata"]["config"]["samples"]
    ):
        selected[key] = result
curves = defaultdict(list)
for (system, kind, seed), result in selected.items():
    for point in result["learning_curve"]:
        curves[(system, kind, point["samples"])].append(
            point["cumulative"][kind]["brier"]
        )
kinds = sorted({key[1] for key in selected})
colors = {
    "oracle": "#338855",
    "marginal": "#4477aa",
    "petta": "#cc6677",
    "petta-link-first": "#aa4499",
}
for system, _, _ in selected:
    if system not in colors:
        colors[system] = colors.get(system.split("@")[0], "#777777")
fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
for ax, kind in zip(axes.flat, kinds):
    partial = []
    for system, color in colors.items():
        samples = sorted(n for s, k, n in curves if s == system and k == kind)
        if not samples:
            continue
        cis = [interval(curves[(system, kind, n)]) for n in samples]
        count = min(ci["n"] for ci in cis)
        if count < 10:
            partial.append(f"{system}: n={count}")
        ax.plot(
            samples, [ci["mean"] for ci in cis], label=system, color=color, marker="."
        )
        ax.fill_between(
            samples,
            [ci["low"] for ci in cis],
            [ci["high"] for ci in cis],
            color=color,
            alpha=0.2,
        )
    ax.set_title(kind)
    if partial:
        ax.text(
            0.03, 0.97, "\n".join(partial), transform=ax.transAxes, va="top", fontsize=7
        )
    ax.set_xlabel("Samples")
    ax.set_ylabel("Cumulative probability Brier")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256, 1024], labels=["64", "256", "1024"])
    ax.grid(alpha=0.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle(
    "Planted mining benchmark: seed-level learning curves with 95% bootstrap intervals"
)
fig.tight_layout()
fig.savefig(root / "learning_curves.svg")
fig.savefig(root / "learning_curves.png", dpi=150)
plt.close(fig)
