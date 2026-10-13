"""Paired bootstrap reports and falsifiable acceptance screens."""

from collections import defaultdict
import random
import statistics


def interval(values, seed=731, repeats=2000):
    """Seed-level percentile bootstrap; no step-level pseudoreplication."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repeats)
    )
    return {
        "mean": statistics.fmean(values),
        "low": means[int(0.025 * repeats)],
        "high": means[int(0.975 * repeats)],
        "n": len(values),
    }


def formatted(values):
    ci = interval(values)
    return f"{ci['mean']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}]"


def gate(name, values, threshold, direction, broken, control):
    """The identical predicate evaluates measured and deliberately broken arms."""

    def passes(xs):
        ci = interval(xs)
        if ci is None:
            return False
        return (
            ci["low"] > threshold if direction == "above" else ci["high"] <= threshold
        )

    ci = interval(values)
    powered = len(values) >= 10
    return {
        "check": name,
        "status": ("PASS" if passes(values) else "FAIL") if powered else "UNMEASURED",
        "evidence": ci,
        "threshold": threshold,
        "direction": direction,
        "negative_control": control,
        "broken_values": broken,
        "discriminates": bool(broken) and not passes(broken),
    }


def broken_system_measurements():
    """Execute small faulty systems through the real scoring functions."""
    from .world import Config, World
    from .scoring import losses, recovery, CostMeter
    from .truth import interpret

    world = World(Config(samples=64, kinds=("xor",)))
    errors, leaked_gaps, baseline_skills = [], [], []
    for sample in world.stream():
        exact = sample["q"]["T0_xor"]
        errors.append(max(abs(p - q) for p, q in zip([0.5, 0.5], exact)))
        oracle = interpret(world, sample["history"], 0)["T0_xor"]
        leaked_gaps.append(
            losses(oracle, exact)["brier"] - losses(oracle, exact)["brier"]
        )
        error = losses([0.5, 0.5], exact)["brier"]
        baseline_skills.append(1 - error / error)
    growing = CostMeter(1024)
    for step in range(1024):
        growing.add(step + 1)
    return {
        "oracle": max(errors),
        "gap": statistics.fmean(leaked_gaps),
        "xor": recovery(world.rules[0], world.rules[0], world.types)["all"]["recall"],
        "difficulty": baseline_skills[-1] - baseline_skills[0],
        "growth": growing.result()["relative_growth"],
    }


def acceptance(results):
    checks = []
    broken = broken_system_measurements()
    base = [r for r in results if r["metadata"].get("level") == "base"]
    oracles = [r for r in base if r["metadata"]["system"] == "oracle"]
    # Independent interpreter error, not oracle-vs-itself probability loss.
    by_seed = defaultdict(list)
    for r in oracles:
        by_seed[r["metadata"]["config"]["seed"]].append(
            r["oracle_independent_max_error"]
        )
    values = [max(v) for v in by_seed.values()]
    checks.append(
        gate(
            "oracle independent truth",
            values,
            1e-12,
            "below",
            [broken["oracle"]] * 10,
            "constant .5 predictor on deterministic exhaustive truth fixtures",
        )
    )
    paired = defaultdict(dict)
    for r in results:
        c = r["metadata"]["config"]
        key = (r["metadata"].get("level"), c["seed"], tuple(c["kinds"]))
        paired[key][r["metadata"]["system"]] = r
    for kind in sorted({k for r in results for k in r["by_type"]} - {"single"}):
        gaps = [
            p["marginal"]["by_type"][kind]["brier"]
            - p["oracle"]["by_type"][kind]["brier"]
            for (level, _, _), p in paired.items()
            if level == "base"
            and "oracle" in p
            and "marginal" in p
            and kind in p["oracle"]["by_type"]
        ]
        checks.append(
            gate(
                "oracle beats marginal: " + kind,
                gaps,
                0.01,
                "above",
                [broken["gap"]] * 10,
                "marginal replaced by oracle predictions",
            )
        )
    xor = [
        r["recovery"]["by_type"]["xor"]["recall"]
        for r in base
        if r["metadata"]["system"] == "marginal" and "xor" in r["by_type"]
    ]
    checks.append(
        gate(
            "marginal XOR recovery fails",
            xor,
            0.05,
            "below",
            [broken["xor"]] * 10,
            "marginal leaks the planted XOR rules",
        )
    )
    systems = sorted({r["metadata"]["system"] for r in results})
    for system in systems:
        if system != "oracle":
            baseline = {
                (
                    r["metadata"]["config"]["seed"],
                    tuple(r["metadata"]["config"]["kinds"]),
                ): r
                for r in base
                if r["metadata"]["system"] == system
            }
            for knob in (
                "distractors",
                "strength",
                "observation_noise",
                "samples",
                "drift",
                "targets",
            ):
                seed_differences = defaultdict(list)
                for r in results:
                    m = r["metadata"]
                    if m["system"] != system or m.get("level") != knob:
                        continue
                    key = (m["config"]["seed"], tuple(m["config"]["kinds"]))
                    b = baseline.get(key)
                    if b:
                        # Paired normalized skill: the irreducible label variance
                        # cancels in both numerator and denominator.
                        difference = b["normalized_skill"] - r["normalized_skill"]
                        seed_differences[key[0]].append(difference)
                values = [statistics.fmean(v) for v in seed_differences.values()]
                checks.append(
                    gate(
                        f"{system}: harder {knob}",
                        values,
                        0.01,
                        "above",
                        [broken["difficulty"]] * 10,
                        "input-ignoring population predictor in both paired levels (skill zero)",
                    )
                )
        long = [
            r
            for r in results
            if r["metadata"]["system"] == system
            and r["metadata"]["config"]["samples"] >= 1000
        ]
        seed_growth = defaultdict(list)
        for r in long:
            seed_growth[r["metadata"]["config"]["seed"]].append(
                r["mining_cost"]["relative_growth"]
            )
        values = [max(v) for v in seed_growth.values()]
        checks.append(
            gate(
                f"{system}: fitted constant-cost screen",
                values,
                0.5,
                "below",
                [broken["growth"]] * 10,
                "miner revisits all previous samples each step (linear cost series)",
            )
        )
    return checks


def report(results, failures=()):
    groups = defaultdict(list)
    for r in results:
        for kind in r["by_type"]:
            groups[
                (r["metadata"].get("level", "run"), r["metadata"]["system"], kind)
            ].append(r)
    lines = [
        "# Pattern mining benchmark",
        "",
        "Intervals are 95% seed bootstrap intervals; paired differences resample paired seeds.",
        "",
        "| Difficulty | System | Type | Seeds | Brier [95% CI] | Run skill [95% CI] | Recall [95% CI] | Precision | Mining ms/step | Fitted growth | Peak MiB |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for (level, system, kind), rows in sorted(groups.items()):

        def values(function):
            return [function(row) for row in rows]

        lines.append(
            f"| {level} | {system} | {kind} | {len(rows)} | "
            + formatted(values(lambda r: r["by_type"][kind]["brier"]))
            + " | "
            + formatted(values(lambda r: r["normalized_skill"]))
            + " | "
            + formatted(values(lambda r: r["recovery"]["by_type"][kind]["recall"]))
            + " | "
            + f"{statistics.fmean(values(lambda r: r['recovery']['by_type'][kind]['precision'])):.3f} | "
            + f"{statistics.fmean(values(lambda r: r['mining_cost']['mean_seconds'])) * 1000:.2f} | "
            + f"{statistics.fmean(values(lambda r: r['mining_cost']['relative_growth'])):.2f} | "
            + f"{max(r['peak_rss_kib'] for r in rows) / 1024:.1f} |"
        )
    lines += [
        "",
        "## Paired arm differences",
        "",
        "| Difficulty | Type | Arm minus marginal Brier [95% CI] | Paired seeds |",
        "|---|---|---|---:|",
    ]
    for (level, system, kind), rows in sorted(groups.items()):
        if system == "marginal":
            continue
        refs = {
            r["metadata"]["config"]["seed"]: r
            for r in groups.get((level, "marginal", kind), [])
        }
        differences = [
            r["by_type"][kind]["brier"]
            - refs[r["metadata"]["config"]["seed"]]["by_type"][kind]["brier"]
            for r in rows
            if r["metadata"]["config"]["seed"] in refs
        ]
        if differences:
            lines.append(
                f"| {level} | {kind} | {system}: {formatted(differences)} | {len(differences)} |"
            )
    association_groups = defaultdict(list)
    for result in results:
        metadata = result["metadata"]
        key = (
            metadata.get("level", "run"),
            metadata["system"],
            tuple(metadata["config"]["kinds"]),
        )
        association_groups[key].append(result["associations"])
    lines += [
        "",
        "## Weak association calibration",
        "",
        "Intervals condition on seeds exporting at least one calibratable weak association.",
        "",
        "| Difficulty | System | Types | Mean weak rules | Calibration MSE [95% CI] | Seeds with associations |",
        "|---|---|---|---:|---|---:|",
    ]
    for (level, system, kinds), rows in sorted(association_groups.items()):
        errors = [
            row["mean_squared_error"]
            for row in rows
            if row["mean_squared_error"] is not None
        ]
        if errors:
            count = statistics.fmean(row["count"] for row in rows)
            lines.append(
                f"| {level} | {system} | {', '.join(kinds)} | {count:.1f} | {formatted(errors)} | {len(errors)} |"
            )
    if failures:
        lines += [
            "",
            "## Incomplete runs",
            "",
            "Incomplete streams have no full-stream score.",
            "",
            "| Difficulty | System | Type | Seed | Exit | Completed steps |",
            "|---|---|---|---:|---:|---:|",
        ]
        for f in failures:
            lines.append(
                f"| {f['level']} | {f['system']} | {', '.join(f['kinds'])} | "
                f"{f['seed']} | {f['returncode']} | {f['completed_steps']} |"
            )
    checks = acceptance(results)
    lines += [
        "",
        "## Acceptance",
        "",
        "| Check | Status | Paired evidence | Rejects broken system |",
        "|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['check']} | {check['status']} | {check['evidence']} | {check['discriminates']} |"
        )
    lines += [
        "",
        "Cost is an empirical slope screen, not a complexity proof. Timings and RSS are not seed deterministic.",
        "Fewer than ten paired seeds is UNMEASURED, never a powered pass.",
        "",
    ]
    return "\n".join(lines), checks
