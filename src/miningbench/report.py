"""Paired bootstrap reports and falsifiable acceptance screens."""

from collections import defaultdict
import random
import statistics


def interval(values, seed=731, repeats=2000):
    """Seed-level percentile bootstrap; no step-level pseudoreplication."""
    if not values:
        return None
    values = sorted(values)  # Artifact paths/order must not change bootstrap draws.
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
    values = [value for value in values if value is not None]
    if not values:
        return "undefined"
    if len(values) == 1:
        return f"{values[0]:.4f} [CI unavailable: one seed]"
    ci = interval(values)
    return f"{ci['mean']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}]"


def gate(
    name,
    values,
    threshold,
    direction,
    broken,
    control,
    positive=None,
    positive_control=None,
):
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
        "positive_control": positive_control,
        "positive_values": positive,
        "positive_passes": passes(positive) if positive is not None else None,
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
    from itertools import islice

    stopped_steps = sum(
        1 for _ in islice(World(Config(samples=1024, kinds=("single",))).stream(), 32)
    )
    return {
        "completion": float(stopped_steps >= 1024),
        "oracle": max(errors),
        "gap": statistics.fmean(leaked_gaps),
        "xor": recovery(world.rules[0], world.rules[0], world.types)["all"]["recall"],
        "difficulty": baseline_skills[-1] - baseline_skills[0],
        "growth": growing.result()["relative_growth"],
    }


def positive_system_measurements():
    """Passing fixtures for six gate families, not independent workload trials.

    The difficulty control intentionally degrades oracle predictions toward the
    population predictor. It checks a passable skill gate, not a claim that every
    natural difficulty intervention must harm every real learner.
    """
    from .world import Config, World
    from .scoring import losses, recovery, CostMeter
    from .truth import interpret

    world = World(Config(samples=64, kinds=("xor",)))
    errors, gaps, skill_losses = [], [], []
    for sample in world.stream():
        truth = sample["q"]["T0_xor"]
        correct = interpret(world, sample["history"], 0)["T0_xor"]
        degraded = [0.5 * p + 0.25 for p in correct]
        base_error = losses([0.5, 0.5], truth)["brier"]
        correct_error = losses(correct, truth)["brier"]
        errors.append(max(abs(p - q) for p, q in zip(correct, truth)))
        gaps.append(base_error - correct_error)
        skill_losses.append(
            (losses(degraded, truth)["brier"] - correct_error) / base_error
        )
    # Known constant-mean 35us series with independent timing jitter, ten types
    # per seed. The measured gate uses this same within-seed mean aggregation.
    growth = []
    for seed in range(10):
        rng = random.Random(seed)
        fits = []
        for _ in range(10):
            meter = CostMeter(1024)
            for _ in range(1024):
                meter.add(35e-6 * rng.uniform(0.1, 1.9))
            fits.append(meter.result()["relative_growth"])
        growth.append(statistics.fmean(fits))
    return {
        "oracle": ([max(errors)] * 10, "independent interpreter on XOR truth"),
        "gap": (
            [statistics.fmean(gaps)] * 10,
            "oracle versus population predictor on XOR",
        ),
        "xor": (
            [recovery([], world.rules[0], world.types)["all"]["recall"]] * 10,
            "single-literal-only learner exports no XOR rule",
        ),
        "difficulty": (
            [statistics.fmean(skill_losses)] * 10,
            "oracle predictions degraded halfway toward population probabilities",
        ),
        "growth": (
            growth,
            "constant-mean 35us work with seeded timing jitter; mean of ten types",
        ),
        "completion": (
            [
                float(
                    sum(
                        1
                        for _ in World(Config(samples=1024, kinds=("single",))).stream()
                    )
                    == 1024
                )
            ]
            * 10,
            "system consumes all 1024 decisions",
        ),
    }


def separate_runtime_cohorts(results, failures=()):
    commits = {
        row["metadata"]["petta_commit"]
        for row in results
        if "petta_commit" in row["metadata"]
    }
    commits.update(
        failure["petta_commit"] for failure in failures if failure.get("petta_commit")
    )
    separated = []
    for row in results:
        metadata = dict(row["metadata"])
        if (
            len(commits) > 1
            and "petta_commit" in metadata
            and "@" not in metadata["system"]
        ):
            metadata["system"] += "@" + metadata["petta_commit"][:8]
        if (
            metadata.get("input_order") == "seeded-per-period-v1"
            and "+shuffled" not in metadata["system"]
        ):
            metadata["system"] += "+shuffled"
        separated.append(dict(row, metadata=metadata))
    separated_failures = []
    for failure in failures:
        row = dict(failure)
        if len(commits) > 1 and row.get("petta_commit") and "@" not in row["system"]:
            row["system"] += "@" + row["petta_commit"][:8]
        if (
            row.get("input_order") == "seeded-per-period-v1"
            and "+shuffled" not in row["system"]
        ):
            row["system"] += "+shuffled"
        separated_failures.append(row)
    return separated, separated_failures


def paired_difficulty_values(by_type):
    """Use the same powered type cohort in every seed-level average."""
    eligible = {k: values for k, values in by_type.items() if len(values) >= 10}
    omitted = {
        ",".join(k): len(values) for k, values in by_type.items() if k not in eligible
    }
    if not eligible:
        return [], [], omitted
    seeds = set.intersection(*(set(values) for values in eligible.values()))
    differences = [
        statistics.fmean(values[seed] for values in eligible.values())
        for seed in sorted(seeds)
    ]
    return differences, [list(k) for k in sorted(eligible)], omitted


def acceptance(results, failures=()):
    results, failures = separate_runtime_cohorts(results, failures)
    checks = []
    broken = broken_system_measurements()
    base = [r for r in results if r["metadata"].get("level") == "base"]
    oracles = [r for r in results if r["metadata"]["system"] == "oracle"]
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
    systems = sorted(
        {r["metadata"]["system"] for r in results}
        | {failure["system"] for failure in failures}
    )
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
                type_differences = defaultdict(dict)
                for r in results:
                    m = r["metadata"]
                    if m["system"] != system or m.get("level") != knob:
                        continue
                    key = (m["config"]["seed"], tuple(m["config"]["kinds"]))
                    b = baseline.get(key)
                    if b:
                        # Paired normalized skill: the irreducible label variance
                        # cancels in both numerator and denominator.
                        if (
                            b["normalized_skill"] is None
                            or r["normalized_skill"] is None
                        ):
                            continue
                        difference = b["normalized_skill"] - r["normalized_skill"]
                        type_differences[key[1]][key[0]] = difference
                values, included_types, omitted_types = paired_difficulty_values(
                    type_differences
                )
                check = gate(
                    f"{system}: harder {knob}",
                    values,
                    0.01,
                    "above",
                    [broken["difficulty"]] * 10,
                    "input-ignoring population predictor in both paired levels (skill zero)",
                )
                check["paired_rule_types"] = included_types
                check["underpowered_rule_types"] = omitted_types
                checks.append(check)
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
        values = [statistics.fmean(v) for v in seed_growth.values()]
        cost_check = gate(
            f"{system}: fitted constant-cost screen",
            values,
            0.5,
            "below",
            [broken["growth"]] * 10,
            "miner revisits all previous samples each step (linear cost series)",
        )
        completed = {
            (
                row["metadata"].get("level"),
                row["metadata"]["config"]["seed"],
                tuple(row["metadata"]["config"]["kinds"]),
            )
            for row in long
        }
        censored = [
            failure
            for failure in failures
            if failure["system"] == system
            and failure["level"] in ("long", "deep_lags")
            and (failure["level"], failure["seed"], tuple(failure["kinds"]))
            not in completed
        ]
        if censored:
            cost_check["evidence"] = {
                "completed_stream_fit": cost_check["evidence"],
                "uncompleted_long_attempts": len(censored),
            }
            if cost_check["status"] == "PASS":
                cost_check["status"] = "UNMEASURED"
        checks.append(cost_check)
        for level, label in (
            ("long", "plain long-stream completion"),
            ("deep_lags", "deep-lag 100-distractor completion"),
        ):
            completion = {
                row["metadata"]["config"]["seed"]: 1.0
                for row in long
                if row["metadata"].get("level") == level
            }
            for failure in censored:
                if failure["level"] == level:
                    completion[failure["seed"]] = 0.0
            checks.append(
                gate(
                    f"{system}: {label}",
                    list(completion.values()),
                    0.99,
                    "above",
                    [broken["completion"]] * 10,
                    "system stops after 32 of 1024 decisions",
                )
            )
    positive = positive_system_measurements()
    for check in checks:
        name = check["check"]
        if name == "oracle independent truth":
            family = "oracle"
        elif name.startswith("oracle beats marginal:"):
            family = "gap"
        elif name == "marginal XOR recovery fails":
            family = "xor"
        elif ": harder " in name:
            family = "difficulty"
        elif name.endswith("cost screen"):
            family = "growth"
        else:
            family = "completion"
        values, label = positive[family]
        tested = gate(
            name,
            check["broken_values"],
            check["threshold"],
            check["direction"],
            check["broken_values"],
            check["negative_control"],
            values,
            label,
        )
        check.update(
            {
                key: tested[key]
                for key in ("positive_values", "positive_control", "positive_passes")
            }
        )
        check["control_family"] = family
    return checks


def report(results, failures=()):
    results, failures = separate_runtime_cohorts(results, failures)
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
    checks = acceptance(results, failures)
    lines += [
        "",
        "## Acceptance",
        "",
        "| Check | Status | Paired evidence | Included types / omitted pair counts | Rejects broken system | Passing control passes |",
        "|---|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['check']} | {check['status']} | {check['evidence']} | "
            f"{check.get('paired_rule_types', [])} / {check.get('underpowered_rule_types', {})} | {check['discriminates']} | {check['positive_passes']} |"
        )
    lines += [
        "",
        "Controls are six reusable fixture families, not independent experiments per gate. The negative gap and difficulty controls are identity constructions (x-x and zero-skill difference). Positive controls test passability, not universal degradation.",
        "Cost is an empirical slope screen, not a complexity proof. Timings and RSS are not seed deterministic.",
        "Fewer than ten paired seeds is UNMEASURED, never a powered pass.",
        "",
    ]
    return "\n".join(lines), checks
