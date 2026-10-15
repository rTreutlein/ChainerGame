"""Prequential evaluation with raw distributions, rules, curves and process cost."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import platform
import resource
import subprocess
import time

from .world import World, serial_sample, rng_for
from .systems import Marginal, BudgetedMarginal
from .scoring import RunningAverage, CostMeter, losses, recovery, association_scores
from .truth import interpret, conditional
from .world import Rule


def revision(path):
    p = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return p.stdout.strip() or None


def source_hash():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def is_new_entity_context(target, row):
    """The scored binding context must actually contain only holdout entities."""
    observed = {head: args[0] for (head, args), value in row.items() if value and args}
    new_entities = {"e4", "e5", "e6", "e7"}
    return all(
        observed.get(target + "_" + role) in new_entities for role in ("At", "Did")
    )


def run(config, system, out, budget=80, max_new=4, max_live=8, adapter=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(
        (out / name).exists() for name in ("world.json", "steps.jsonl", "result.json")
    ):
        raise FileExistsError(f"refusing to overwrite benchmark evidence in {out}")
    census_steps = {
        int(n) for n in os.environ.get("MININGBENCH_STORE_CENSUS", "").split(",") if n
    }
    world = World(config)
    metadata = world.metadata()
    metadata.update(
        {
            "system": system,
            "python": platform.python_version(),
            "source_sha256": source_hash(),
            "chainer_game_commit": revision(Path(__file__).parents[2]),
            "budget": budget,
            "max_new": max_new,
            "max_live": max_live,
        }
    )
    if system in ("petta", "petta-link-first"):
        from .petta import Petta

        learner = Petta(
            world.features,
            world.domains,
            budget,
            max_new,
            max_live,
            link_first=system == "petta-link-first",
            seed=config.seed,
        )
        import pettachainer

        metadata["input_order"] = learner.input_order
        metadata["petta_path"] = str(Path(pettachainer.__file__).resolve())
        metadata["petta_commit"] = revision(Path(pettachainer.__file__).parents[1])
    elif system == "marginal":
        learner = Marginal(world.features, world.domains)
    elif system in ("budgeted", "budgeted-flat"):
        reference_features = list(world.features)
        rng_for(config.seed, "budgeted-schema").shuffle(reference_features)
        metadata["reference_feature_order_sha256"] = hashlib.sha256(
            repr(reference_features).encode()
        ).hexdigest()
        learner = BudgetedMarginal(
            reference_features, world.domains, budget, flat=system == "budgeted-flat"
        )
    elif system == "oracle":
        learner = None
    elif adapter:
        learner = adapter(world.features, world.domains)
    else:
        raise ValueError(f"unknown system {system}")
    (out / "world.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_scores = RunningAverage()
    times, learn_times = CostMeter(config.samples), CostMeter(config.samples)
    mining_times = CostMeter(config.samples)
    curves = []
    relevance = {
        target: 1.0 if i == 0 else 0.05 for i, target in enumerate(world.domains)
    }
    weighted_error = weighted_mass = oracle_max_error = 0.0
    base_scores = RunningAverage()
    holdout_scores = RunningAverage()
    prior = {
        phase: {
            t: [conditional(world, Rule(t, (), v), phase) for v in domain]
            for t, domain in world.domains.items()
        }
        for phase in (0, 1)
    }
    work_totals = {}
    frozen_rules = None
    recovery_points = []
    by_type = {k: RunningAverage() for k in sorted(set(world.types.values()))}
    windows = {k: RunningAverage() for k in by_type}
    holdout_by_type = {k: RunningAverage() for k in by_type}
    digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        with (
            (out / "steps.jsonl").open("w") as raw,
            (out / "samples.jsonl").open("w") as samples_file,
        ):
            for sample in world.stream():
                serialized = json.dumps(serial_sample(sample), sort_keys=True)
                digest.update(serialized.encode())
                samples_file.write(serialized + "\n")
                t0 = time.perf_counter()
                if learner and hasattr(learner, "set_targets"):
                    learner.set_targets(relevance)
                predictions = (
                    learner.predict(sample["history"])
                    if learner
                    else interpret(world, sample["history"], sample["phase"])
                )
                t1 = time.perf_counter()
                if sample["holdout"] and frozen_rules is None:
                    frozen_rules = (
                        learner.rules() if learner else world.rules[sample["phase"]]
                    )
                step_events = (
                    learner.learn(sample["history"], sample["y"])
                    if learner and not sample["holdout"]
                    else []
                )
                t2 = time.perf_counter()
                times.add(t2 - t0)
                learn_times.add(t2 - t1)
                mining_seconds = (
                    learner.ingest_seconds + t2 - t1
                    if system.startswith("petta")
                    else (t2 - t0 if learner else 0.0)
                )
                mining_times.add(mining_seconds)
                scores = {t: losses(predictions[t], q) for t, q in sample["q"].items()}
                held_targets = [
                    target
                    for target in world.domains
                    if sample["holdout"]
                    and (
                        world.types[target] != "binding"
                        or is_new_entity_context(target, sample["history"][-1])
                    )
                ]
                for t, score in scores.items():
                    by_type[world.types[t]].add(score)
                    windows[world.types[t]].add(score)
                    all_scores.add(score)
                    base_scores.add(losses(prior[sample["phase"]][t], sample["q"][t]))
                    weighted_error += relevance[t] * score["brier"]
                    weighted_mass += relevance[t]
                    if t in held_targets:
                        holdout_scores.add(score)
                        holdout_by_type[world.types[t]].add(score)
                    if learner is None:
                        oracle_max_error = max(
                            oracle_max_error,
                            max(
                                abs(p - q)
                                for p, q in zip(predictions[t], sample["q"][t])
                            ),
                        )
                row = {
                    "step": sample["step"],
                    "phase": sample["phase"],
                    "predicted": predictions,
                    "exact": sample["q"],
                    "label": sample["y"],
                    "scores": scores,
                    "predict_seconds": t1 - t0,
                    "learn_seconds": t2 - t1,
                    "total_seconds": t2 - t0,
                    "mining_path_seconds": mining_seconds,
                    "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "work_units": learner.work if learner else 0,
                    "events": step_events,
                    "relevance": relevance,
                    "holdout": sample["holdout"],
                    "holdout_scored_targets": held_targets,
                }
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                if system.startswith("petta") and sample["step"] + 1 in census_steps:
                    from .census import store_census

                    with (out / "store_census.jsonl").open("a") as census_file:
                        census_file.write(
                            json.dumps(store_census(sample["step"] + 1)) + "\n"
                        )
                if learner and isinstance(learner.work, dict):
                    for key, value in learner.work.items():
                        if isinstance(value, (int, float)):
                            work_totals[key] = work_totals.get(key, 0) + value
                        elif key == "by_lag":
                            for lag, count in value.items():
                                work_totals["lag_" + lag] = (
                                    work_totals.get("lag_" + lag, 0) + count
                                )
                n = sample["step"] + 1
                if n in (64, 256, 1024) or n == config.samples:
                    learned = (
                        frozen_rules
                        if frozen_rules is not None
                        else learner.rules()
                        if learner
                        else world.rules[sample["phase"]]
                    )
                    rec = recovery(
                        learned,
                        world.rules[sample["phase"]],
                        world.types,
                        world,
                        sample["phase"],
                    )
                    point = {
                        "samples": n,
                        "phase": sample["phase"],
                        "recovery": rec,
                        "window": {k: v.result() for k, v in windows.items()},
                        "cumulative": {k: v.result() for k, v in by_type.items()},
                    }
                    curves.append(point)
                    recovery_points.append(rec)
                    windows = {k: RunningAverage() for k in by_type}
        learned = (
            frozen_rules
            if frozen_rules is not None
            else learner.rules()
            if learner
            else world.rules[sample["phase"]]
        )
        result = {
            "status": "complete",
            "metadata": metadata,
            "data_sha256": digest.hexdigest(),
            "scores": all_scores.result(),
            "by_type": {k: v.result() for k, v in by_type.items()},
            "recovery": recovery_points[-1],
            "learning_curve": curves,
            "cost": times.result(),
            "feedback_cost": learn_times.result(),
            "mining_cost": mining_times.result(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_seconds": time.perf_counter() - started,
            "learned_rules": [asdict(r) for r in learned],
            "associations": association_scores(
                learned,
                world.rules[sample["phase"]],
                world,
                sample["phase"],
                out / "samples.jsonl",
            ),
            "baserate_scores": base_scores.result(),
            "normalized_skill": (
                1 - all_scores.result()["brier"] / base_scores.result()["brier"]
                if base_scores.result()["brier"] > 0
                else None
            ),
            "relevance_weighted_brier": weighted_error / weighted_mass,
            "holdout_scores": holdout_scores.result(),
            "holdout_by_type": {
                kind: score.result()
                for kind, score in holdout_by_type.items()
                if score.count
            },
            "holdout_scored_predictions_by_type": {
                kind: score.count
                for kind, score in holdout_by_type.items()
                if score.count
            },
            "work_totals": work_totals,
            "fixed_update_budget": budget if system.startswith("budgeted") else None,
            "oracle_independent_max_error": oracle_max_error
            if learner is None
            else None,
        }
        if system in ("petta", "petta-link-first"):
            result["petta"] = {
                "unparsed": learner.unparsed,
                "last_verdicts": learner.raw,
                "answer_coverage_count": learner.coverage,
                "anticipated_negative_count": learner.anticipated,
                "live_rules": [asdict(r) for r in learner.live.values()],
            }
    except Exception as exc:
        (out / "error.json").write_text(
            json.dumps({"error": repr(exc), "completed_steps": times.count}, indent=2)
        )
        raise
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
