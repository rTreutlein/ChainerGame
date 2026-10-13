"""Prequential evaluation with raw distributions, rules, curves and process cost."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import resource
import subprocess
import time

from .world import Config, World, serial_sample
from .systems import Marginal
from .scoring import RunningAverage, CostMeter, losses, recovery
from .trace import TraceAudit, target_requests


def revision(path):
    p = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return p.stdout.strip() or None


def source_hash():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def run(config, system, out, budget=80, max_new=4, max_live=8, adapter=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('world.json','steps.jsonl','result.json')):
        raise FileExistsError(f'refusing to overwrite benchmark evidence in {out}')
    world = World(config)
    metadata = world.metadata()
    metadata.update({'system': system, 'python': platform.python_version(), 'source_sha256': source_hash(),
                     'chainer_game_commit': revision(Path(__file__).parents[2]),
                     'budget': budget, 'max_new': max_new, 'max_live': max_live})
    if system == 'petta':
        from .petta import Petta
        learner = Petta(world.features, world.domains, budget, max_new, max_live)
        import pettachainer
        metadata['petta_path'] = str(Path(pettachainer.__file__).resolve())
        metadata['petta_commit'] = revision(Path(pettachainer.__file__).parents[1])
    elif system == 'marginal':
        learner = Marginal(world.features, world.domains)
    elif system == 'oracle':
        learner = None
    elif adapter:
        learner = adapter(world.features, world.domains)
    else:
        raise ValueError(f'unknown system {system}')
    (out/'world.json').write_text(json.dumps(metadata, indent=2)+'\n')
    all_scores = RunningAverage()
    times, learn_times = CostMeter(config.samples), CostMeter(config.samples)
    curves, audit = [], TraceAudit()
    previous_predictions, previous_labels, learned_targets = {}, {}, set()
    recovery_points = []
    by_type = {k: RunningAverage() for k in sorted(set(world.types.values()))}
    windows = {k: RunningAverage() for k in by_type}
    digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        with (out/'steps.jsonl').open('w') as raw, (out/'samples.jsonl').open('w') as samples_file:
            for sample in world.stream():
                serialized = json.dumps(serial_sample(sample), sort_keys=True)
                digest.update(serialized.encode())
                samples_file.write(serialized+'\n')
                requests = target_requests(previous_predictions, previous_labels, world.domains, learned_targets)
                t0 = time.perf_counter()
                if learner and hasattr(learner, 'set_targets'):
                    learner.set_targets(requests)
                predictions = learner.predict(sample['history']) if learner else world.probabilities(sample['history'], sample['phase'])
                t1 = time.perf_counter()
                step_events = learner.learn(sample['history'], sample['y']) if learner else []
                t2 = time.perf_counter()
                times.add(t2-t0)
                learn_times.add(t2-t1)
                for e in step_events:
                    audit.add(dict(e, step=sample['step']))
                previous_predictions, previous_labels = predictions, sample['y']
                scores = {t: losses(predictions[t], q) for t, q in sample['q'].items()}
                for t, score in scores.items():
                    by_type[world.types[t]].add(score)
                    windows[world.types[t]].add(score)
                    all_scores.add(score)
                row = {'step': sample['step'], 'phase': sample['phase'], 'predicted': predictions,
                       'exact': sample['q'], 'label': sample['y'], 'scores': scores,
                       'predict_seconds': t1-t0, 'learn_seconds': t2-t1, 'total_seconds': t2-t0,
                       'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'work_units': learner.work if learner else 0, 'events': step_events, 'target_requests': requests}
                raw.write(json.dumps(row)+'\n')
                raw.flush()
                n = sample['step']+1
                if n % max(8, config.samples//8) == 0 or n == config.samples:
                    learned = learner.rules() if learner else world.rules[sample['phase']]
                    learned_targets = {r.target for r in learned}
                    rec = recovery(learned, world.rules[sample['phase']], world.types)
                    point = {'samples': n, 'phase': sample['phase'], 'recovery': rec,
                             'window': {k: v.result() for k, v in windows.items()},
                             'cumulative': {k: v.result() for k, v in by_type.items()}}
                    curves.append(point)
                    recovery_points.append(rec)
                    windows = {k: RunningAverage() for k in by_type}
        learned = learner.rules() if learner else world.rules[sample['phase']]
        result = {'status': 'complete', 'metadata': metadata, 'data_sha256': digest.hexdigest(),
                  'scores': all_scores.result(), 'by_type': {k: v.result() for k, v in by_type.items()},
                  'recovery': recovery_points[-1], 'learning_curve': curves,
                  'cost': times.result(), 'feedback_cost': learn_times.result(),
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'elapsed_seconds': time.perf_counter()-started,
                  'learned_rules': [asdict(r) for r in learned], 'trace_audit': audit.result()}
        if system == 'petta':
            result['petta'] = {'unparsed': learner.unparsed, 'last_verdicts': learner.raw,
                               'answer_coverage_count': learner.coverage,
                               'anticipated_negative_count': learner.anticipated,
                               'live_rules': [asdict(r) for r in learner.live.values()]}
    except Exception as exc:
        (out/'error.json').write_text(json.dumps({'error': repr(exc), 'completed_steps': times.count}, indent=2))
        raise
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
