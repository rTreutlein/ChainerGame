"""Reports distinguish measured discrimination from acceptance expectations."""
from collections import defaultdict
import statistics


def report(results):
    groups = defaultdict(list)
    for r in results:
        c = r['metadata']['config']
        level = r['metadata'].get('level', 'run')
        for kind in r['by_type']:
            groups[(level, r['metadata']['system'], kind)].append(r)
    lines = ['# Pattern mining benchmark', '',
             '| Difficulty | System | Type | Seeds | Brier | Excess log loss | Precision | Recall | ms/step | Late/early | Peak MiB |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (level, system, kind), rows in sorted(groups.items()):
        mean = lambda f: statistics.fmean(f(r) for r in rows)
        lines.append(f'| {level} | {system} | {kind} | {len(rows)} | '
                     f'{mean(lambda r:r["by_type"][kind]["brier"]):.5f} | '
                     f'{mean(lambda r:r["by_type"][kind]["excess_log_loss"]):.5f} | '
                     f'{mean(lambda r:r["recovery"]["by_type"][kind]["precision"]):.3f} | '
                     f'{mean(lambda r:r["recovery"]["by_type"][kind]["recall"]):.3f} | '
                     f'{mean(lambda r:r["cost"]["mean_seconds"])*1000:.3f} | '
                     f'{mean(lambda r:r["cost"]["late_over_early"]):.2f} | '
                     f'{max(r["peak_rss_kib"] for r in rows)/1024:.1f} |')
    checks = acceptance(results)
    lines.extend(['', '## Acceptance', '', '| Check | Status | Evidence |', '|---|---|---|'])
    for c in checks:
        lines.append(f'| {c["check"]} | {c["status"]} | {c["evidence"]} |')
    lines.extend(['', 'Timing and RSS are measurements, not seed-deterministic values. Late/early is a diagnostic, not a complexity proof.',
                  'Brier is squared error against the exact distribution; excess log loss subtracts its entropy.', ''])
    return '\n'.join(lines), checks


def acceptance(results):
    checks = []
    oracles = [r for r in results if r['metadata']['system'] == 'oracle']
    checks.append({'check': 'oracle exact', 'status': 'PASS' if oracles and all(
        r['scores']['brier'] < 1e-15 and r['recovery']['all']['recall'] == 1 for r in oracles) else ('FAIL' if oracles else 'UNMEASURED'),
        'evidence': f'{len(oracles)} oracle runs'})
    pairs = defaultdict(dict)
    for r in results:
        m = r['metadata']
        c = m['config']
        key = (m.get('level', 'run'), c['seed'], tuple(c['kinds']), c['samples'], c['distractors'],
               c['targets'], c['strength'], c['observation_noise'], c['drift'])
        pairs[key][m['system']] = r
    for kind in sorted({k for r in results for k in r['by_type']} - {'single'}):
        gaps = [p['marginal']['by_type'][kind]['brier']-p['oracle']['by_type'][kind]['brier']
                for p in pairs.values() if 'oracle' in p and 'marginal' in p and kind in p['oracle']['by_type']
                and p['oracle']['metadata'].get('level', 'base') in ('base', 'run')]
        checks.append({'check': f'oracle beats marginal: {kind}',
                       'status': 'PASS' if gaps and min(gaps) > .01 else ('FAIL' if gaps else 'UNMEASURED'),
                       'evidence': f'min paired Brier gap {min(gaps):.5f} > 0.01' if gaps else 'no paired base runs'})
    xor = [r['recovery']['by_type']['xor']['recall'] for r in results
           if r['metadata']['system'] == 'marginal' and 'xor' in r['by_type']]
    checks.append({'check': 'marginal XOR recovery fails', 'status': 'PASS' if xor and max(xor) == 0 else ('FAIL' if xor else 'UNMEASURED'),
                   'evidence': f'max recall {max(xor) if xor else None}'})
    for system in sorted({r['metadata']['system'] for r in results}-{'oracle'}):
        base = {(r['metadata']['config']['seed'],tuple(r['metadata']['config']['kinds'])): r for r in results
                if r['metadata']['system'] == system and r['metadata'].get('level') == 'base'}
        for knob in ('distractors', 'strength', 'observation_noise', 'samples', 'drift', 'targets'):
            high = [r for r in results if r['metadata']['system'] == system and r['metadata'].get('level') == knob]
            changes = []
            for r in high:
                b = base.get((r['metadata']['config']['seed'],tuple(r['metadata']['config']['kinds'])))
                if not b:
                    continue
                # Noise is expected to worsen label prediction, but can make q easier
                # to approximate. Keep both columns in raw results; do not tune gates.
                metric = 'expected_brier' if knob in ('strength', 'observation_noise') else 'brier'
                delta = r['scores'][metric]-b['scores'][metric]
                cost_ratio = r['cost']['mean_seconds']/b['cost']['mean_seconds']
                changes.append((delta, cost_ratio))
            degraded = bool(changes) and (statistics.fmean(x[0] for x in changes) > .002 or
                knob in ('distractors', 'targets') and statistics.fmean(x[1] for x in changes) > 1.2)
            checks.append({'check': f'{system}: harder {knob}',
                           'status': 'PASS' if degraded else ('FAIL' if changes else 'UNMEASURED'),
                           'evidence': f'mean error delta {statistics.fmean(x[0] for x in changes):+.5f}; '
                                       f'cost ratio {statistics.fmean(x[1] for x in changes):.2f}' if changes else 'no matched sweep'})
    return checks
