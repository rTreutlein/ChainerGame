"""Verify deterministic replays and oracle discrimination from retained artifacts."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
checks = []
for system in ('oracle','marginal','petta'):
    for kind in ('single','xor'):
        first = root/('petta-final' if system == 'petta' else 'matrix')/f'base-{system}-1-{kind}'
        again = root/('replay-final' if system == 'petta' else 'replay')/first.name
        a,b = [json.loads((p/'result.json').read_text()) for p in (first,again)]
        fields = ['data_sha256','scores','recovery','learned_rules']
        equal = all(a[key] == b[key] for key in fields)
        predictions = [[json.loads(line)['predicted'] for line in (p/'steps.jsonl').read_text().splitlines()]
                       for p in (first,again)]
        checks.append({'system':system,'kind':kind,'identical':equal and predictions[0] == predictions[1],
                       'checked':fields+['every prediction'], 'sources':[str(first),str(again)]})
gaps = []
for p in sorted((root/'final').glob('*-marginal-*/result.json')):
    a = json.loads(p.read_text())
    oracle = p.parent.name.replace('-marginal-', '-oracle-')
    b = json.loads((root/'final'/oracle/'result.json').read_text())
    for kind, scores in a['by_type'].items():
        if kind != 'single':
            gaps.append({'case':p.parent.name,'kind':kind,'gap':scores['brier']-b['by_type'][kind]['brier']})
result = {'replays':checks, 'all_replays_identical':all(c['identical'] for c in checks),
          'oracle_gaps_all_difficulties':gaps,
          'all_oracle_gaps_above_0_01':all(c['gap'] > .01 for c in gaps),
          'minimum_oracle_gap':min(c['gap'] for c in gaps)}
(root/'reproducibility.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('replays','oracle_gaps_all_difficulties')}))
if not result['all_replays_identical']:
    raise SystemExit(1)
