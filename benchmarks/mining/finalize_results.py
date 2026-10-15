"""Assemble immutable raw runs, report and replay verification in a fresh folder.

Usage: python benchmarks/mining/finalize_results.py RESULT_ROOT
Baseline streams are unchanged by the recovery extraction correction. Preserve
all original artifacts and combine them with the corrected live PeTTa runs.
"""
from pathlib import Path
import json
import shutil
import sys
from miningbench.report import report

root = Path(sys.argv[1])
final = root/'final'
final.mkdir(exist_ok=False)
provenance = []
for source in (root/'matrix', root/'petta-final'):
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not (folder/'status.json').exists():
            continue
        status = json.loads((folder/'status.json').read_text())
        if source.name == 'matrix' and status['system'] == 'petta':
            continue
        shutil.copytree(folder, final/folder.name)
        provenance.append({'output':folder.name, 'source':str(folder), 'system':status['system']})
results, failures = [], []
for p in sorted(final.glob('*/status.json')):
    status = json.loads(p.read_text())
    if status['returncode']:
        trace = p.parent/'steps.jsonl'
        status['completed_steps'] = sum(1 for _ in trace.open()) if trace.exists() else 0
        failures.append(status)
    else:
        results.append(json.loads((p.parent/'result.json').read_text()))
text, checks = report(results, failures)
(final/'report.md').write_text(text)
(final/'acceptance.json').write_text(json.dumps(checks,indent=2)+'\n')
(final/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
(final/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'complete':len(results),'incomplete':len(failures),'output':str(final)}))
