"""Verify final sweep provenance, complete streams and planning bounds."""
import hashlib
import json
from pathlib import Path
import sys
from sweep import jobs

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/nexus/Dev/OpenCog/bench/results/supplynet-stage5/partial-network')
source=hashlib.sha256()
for p in sorted((Path(__file__).resolve().parents[2]/'src/supplynet').glob('*.py')):
    source.update(p.name.encode()+p.read_bytes())
verified=[]
unavailable=[]
commits=set()
summaries={}
for name,python,args in jobs('all'):
    status=json.loads((root/(name+'.status.json')).read_text())
    assert status['fingerprint']['source_sha256']==source.hexdigest(),name
    commits.add(status['fingerprint']['chainer_commit'])
    expected=[python,'-m','supplynet.cli','run','--stage','5','--rounds','30','--problog-timeout','15','--stream',*args]
    assert status['fingerprint']['command']==expected,name
    if status['returncode']:
        assert name.startswith('mm2-'),(name,status['returncode'])
        unavailable.append(name)
        continue
    rows=[json.loads(line) for line in (root/(name+'.jsonl')).read_text().splitlines()]
    summary=json.loads((root/(name+'.json')).read_text())
    summaries[name]=summary
    assert rows[-1]==summary and summary['type']=='run-summary',name
    count=5 if name.startswith('growth-') else 30
    assert summary['rounds']==count and len(rows)==count+1,name
    assert [r['round'] for r in rows[:-1]]==list(range(1,count+1)),name
    assert all(r['type']=='round' and r['reference_seconds']>0 for r in rows[:-1]),name
    if summary['backend']=='pettachainer':
        assert all(r['candidate_pairs']<=summary['pair_cap'] for r in rows[:-1]),name
        assert all(r['context_bound']<=summary['pair_cap']+2 for r in rows[:-1]),name
    if summary['backend']=='problog':
        assert all(r.get('problog_reference_max_error',0)<1e-8 for r in rows[:-1]),name
    if summary.get('backend_unavailable'):unavailable.append(name)
    verified.append(name)
result={'type':'sweep-audit','source_sha256':source.hexdigest(),'cases':len(verified)+len(set(unavailable)-set(verified)),
        'successful':len(verified),'unavailable':unavailable,'verified_cases':verified,'chainer_commits':sorted(commits),
        'budget_actions_identical':all(len({tuple(summaries[f'pettachainer-extend-b{budget}-s{seed}']['actions']) for budget in (50,100,200,400)})==1 for seed in range(1,5))}
print(json.dumps(result,indent=2))
