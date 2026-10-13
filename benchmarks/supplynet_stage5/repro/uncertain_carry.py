"""Cold shared-ancestor conjunction through a positive-STV Not alias.

Each case gets a fresh process: querying other heads first can change the
answer. D-now=0 checks certainty; independent E and direct Not are controls.
The larger original production graph retains the negative-carry probe.
Run the parent through capped; its serial children inherit that admission.
"""
import json
import subprocess
import sys


def probe(case):
    from pettachainer import PeTTaChainer
    c=PeTTaChainer()
    p,variant,budget=case['p'],case['variant'],case['budget']
    fuel_rate=case.get('fuel_rate',1)
    if variant=='negative-carry' or case.get('graph')=='production':
        for head in ('Producing','Fuel','Operational'): c.set_complete_predicate(head)
        c.set_positive_only_predicate('Service')
        ancestor='E' if variant=='independent' else 'D'
        operand='(Not (D later))' if variant=='direct-not' else '(Operational mine later)'
        body=f'(And ({ancestor} now) (Service now))' if case.get('compound') else f'({ancestor} now)'
        atoms=[
            f'(: d (D now) (STV {p} 1))',
            '(: seed (Seed now) (STV 1 1))',
            '(: service (Service now) (STV 1 1))',
            '(: power (Available later) (STV 0.95 1))',
            f'(: (no_inverse carry) (Implication {body} ({ancestor} later)) (CTV (STV 0.75 1) (STV 0.08 1)))',
            '(: (no_inverse mine-now) (Implication (And (Seed now) (Not (D now)) (Service now)) (Producing mine now)) (CTV (STV 1 1) (STV 0 1)))',
            f'(: (no_inverse fuel) (Implication (Producing mine now) (Fuel later)) (STV {fuel_rate} 1))',
            '(: (no_inverse power-later) (Implication (And (Fuel later) (Available later)) (Producing power later)) (CTV (STV 1 1) (STV 0 1)))',
            f'(: (no_inverse operational) (Implication (Not ({ancestor} later)) (Operational mine later)) (STV 1 1))',
            f'(: (no_inverse mine-later) (Implication (And (Producing power later) {operand}) (Producing mine later)) (CTV (STV 1 1) (STV 0 1)))',
        ]
        if ancestor=='E': atoms.insert(1,f'(: e (E now) (STV {p} 1))')
        if case.get('assumed'): atoms.remove('(: service (Service now) (STV 1 1))')
        c.add_atoms_no_check(atoms)
        goals=['(And (Not (D now)) (Not (D later)))','(And (Producing power later) (Not (D later)))','(Producing mine later)']
        exact=[.46,.437,.437]
        if case.get('graph')=='production':
            goals=['(Producing mine later)']
            exact=[.95*fuel_rate*(1-p)*(1-(p*.75+(1-p)*.08)) if ancestor=='E' else .95*fuel_rate*(1-p)*.92]
    else:
        for head in ('Fuel','Operational'): c.set_complete_predicate(head)
        ancestor='E' if variant=='independent' else 'D'
        operand='(Not (D later))' if variant=='direct-not' else '(Operational mine later)'
        atoms=[f'(: d (D now) (STV {p} 1))',
               f'(: (no_inverse carry) (Implication ({ancestor} now) ({ancestor} later)) (CTV (STV 0.75 1) (STV 0.08 1)))',
               f'(: (no_inverse fuel) (Implication (Not (D now)) (Fuel later)) (STV {fuel_rate} 1))',
               f'(: (no_inverse operational) (Implication (Not ({ancestor} later)) (Operational mine later)) (STV 1 1))']
        if ancestor=='E': atoms.insert(1,f'(: e (E now) (STV {p} 1))')
        if case.get('fuel_ctv'):
            atoms[2 if ancestor=='D' else 3] = f'(: (no_inverse fuel) (Implication (D now) (Fuel later)) (CTV (STV 0 1) (STV {fuel_rate} 1)))'
        c.add_atoms_no_check(atoms)
        goals=[f'(And (Fuel later) {operand})']
        exact=[fuel_rate*(1-p)*(1-(p*.75+(1-p)*.08)) if ancestor=='E' else fuel_rate*(1-p)*.92]
    queries=[f'(: $p (Assuming (Service now) {goal}) $tv)' if case.get('assumed') else f'(: $p {goal} $tv)' for goal in goals]
    rows=c.query_many_refs(queries,steps=budget*len(goals))
    print(json.dumps({**case,'goals':goals,'exact':exact,'strengths':[[float(a.tv[1]) for a in row] for row in rows]}),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1:
        probe(json.loads(sys.argv[1]))
    else:
        cases=[{'p':p,'variant':v,'budget':b} for p in (0,.5) for v in ('shared-alias','independent','direct-not') for b in (400,2000)]
        cases += [{'p':.5,'variant':'negative-carry','budget':b,'compound':compound} for compound in (False,True) for b in (400,2000)]
        cases += [{'p':p,'variant':v,'budget':b,'graph':'production'} for p in (0,.5) for v in ('shared-alias','independent','direct-not') for b in (400,2000)]
        # Nontrivial fuel probability prevents the certain-D conjunction
        # from simplifying to its other conjunct.
        cases += [{'p':p,'variant':v,'budget':b,'fuel_rate':.95} for p in (0,.5) for v in ('shared-alias','independent','direct-not') for b in (400,2000)]
        cases += [{'p':p,'variant':v,'budget':b,'fuel_rate':.95,'fuel_ctv':True} for p in (0,.5) for v in ('shared-alias','independent','direct-not') for b in (400,2000)]
        for case in cases:
            subprocess.run([sys.executable,__file__,json.dumps(case)],check=True,timeout=120)
