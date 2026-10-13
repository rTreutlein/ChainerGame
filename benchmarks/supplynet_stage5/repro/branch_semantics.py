"""Two sufficient causes must not supply negative CTV conclusions."""
from pettachainer import PeTTaChainer
for positive in (False,True):
 c=PeTTaChainer()
 c.set_complete_predicate('Fuelled')
 tv='(STV 1 1)' if positive else '(CTV (STV 1 1) (STV 0 1))'
 c.add_atoms_no_check([f'(: (no_inverse stock) (Implication (Stock now) (Fuelled now)) {tv})',f'(: (no_inverse natural) (Implication (And (Unstocked now) (Arrived now)) (Fuelled now)) {tv})','(: absent (Arrived now) (STV 0 1))'])
 print('positive alternatives',positive,c.query_many_refs(['(: $p (Assuming (And (Stock now) (Not (Unstocked now))) (Fuelled now)) $tv)'],steps=400),flush=True)
