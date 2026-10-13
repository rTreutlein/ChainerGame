"""Current assumed production supplies the second period; cold versus warm."""
from pettachainer import PeTTaChainer
for warm in (False,True):
 c=PeTTaChainer()
 for head in ('Producing','Arrived'):c.set_complete_predicate(head)
 c.add_atoms_no_check(['(: (no_inverse mine) (Implication (Stock now) (Producing mine now)) (CTV (STV 1 1) (STV 0 1)))','(: (no_inverse fuel) (Implication (And (Producing mine now) (Open now)) (Arrived later)) (CTV (STV 1 1) (STV 0 1)))'])
 if warm:c.query_many_refs(['(: $p (Assuming (And (NoStock now) (Open now)) (Arrived later)) $tv)'],steps=400)
 print('warm',warm,c.query_many_refs(['(: $p (Assuming (And (Stock now) (Open now)) (Arrived later)) $tv)'],steps=400),flush=True)
