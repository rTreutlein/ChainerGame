"""Shared-cause conjunction: compare P(A and B)=.8 to independent .64."""
from pettachainer import PeTTaChainer
c=PeTTaChainer()
c.add_atoms_no_check(['(: a (A now) (STV 0.8 1))','(: (no_inverse b) (Implication (A now) (B now)) (CTV (STV 1 1) (STV 0 1)))'])
print('exact .8',c.query_many_refs(['(: $p (And (A now) (B now)) $tv)'],steps=400),flush=True)
c=PeTTaChainer()
c.add_atoms_no_check(['(: d (Degraded now) (STV 0.8 1))','(: (no_inverse carry) (Implication (Degraded now) (Degraded later)) (CTV (STV 0.75 1) (STV 0.05 1)))'])
for goal,exact in [('(And (Degraded now) (Degraded later))',.6),('(And (Not (Degraded now)) (Not (Degraded later)))',.19)]:
    print('exact',exact,goal,c.query_many_refs([f'(: $p {goal} $tv)'],steps=400),flush=True)
