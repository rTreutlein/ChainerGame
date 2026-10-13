"""Shared uncertain degradation through production and next-period carry.

The direct negative carry control is .5*.92=.46. Production now supplies
later power with .95 availability, so later mine output is .46*.95=.437.
Compare a single-condition carry with the game's in-service conjunction,
both directly and under an assumed service fact. No chainer files change.
"""
from pettachainer import PeTTaChainer

for compound in (False, True):
    for assumed in (False, True):
        c=PeTTaChainer()
        for head in ('Producing','Fuel','Operational'):
            c.set_complete_predicate(head)
        c.set_positive_only_predicate('Service')
        antecedent='(And (D now) (Service now))' if compound else '(D now)'
        atoms=[
            '(: d (D now) (STV 0.5 1))',
            '(: seed (Seed now) (STV 1 1))',
            '(: power (Available later) (STV 0.95 1))',
            f'(: (no_inverse carry) (Implication {antecedent} (D later)) (CTV (STV 0.75 1) (STV 0.08 1)))',
            '(: (no_inverse mine-now) (Implication (And (Seed now) (Not (D now)) (Service now)) (Producing mine now)) (CTV (STV 1 1) (STV 0 1)))',
            '(: (no_inverse fuel) (Implication (Producing mine now) (Fuel later)) (STV 1 1))',
            '(: (no_inverse power-later) (Implication (And (Fuel later) (Available later)) (Producing power later)) (CTV (STV 1 1) (STV 0 1)))',
            '(: (no_inverse operational) (Implication (Not (D later)) (Operational mine later)) (STV 1 1))',
            '(: (no_inverse mine-later) (Implication (And (Producing power later) (Operational mine later)) (Producing mine later)) (CTV (STV 1 1) (STV 0 1)))',
        ]
        if not assumed: atoms.append('(: service (Service now) (STV 1 1))')
        c.add_atoms_no_check(atoms)
        goals=[('(And (Not (D now)) (Not (D later)))',.46),
               ('(And (Producing power later) (Not (D later)))',.437),
               ('(Producing mine later)',.437)]
        for budget in (400,2000):
            queries=[f'(: $p {"(Assuming (Service now) "+goal+")" if assumed else goal} $tv)' for goal,exact in goals]
            answers=c.query_many_refs(queries,steps=budget*len(queries))
            for (goal,exact),row in zip(goals,answers):
                print('compound',compound,'assumed',assumed,'budget',budget,'exact',exact,goal,row,flush=True)
