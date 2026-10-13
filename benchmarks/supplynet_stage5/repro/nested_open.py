"""Small frozen-chainer probe: nested Assuming, including an open rule goal."""
from pettachainer import PeTTaChainer

c = PeTTaChainer()
c.add_atoms_no_check([
    "(: (no_inverse r) (Implication (And (A now) (B later)) (P later)) (CTV (STV 1 1) (STV 0 1)))",
    "(: (no_inverse open) (Implication (Assuming (B later) (P later)) (Choice b)) (CTV (STV 1 1) (STV 0 1)))",
])
for goal in [
    "(Assuming (And (A now) (B later)) (P later))",
    "(Assuming (A now) (Assuming (B later) (P later)))",
    "(Assuming (A now) (Choice $b))",
]:
    answers = c.query_many_refs([f"(: $p {goal} $tv)"], steps=400)
    print(goal, answers, flush=True)
