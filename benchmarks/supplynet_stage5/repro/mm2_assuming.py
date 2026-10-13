"""Use mm2's Python 3.13 venv and its python/ on PYTHONPATH."""
from mm2_chainer.petta import PeTTaChainer

c = PeTTaChainer()
c.add_atoms_no_check(["(: empty (Outcome now Empty 0) (STV 1 1))"])
for goal in ["(Outcome now Empty 0)", "(Assuming (Decision now) (Outcome now Empty 0))"]:
    print(goal, c.query_many_refs([f"(: $p {goal} $tv)"], steps=400), flush=True)
for setting in ('set_complete_predicate','set_positive_only_predicate'):
    print(setting,'available',hasattr(c,setting),flush=True)
