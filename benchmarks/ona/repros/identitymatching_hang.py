"""Stops PLN identity matching (seed 1, sample-pairs) at the review query of trial 72
and prints its statements. Running that query (remove the sys.exit) does not
return within 13 minutes:
    python identitymatching_hang.py identitymatching 1 72
"""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../src"))
import onabench.mts as m
task, seed, stop = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = m.PLNAgent(seed, 200, space="sample-pairs")
orig_qm = a.h.query_many
def qm(atoms, steps=100, timeout_sec=None):
    if a.t == stop:
        print("REVIEW AT", a.t, "steps", steps, file=sys.stderr)
        for x in atoms: print("   ", x, file=sys.stderr)
        print("HYPS", sorted(a.hyps), file=sys.stderr, flush=True)
        sys.exit(0)
    return orig_qm(atoms, steps=steps, timeout_sec=timeout_sec)
a.h.query_many = qm
m.run(task, a, seed)
