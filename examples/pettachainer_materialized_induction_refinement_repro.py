"""Reproduce stale zero-budget reuse after materialized induction refinement.

Run with PeTTaChainer on ``PYTHONPATH``:

    PYTHONPATH=/path/to/PeTTaChainer \
      /path/to/PeTTaChainer/.venv/bin/python \
      examples/pettachainer_materialized_induction_refinement_repro.py

The initial materialized result is reused correctly.  A distinct alarm root
correctly remains unanswered.  After a newly resolved state changes the
induced FoldAll result, however, the refining materialization query returns an
updated answer while a zero-budget repeat still returns the old answer.
"""

import re

from pettachainer import PeTTaChainer


NORMAL_ROOT = (
    "(: $proof (Inheritance (PressureNormal old oxygen-scrubber) "
    "(SealLeak old oxygen-scrubber)) $tv)"
)
ALARM_ROOT = (
    "(: $proof (Inheritance (PressureAlarm old oxygen-scrubber) "
    "(SealLeak old oxygen-scrubber)) $tv)"
)
TV_RE = re.compile(r"\(STV ([^ ]+) ([^)]+)\)")


def state(name, leak, normal):
    terms = [
        f"(Member {name} (EquipmentState old oxygen-scrubber))",
        f"(Member {name} (SealLeak old oxygen-scrubber))",
        f"(Member {name} (PressureNormal old oxygen-scrubber))",
    ]
    statements = [
        f"(: {name}-equipment {terms[0]} (STV 1 1))",
        f"(: {name}-leak {terms[1]} (STV {leak} 1))",
        f"(: {name}-normal {terms[2]} (STV {normal} 1))",
    ]
    return statements, terms


def add_and_forward(handler, rows):
    statements = []
    terms = []
    for row in rows:
        row_statements, row_terms = state(*row)
        statements.extend(row_statements)
        terms.extend(row_terms)
    handler.add_atoms_no_check(statements)
    seeds = handler.select_facts(terms)
    handler.forward_chain(seeds, steps=2 * len(seeds))


def result_tvs(batch):
    result = []
    for proof in batch[0]:
        matches = TV_RE.findall(proof)
        result.append(tuple(map(float, matches[-1])))
    return result


def strongest(batch):
    values = result_tvs(batch)
    return max(values, key=lambda tv: (tv[0], tv[1])) if values else None


handler = PeTTaChainer()
add_and_forward(handler, [
    ("s1", 1, 0),
    ("s2", 0, 1),
    ("s3", 0, 1),
])

initial = handler.query_many_materialization([NORMAL_ROOT], steps=100)
initial_zero = handler.query_many_materialization([NORMAL_ROOT], steps=0)
alarm_zero = handler.query_many_materialization([ALARM_ROOT], steps=0)

assert initial[0], "the initial induction query produced no answer"
assert initial_zero == initial, "the initial materialized answer was not reused"
assert alarm_zero == [[]], "the distinct alarm root was incorrectly answered"

add_and_forward(handler, [("s4", 1, 1)])
refined = handler.query_many_materialization([NORMAL_ROOT], steps=100)
refined_zero = handler.query_many_materialization([NORMAL_ROOT], steps=0)

summary = {
    "initial": strongest(initial),
    "initial_at_zero_budget": strongest(initial_zero),
    "alarm_at_zero_budget": strongest(alarm_zero),
    "refined": strongest(refined),
    "refined_at_zero_budget": strongest(refined_zero),
}
print(summary)

assert refined[0], "refinement turned the root unanswered"
assert strongest(refined) != strongest(initial), (
    "the new resolved state did not produce a refined result"
)
assert strongest(refined_zero) == strongest(refined), (
    "the refining query returned an updated answer, but the next zero-budget "
    "query reused the old materialized answer"
)
