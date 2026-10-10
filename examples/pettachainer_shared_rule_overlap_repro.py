"""Repro: two applications of one rule to different facts count as overlapping
evidence, so a merge keeps only the more confident view.

Run with PeTTaChainer's virtual environment (memory-capped, no display):

    PYTHONPATH=/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer \
      /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
      examples/pettachainer_shared_rule_overlap_repro.py < /dev/null

A system w is up when both of its parts are (rule ``sys``); sources s, r and
q each have a trust rule ``(Implication (Claims s (Up $n $t)) (Up $n $t))``
with confidence 0.9. s claims c1 up, r claims c2 up, and a third claim says w
is down. When q makes that claim, the answer for ``(Up w n)`` revises the
forward view through ``sys`` with q's view. When s makes it, s's view of w
is dropped: it shares ``trust-s`` with the forward view's proof of c1, though
the two rest on different claims. With the trust rules stated at confidence
1 both cases revise. At PeTTaChainer master b43ad3de:

    q: (merge/revision (by sys ...) (by trust-q kw)) (STV 0.568 0.965)
    s: (by sys (conjunction (by trust-s k1) (by trust-r k2)))  (STV 0.748 0.948)

The same holds for refined (hypothesis) trust rules, whose views carry their
samples: in ChainerGame's conflicting-sources stage a node's own claims are
replaced by a view through a system rule whenever both use one source's rule.
"""

import random
import sys

import pettachainer


def answer(claimant: str, confidence: str) -> str:
    handler = pettachainer.PeTTaChainer()
    handler.set_evidence_confidence_k(5)
    rng = random.Random(3)
    statements = ["(: sys (Implication (And (Up c1 $t) (Up c2 $t)) (Up w $t)) (CTV (STV 0.9 1) (STV 0.1 1)))"]
    statements += [
        f"(: trust-{s} (Implication (Claims {s} (Up $n $t)) (Up $n $t)) (CTV (STV 0.9 {confidence}) (STV 0.2 {confidence})))" for s in "srq"
    ]
    for i in range(30):  # labelled history, for base rates
        a, b = rng.random() < 0.7, rng.random() < 0.7
        w = rng.random() < (0.9 if a and b else 0.1)
        statements += [f"(: l1-{i} (Up c1 h{i}) (STV {int(a)} 1))", f"(: l2-{i} (Up c2 h{i}) (STV {int(b)} 1))", f"(: lw-{i} (Up w h{i}) (STV {int(w)} 1))"]
    handler.add_atoms_no_check(statements)
    handler.add_atoms_no_check(
        ["(: k1 (Claims s (Up c1 n)) (STV 1 1))", "(: k2 (Claims r (Up c2 n)) (STV 1 1))", f"(: kw (Claims {claimant} (Up w n)) (STV 0 1))"]
    )
    return handler.query_many(["(: $prf (Up w n) $tv)"], steps=200, timeout_sec=0)[0][0]


for confidence in ("0.9", "1"):
    for claimant in ("q", "s"):
        print(f"trust confidence {confidence}, w's claim by {claimant}:", answer(claimant, confidence), file=sys.stdout)
