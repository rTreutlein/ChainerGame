"""ONA's matching-to-sample tasks (misc/Python/conditioning.py and
identitymatching.py) with a pluggable agent, so ONA and PeTTaChainer play
the same trials.

The protocol copies ONA's scripts: the same conditions, phases, block
counts, feedback, cycles between trials and the same `random.seed(seed)` /
`random.sample` calls, so a seed gives the trial order of the original
script. Agents draw their own randomness from a separate generator.

A trial shows a sample and a left and a right comparison stimulus; the agent
is asked for G and may execute ^left or ^right (or nothing). In feedback
phases a correct operation is followed by `G. :|:`, a wrong one by
`G. :|: {0.0 0.9}`, no operation by nothing. A block's score is the share
of its four trials answered with the expected operation; no operation counts
as wrong.

Tasks:
- conditioning: ONA's conditioning.py (arbitrary mapping A1->B1, A2->B2).
- identitymatching: ONA's identitymatching.py (A1->A1, A2->A2, then a test
  with novel stimuli A3, A4 and no feedback: generalised identity).
- reversal: an extension, not in ONA's suite: conditioning, then the mapping
  is reversed (A1->B2, A2->B1) for 20 feedback blocks and tested again.
- rereversal: reversal, then the original mapping returns for 20 feedback
  blocks and is tested again: a regime that recurs.
"""
import argparse
import itertools
import json
import os
import random
import re
import sys
import time

COND = [("A1", "B1", "B2", "^left"), ("A1", "B2", "B1", "^right"),
        ("A2", "B1", "B2", "^right"), ("A2", "B2", "B1", "^left")]
COND_REV = [(s, l, r, "^right" if op == "^left" else "^left") for s, l, r, op in COND]
IDENT = [("A1", "A1", "A2", "^left"), ("A1", "A2", "A1", "^right"),
         ("A2", "A1", "A2", "^right"), ("A2", "A2", "A1", "^left")]
IDENT_NOVEL = [("A3", "A3", "A4", "^left"), ("A3", "A4", "A3", "^right"),
               ("A4", "A3", "A4", "^right"), ("A4", "A4", "A3", "^left")]

# ONA's scripts ask these questions after the baseline and after every
# training block; asking them can change ONA's attention, so its agent asks
# them at the same points. The PLN agent ignores them.
COND_QUESTIONS = ["<((<A1 --> [sample]> &/ <B1 --> [left]>) &/ ^left) =/> G>?",
                  "<((<A1 --> [sample]> &/ <B1 --> [right]>) &/ ^right) =/> G>?",
                  "<((<A2 --> [sample]> &/ <B2 --> [left]>) &/ ^left) =/> G>?",
                  "<((<A2 --> [sample]> &/ <B2 --> [right]>) &/ ^right) =/> G>?"]
IDENT_QUESTIONS = ["<((<A1 --> [sample]> &/ <A1 --> [left]>) &/ ^left) =/> G>?",
                   "<((<A1 --> [sample]> &/ <A1 --> [right]>) &/ ^right) =/> G>?",
                   "<((<A2 --> [sample]> &/ <A2 --> [left]>) &/ ^left) =/> G>?",
                   "<((<A2 --> [sample]> &/ <A2 --> [right]>) &/ ^right) =/> G>?",
                   "<((<#1 --> [sample]> &/ <#1 --> [left]>) &/ ^left) =/> G>?",
                   "<((<#1 --> [sample]> &/ <#1 --> [right]>) &/ ^right) =/> G>?"]

# phase: (name, blocks, conditions, feedback, cycles after a trial,
#         questions after the phase's blocks: "each" | "end" | None, scored)
TASKS = {
    "conditioning": [("baseline", 10, COND, False, 100, "end", False),
                     ("train", 20, COND, True, 100, "each", False),
                     ("test", 4, COND, False, 100, None, True)],
    "identitymatching": [("baseline", 10, IDENT, False, 50, "end", False),
                         ("train", 10, IDENT, True, 50, "each", False),
                         ("test", 4, IDENT, False, 50, "each", True),
                         ("novel", 4, IDENT_NOVEL, False, 50, "each", True)],
    "reversal": [("baseline", 10, COND, False, 100, "end", False),
                 ("train", 20, COND, True, 100, "each", False),
                 ("test", 4, COND, False, 100, None, True),
                 ("reverse", 20, COND_REV, True, 100, "each", False),
                 ("retest", 4, COND_REV, False, 100, None, True)],
    "rereversal": [("baseline", 10, COND, False, 100, "end", False),
                   ("train", 20, COND, True, 100, "each", False),
                   ("test", 4, COND, False, 100, None, True),
                   ("reverse", 20, COND_REV, True, 100, "each", False),
                   ("retest", 4, COND_REV, False, 100, None, True),
                   ("rereverse", 20, COND, True, 100, "each", False),
                   ("final", 4, COND, False, 100, None, True)],
}
QUESTIONS = {"conditioning": COND_QUESTIONS, "identitymatching": IDENT_QUESTIONS,
             "reversal": COND_QUESTIONS, "rereversal": COND_QUESTIONS}


class ONAAgent:
    """ONA through its own misc/Python/NAR.py, with the scripts' settings."""
    name = "ona"

    def __init__(self, ona_dir, seed):
        os.chdir(os.path.join(ona_dir, "misc", "Python"))
        sys.path.insert(0, os.getcwd())
        import NAR  # spawns ./../../NAR shell
        self.nar = NAR
        for line in ["*volume=0", "*babblingops=2", "*motorbabbling=0.2",
                     "*setopname 1 ^left", "*setopname 2 ^right"]:
            NAR.AddInput(line, Print=False)

    def trial(self, sample, left, right):
        self.nar.AddInput(f"<{sample} --> [sample]>. :|:", Print=False)
        self.nar.AddInput(f"<{left} --> [left]>. :|:", Print=False)
        self.nar.AddInput(f"<{right} --> [right]>. :|:", Print=False)
        ex = self.nar.AddInput("G! :|:", Print=False)["executions"]
        return ex[0]["operator"] if ex else None

    def feedback(self, op, success):
        self.nar.AddInput("G. :|:" if success else "G. :|: {0.0 0.9}", Print=False)

    def wait(self, cycles):
        self.nar.AddInput(str(cycles), Print=False)

    def ask(self, questions):
        for q in questions:
            self.nar.AddInput(q, Print=False)


def parse_tvs(answers):
    out = []
    for a in answers:
        m = re.search(r"\((?:STV) ([-0-9.e]+) ([-0-9.e]+)\)\)\s*$", a)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


class PLNAgent:
    """PeTTaChainer learning P(G | context, operation) from its own trials.

    Each trial t stores (Sample s t), (LeftStim l t), (RightStim r t); an
    executed operation's outcome is stored as (Reward op t), (STV 1 1) after
    `G. :|:` and (STV 0 1) after the negative feedback. Nothing is stored
    without feedback (open world).

    Hypotheses: when an operation gets its first outcome in a context,
    conjunctions of the context's features become refined rules
    (Implication <features> (Reward op $t)) with the weak prior
    (STV 0.5 0.01); two features with the same stimulus also give the
    relational form with a shared variable, e.g.
    (And (Sample $x $t) (LeftStim $x $t)). ONA likewise forms a hypothesis
    from an experienced sequence. Creating them for both operations when a
    context is first seen works too since PeTTaChainer pln-gaps (a refined
    rule reviewed before its outcomes no longer keeps its prior), but scores
    lower (identity matching 0.847 against 0.956, 10 seeds) and takes twice
    as long: the hypotheses never tried stay at their prior and join every
    review and merge. The data decides their strengths
    (set-rule-refinement); no rate is handed in. Evidence k = 1, NARS's
    evidential horizon. Hypothesis spaces:
    - all: every single feature and every pair of features (ONA's sequence
      hypotheses are up to three events long);
    - sample-pairs: the sample with one comparison stimulus (the pairs
      ONA's scripts ask about).

    With ``mined`` (PeTTaChainer ``set-hypothesis-mining``) the harness
    creates no hypotheses: the knowledge base proposes them itself from the
    facts it is given (at most ``max_new`` per fact, ``max_live`` per
    consequent pattern), and the review is its ``review-hypotheses`` of the
    trial's ``(Reward $op t)``, which also drops the ones that do not predict.
    Every fact refers to its trial, and within a trial the stimuli come before
    the reward (``set_predicate_time``), so the miner relates only a trial's
    own facts, the stimuli as the causes.

    Decision: first the truths of the hypotheses that match the trial are
    read (their RuleTruth goals queried in one query-many), with budget only
    when one was never read, to fold its instances: PeTTaChainer takes each
    new outcome into the truths at its next query (docs/metta/
    hypothesis_rules.md, "Instance tallies"), so a read needs no budget. The
    desire of op is then the expectation c(f - 0.5) + 0.5 of
    the best-confidence answer to (Reward op t), the merged application of
    the hypotheses, read as terms (query-many-refs) without printing its
    proof. Each operation has its own query: asking both in one query-many
    shares their budget and changed decisions in 2 of 10 seeds (reversal
    seed 5 0.906 -> 0.781). The choice copies ONA's Decision_Suggest: with probability
    0.2 babble a random operation unless the best desire exceeds 0.55;
    otherwise execute the best operation when its desire is at least 0.501,
    else nothing.

    Outcome evidence (``outcome_evidence``). The agent sees only the chosen
    operation's outcome, so the rewards stored for an operation come from the
    trials where the agent expected it to pay, and an operation's population
    prior (the answer to (Reward op t) when no hypothesis matches the trial)
    is biased by the agent's choices. In these tasks exactly one of the two
    operations is correct in a trial, so the outcome of the one not chosen
    follows from the outcome of the chosen one:
    - none: nothing about the unchosen operation (open world);
    - rule: the knowledge base states the task structure, one rule per
      direction, (Implication (Reward right $t) (Reward left $t))
      (CTV (STV 0 1) (STV 1 1)) and the converse: the unchosen outcome is
      derived. The outcome it determines counts as a hypothesis sample, and
      in the population prior, at the next query, as a stored one does
      (PeTTaChainer docs/metta/runtime_design_assumptions.md, "Determined
      rows"). The rule also couples the current trial's two queries: each
      operation's answer merges its own hypotheses with the complement of the
      other's;
    - facts: the harness stores the unchosen outcome as a fact
      (Reward other t), (STV 1-s 1), and the unchosen operation gets its
      hypotheses for the context as the chosen one does.

    Desire (``desire``): ``all`` reads it from every answer to (Reward op t),
    the population prior included; ``matching`` reads it only through the
    hypotheses that match the trial, and an operation with none has desire
    0.5 without a query.

    Each trial's timing also splits the chainer's time into inference
    (activation, search and evaluation) and overhead (the rest of its calls:
    the Python layer, reading calls, query setup and teardown, results,
    storing facts), from PeTTaChainer.timing().
    """
    name = "pln"
    OPS = ("left", "right")

    def __init__(self, seed, steps, pettachainer_path=None, relational=True, space="sample-pairs", mined=None,
                 change_hazard=None, outcome_evidence="none", desire="all"):
        if pettachainer_path:
            sys.path.insert(0, pettachainer_path)
        from pettachainer import PeTTaChainer
        self.h = PeTTaChainer()
        self.h.set_evidence_confidence_k(1)
        self.h.set_rule_refinement(True)
        if change_hazard is not None:
            self.h.set_rule_change_hazard(change_hazard)
        for p in ("Sample", "LeftStim", "RightStim"):
            self.h.set_positive_only_predicate(p)
        self.mined = mined
        self.dropped = 0
        if mined:
            self.h.set_hypothesis_mining(*mined)
            for p in ("Sample", "LeftStim", "RightStim"):
                self.h.set_predicate_time(p, 2, 0)
            self.h.set_predicate_time("Reward", 2, 1)
        self.outcome_evidence = outcome_evidence
        self.desire_from = desire
        if outcome_evidence == "rule":
            self.h.add_atoms_no_check([f"(: one-of-{a} (Implication (Reward {a} $t) (Reward {b} $t)) (CTV (STV 0 1) (STV 1 1)))"
                                       for a, b in (self.OPS, self.OPS[::-1])])
        self.rng = random.Random(seed * 7919 + 1)
        self.steps = steps
        self.relational = relational
        self.space = space
        self.t = 0
        self.hyps = set()
        self.reviewed = set()
        self.timing = []

    def _antecedents(self, sample, left, right):
        feats = [("Sample", sample), ("LeftStim", left), ("RightStim", right)]
        pairs = list(itertools.combinations(feats, 2))
        if self.space == "all":
            ants = [f"({p} {a} $t)" for p, a in feats]
        else:
            ants = []
            pairs = pairs[:2]
        for (p1, a1), (p2, a2) in pairs:
            ants.append(f"(And ({p1} {a1} $t) ({p2} {a2} $t))")
            if self.relational and a1 == a2:
                ants.append(f"(And ({p1} $x $t) ({p2} $x $t))")
        return ants

    def trial(self, sample, left, right):
        self.t += 1
        t = self.t
        clock = self.h.timing()
        t0 = time.perf_counter()
        self.context = (sample, left, right)
        matching = [(a, op) for a in self._antecedents(sample, left, right) for op in self.OPS if (a, op) in self.hyps]
        steps = self.steps if any(h not in self.reviewed for h in matching) else 0
        self.reviewed.update(matching)
        review = [f"(: $p (RuleTruth (Implication {a} (Reward {op} $t))) $tv)" for a, op in matching]
        self.h.add_atoms_no_check([f"(: s{t} (Sample {sample} t{t}) (STV 1 1))",
                                   f"(: l{t} (LeftStim {left} t{t}) (STV 1 1))",
                                   f"(: r{t} (RightStim {right} t{t}) (STV 1 1))"])
        t1 = time.perf_counter()
        if self.mined:
            verdicts = self.h.review_hypotheses(f"(Reward $op t{t})", steps=self.steps)
            self.dropped += sum(v.endswith(" dropped)") for v in verdicts)
            review = verdicts
        elif review:
            print(f"  trial {t} reviewing {len(review)}", file=sys.stderr, flush=True)
            self.h.query_many_refs(review, steps=steps)
        t1r = time.perf_counter()
        desire = {}
        prior = []
        for op in self.OPS:
            if self.desire_from == "matching" and not any(o == op for _, o in matching):
                desire[op] = 0.5
                continue
            (rows,) = self.h.query_many_refs([f"(: $p (Reward {op} t{t}) $tv)"], steps=self.steps)
            answers = [(float(a.tv[1]), float(a.tv[2]), a.proof) for a in rows if a.tv[0] == "STV"]
            f, c, proof = max(answers, key=lambda x: x[1]) if answers else (0.5, 0.0, None)
            desire[op] = c * (f - 0.5) + 0.5
            if isinstance(proof, list) and proof[0] == "population-prior":
                prior.append(op)
        t2 = time.perf_counter()
        best = max(self.OPS, key=lambda o: (desire[o], self.rng.random()))
        babble = self.rng.choice(self.OPS) if self.rng.random() < 0.2 else None
        if babble and desire[best] <= 0.55:
            op = babble
        elif desire[best] >= 0.501:
            op = best
        else:
            op = None
        self.timing.append({"t": t, "add": t1 - t0, "review": t1r - t1, "decide": t2 - t1r, "reviewed": len(review),
                            "desire": {k: round(v, 4) for k, v in desire.items()}, "prior": prior})
        self._split(clock)
        print(f"  trial {t} {sample} {left} {right} review {t1r - t1:.2f}s decide {t2 - t1r:.2f}s", file=sys.stderr, flush=True)
        return None if op is None else "^" + op

    def feedback(self, op, success):
        t0 = time.perf_counter()
        chosen = op[1:]
        (other,) = [o for o in self.OPS if o != chosen]
        outcomes = [(chosen, int(success))] + ([(other, int(not success))] if self.outcome_evidence == "facts" else [])
        new = []
        for a in ([] if self.mined else self._antecedents(*self.context)):
            for o, _ in outcomes:
                if (a, o) not in self.hyps:
                    self.hyps.add((a, o))
                    new.append(f"(: h{len(self.hyps)} (Implication {a} (Reward {o} $t)) (STV 0.5 0.01))")
        clock = self.h.timing()
        self.h.add_atoms_no_check(new + [f"(: {n}{self.t} (Reward {o} t{self.t}) (STV {s} 1))" for n, (o, s) in zip("oc", outcomes)])
        self.timing[-1]["feedback"] = time.perf_counter() - t0
        self._split(clock)

    def _split(self, clock):
        """Add the chainer's inference and overhead seconds since `clock` to
        the current trial."""
        now = self.h.timing()
        for part in ("inference", "overhead"):
            self.timing[-1][part] = self.timing[-1].get(part, 0.0) + now[part] - clock[part]

    def wait(self, cycles):
        pass

    def ask(self, questions):
        pass


def run(task, agent, seed):
    random.seed(seed)
    seq = [0, 1, 2, 3]
    blocks = []
    for name, nblocks, conds, fb, cycles, questions, scored in TASKS[task]:
        for b in range(nblocks):
            correct = 0
            trials = []
            for i in random.sample(seq, 4):
                sample, left, right, expected = conds[i]
                t0 = time.perf_counter()
                op = agent.trial(sample, left, right)
                if op is not None and fb:
                    agent.feedback(op, op == expected)
                agent.wait(cycles)
                trials.append({"cond": [sample, left, right], "expected": expected, "op": op,
                               "wall": round(time.perf_counter() - t0, 4)})
                correct += op == expected
            if questions == "each":
                agent.ask(QUESTIONS[task])
            blocks.append({"phase": name, "block": b, "correct": correct / 4, "trials": trials})
            print(f"{name} {b} correct={correct / 4}", file=sys.stderr, flush=True)
        if questions == "end":
            agent.ask(QUESTIONS[task])
    scored = [b["correct"] for b in blocks if [p for p in TASKS[task] if p[0] == b["phase"]][0][6]]
    return {"task": task, "agent": agent.name, "seed": seed, "blocks": blocks,
            "score": sum(scored) / len(scored),
            "phase_scores": {p[0]: sum(b["correct"] for b in blocks if b["phase"] == p[0]) / p[1]
                             for p in TASKS[task]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--agent", choices=("ona", "pln"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=200, help="PLN query budget")
    ap.add_argument("--no-relational", action="store_true")
    ap.add_argument("--space", choices=("all", "sample-pairs"), default="sample-pairs")
    ap.add_argument("--mined", nargs=2, type=int, metavar=("MAX_NEW", "MAX_LIVE"),
                    help="the knowledge base mines its own hypotheses (set-hypothesis-mining)")
    ap.add_argument("--change-hazard", type=float, metavar="HAZARD",
                    help="prior probability that a sample starts a new regime of a refined rule's rate "
                         "(PeTTaChainer set-rule-change-hazard; default PeTTaChainer's, 0 keeps one regime)")
    ap.add_argument("--outcome-evidence", choices=("none", "rule", "facts"), default="none",
                    help="what the PLN agent learns about the operation it did not choose (PLNAgent docstring)")
    ap.add_argument("--desire", choices=("all", "matching"), default="all",
                    help="read desires from every answer, or only through hypotheses matching the trial")
    ap.add_argument("--ona-dir", default="/nexus/Dev/OpenCog/ONA")
    ap.add_argument("--pettachainer-path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mined and a.desire == "matching":
        ap.error("--desire matching needs harness hypotheses, not --mined")
    out = os.path.abspath(a.out)
    agent = (ONAAgent(a.ona_dir, a.seed) if a.agent == "ona"
             else PLNAgent(a.seed, a.steps, a.pettachainer_path, not a.no_relational, a.space, a.mined,
                      a.change_hazard, a.outcome_evidence, a.desire))
    t0 = time.perf_counter()
    res = run(a.task, agent, a.seed)
    res["wall"] = time.perf_counter() - t0
    if a.agent == "pln":
        res["steps"] = a.steps
        res["space"] = a.space
        res["hypotheses"] = len(agent.hyps)
        res["mined"] = a.mined
        res["change_hazard"] = a.change_hazard
        res["dropped"] = agent.dropped
        res["outcome_evidence"] = a.outcome_evidence
        res["desire"] = a.desire
        res["timing"] = agent.timing
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("task", "agent", "seed", "score", "phase_scores", "wall")}))


if __name__ == "__main__":
    main()
