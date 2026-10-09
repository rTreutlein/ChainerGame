# CombiNet: how forward rules into one consequent should combine

Status: built (`src/combinet`), October 2026. Measures whether the remedies
of PeTTaChainer's `docs/metta/combination_hypotheses.md` (branch
`combination-hypotheses-design`) are worth building: what each one gains in
accuracy and what it costs.

## Question

Several rules `A_i → C` give several forward views of `(C x)`. PeTTaChainer
revises them, which averages them. The right combination depends on how the
A_i relate to C: the odds rule for independent signs of C, noisy-OR for
independent causes, revision for redundant evidence, anything for
interactions. The design note proposes two remedies:

- **mode learning:** per consequent, choose revision, odds or noisy-OR by the
  likelihood of the labelled history; cheap, cannot express interactions;
- **per-combination hypotheses:** refined rules such as
  `(Implication (And (A $x) (Not (B $x))) (C $x))` learned from the data by
  the instance fold; exact in the limit, up to 3ⁿ rules.

The benchmark answers: how large is the gap between revision and the exact
posterior per relation, how much of it each remedy closes, and at what cost
as n grows.

## World

- **Consequents.** `C_j`, each with n parent predicates `A_j_i` and one
  relation, drawn per consequent (`world.py`):
  - `signs`: C from a base rate in [0.2, 0.4], then A_i given C
    (P(A|C) in [0.55, 0.85], P(A|¬C) in [0.1, 0.35]);
  - `causes`: A_i independent (P in [0.2, 0.5]), C = noisy-OR of a leak
    (0.05–0.15) and weights 0.4–0.8;
  - `redundant`: A_i are copies of a latent L (P(L) in [0.3, 0.5]), each
    faithful with probability 0.85–0.95; P(C|L) 0.7–0.9, P(C|¬L) 0.05–0.2;
  - `interacting`: A_i independent, C logistic with main effects 1–2 and
    antagonistic pairs (A_1 with A_2, A_3 with A_4, ...; −2.5 to −4): each
    raises C alone, together they lower it. At n = 1 there is no pair.
- **Mixed KB.** By default one consequent per relation (`--relations`,
  `--per-relation`), all in one KB; results are reported per relation.
- **Cases.** An entity carries every predicate of every consequent. Each
  parent is observed with probability `--observe-rate` (0.8), else unknown.
  A labelled history (`--history`, 50) is given up front: observed parents
  plus the revealed C. Each round brings `--cases` (10) new entities; the
  reasoner states P(C_j | observed parents) for each; after scoring C is
  revealed and the cases join the history, so learners improve over rounds
  (`--rounds`, 10).
- **Exact posterior.** P(C | observed parents) under the true model, closed
  form for signs, causes and redundant, enumeration of the unknown parents for
  interacting; tested against brute-force enumeration of the joint.
- **What the reasoner gets.** One rule per parent,
  `(Implication (A_i $x) (C $x))` with `CTV (P(C|A_i), P(C|¬A_i))`, the history
  as facts, and the current cases' observed parents. The rates are the true
  marginals (`--rates true`, default: computed exactly from the model) or
  Laplace-smoothed frequencies in the initial history (`--rates learned`).
  Stated at confidence 1 (`--rule-confidence`), as given rates.

## Reasoners

| backend | what it is |
|---|---|
| `exact` | the posterior under the true model |
| `base-rate` | C's frequency in the history, ignoring parents (floor) |
| `revision` | mean of the observed parents' views, P(C) with none: what the chainer does |
| `odds` | prior odds × each view's likelihood ratio; exact for signs |
| `noisy-or` | prior survival 1−P(C) × each view's survival ratio; exact for noisy-OR causes |
| `mode` | per consequent, the one of revision/odds/noisy-OR with the highest history likelihood, re-chosen each round (ties keep revision) |
| `cells` | C's frequency among history cases matching every observed parent of the case, smoothed towards revision with 2 pseudo-cases: what an exact per-combination hypothesis converges to |
| `cell-mode` | the same, smoothed towards the learned mode |
| `pettachainer` | PeTTaChainer master, marginal rules only |
| `pettachainer-pairs` | + a hypothesis for every two observed parents with their values that occur in ≥ `--min-support` (5) history cases |
| `pettachainer-cells` | + a hypothesis for each full observed pattern (≥ 2 parents) of a history case with ≥ 5 instances |
| `pettachainer-cells-read` | `cells`, but where a review gave the case's exact cell a learned truth, the belief is that truth: the fold's learning, without the merge |

The odds and noisy-OR rules are both products of per-view ratios (in odds and
in survival space) that divide out the shared prior once per view; each is
exact for its relation from the marginals alone (`test_each_mode_is_exact_for_its_relation`).

### Hypotheses in the chainer

Written by the backend; no chainer change (`metta.py`, `backends.py`):

- `set-rule-refinement` on, `set-evidence-confidence-k 5`;
- `(: h-C1-1T-2F (Implication (And (A1_1 $x) (NotA1_2 $x)) (C1 $x)) (STV p 0.02))`
  with p the revision of the literals' views: no data, no change;
- before a round's queries, each hypothesis is reviewed by its own
  `(RuleTruth ...)` query with 20 steps (`--review-steps`); a folded
  hypothesis is counted as one fold;
- new combinations crossing the support threshold are added each round (the
  design note's "lazy, by occurrence" bound).

Two encoding choices were forced by the chainer (see Chainer problems):

- **Negative literals as complement facts.** `(Not (A $x))` in a hypothesis is
  never folded. The backend states each observation also as its complement
  under the same proof name, `(: o-A1_2-e5 (A1_2 e5) (STV 0 1))` and
  `(: o-A1_2-e5 (NotA1_2 e5) (STV 1 1))`, so a hypothesis over `NotA1_2`
  carries the same evidence as the marginal rule's view (`--form not` writes
  the design note's form instead).
- **Reviews one by one.** In one shared `query-many` batch the first goals
  take the budget and the later ones stay at their prior.

## Running

```sh
PYTHONPATH=src python -m combinet.cli run --backend mode --parents 3
# PeTTaChainer: its venv, its checkout importable, under a memory limit
ulimit -v 16000000; unset DISPLAY
PYTHONPATH=src PETTACHAINER_PYTHONPATH=/path/to/PeTTaChainer \
  /path/to/PeTTaChainer/.venv/bin/python -m combinet.cli sweep \
  --backends pettachainer,pettachainer-cells --parents 1,2,3 --seeds 1,2,3 \
  --history 30 --rounds 5 --cases 5 --jobs 2 --out results.jsonl
PYTHONPATH=src python -m combinet.cli table results.jsonl
```

`sweep` runs each (backend, n, seed) in its own process, so a run's time and
peak memory (`max_rss_mb`) are its own; crashed and timed-out runs are
recorded and listed. `error` below is the mean |belief − exact posterior|;
an unanswered key counts as 0.5 (coverage < 1).

## Results

October 9, 2026, PeTTaChainer master 28473546, Ryzen 9950X.

### The gap and the Python remedies

Default world (history 50 + 10 rounds of 10 cases, observe 0.8, true rates),
5 seeds, error against the exact posterior:

| n | revision | odds | noisy-or | mode | cells | cell-mode |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.041 | 0.041 |
| 2 | 0.076 | 0.048 | 0.057 | 0.045 | 0.067 | 0.067 |
| 3 | 0.108 | 0.052 | 0.071 | 0.046 | 0.079 | 0.078 |
| 5 | 0.139 | 0.066 | 0.088 | 0.060 | 0.115 | 0.091 |
| 10 | 0.150 | 0.070 | 0.099 | 0.064 | 0.145 | 0.069 |

Per relation, n = 3 and n = 10:

| n | backend | signs | causes | redundant | interacting |
|---|---|---|---|---|---|
| 3 | revision | 0.114 | 0.129 | 0.068 | 0.120 |
| 3 | mode | 0.023 | 0.002 | 0.071 | 0.089 |
| 3 | cells | 0.064 | 0.097 | 0.073 | 0.082 |
| 10 | revision | 0.257 | 0.071 | 0.120 | 0.151 |
| 10 | mode | 0.000 | 0.000 | 0.120 | 0.137 |
| 10 | cell-mode | 0.003 | 0.013 | 0.123 | 0.139 |

Brier: exact 0.157, revision 0.180, mode 0.163 at n = 3; exact 0.083,
revision 0.123, mode 0.094 at n = 10.

**Learning curve of the cells** (`--history`, then 10 rounds of 10; 5 seeds;
overall error, and interacting / redundant):

| n | history | mode | cells | cells: interacting | cells: redundant |
|---|---|---|---|---|---|
| 2 | 50 | 0.045 | 0.067 | 0.067 | 0.080 |
| 2 | 200 | 0.036 | 0.040 | 0.057 | 0.044 |
| 2 | 1000 | 0.033 | 0.018 | 0.021 | 0.021 |
| 2 | 5000 | 0.032 | 0.009 | 0.007 | 0.010 |
| 3 | 200 | 0.041 | 0.054 | 0.062 | 0.038 |
| 3 | 1000 | 0.039 | 0.025 | 0.026 | 0.021 |
| 3 | 5000 | 0.039 | 0.011 | 0.014 | 0.010 |
| 5 | 200 | 0.058 | 0.092 | 0.116 | 0.076 |
| 5 | 1000 | 0.056 | 0.049 | 0.061 | 0.039 |
| 5 | 5000 | 0.061 | 0.025 | 0.031 | 0.023 |
| 10 | 1000 | 0.062 | 0.111 | 0.137 | 0.084 |
| 10 | 5000 | 0.059 | 0.077 | 0.108 | 0.052 |

Mode stays at its floor (interacting ≈ 0.09–0.13, redundant ≈ 0.04–0.11:
noisy copies are none of the three modes) however long the history.

**Learned rates** (`--rates learned`, history 50): estimation error
compounds in the product rules. At n = 10, odds 0.178 and noisy-OR 0.240 are
worse than revision 0.166; mode (0.117) protects against it by choosing
revision where the products overshoot. With history 1000 the picture of the
true rates returns (mode 0.071, cell-mode 0.068).

### The chainer

Small world for the chainer (history 30 + 5 rounds of 5 cases, 4 consequents,
3 seeds), error against the exact posterior, Python references on the same
runs:

| n | revision | mode | cells | pettachainer | + pairs | + cells | + cells-read |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.064 | 0.036 | 0.036 | 0.036 | 0.036 |
| 2 | 0.069 | 0.059 | 0.092 | 0.077 | 0.079 | 0.079 | 0.064 |
| 3 | 0.095 | 0.046 | 0.094 | 0.096 | 0.100 | 0.103 | 0.104 |
| 5 | 0.141 | 0.068 | 0.135 | 0.140 | 0.144 | crash | crash (0.142 with ≤ 4 literals) |
| 10 | 0.154 | 0.069 | 0.148 | 0.148 | crash | crash | crash |

`pettachainer` alone is revision: per key it equals the mean of the views
(live test `test_marginal_rules_are_revised`); its error above revision at
n ≤ 2 is the cases with no observed parent, which it does not answer
(coverage 0.81 at n = 1, 0.95 at n = 2). With three or more views it differs
from the mean (see Chainer problems).

History 200 at n = 2 (5 rounds of 5, 3 seeds): revision 0.075, mode 0.035,
cells 0.041, pettachainer 0.081, + cells 0.080, + cells-read 0.071.

**Cost** (same small world; per round = review + query + adding the round's
facts):

| n | backend | hypotheses | folds/round | s/round | of it review | RSS MB |
|---|---|---|---|---|---|---|
| 1 | pettachainer | 0 | 0 | 0.04 | 0 | 162 |
| 2 | pettachainer | 0 | 0 | 0.17 | 0 | 202 |
| 3 | pettachainer | 0 | 0 | 0.28 | 0 | 209 |
| 5 | pettachainer | 0 | 0 | 0.44 | 0 | 307 |
| 10 | pettachainer | 0 | 0 | 1.05 | 0 | 537 |
| 2 | + pairs | 13 | 10 | 3.5 | 2.0 | 637 |
| 3 | + pairs | 35 | 28 | 12.3 | 8.5 | 1288 |
| 5 | + pairs | 123 | 99 | 68.5 | 61.3 | 2494 |
| 10 | + pairs | — | — | > 180 | | stack overflow (7.5 GB) / timeout (batch reviews) |
| 2 | + cells | 13 | 10 | 2.9 | 1.7 | 551 |
| 3 | + cells | 39 | 29 | 22.7 | 17.0 | 1590 |
| 5 | + cells, ≤ 4 literals | 47 | 30 | 95.8 | 81.0 | 2675 |
| 2, history 200 | + cells | 16 | 16 | 4.6 | 2.7 | 981 |

The default world (history 50, 10 rounds of 10) is out of reach with
hypotheses (first runs, with batch reviews): `+ pairs` at n = 3 took 18.7 s
per round and 4.7 GB, and `+ cells` overflowed the 7.5 GB Prolog stack in round 8.

## Chainer problems

Found while building the backend; reported, not fixed. Probes are small
Python scripts against master 28473546.

1. **Revision is order-dependent with confidence-1 rules.** Views from rules
   stated at confidence 1 come out at confidence 0.9999 and the merged view
   stays there, so each pairwise revision weights its two inputs equally: four
   views 0.9, 0.7, 0.5, 0.3 give 0.475 (((0.7+0.9)/2+0.5)/2+0.3)/2, not 0.6.
   At rule confidence 0.99 the answer is the mean. In the benchmark this moves
   `pettachainer` away from `revision` at n ≥ 3 (n = 10: 0.148 against 0.154,
   signs 0.231 against 0.254) only by luck of the order.
2. **`(Not (A $x))` hypotheses are never folded.** A review of
   `(RuleTruth (Implication (And (A $x) (Not (B $x))) (C $x)))` returns no
   answer at any budget; such a hypothesis keeps its prior.
3. **Hypotheses with three or more literals are never folded.** The review
   returns the axiom (confidence 0.02) at any budget, so at n ≥ 3 the cells
   that matter most, the full patterns, never learn.
4. **A refined rule with five conjuncts does not compile:** `compileadd` of
   `(Implication (And (A0 $x) ... (A4 $x)) (C $x))` with refinement on raises
   `Type mismatch: got : but expected ['|','STVType','CTVType','NoEvidenceType']`;
   four conjuncts, or five unrefined, compile. Every `cells` run at n ≥ 5
   crashed on it.
5. **Learned hypotheses do not reach the answers.** `+ pairs` and `+ cells`
   answer as `pettachainer` does (n = 2: 0.079 against 0.077; per relation
   within 0.005), while `+ cells-read`, reading the same reviews, moves
   (0.064). Probes show three behaviours, depending on the order the search
   finds views in:
   - the marginal views are merged first, at confidence 0.9999, and the
     hypothesis is revised in with negligible weight (the common case);
   - the hypothesis's view is found first and, its evidence containing the
     marginal views' facts, replaces them, which is the dominance the design
     note asks for (seen in small KBs);
   - after a review with a larger budget, applications use the hypothesis's
     axiom (0.5, confidence 0.02) instead of its learned truth, and that view
     replaces the marginal views: the answer becomes the prior. Same KB, review
     budget 40 in one batch: x1 → 0.939 (learned); 80: x1 → 0.500.
   A hypothesis whose antecedent is false for the case (conjunction of a
   false fact) also yields a view (strength 0.33, confidence 1e-6) that can
   replace the marginal views when it is the only one found.
6. **A shared review batch starves its later goals.** With 10 steps per goal
   in one `query-many` of 16 reviews, only the first consequent's 4
   hypotheses folded; with 60 steps, 8. Hence the one-by-one reviews.
7. **Folded truths drift from the instance frequencies.** Reviewed one by one
   at 20 steps, most cells read their frequency exactly (0.094 for 3/32), but
   some cells come back partial and wrong (a cell with frequency 0.474 over 19
   instances read 0.021 at confidence 0.29; 0.286 over 14 read 0.700); at 50
   steps values drift (0.286 → 0.442) at confidence 0.81, which is more than
   14 instances give at k = 5 (0.74): something beyond the instances is
   counted, probably the cross-feeding noted in `hypothesis_rules.md`. This
   is why `+ cells-read` (0.071 at history 200) trails the Python `cells`
   (0.041) that it should equal.
8. **Cost and memory.** Each fold costs about 0.1–0.6 s and grows with the
   history; reviews dominate the round (60 of 68 s at n = 5 with pairs). Peak
   memory grows from 0.2–0.3 GB (rules only) to 2.5–2.7 GB with 50–120 hypotheses
   (5.5 GB with batch reviews),
   and the Prolog stack (7.5 GB) overflows at n = 10 (pairs, cells) and in the
   default world (cells, 4 consequents × 3 parents, ~130 cases).

## Conclusion

- **The gap is real and grows with n.** Revision's error against the exact
  posterior is 0.08 at n = 2, 0.11 at n = 3 and 0.15 at n = 5–10 (Brier
  0.180 against 0.157 at n = 3). It is worst for signs (0.26 at n = 10) and
  causes, where it throws away the multiplication of evidence, and smallest
  for redundant evidence, where averaging is roughly right.
- **Mode learning closes most of it, cheaply.** It removes the signs and
  causes gap entirely (error ≤ 0.02 from history 50 on), keeps redundant at
  revision's level, and halves the overall error at every n (0.046 against
  0.108 at n = 3; 0.064 against 0.150 at n = 10). Its cost is a few
  log-likelihood sums per consequent per round and no rules. It fails safe:
  with learned rates, where the products overshoot, it picks revision.
  What it leaves is the interacting (≈ 0.09–0.14) and noisy-redundant
  (≈ 0.07–0.12) error.
- **Per-combination hypotheses pay only with much data and few parents.**
  The ideal (Python `cells`) beats mode only from ~200 labelled cases at
  n = 2, ~1000 at n = 3–5, and not at n = 10 within 5000 cases (0.077
  against 0.059: with 8 of 10 parents observed, few history cases match). Below
  that it is worse than mode; smoothing towards the mode (`cell-mode`) makes
  it never worse than mode, so the two remedies compose.
- **In the chainer today they cost far more than they return.** Even
  bounded by occurrence (≥ 5 instances), hypotheses multiply the round time by
  20–200 (0.28 → 12–23 s at n = 3; 0.44 → 69–96 s at n = 5), memory by 3–9,
  crash at n ≥ 5, and do not improve the answers (problems 2–5, 7).

**Recommendation.** Build mode learning per consequent first, as the design
note proposes. Treat per-combination hypotheses as a later, targeted
addition with these bounds: only pairs (or patterns) over at most 3–4
parents; only for consequents whose learned mode keeps mispredicting on the
history; only cells with enough instances to beat the mode (tens, not 5);
and smoothed towards the mode's value rather than revision's. Before any of
that, the chainer needs: folds of negative and ≥ 3-literal antecedents
(2, 3), refined rules of any arity (4), a dominance rule that makes a
confirmed hypothesis covering all observed parents replace the marginal
views and never lets an axiom or false-antecedent view do so (5), and
reviews whose cost does not grow with the history on every round (8).
