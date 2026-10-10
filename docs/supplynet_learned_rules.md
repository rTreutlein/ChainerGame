# SupplyNet with learned rules

Status: built (`supplynet run --learned-rules`), October 2026. Runner:
`benchmarks/learned/run_grid.sh`; raw runs, probes and `summary.json`:
`/nexus/Dev/OpenCog/bench/results/supplynet-learned/`.

## Question

In SupplyNet stages 1 to 3 as built, every rule reaches the reasoner with its
exact rates: `(Storm north $p) → (Blocked r1 $p)` with P(blocked | storm) =
0.8 and P(blocked | no storm) = 0.03, and so on. Only base rates (stage 1's
storm rate, the labels' frequencies) come from the labelled history. That
favours the reasoners whose home ground is exact inference over given
parameters (ProbLog, and PLN with certain CTVs) and leaves out what NARS is
built for: learning from experience.

The variant keeps the same worlds, observations, queries and scoring, and
hands every reasoner the rules' structure but not their rates. Each has to
learn the rates from the labelled history: 30 periods before the first round,
and each period of the game once it resolves.

## What the reasoners get

`--learned-rules` (stages 1–3; `src/supplynet/metta.py`, `rules(...,
learned=True)`) states every rule as before, with its rates replaced by one
weak prior on both branches:

    (: block-r1 (Implication (Storm north $p) (Blocked r1 $p)) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))

**Learned** (one rule per node, both CTV branches):

| rule | rates left to the data | per |
|---|---|---|
| `block-<route>`: `(Storm region $p) → (Blocked route $p)` | P(blocked \| storm), P(blocked \| no storm) | route, the fuel route `rf` included |
| `late-<shipment>`: `(Blocked route $p) → (Late shipment $p)` | P(late \| blocked), P(late \| open) | shipment |
| `persist-<region>` (stages 2–3): `(And (NextPeriod $p $t) (Storm region $p)) → (Storm region $t)` | persistence, onset | region |
| `persist-degraded-<site>` (stage 3): the same for `Degraded site` | persistence, onset | site |
| stage 1's storm rate (no rule: periods are independent) | P(storm) | region |

**Given** (unchanged):

- the stage 3 production rules (`fuel-arrives`, `fuelled`, `runs-*`), certain
  CTVs `(CTV (STV 1 1) (STV 0 1))`, and the declaration that `Producing`,
  `Fuelled` and `FuelArrived` are complete. They are recipes: a plant runs when
  it is powered, supplied and not degraded. An operator knows them, they are
  deterministic, and the simulator's production is exactly their least fixed
  point. Learning them would be learning logic from a handful of
  combinations, not rates;
- the structure: which parent each node has, the `NextPeriod` links, the
  transit times (lateness is keyed by departure period), and the observation
  facts' meaning;
- nothing else. The fuel stock (rate 0.4) and the plants' inputs are observed
  every period, so no rule states them.

The rates are learned **per node**, not per class: the reasoner is not told
which routes are exposed or sheltered, which regions are coastal or inland, or
which sites share a hazard, so each node learns from its own samples (about
3–10 positive samples per block rule in 30 periods). Pooling by class would
need the class as a fact and a rule form over it; it would make every learner
better and is left out.

**Samples.** A rule's sample is a labelled period of its consequent whose
antecedent is labelled; the `NextPeriod` link must hold in both branches, so a
persistence rule has no sample in the first history period. It counts towards
P(B | A) when the antecedent holds and towards P(B | not A) otherwise.
Unresolved periods are not samples, although some of their statements
(inspected routes, late shipments) are observed: they are a biased subset.

## Encodings

| reasoner | how it learns | code |
|---|---|---|
| **Exact, learned rates** | the exact posterior (`world.Knowledge`) under each node's rates counted from the labelled periods by Laplace's rule, (k + 1)/(n + 2) per branch, re-estimated before every round; counted from the simulator's labels (`world.counts`, `world.learned_rates`) | `backends.LearnedReferenceBackend` |
| **ProbLog** | the same translation as with given rules (`docs/problog_backend.md`), with each learned rule's two branch probabilities and stage 1's storm rate counted each round from the MeTTa facts of the labelled periods it holds (`problog_backend.Learner`: Laplace counts) | `ProblogBackend(learned=True)` |
| **PLN (PeTTaChainer)** | the rules as refined rules (`set-rule-refinement`, PeTTaChainer `docs/metta/hypothesis_rules.md`) with the weak prior, the labelled periods as facts; before a round's queries, when periods resolved since the last round, each learned rule is reviewed by its own `RuleTruth` query of 200 steps, which folds its instances | `PeTTaChainerBackend(learned=True)` |
| **NARS (ONA)** | each learned branch as the evidence NARS induction and revision would sum over the samples: frequency k/n, confidence n/(n+1), counted by the client (`world.counts`); base rates as before | `NarsBackend(learned=True)` |
| **Base rates** | each statement's Laplace frequency in the labelled periods (unchanged) | `PriorBackend` |

- **ProbLog equals the exact posterior under counted rates.** Laplace counts
  are the posterior mean of each rate under a uniform prior, which is exact
  for fully observed periods. `benchmarks/learned/validate.py` plays every
  stage and seed in lockstep: over 7487 keys (seeds 1–4, 30 rounds, all four
  stage configurations) the largest difference is 1.5e-10 (ProbLog clauses
  carry 10 significant digits). ProbLog counts from the MeTTa facts it
  receives and the reference from the simulator's labels, so this checks the
  counting as well as the inference. ProbLog's LFI was not used: for fully
  observed data it returns the maximum-likelihood counts k/n (no Laplace
  prior), after an EM loop over one compiled example per period.
- **PLN.** The review's budget matters. The first refined rule of a shape folds
  its instances within 20 steps; after that review, a second rule of the same
  shape keeps its prior at 20 steps and needs about 50
  (`probes/two_persistence_rules.metta`; reviewed alone, 20 suffice:
  `probes/b_alone.metta`). In the game a few block and late rules still kept
  their prior at 50 steps (`probes/probe_failed_reviews.py`), none at 200. A
  review stops when its fold is complete, so 200 steps cost no more than 50.
  Reviews take 0.7–1.2 s of a round's 1.1–2.3 s; the first round's review,
  which folds the whole history, takes 3–8 s. Evidence k is 5, as in the
  given-rules runs.
- **NARS learns from counts, not from the facts.** NARS's own route would be
  NAL-1 induction from facts sharing a subject: from `<h3 --> storm_north>`
  and `<h3 --> blocked_r1>`, `<storm_north --> blocked_r1>`, revised over the
  periods (the product form `<(north * h3) --> storm>` gives no such
  induction: ONA introduces variables over the region instead,
  `probes/ona_toy.nal`). With 30 periods of one region and one route, ONA
  learns P(blocked | storm) = 0.82 with confidence 0.92 from the facts alone
  (`probes/ona_toy2.nal`). With a whole stage-1 history (30 periods, 3
  regions, 10 routes, 21 shipments; 1474 statements, the complementary terms
  `nostorm_north`, `open_r1` included for the negative branches), ONA took
  9 min 12 s for one pass and left 37 of the 62 rule branches unanswered; the
  answered ones are far from the counts, e.g. P(blocked r5 | no storm) 0.95
  where the data say 0 of 20 (`probes/ona_probe.py`, `ona_s1.out`). A round
  would need that pass again (ONA cannot hold the history across rounds: 255
  atoms, 4096 concepts), at 9 minutes against the given-rules rounds' 35–70
  seconds, and still without the rules. So the backend states each branch as
  the evidence the induction would sum to, as it already does for base
  rates. NARS gets the learned rules as counts; PLN gets the raw facts.

## Scoring

As in stages 1–3: error is the mean absolute difference from the exact
posterior under the **true** rates, an unanswered query counts as 0.5;
Brier and log loss against the truth. The "Exact, learned rates" row is the
ceiling for any learner given the same history: its error is the learning
error alone, and a reasoner's distance above it is inference error (or worse
learning).

## Results

ChainerGame `supplynet-learned-rules` 9041392, PeTTaChainer 29c78d00 (frozen;
the same code as master 66a684f1) and 90264687 (master with the refined-rule
fixes), ProbLog 2.3.0 (d-DNNF), ONA v0.9.3. Seeds
1–4, 3 regions, 30 history periods, 30 rounds, inspection rate 0.25, budget
100, evidence k 5; PLN reviews 200 steps per rule; NARS 0 extra cycles,
frequency reading. "Given rules" is the same game with the rates given
(`/nexus/Dev/OpenCog/bench/results/rerun-2026-10-10`, PLN on 29c78d00 as
well). Error is to the exact posterior under the true rates; seconds are per
round and indicative (other agents' benchmarks shared the machine).

TABLE

**Reading.**

- **The learning cost is visible and the same for every exact learner.**
  Counting from 30 periods costs 0.02 (stage 1) to 0.07 (stages 2–3) of error
  against the true-rates posterior; ProbLog with counted rates sits exactly
  on that ceiling. Brier moves less (0.027 → 0.029 in stage 1, 0.154 → 0.162
  in stage 2): the learned rates are noisy per node but unbiased.
- **PLN loses most of its given-rules lead.** With given rates PLN is within
  0.008 of exact everywhere. With learned rules on 29c78d00 it is 0.03–0.06
  above the learned ceiling: halfway from the ceiling to the base rates in
  stage 2, a sixth of the way in stage 1 and a quarter in stage 3. The gap is
  inference with uncertain rules, not learning: PLN's P(B|A) is the counts',
  and with the counted rates stated as certain rules it is exact (see below).
  On 90264687 (the refined-rule fixes) stage 1 reaches the ceiling (0.024
  against 0.022) and stage 3 gains 0.012, but stage 2 stays at 0.121 against
  0.071: the storm queries that combine persistence with the window's
  evidence (`storm`, `previous_storm`) still drop evidence.
- **NARS stays at the base rates**, with learned rules as with given ones:
  its errors move by at most 0.01 between the two, and its answers stay within
  0.02 of the base rates' error, for the reasons in `docs/nars_backend.md`
  (abduction ignores P(B | not A), attention never reaches the second hop).
  Learned rates make little difference to a reasoner that hardly uses its
  rules.
- **Cost.** Learning adds no cost to ProbLog (counting is cheap). PLN's
  rounds take 1.1–2.1 s instead of 0.15–0.48 s, more than half of it
  the reviews; NARS's rounds are unchanged (30–70 s, input processing).

**More history.** The same grid with 120 labelled periods instead of 30
(`history120/summary.json`; NARS and ProbLog not rerun, ProbLog equals the
ceiling):

| stage | Exact, learned rates: 30 / 120 | PLN: 30 / 120 | Base rates: 30 / 120 | PLN s per round: 30 / 120 |
|---|---|---|---|---|
| Stage 1 | 0.022 / 0.014 | 0.055 / 0.046 | 0.239 / 0.229 | 1.09 / 2.26 |
| Stage 2 | 0.071 / 0.041 | 0.129 / 0.117 | 0.183 / 0.176 | 1.57 / 3.34 |
| Stage 3 timed | 0.067 / 0.036 | 0.108 / 0.091 | 0.229 / 0.215 | 2.08 / 4.49 |
| Stage 3 untimed | 0.064 / 0.034 | 0.104 / 0.087 | 0.214 / 0.205 | 2.11 / 4.48 |

The ceiling halves with four times the data; PLN improves by 0.01–0.02 and
its cost per round doubles. PLN's error is dominated by how it combines
uncertain rules, which more data does not fix.

## What limits PLN

Found on PeTTaChainer 29c78d00, then rechecked on 90264687 (master with the
refined-rule fixes; "After the fixes" below). Each repro is under
`/nexus/Dev/OpenCog/bench/results/supplynet-learned/probes/` (each MeTTa file
imports the frozen build; run it from the build's directory with
`.venv/bin/petta <file>`).

**PLN's inference is exact with the learned rates stated as certain rules.**
The round-1 ablation (`probes/probe_ablation.py`, `probes/ablation/`) states
the same Laplace rates the learned reference uses as certain CTVs, with no
refinement. Against the exact posterior under those rates (not the true
ones), seeds 1–4, round 1:

| stage | PLN, refined rules | PLN, counted rates as certain CTVs |
|---|---|---|
| 1 | 0.065 | 0.002 |
| 2 | 0.020 | 0.000 |
| 3 timed | 0.028 | 0.000 |

So the gap to the ceiling is not inference over given rates, which matches
the given-rules results. It is what happens when the rules are uncertain.

**1. Applications of one uncertain rule to different facts overlap, and the
merge keeps one view.** A rule certain in every branch carries no rule
evidence (PeTTaChainer c2fdc232); a learned rule does, so every view that
applies it shares that evidence. The prior-factored merge does not multiply
views that share evidence; it keeps the most confident. In stage 2 the
persistence view of a storm runs through the previous period's evidence,
which inverts the same block rules a later inspection inverts, so the later
inspection is dropped. Repro `persistence_drops_later_evidence.metta` (one
region, one route, one shipment, 30 labelled periods; t1 on time, t2 inspected
blocked):

| KB | P(Storm t2) | proof |
|---|---|---|
| exact, counted rates | 0.872 | |
| PLN, counted rates as certain CTVs (`_given`) | 0.872 | persistence factored with the t2 inversion |
| PLN, refined rules (rule truths within 0.07 of the counts) | 0.457 | `(by persist (conjunction next-t2 (factored-revision (by persist … storm-h30) (prior …) ((inverted block …) ((inverted late …) late-t1)))))`: the persistence chain alone; `blocked-t2` is not used |
| PLN, counted rates at confidence 0.9, no refinement (`_given09`) | 0.362 | the same drop |

The last row shows it is the rules' uncertainty, not the refinement. The
same mechanism keeps the storm errors flat with more data (below):
`previous_storm` 0.193 with 30 history periods, 0.189 with 120, while the
ceiling drops from 0.081 to 0.047. Seed 1, stage 2, round 2 in the game:
`Storm north t2` exact (learned rates) 0.829, PLN 0.357, the persistence
chain alone, with `(Blocked r1 t2)` inspected true and unused.

**2. A learned CTV's P(B | not A) comes from base rates, not from the
antecedent-false instances.** `rule_rates-29c78d00.json` compares every
rule's last reviewed CTV with the counts of the same periods (489 rules over
the 16 runs):

| rule kind | P(B\|A): PLN to ML counts | PLN to true | Laplace to true | P(B\|¬A): PLN to ML counts | PLN to true | Laplace to true |
|---|---|---|---|---|---|---|
| block | 0.003 | 0.087 | 0.080 | 0.039 | 0.036 | 0.028 |
| late | 0.005 | 0.071 | 0.077 | 0.044 | 0.040 | 0.026 |
| storm persistence | 0.006 | 0.089 | 0.086 | 0.063 | 0.077 | 0.076 |
| degradation persistence | 0.024 | 0.080 | 0.086 | 0.070 | 0.081 | 0.045 |

P(B | A) is the instance fold, within 0.003–0.024 of the maximum-likelihood
counts. P(B | not A) is off by 0.04–0.07, and worse than Laplace against the
truth. A rule whose antecedent never holds in the history has no instance
at all and keeps its prior on both branches: stage 3 untimed seed 3, the power
plant is never degraded in 30 periods, and `persist-degraded-power-plant`
stays (0.5, 0.02) / (0.5, 0.02), where the counts give 0 of 29 for its onset.

The two combine in stage 1. Repro `weak_rule_inversion.metta` (one route, a
storm rule and three shipments; the route blocked once in 30 periods; all
three shipments late at t1): exact P(Blocked r t1) under the counted rates
0.990, PLN with them as certain CTVs (`_given`) 0.977, PLN refined 0.036,
the base rate, with confidence 0.41. The learned late rules read P(late |
open) = 0.203 where the data say 1 of 29 (Laplace 0.065), and with one late
shipment instead of three (`_one`) PLN gives 0.076 against an exact 0.489:
three pieces of evidence give a lower belief than one.

**3. A review's budget depends on earlier reviews.** Repro
`two_persistence_rules.metta`: two refined rules of the same shape; the first
review folds within 20 steps, the second then keeps its prior at 20 and needs
50 (`b_alone.metta`: reviewed alone, 20 suffice). In the game 50 still left
some block and late rules at their prior; the runs use 200, which no review
exhausted (`probes/probe_failed_reviews.py`).

**4. The reviews' cost grows with the history.** With 120 history periods
instead of 30, the reviews take 1.60 s per round instead of 0.73 (stage 1)
and 2.8 s instead of 1.2 (stage 3); the round 2.3 s instead of 1.1 and 4.5 s
instead of 2.1. Each review folds all the rule's instances again, not only
the new period's.

### After the fixes (PeTTaChainer 90264687)

90264687 revises views that share only a rule instead of dropping one, and
learns a CTV's P(B | not A) from the antecedent-false instances. The client
already reviews the rules before it adds the round's observations, as the
fixes require (a rule reviewed with the round's facts already in its fold
reads its prior for them, PeTTaChainer's open problem 6).

- **Stage 1 reaches the ceiling:** error 0.024 against 0.022 (was 0.055);
  `blocked` 0.017 as the ceiling, `storm` 0.042 against 0.032.
- **P(B | not A)** is now as good as Laplace against the truth
  (`rule_rates-90264687.json`: block 0.022, late 0.027, storm persistence
  0.073, degradation 0.038; Laplace 0.028, 0.026, 0.076, 0.045).
- **Repros:** `weak_rule_inversion` with the reviews before the round's facts
  (`probes/90264687/weak_rule_inversion_reviewed_first.metta`) gives 0.998
  (exact 0.990), one late shipment 0.460 (exact 0.489).
  `persistence_drops_later_evidence_given09` (uncertain given rules) gives
  0.869 (exact 0.872).

**Still open on 90264687:**

1. **With refined rules, a later inspection is still dropped next to the
   persistence view.** `probes/90264687/persistence_drops_later_evidence_reviewed_first.metta`
   (reviews before t1 and t2's facts): P(Storm t2) 0.327 against 0.872, proof
   `(by persist (conjunction next-t2 (factored-revision (by persist … storm-h30)
   (prior …) ((inverted block …) ((inverted late …) late-t1)))))`, no
   `blocked-t2`. The same KB with uncertain given rules is fixed (0.869), so
   what still overlaps is the refined rules' evidence (their instance folds),
   not the rules themselves. In the grid this is the storm error of stages 2
   and 3: `storm` 0.131–0.145 and `previous_storm` 0.162–0.172 against the
   ceiling's 0.073–0.081, and stage 2 improves only from 0.129 to 0.121.
2. **A rule with no antecedent-true instance keeps its prior on both
   branches**, although its negative branch now has samples: stage 1 seed 3,
   route r10 is never blocked in the labelled periods, and the three
   `late-s10-*` rules stay (0.5, 0.02) / (0.5, 0.02) with 3 of 59
   antecedent-false samples late.
3. **The review budget still depends on earlier reviews**
   (`two_persistence_rules.metta` on 90264687: the second rule keeps its
   prior at 20 steps, folds at 50).
4. **Cost.** PLN's rounds take 0.95–2.87 s (29c78d00: 1.09–2.11 s); the
   reviews' share is unchanged (0.63–1.15 s per round).

`scripts/rerun_pln.sh <build>` reruns the PLN side on another build and adds
its rows to `summary.json`.

## Limits of the variant

- Rates are learned per node, so small nodes stay noisy: with 30 periods a
  route of a calm region may see 3 storms. Pooling by class is not offered.
- The rules' structure is complete and correct; no spurious or missing rule
  (that is SupplyNet stage 7, hypotheses).
- NARS's numbers use client-side counts. They show how NARS combines learned
  evidence, not how well ONA learns.
- PLN's review budget (200 steps per rule) is a workaround for the review
  cost depending on earlier reviews; a smaller one silently keeps priors.
- ProbLog's learning is counting outside ProbLog; ProbLog LFI would give
  maximum-likelihood rates (no Laplace prior) at a higher cost.

## Rerun

    cd ChainerGame-learned        # branch supplynet-learned-rules
    R=/nexus/Dev/OpenCog/bench/results/supplynet-learned
    PETTACHAINER_PATH=/nexus/Dev/OpenCog/bench/chainers/<build> JOBS=1 \
      benchmarks/learned/run_grid.sh $R/runs "learned-reference reference prior problog" "1 2 3 4"
    PETTACHAINER_PATH=/nexus/Dev/OpenCog/bench/chainers/<build> JOBS=1 \
      benchmarks/learned/run_grid.sh $R/pln-<build> "pettachainer" "1 2 3 4"
    PETTACHAINER_PATH=... JOBS=2 benchmarks/learned/run_grid.sh $R/runs "nars" "1 2 3 4" --nars-cycles-per-step 0
    PYTHONPATH=src .venv-problog/bin/python benchmarks/learned/validate.py $R/validation.json 30
    PYTHONPATH=src python benchmarks/learned/rule_rates.py $R/pln-<build> $R/rule_rates-<build>.json
    python3 benchmarks/learned/summarize.py $R $R/scripts/meta.json /nexus/Dev/OpenCog/bench/results/rerun-2026-10-10/summary.json

The PLN side alone, on a new build: `$R/scripts/rerun_pln.sh <build>`.
