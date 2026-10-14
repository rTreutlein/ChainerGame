# Planted temporal rule mining benchmark

`python -m miningbench run|sweep|report` evaluates prequential predictions and rule
recovery against generated ground truth. This is benchmark code. PeTTaChainer is
read through its shared interpreter; no upstream implementation is changed.

## Review corrections

1. **Sufficiency and association are separate.** A learned rule enters recovery
   only when its strength is at least its target's minimum planted strength in
   the observed outcome channel minus **0.10**. Boolean flip noise transforms
   strength to `eta+(1-2*eta)*s`; numeric uniform noise/failure uses
   `s*(1-eta)+(1-s*(1-eta))/K`. Thus a correctly calibrated .8 rule under
   20% flip noise is not mistakenly rejected against a latent strength of 1.
   Only that subset undergoes DNF consensus and absorption. Weak singles
   cannot erase a sufficient conjunction. Remaining associations receive exact
   conditional calibration scores, including their individual probabilities and
   squared errors. Entity-holdout association calibration uses the training
   population; novel-entity prediction is measured separately. The threshold is stated in the raw result, not tuned per run.
2. **Numeric truth uses observables.** Position and signed velocity are observed.
   Rules bind those two values and branch on their sum being below, within, or
   above the bounds. Numeric recovery measures equivalence over all 198 reachable
   position/velocity contexts, crediting correct grounded transitions as well as
   formulas. Random initial positions and velocities, a 32-unit domain, and
   random velocity interventions remove the old period-eight shortcut.
3. **Normalized skill replaces raw noisy-label error gates.** Population skill
   is `(E_baserate-E_system)/(E_baserate-E_oracle)`, calculated within each seed
   before taking paired differences. The independent oracle interprets planted
   rules rather than calling the generator's predictive-probability function.
4. **Two temporal arms.** `petta` uses observation-first/link-last and observed
   predicates. `petta-link-first` adds each incoming link before inputs, disables
   anticipation, and states all negative labels explicitly. Warmup follows the
   corresponding order. Neither arm filters proofs or repairs withdrawn rules.
5. **Cost and timeouts.** Long streams contain 1024 decisions. Cost growth is a
   fitted least-squares slope multiplied by stream length and divided by mean
   step cost. The fit excludes the first 32 steps so initial history ingestion
   cannot hide later growth; the overall mean still includes them. Ordinary runs
   allow 600 seconds. The CLI default allows 3600 for stress levels; the powered live rerun uses 1800 for distractor, three-target and
   numeric cases, and 3600 for the 1024-step deep-lag case. Incomplete runs retain
   partial observations and receive no fabricated full-stream score.
6. **Power.** Base 256, sample-count comparison 64, long 1024; ten paired seeds.
   Curves are recorded at 64, 256, and 1024 when reached. The 16-sample comparison
   has been removed. Reports use 2000 seed-level bootstrap resamples, fixed seed
   731, with 95% percentile intervals. Per-seed values are sorted before
   resampling so artifact collection order cannot change the interval. Differences
   resample paired seeds, not independent steps. Fewer than ten pairs cannot earn an acceptance pass. Difficulty averages use
   only type cohorts with ten paired seeds and the intersection of their seed
   sets; incomplete types are listed explicitly and cannot change the type mix
   from seed to seed.
7. **Directed design measured through behaviour.** Fixed-budget reference arms,
   relevance-weighted error, deep-lag work allocation, and novel-entity holdout
   replace the previous self-reported trace contract. `trace.py` is removed.
8. **Readability and evidence.** Scoring, reporting, and the temporal protocol have
   named operations and documented units. Historical runs remain immutable.
   `benchmarks/mining/analyze_review.py` collects the corrected results and raw
   behavioural evidence into a fresh report directory.

## Generator and exact truth

Each seed has independent RNG channels for targets, labels, and distractors.
Adding distractors or target copies preserves existing channels. Boolean inputs
are independent fair bits; relational inputs independently choose `At` and `Did`
entities. The generator keeps a bounded history. Default lags are 1–3; deep-lag
experiments expose distractor features through lag 12.

| Type | Mechanism |
|---|---|
| single | One positive literal |
| and2, and3, and4 | Two, three, or four positive literals |
| negation | `F0 AND NOT F1` |
| lag | `F0@1 AND F1@2 AND F2@3` |
| xor | `(F0 AND NOT F1) OR (NOT F0 AND F1)` |
| or | `(F0 AND F1) OR (F2 AND F3)`; overlapping cases use noisy OR |
| binding | `At($x) AND Did($x)` with one shared entity variable |
| numeric | `Pos($x1) AND Vel($v1)`; reflected motion with boundary exceptions |

Active Boolean rules fire independently with configured strength `s`:
`p = 1 - product(1-s_r)`. Outcome-flip noise `eta` changes that to
`eta + (1-2*eta)*p`. Inputs themselves remain observed without corruption.

For numeric worlds, `x` is in 0–32 and velocity is one of ±1, ±2, ±3. The next
position is `z=x+v` inside the domain, `-z` below zero, or `64-z` above 32.
Twenty percent of input steps redraw velocity; otherwise velocity persists and
flips on reflection. Rule failure draws a uniform categorical outcome. Numeric
observation noise also redraws uniformly. Thus each exact class probability is
`(1-s*(1-eta))/33 + s*(1-eta)*I(class=next_position)`.
The outcome channel is sampled separately from the exogenous position stream.

Drift changes the Boolean F0 literal's polarity halfway through; binding acquires
an exception for e0; numeric outcomes reverse the signed velocity before
reflection. Ground truth records both regimes. No learner is told the transition.

The independent oracle runs a rule interpreter in `truth.py`. Tests separately
check exhaustive Boolean truth tables, lag placement, equal-entity bindings,
noise composition and arithmetic reflection. Deliberately corrupting the numeric
formula makes the independent oracle check fail.

## Rule matching and association calibration

Recovery is structural for Boolean/relational rules and behavioural for numeric
rules. Each rule is grouped by target and predicted value. Its conjunction is
sorted, duplicate literals removed, contradictions discarded, and variables
canonically renamed. Sufficient ground Boolean rules use the Blake cover:
consensus and absorption produce all prime implicants. Existential variable
scopes are local to each rule; consensus never crosses independent scopes.
Numeric formulas use exact affine coefficient normalization, and their condition
is part of the rule. Formula roles `x1`, `x2`, `x3` denote positions at lags
1–3; `v1` is the observed current velocity. Unconditional formulas and weak
formula associations are evaluated too; out-of-domain predictions are wrong
outcomes, not silently clipped coordinates. CTV negative branches retain their whole conditioning event:
`P(Y | NOT(A AND B))` is not copied onto each disjunct as `P(Y | NOT A)`.
Only sufficient ground complements expand by De Morgan for DNF matching;
association calibration uses the original pooled event. Quantified complements
remain `NOT EXISTS`, preserving their variable scope. Recovery precision and
recall use the normalized sets.

For numeric rules, a normalized element is `(target, context, predicted_value)`.
Each current position/velocity state has total weight one. When learned rules
inspect prior positions or other observations, the evaluator expands reachable
pasts and independent auxiliary observations, distributing that weight uniformly.
Unmentioned distractors integrate out. Thus lagged or distractor-constrained rules
are evaluated on legal contexts, rather than treating missing values as false.
Exact expansion has a one-million-context guard and fails explicitly beyond it.
Precision penalizes incorrect predictions on reachable contexts; recall measures
coverage of correct context/value pairs. Two wall transitions receive partial
credit. An incorrect extra rule adds false positives, rather than canceling a
correct rule. This coverage definition deliberately differs from Boolean rule
counts; compare each type separately.

Weak associations are **not false sufficient rules**. The evaluator enumerates
independent Boolean assignments and categorical entity choices required by the
body and planted target rules, then averages exact outcome probabilities among
satisfying contexts. Unrelated predicates are integrated out. Impossible bodies
have no conditional probability and are reported with null calibration. Numeric
association calibration averages exact conditional outcome distributions over
satisfying **observed histories in the current regime**, never sampled labels.
This finite-stream reference handles correlated numeric lags and unrelated
conjuncts without pretending the trajectory is IID. Each row names its reference
measure. Numeric behavioural matching still uses the uniform reachable domain. Enumeration
is evaluator work, outside miner timing, and refuses oversized exact problems rather than silently approximating.

## Prediction, relevance, generalization and cost

Every step predicts before revealing labels. Raw JSONL contains predictions,
exact distributions, labels, per-target losses, mining-path and total latency,
RSS high-water mark, work counts, relevance weights, and actual review verdicts.
Brier is the category-mean squared distance to exact probabilities. Expected
label Brier adds irreducible variance; excess log loss subtracts true entropy.

Population predictions are computed independently by enumeration. Normalized
skill subtracts the oracle error from the same seed's population error; zero is
population-only performance and one is exact prediction. If the population and
oracle errors coincide (for example, Boolean flip noise .5), skill is undefined;
the report keeps the proper losses and does not fabricate a normalized value.
Numeric population probabilities use the uniform reachable-context reference described above.

The highest-relevance target receives weight 1, the others .05. Weights are public
and independent of current labels. Reference `budgeted` prioritizes relevance then
lag; `budgeted-flat` rotates through the same tables. Both use a 40-unit search/read
allowance. With six targets, six input reads are reserved for their learned predictors and 34 candidate tables are scheduled
for updates. Repeated input reads are cached; actual reads, candidate evaluations,
candidate updates and target-base-rate updates are counted separately. A stored
best rule remains usable when search moves to another table. Candidate quality
is empirical Gini gain on that table's sampled observations, with sample-count
tie breaking. Both policies receive the same seeded feature permutation, using
the `budgeted-schema` RNG channel, so a planted schema prefix cannot favor either.
Schedules advance on predictions, including periods without labels. Schedules are
built once, not repeatedly scanned or sorted. Both arms rotate among tables with equal
priority; a fixed feature-schema prefix cannot exclude equal-priority candidates
forever. The priority arm does not adaptively escalate after stalled improvement.
These are simple single-literal reference learners, not implementations of the directed miner. Actual feature reads, table
updates, and updates by lag are counted. Predictive comparisons use paired seeds
and equal budgets. PeTTa receives the same query allowance but has no public target
priority API or total mining-work cap: its per-hypothesis review allowance and
per-fact proposal cap are **not** a fixed total compute budget.

The holdout experiment trains on entities e0–e3, then introduces e4–e7 halfway
through. Holdout entities are absent from training facts (including false facts)
and from the initial grounded feature schema. Predicate heads and arities are
shared. No holdout labels are supplied to the learner. Because prediction uses
lagged inputs, the first label-withheld period still has an old-entity context;
it is excluded from the new-entity score by inspecting the actual input entities.
At 256 steps, each binding target has 128 training labels and 127 scored novel
contexts. Prediction-only periods count reads and input ingestion, but zero
label submissions, reviews and table updates. Recovery is frozen at the
training boundary and prediction on new entities is scored separately. PeTTa
still ingests test inputs, so this is label-free transfer, not a frozen engine
snapshot; the distinction is visible in the adapter. Memorized old constants
cannot satisfy the new contexts. General shared-variable rules can.

Cost includes input ingestion and feedback/review; prediction query time is also
reported separately through total latency. A least-squares positive trend greater
than half the mean cost over a 1024-step stream fails the empirical constant-cost
screen. Timing does not prove asymptotic complexity. RSS includes the evaluator
and engine; wall-clock and RSS are inherently not reproducible from a seed.
PeTTa work counters cover submitted atoms, requested query allowance, and returned
review rows; internal candidate updates remain null. They must not be presented
as measured lag-priority internals.

## Directed-design coverage and limits

| Design decision | Behavioural test or exposed limit |
|---|---|
| Decision relevance | Weighted prediction error with competing targets at equal table-update budgets |
| Residual specialization | Conjunction size, negation, exception and drift recovery/loss curves |
| Lag priority | Extra lag-4–12 distractors; counted table updates by lag; paired priority/flat arms |
| Numeric conditions/formulas | Observable position/velocity rules; reachable-context equivalence |
| Pair interactions | XOR recovery and prediction versus the best marginal learner |
| Overlapping disjuncts | Exact noisy-OR probabilities and OR prediction/calibration |
| Anti-unification | Shared-variable recovery and unseen-entity predictions without labels |
| Promotion/retirement | Rule recovery over time and post-drift prediction; raw actual verdicts |
| Bounded cost | 1024-step fitted cost and memory, including 100-distractor streams |
| Soft input evidence, learned template inventions | Not generated; no claim of coverage |
| Internal queue ordering, compilation/backfill | Not directly identifiable from public predictions; no trace-based certification |

The initial requirement to judge every internal decision cannot be fulfilled by
self-reported fields. These behavioural workloads can discriminate outcomes but
cannot establish that a particular internal queue or refinement algorithm was
used. This limitation is now explicit.

## Acceptance and deliberate failures

The exact same gate predicates evaluate measured systems and deliberately broken
fixtures passed through real scoring functions:

- Constant .5 predictions fail independent deterministic oracle truth checks.
- An oracle masquerading as the marginal baseline removes the required gap.
- A marginal system leaking planted XOR rules fails the zero-recall requirement.
- A population-only system has zero skill at both paired difficulty levels and
  fails the required degradation test.
- A miner revisiting all previous observations has linear per-step cost and fails
  the fitted-slope screen, even when a large first-step spike is added.
- A system stopping after 32 of 1024 decisions fails long-stream completion.
  Uncompleted stress cohorts prevent a constant-cost pass inferred only from
  easier completed streams. Superseded interrupted attempts remain in the audit
  but do not invalidate a subsequently completed identical case.

Tests execute every generated acceptance gate on its broken values and require
FAIL. `acceptance.json` includes the fixture, measured evidence, threshold, and
whether the deliberately broken arm was rejected. These are gate-sensitivity
checks, not evidence that every difficulty must make every actual learner worse.
Some robust systems, or systems already at chance, legitimately fail a universal
monotonic-degradation requirement; such failures are retained.

## Reproduction

All engine, benchmark and test processes must use the shared Python, capped
admission, DISPLAY unset and the virtual-memory limit. Example:

```bash
unset DISPLAY
ulimit -v 16000000
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MININGBENCH_CAPPED=1
export TMPDIR="$PWD/.scratch"
/nexus/Dev/OpenCog/bench/capped --slots 2 --peak-mb 256 \
 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
 -m miningbench sweep --out /nexus/Dev/OpenCog/bench/results/mining-bench/NEW-REFERENCE \
 --systems oracle marginal --separate-kinds --levels base samples long \
 --samples 256 --seeds 1 2 3 4 5 6 7 8 9 10 --budget 40 --jobs 2

/nexus/Dev/OpenCog/bench/capped --slots 1 --peak-mb 128 \
 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
 benchmarks/mining/powered_sweep.py \
 --evidence /nexus/Dev/OpenCog/bench/results/mining-bench/NEW-REFERENCE \
 --out /nexus/Dev/OpenCog/bench/results/mining-bench/NEW-LIVE \
 --levels base --jobs 2
```

Choose fresh output directories. The live coordinator admits each child through
`capped` separately, reserving 15 GB for a numeric attempt, 6 GB for binding stress,
3 GB for ordinary binding, and 1.5 GB for other ordinary runs. Other stress attempts
default to 6 GB; `--stress-peak-mb` adjusts that reservation from measured peaks.
The early resized cohorts used 3.5 GB after four deep-lag timeout attempts logged
peaks below 2.4 GiB. After the reboot, ordinary stress reservations increased to
5 GB because three-target runs exceeded 4 GiB. Execution limits and seeds are unchanged. Admission waiting
does not consume the execution timeout. Numeric attempts use one worker so a
second queued 15 GB reservation cannot block smaller jobs behind it. The
remaining difficulty grid uses eight waiting workers after completing strength
and noise; each child still needs its own one-slot capped admission.
`--keep-timeouts` reuses completed timeout attempts. `--keep-resource-failures`
also reuses exit-137 or exit-250 attempts whose retained log explicitly reports memory
allocation failure. Interrupted attempts are recorded separately and rerun in
fresh directories.

Master was `4e6b059b` when the powered experiment started. It advanced to
`6ffb1404` during the resumed sweep, changing implication-query semantics without
landing the withdrawal fix. Remaining powered runs use an unmodified local clone
at the original commit, selected through `PYTHONPATH` while keeping the shared
interpreter. Separate single/AND2 runs and the prediction-read repro check the
new master. Runtime commits and import paths are recorded per run; these cohorts
are not pooled.
All 40 new-master runs completed (single/AND2, both ingestion arms, ten seeds,
256 steps). Their predictions, exact probabilities, labels, scores, learned
rules and recovery match the original runtime exactly. See the
[separate current-master table](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-master-6ff/report.md)
and [per-run comparison](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-master-comparison.json).
The full suite also passes on `6ffb1404`: 188 passed, 25 skipped and 13 subtests.
The prediction-read repro still exposes the same seven withdrawn hypotheses.
After that check, master advanced to `7925f8d5`, including the withdrawal and
late-link replay fix. The unchanged repro now predicts .99985 versus .00016,
and its expanded proofs contain no withdrawn hypotheses. The full suite also
passes on that revision: 188 passed, 25 skipped, 13 subtests. A separate powered
cohort is running from an unmodified clone of this revision, using the shared
interpreter. Its results will remain separate from the original runtime.
For long streams add `--levels long`; for the 100-distractor, lag-12 stress case
use `--levels deep_lags --stress-seconds 3600`. For transfer use
`--kinds binding --levels entity_holdout`. Report collection also uses capped.
The suite uses `PYTHONPATH=src:.test-deps` with local pytest dependencies and the
shared interpreter. Live tests require `MININGBENCH_LIVE_PETTACHAINER=1`.

Historical v1 evidence remains under `final/` and other earlier directories. It
contains the matching and numeric-world artefacts described above and is not the
corrected result. The v2 evidence root is
`/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2/`.

## Upstream bug and genuine gaps

The flat ~.454 prediction repro is an upstream withdrawal bug: retained skeleton
rows still apply dropped hypotheses. Expanded revision proofs expose them.
`benchmarks/mining/repros/prediction_read.py` now prints expanded proofs, and its
README records the cause. There is no harness workaround.

AND2 recall zero in the earlier table was a matching artefact, not proof of a
candidate-language gap. Numeric zero-recall claims based on unobserved `Region`
predicates were also invalid. The miner can propose positive pairs and shared
entity variables; grounded numeric transitions must receive behavioural credit.
Actual limits to test are higher-order conjunction proposals, mixed-polarity XOR,
multiple-link temporal conjunctions, numeric formula discovery, relevance-directed
budget allocation, and runtime retention/cost. The two ingestion arms separate
link availability from those language limits.

## Powered results

The [full results table](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-reference-powered/report.md) contains systems × rule types × difficulty, paired differences, weak-association calibration, costs, learning curves and censored attempts.

There are 3870 completed runs in this report. Historical interrupted attempts remain in the audit; completed replacements determine coverage. PeTTa commits: 4e6b059be39537d2c7c8a28e5dc807fb4f3a83c9.

This is an interim evidence snapshot. The original-runtime cohort is now fully
attempted: 530 completed streams, 209 timeouts and one reproducible allocation
failure among 740 planned live conditions. The updated-runtime cohort is still
running. The complete original-runtime audit is in
[review-v2-final](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-final/report.md),
with [unique-condition coverage](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-live-coverage.json).

Intervals below are 95% seed bootstrap intervals. An underpowered cell retains its actual n and cannot earn an acceptance pass. Oracle base probability Brier is zero and recovery is one for every type.

| Type | Marginal base Brier [95% CI] | petta base Brier [95% CI] | petta-link-first base Brier [95% CI] |
|---|---|---|---|
| and2 | 0.1296 [0.1243, 0.1356]; n=10 | 0.1742 [0.1634, 0.1851]; n=10 | 0.1742 [0.1634, 0.1851]; n=10 |
| and3 | 0.1025 [0.0944, 0.1109]; n=10 | 0.0872 [0.0820, 0.0924]; n=10 | 0.0872 [0.0820, 0.0924]; n=10 |
| and4 | 0.0639 [0.0546, 0.0731]; n=10 | 0.0426 [0.0374, 0.0473]; n=10 | 0.0426 [0.0374, 0.0473]; n=10 |
| binding | 0.1984 [0.1917, 0.2052]; n=10 | 0.1863 [0.1790, 0.1943]; n=10 | 0.1866 [0.1791, 0.1948]; n=10 |
| lag | 0.0983 [0.0922, 0.1040]; n=10 | 0.1078 [0.0998, 0.1161]; n=10 | 0.1078 [0.0998, 0.1161]; n=10 |
| negation | 0.1293 [0.1253, 0.1338]; n=10 | 0.1754 [0.1704, 0.1812]; n=10 | 0.1754 [0.1704, 0.1812]; n=10 |
| numeric | 0.0295 [0.0295, 0.0296]; n=10 | no completed stream | no completed stream |
| or | 0.2182 [0.2105, 0.2275]; n=10 | 0.1897 [0.1741, 0.2050]; n=10 | 0.1897 [0.1741, 0.2050]; n=10 |
| single | 0.0032 [0.0032, 0.0032]; n=10 | 0.2526 [0.2497, 0.2551]; n=10 | 0.2526 [0.2497, 0.2551]; n=10 |
| xor | 0.2677 [0.2656, 0.2700]; n=10 | 0.2516 [0.2491, 0.2544]; n=10 | 0.2516 [0.2491, 0.2544]; n=10 |

| Type | petta base recall [95% CI] | petta-link-first base recall [95% CI] |
|---|---|---|
| and2 | 0.2000 [0.0000, 0.5000]; n=10 | 0.2000 [0.0000, 0.5000]; n=10 |
| and3 | 0.0000 [0.0000, 0.0000]; n=10 | 0.0000 [0.0000, 0.0000]; n=10 |
| and4 | 0.0000 [0.0000, 0.0000]; n=10 | 0.0000 [0.0000, 0.0000]; n=10 |
| binding | 0.8000 [0.5000, 1.0000]; n=10 | 0.8000 [0.5000, 1.0000]; n=10 |
| lag | 0.0000 [0.0000, 0.0000]; n=10 | 0.0000 [0.0000, 0.0000]; n=10 |
| negation | 0.0000 [0.0000, 0.0000]; n=10 | 0.0000 [0.0000, 0.0000]; n=10 |
| numeric | no completed stream | no completed stream |
| or | 0.3500 [0.1500, 0.5500]; n=10 | 0.3500 [0.1500, 0.5500]; n=10 |
| single | 1.0000 [1.0000, 1.0000]; n=10 | 1.0000 [1.0000, 1.0000]; n=10 |
| xor | 0.0000 [0.0000, 0.0000]; n=10 | 0.0000 [0.0000, 0.0000]; n=10 |

Difficulty comparisons use paired normalized-skill loss (base minus harder), averaged within seed over the same fully paired rule types. Incomplete type cohorts are excluded and listed in the full acceptance table. The full table states individual type coverage; the live matrix is not a complete factorial experiment.

| System | Harder level | Included types | Skill loss [95% paired CI] | Gate |
|---|---|---|---|---|
| budgeted | distractors | all ten | 0.2044 [0.1851, 0.2253]; n=10 | PASS |
| budgeted | strength | all ten | 0.0503 [0.0438, 0.0569]; n=10 | PASS |
| budgeted | observation_noise | all ten | 0.1996 [0.1747, 0.2277]; n=10 | PASS |
| budgeted | samples | all ten | 0.1344 [0.1155, 0.1569]; n=10 | PASS |
| budgeted | drift | all ten | 0.1046 [0.1004, 0.1089]; n=10 | PASS |
| budgeted | targets | all ten | 0.3630 [0.3420, 0.3861]; n=10 | PASS |
| budgeted-flat | distractors | all ten | 0.7135 [0.6727, 0.7632]; n=10 | PASS |
| budgeted-flat | strength | all ten | 0.0508 [0.0443, 0.0572]; n=10 | PASS |
| budgeted-flat | observation_noise | all ten | 0.2000 [0.1750, 0.2281]; n=10 | PASS |
| budgeted-flat | samples | all ten | 0.1343 [0.1155, 0.1568]; n=10 | PASS |
| budgeted-flat | drift | all ten | 0.1044 [0.1001, 0.1087]; n=10 | PASS |
| budgeted-flat | targets | all ten | 0.1491 [0.1396, 0.1592]; n=10 | PASS |
| marginal | distractors | all ten | 0.0476 [0.0425, 0.0536]; n=10 | PASS |
| marginal | strength | all ten | 0.0518 [0.0445, 0.0597]; n=10 | PASS |
| marginal | observation_noise | all ten | 0.1893 [0.1640, 0.2151]; n=10 | PASS |
| marginal | samples | all ten | 0.1220 [0.0990, 0.1489]; n=10 | PASS |
| marginal | drift | all ten | 0.1046 [0.0996, 0.1093]; n=10 | PASS |
| marginal | targets | all ten | 0.0182 [0.0133, 0.0234]; n=10 | PASS |
| petta | strength | and2; binding; single; xor | 0.0078 [-0.0061, 0.0222]; n=10 | FAIL |
| petta | observation_noise | and2; binding; single; xor | 0.0357 [0.0150, 0.0579]; n=10 | PASS |
| petta | samples | and2; binding; single; xor | 0.0577 [0.0479, 0.0692]; n=10 | PASS |
| petta | drift | and2; binding; single; xor | -0.0038 [-0.0244, 0.0107]; n=10 | FAIL |
| petta-link-first | strength | and2; binding; single; xor | 0.0081 [-0.0060, 0.0226]; n=10 | FAIL |
| petta-link-first | observation_noise | and2; binding; single; xor | 0.0312 [0.0083, 0.0557]; n=10 | FAIL |
| petta-link-first | samples | and2; binding; single; xor | 0.0584 [0.0491, 0.0697]; n=10 | PASS |
| petta-link-first | drift | and2; binding; single; xor | -0.0040 [-0.0247, 0.0101]; n=10 | FAIL |

The fixed-budget comparison retains learned predictors, rotates candidate updates, and uses a shared seeded feature order. Earlier fixed-prefix and batch-only-prediction results are superseded.

| Cohort | Priority advantage in weighted Brier [95% paired CI] |
|---|---|
| base | 0.0000 [0.0000, 0.0000]; n=100 |
| directed_base | 0.1379 [0.1150, 0.1654]; n=10 |
| directed_deep_lags | 0.2636 [0.2606, 0.2667]; n=10 |
| directed_entity_holdout | 0.2061 [0.1531, 0.2598]; n=10 |
| distractors | 0.0716 [0.0635, 0.0794]; n=100 |
| drift | 0.0000 [0.0000, 0.0000]; n=100 |
| long | 0.0000 [0.0000, 0.0000]; n=100 |
| observation_noise | 0.0000 [0.0000, 0.0000]; n=100 |
| samples | 0.0000 [-0.0000, 0.0000]; n=100 |
| strength | 0.0000 [0.0000, 0.0000]; n=100 |
| targets | 0.0128 [0.0097, 0.0160]; n=100 |

| New-entity holdout cohort | Brier [95% CI] |
|---|---|
| ('budgeted', ('binding',), 3) | 0.1872 [0.1805, 0.1945]; n=10 |
| ('budgeted', ('single',), 3) | 0.1683 [0.1671, 0.1697]; n=10 |
| ('budgeted-flat', ('binding',), 3) | 0.3333 [0.2952, 0.3813]; n=10 |
| ('budgeted-flat', ('single',), 3) | 0.1725 [0.1420, 0.2108]; n=10 |
| ('marginal', ('binding',), 1) | 0.1882 [0.1759, 0.2015]; n=10 |
| ('marginal', ('binding',), 3) | 0.1934 [0.1848, 0.2027]; n=10 |
| ('marginal', ('single',), 3) | 0.0001 [0.0001, 0.0001]; n=10 |
| ('oracle', ('binding',), 1) | 0.0000 [0.0000, 0.0000]; n=10 |
| ('oracle', ('binding',), 3) | 0.0000 [0.0000, 0.0000]; n=10 |
| ('oracle', ('single',), 3) | 0.0000 [0.0000, 0.0000]; n=10 |
| ('petta', ('binding',), 1) | 0.1828 [0.1695, 0.1973]; n=10 |

| Numeric exact-context memorizer | Brier [95% CI] |
|---|---|
| base | 0.0163 [0.0159, 0.0168]; n=10 |
| long | 0.0056 [0.0055, 0.0056]; n=10 |

The widened stochastic trajectory removes the eight-context shortcut. Its finite 198-context domain remains eventually memorizable.

| Acceptance screen | Result | Rejects broken control |
|---|---|---|
| oracle independent truth | PASS | True |
| oracle beats marginal: and2 | PASS | True |
| oracle beats marginal: and3 | PASS | True |
| oracle beats marginal: and4 | PASS | True |
| oracle beats marginal: binding | PASS | True |
| oracle beats marginal: lag | PASS | True |
| oracle beats marginal: negation | PASS | True |
| oracle beats marginal: numeric | PASS | True |
| oracle beats marginal: or | PASS | True |
| oracle beats marginal: xor | PASS | True |
| marginal XOR recovery fails | PASS | True |
| budgeted: harder distractors | PASS | True |
| budgeted: harder strength | PASS | True |
| budgeted: harder observation_noise | PASS | True |
| budgeted: harder samples | PASS | True |
| budgeted: harder drift | PASS | True |
| budgeted: harder targets | PASS | True |
| budgeted: fitted constant-cost screen | FAIL | True |
| budgeted: long-stream completion | PASS | True |
| budgeted-flat: harder distractors | PASS | True |
| budgeted-flat: harder strength | PASS | True |
| budgeted-flat: harder observation_noise | PASS | True |
| budgeted-flat: harder samples | PASS | True |
| budgeted-flat: harder drift | PASS | True |
| budgeted-flat: harder targets | PASS | True |
| budgeted-flat: fitted constant-cost screen | FAIL | True |
| budgeted-flat: long-stream completion | PASS | True |
| marginal: harder distractors | PASS | True |
| marginal: harder strength | PASS | True |
| marginal: harder observation_noise | PASS | True |
| marginal: harder samples | PASS | True |
| marginal: harder drift | PASS | True |
| marginal: harder targets | PASS | True |
| marginal: fitted constant-cost screen | FAIL | True |
| marginal: long-stream completion | PASS | True |
| oracle: fitted constant-cost screen | PASS | True |
| oracle: long-stream completion | PASS | True |
| petta: harder distractors | UNMEASURED | True |
| petta: harder strength | FAIL | True |
| petta: harder observation_noise | PASS | True |
| petta: harder samples | PASS | True |
| petta: harder drift | FAIL | True |
| petta: harder targets | UNMEASURED | True |
| petta: fitted constant-cost screen | UNMEASURED | True |
| petta: long-stream completion | FAIL | True |
| petta-link-first: harder distractors | UNMEASURED | True |
| petta-link-first: harder strength | FAIL | True |
| petta-link-first: harder observation_noise | FAIL | True |
| petta-link-first: harder samples | PASS | True |
| petta-link-first: harder drift | FAIL | True |
| petta-link-first: harder targets | UNMEASURED | True |
| petta-link-first: fitted constant-cost screen | UNMEASURED | True |
| petta-link-first: long-stream completion | FAIL | True |

All 53 of 53 generated screens reject their broken controls. Actual gate outcomes: {'PASS': 37, 'FAIL': 10, 'UNMEASURED': 6}. See [acceptance evidence](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-reference-powered/acceptance.json) for thresholds, paired intervals and control values.

![Learning curves](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-reference-powered/learning_curves.svg)

Full ChainerGame suite: 188 passed, 25 skipped, 13 subtests passed, with mining live tests enabled. Independent reproducibility verification matched 420 base/long prefixes exactly. Timing and RSS remain machine-dependent.

PeTTa stress censoring prevents a universal constant-cost claim. Partial timing excludes unfinished stalled steps. The current upstream withdrawal bug invalidates predictive reads even where learned rules recover correctly; no harness workaround was applied. Numeric formula discovery, higher-order or mixed-polarity conjunction proposals, multi-link conjunction mining, and an auditable relevance-directed total work budget remain separate capability limits.
