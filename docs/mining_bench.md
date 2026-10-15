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
   fitted least-squares slope multiplied by the fitted portion's length and
   divided by its mean step cost. The fit excludes the first 32 steps so initial history ingestion
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
   behavioural evidence into a fresh report directory. PeTTa checkpoint recovery
   is reconstructed from raw verdict events. Final recovery uses saved rules for
   every system. Older reference checkpoints without saved rule snapshots retain
   their original values as `recorded_recovery`; their current-matcher recovery
   is left unmeasured. Prediction curves do not depend on this reconstruction.

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
screen. Rule-type fits are averaged within each seed before the bootstrap, rather
than selecting the noisiest maximum. A known-constant 35-microsecond series with
seeded jitter passes the same predicate, and the actual constant-work reference
learners calibrate this screen against measured timing noise. Timing does not prove asymptotic complexity. RSS includes the evaluator
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

These are **six reusable negative-control families**, repeated across gates;
the oracle-versus-oracle gap (`x-x`) and zero-skill difference (`1-error/error`)
are identity constructions. They test threshold sensitivity, not system behaviour.
Positive controls also exercise each family: independently interpreted truth,
oracle versus population prediction, no exported XOR rules, deliberately degraded
oracle predictions, noisy constant work, and full-stream consumption. The
constructed degradation control demonstrates passability, not that every real
system must degrade on every knob. Plain 1024-step completion and 100-distractor,
lag-12 completion are now separate gates.

Tests execute every generated acceptance gate on its broken values and require
FAIL, and execute its passing control and require PASS. `acceptance.json` includes the fixture, measured evidence, threshold, and
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

PYTHONPATH="$PWD/.scratch/petta-171b2820:$PWD/src" \
/nexus/Dev/OpenCog/bench/capped --slots 1 --peak-mb 128 \
 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
 benchmarks/mining/powered_sweep.py \
 --evidence /nexus/Dev/OpenCog/bench/results/mining-bench/NEW-REFERENCE \
 --out /nexus/Dev/OpenCog/bench/results/mining-bench/NEW-LIVE \
 --jobs 1 --required-petta-commit 171b2820744ff778cadba32213654fe7002f0281 \
 --tasks-file /nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-plan/numeric-tasks.json \
 --peaks /nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-plan/measured-peaks.json
```

Choose fresh output directories. The live example uses the unmodified local
checkout of `171b2820` at `.scratch/petta-171b2820` and the saved two-condition
numeric manifest. A different workload needs measured profiles for its own
labels. Each child has the stable label
`mining-{system}-{level}-{kind}` and explicit sweep priority, including
`BENCH_PRIORITY=sweep`. The coordinator now requires a `--peaks` JSON file keyed
by those labels. Each entry records `observed_peak_mib` and its source; admission
reserves `ceil(1.2 * observed_peak_mib)` MiB. There are no fixed numeric, binding,
or stress reservations. Censored peaks are lower bounds, so the source and
completion flag remain in the saved admission profile. The daemon also learns
measured peaks under the stable label.

`--tasks-file` accepts explicit `[level, kind, seed, system]` rows, preventing a
small follow-up from restarting the full grid. The default is one worker.
`--required-petta-commit` checks the actual import path's commit before launching
children, as well as filtering reusable results. Admission waiting does not
consume the execution timeout. `--keep-timeouts` and
`--keep-resource-failures` reuse terminal evidence only for the requested runtime;
resource failures require an explicit allocation-failure message. Interrupted
attempts retain their logs and are not silently treated as timeouts.

Earlier cohorts used fixed declarations (up to 15 GB) and larger worker queues;
those historical settings remain recorded in their raw status files. The
171b2820 follow-up permits at most two coordinators, one worker each, at sweep
priority. Its declared peaks come from the maximum saved RSS for each matching
label, plus the stated margin, until new measurements justify a replacement.

Master was `4e6b059b` when the powered experiment started. It advanced to
`6ffb1404` during the resumed sweep, changing implication-query semantics without
landing the withdrawal fix. At that stage, remaining powered runs used an
unmodified local clone at the original commit, selected through `PYTHONPATH`
while keeping the shared interpreter. Separate single/AND2 runs and the
prediction-read repro checked `6ffb1404`. Runtime commits and import paths are
recorded per run; these cohorts
are not pooled.
All 40 `6ffb1404` runs completed (single/AND2, both ingestion arms, ten seeds,
256 steps). Their predictions, exact probabilities, labels, scores, learned
rules and recovery match the original runtime exactly. See the
[separate 6ffb1404 table](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-master-6ff/report.md)
and [per-run comparison](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-master-comparison.json).
The full suite also passes on `6ffb1404`: 188 passed, 25 skipped and 13 subtests.
The prediction-read repro still exposes the same seven withdrawn hypotheses.
After that check, master advanced to `7925f8d5`, including the withdrawal and
late-link replay fix. The unchanged repro now predicts .99985 versus .00016,
and its expanded proofs contain no withdrawn hypotheses. The full suite also
passes on that revision: 188 passed, 25 skipped, 13 subtests. The separate powered cohort used an unmodified clone of this revision and the
shared interpreter. The coordinator stopped its remaining jobs after the CLI
failure on 15 October. Its results remain separate from the original runtime.
For a full grid instead of an explicit manifest, omit `--tasks-file`. Choose
`--levels long` for long streams; for the 100-distractor, lag-12 stress case
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
budget allocation, and runtime retention/cost. The two ingestion arms **separate
nothing in the saved lag experiment**: seed 1 predictions have identical SHA-256
`fad0dba30642565bc9f49f9bec587339fd2f5534679a1473dfab959bf3e53303`.
The miner's “present first” policy suppresses cross-link candidates whenever a
live same-period hypothesis already concludes the target. Consistent ingestion
order alone cannot expose the candidate-language difference in this workload.

The corrected AND2 recall of .20 is a **review/retirement defect, not a proposal
gap**. In seed 2, `F0 AND F1 -> T` (`hypothesis-3`) is proposed at zero-based step
1 and dropped at step 7, and the tried-rule store prevents re-proposal. At that
review the pair reads .9987529 on about four effective observations, versus
F1's .6663679 on about six. Their .3324 difference falls below the two-standard-
error threshold (about .471), so the general rule screens off the correct pair.
F1 read .9975 earlier at step 2; it did **not** read .998 at retirement. Raw
[verdicts and hashes](/nexus/Dev/OpenCog/bench/results/mining-bench/review4-diagnoses.json)
retain this distinction. No backend policy is changed here.

## Report from saved evidence, before the 171b2820 rerun

This snapshot was written before starting new engine work. The stopped sweeps'
recorded runtime is **7925f8d5**, not 4e6b059b. The earlier 4e6b059b cohort remains
separate. The saved evidence contains 4445 completed runs across reference and
live cohorts. No mining processes were running when this snapshot was assembled.
The immutable [snapshot report](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-disk-final/report.md)
and [coverage manifest](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-disk-coverage.json)
remain available independently of the follow-up below.

For the 7925f8d5 plan, 575 of 740 conditions completed; 73 timed out, one had a
confirmed allocation failure, and 91 have no terminal comparable result
(including interrupted and unstarted conditions). No partial stream is scored
as a complete prediction experiment.

**Powered** means ten completed seeds per arm; **underpowered** means fewer than
ten. Counts below are link-last / link-first. Other type/level combinations were
not in the live plan, although all ten types have powered reference results.

| Live level | Kind(s) | Completed seeds | Power |
|---|---|---|---|
| Base, 256 | All except numeric | 10 / 10 each | Powered |
| Base, 256 | Numeric | 0 / 0 | Unmeasured |
| Samples 64; strength .7; noise .2; drift | Single, AND2, XOR, binding | 10 / 10 each | Powered |
| Three targets | Single | 6 / 8 | Underpowered |
| Three targets | AND2 | 6 / 6 | Underpowered |
| Three targets | XOR | 2 / 2 | Underpowered |
| Three targets | Binding | 0 / 0 | Unmeasured |
| 100 distractors | Single | 7 / 6 | Underpowered |
| 100 distractors | AND2, XOR, binding | 0 / 0 each | Unmeasured |
| Long, 1024 | Single | 10 / 10 | Powered |
| Deep lag 12, 100 distractors, 1024 | Single | 0 / 0 | Unmeasured |
| New-entity holdout | Binding | 10 / 2 | Powered / underpowered |

The original snapshot's acceptance counts are superseded by the corrected gates
below. Its three reference cost failures were selection-of-the-maximum artefacts;
its two PeTTa “long” failures were deep-lag stress timeouts mislabeled as plain
long streams. Strength and noise remain passable with ten paired seeds. The old
“101/101” and later “117/117” counts repeated six fixture families; they were
never that many independent discrimination experiments.

The numeric task is now winnable in its observed language: behavioural matching
credits correct grounded transitions, and the oracle has zero probability
error. No completed **256-step base** live numeric stream supports a learned-rule conclusion.
The period-eight shortcut is gone; the widened finite domain remains eventually
memorizable. Corrected recall credits AND2 and OR discoveries without allowing
weak singles to absorb them. Mixed-polarity, higher-arity and mixed-lag recovery
remains poor in the powered base data. The ingestion arms produce identical lag predictions; the fixed upstream
late-link repro alone does not establish that this benchmark separates ordering
from proposal policy.

The full suite passed on 7925f8d5: **188 passed, 25 skipped, 13 subtests passed**.
All **440** independently executed base/long prefixes match exactly. Intermediate
reference recall without saved rule snapshots is explicitly historical; final
recovery and PeTTa checkpoint recovery use the corrected matcher.

## Powered results

The [historical full results table](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-final/report.md) contains systems × rule types × difficulty, paired differences, weak-association calibration, costs, learning curves and censored attempts.

There are 4451 completed runs in this report. Historical interrupted attempts remain in the audit; completed replacements determine coverage. PeTTa commits: 171b2820744ff778cadba32213654fe7002f0281, 4e6b059be39537d2c7c8a28e5dc807fb4f3a83c9, 7925f8d588b6aba5b868e7518c55f6e1249c0464.

The withdrawal-fix cohort covers 740 planned conditions: {'completed': 575, 'timeout': 73, 'unresolved': 91, 'memory_failure': 1}. Earlier runtime results remain separate in the full report; the concise tables below use 7925f8d5.

Intervals below are 95% seed bootstrap intervals. An underpowered cell retains its actual n and cannot earn an acceptance pass. Oracle base probability Brier is zero and recovery is one for every type.

| Type | Marginal base Brier [95% CI] | petta-link-first@7925f8d5 base Brier [95% CI] | petta@7925f8d5 base Brier [95% CI] |
|---|---|---|---|
| and2 | 0.1296 [0.1243, 0.1356]; n=10 | 0.0428 [0.0288, 0.0636]; n=10 | 0.0428 [0.0288, 0.0636]; n=10 |
| and3 | 0.1025 [0.0944, 0.1109]; n=10 | 0.0917 [0.0448, 0.1502]; n=10 | 0.0917 [0.0448, 0.1502]; n=10 |
| and4 | 0.0639 [0.0546, 0.0731]; n=10 | 0.1161 [0.0648, 0.1715]; n=10 | 0.1161 [0.0648, 0.1715]; n=10 |
| binding | 0.1984 [0.1917, 0.2052]; n=10 | 0.1917 [0.1817, 0.2037]; n=10 | 0.1911 [0.1816, 0.2022]; n=10 |
| lag | 0.0983 [0.0922, 0.1040]; n=10 | 0.1695 [0.1319, 0.2091]; n=10 | 0.1695 [0.1319, 0.2091]; n=10 |
| negation | 0.1293 [0.1253, 0.1338]; n=10 | 0.1288 [0.1218, 0.1354]; n=10 | 0.1288 [0.1218, 0.1354]; n=10 |
| numeric | 0.0295 [0.0295, 0.0296]; n=10 | no completed stream | no completed stream |
| or | 0.2182 [0.2105, 0.2275]; n=10 | 0.1429 [0.1117, 0.1726]; n=10 | 0.1429 [0.1117, 0.1726]; n=10 |
| single | 0.0032 [0.0032, 0.0032]; n=10 | 0.0261 [0.0230, 0.0298]; n=10 | 0.0261 [0.0230, 0.0298]; n=10 |
| xor | 0.2677 [0.2656, 0.2700]; n=10 | 0.2021 [0.1813, 0.2276]; n=10 | 0.2021 [0.1813, 0.2276]; n=10 |

| Type | petta-link-first@7925f8d5 base recall [95% CI] | petta@7925f8d5 base recall [95% CI] |
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

All 180 base comparisons between the original and fixed runtimes have identical generated data and outcomes. For the link-last arm, Brier improvement (old minus fixed) is 0.2265 [0.2225, 0.2309] for single literals and 0.1314 [0.1040, 0.1549] for AND2, but -0.0735 [-0.1284, -0.0200] for AND4. Fixing invalid proof reuse does not uniformly improve prediction. See the [paired runtime comparison](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-792-base-comparison.json).

Current-runtime completion is assessed per planned condition, including all ten seeds in each arm. Timeouts and confirmed allocation failures retain partial cost and RSS evidence, not full-stream prediction scores.

| Live difficulty | Completed | Timeout | Allocation failure | Unresolved |
|---|---:|---:|---:|---:|
| base | 180 | 9 | 0 | 11 |
| samples | 80 | 0 | 0 | 0 |
| strength | 80 | 0 | 0 | 0 |
| observation_noise | 80 | 0 | 0 | 0 |
| drift | 80 | 0 | 0 | 0 |
| targets | 30 | 50 | 0 | 0 |
| distractors | 13 | 0 | 0 | 67 |
| long | 20 | 0 | 0 | 0 |
| deep_lags | 0 | 6 | 1 | 13 |
| entity_holdout | 12 | 8 | 0 | 0 |

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
| petta-link-first@7925f8d5 | strength | and2; binding; single; xor | 0.1728 [0.1280, 0.2201]; n=10 | PASS |
| petta-link-first@7925f8d5 | observation_noise | and2; binding; single; xor | 0.2770 [0.1758, 0.3809]; n=10 | PASS |
| petta-link-first@7925f8d5 | samples | and2; binding; single; xor | 0.1832 [0.1657, 0.2028]; n=10 | PASS |
| petta-link-first@7925f8d5 | drift | and2; binding; single; xor | 0.2618 [0.2420, 0.2806]; n=10 | PASS |
| petta@7925f8d5 | strength | and2; binding; single; xor | 0.1711 [0.1246, 0.2233]; n=10 | PASS |
| petta@7925f8d5 | observation_noise | and2; binding; single; xor | 0.2778 [0.1759, 0.3833]; n=10 | PASS |
| petta@7925f8d5 | samples | and2; binding; single; xor | 0.1841 [0.1674, 0.2030]; n=10 | PASS |
| petta@7925f8d5 | drift | and2; binding; single; xor | 0.2619 [0.2424, 0.2804]; n=10 | PASS |

The fixed-budget comparison retains learned predictors, rotates candidate updates, and uses a shared seeded feature order. Earlier fixed-prefix and batch-only-prediction results are superseded.

| Cohort | Priority advantage in weighted Brier [95% paired CI] |
|---|---|
| base | 0.0000 [0.0000, 0.0000]; n=10 |
| directed_base | 0.1379 [0.1150, 0.1654]; n=10 |
| directed_deep_lags | 0.2636 [0.2606, 0.2667]; n=10 |
| directed_entity_holdout | 0.2061 [0.1531, 0.2598]; n=10 |
| distractors | 0.0716 [0.0658, 0.0783]; n=10 |
| drift | 0.0000 [0.0000, 0.0000]; n=10 |
| long | 0.0000 [0.0000, 0.0000]; n=10 |
| observation_noise | 0.0000 [0.0000, 0.0000]; n=10 |
| samples | 0.0000 [-0.0000, 0.0000]; n=10 |
| strength | 0.0000 [0.0000, 0.0000]; n=10 |
| targets | 0.0128 [0.0111, 0.0146]; n=10 |

These ancillary contrasts average rule types within each seed before
bootstrapping: ten types do not create one hundred independent seeds. The
fresh report corrects the frozen snapshot's ancillary sample counts; the
dedicated directed-design probes and primary acceptance conclusions are unchanged.

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
| ('petta-link-first@7925f8d5', ('binding',), 1) | 0.2025 [0.1833, 0.2217]; n=2 |
| ('petta@7925f8d5', ('binding',), 1) | 0.1939 [0.1747, 0.2163]; n=10 |

| Numeric exact-context memorizer | Brier [95% CI] |
|---|---|
| base | 0.0163 [0.0159, 0.0168]; n=10 |
| long | 0.0056 [0.0055, 0.0056]; n=10 |

The widened stochastic trajectory removes the eight-context shortcut. Its finite 198-context domain remains eventually memorizable.

| Acceptance screen | Result | Rejects failing control | Passing control passes |
|---|---|---|---|
| oracle independent truth | PASS | True | True |
| oracle beats marginal: and2 | PASS | True | True |
| oracle beats marginal: and3 | PASS | True | True |
| oracle beats marginal: and4 | PASS | True | True |
| oracle beats marginal: binding | PASS | True | True |
| oracle beats marginal: lag | PASS | True | True |
| oracle beats marginal: negation | PASS | True | True |
| oracle beats marginal: numeric | PASS | True | True |
| oracle beats marginal: or | PASS | True | True |
| oracle beats marginal: xor | PASS | True | True |
| marginal XOR recovery fails | PASS | True | True |
| budgeted: harder distractors | PASS | True | True |
| budgeted: harder strength | PASS | True | True |
| budgeted: harder observation_noise | PASS | True | True |
| budgeted: harder samples | PASS | True | True |
| budgeted: harder drift | PASS | True | True |
| budgeted: harder targets | PASS | True | True |
| budgeted: fitted constant-cost screen | PASS | True | True |
| budgeted: plain long-stream completion | PASS | True | True |
| budgeted: deep-lag 100-distractor completion | PASS | True | True |
| budgeted-flat: harder distractors | PASS | True | True |
| budgeted-flat: harder strength | PASS | True | True |
| budgeted-flat: harder observation_noise | PASS | True | True |
| budgeted-flat: harder samples | PASS | True | True |
| budgeted-flat: harder drift | PASS | True | True |
| budgeted-flat: harder targets | PASS | True | True |
| budgeted-flat: fitted constant-cost screen | PASS | True | True |
| budgeted-flat: plain long-stream completion | PASS | True | True |
| budgeted-flat: deep-lag 100-distractor completion | PASS | True | True |
| marginal: harder distractors | PASS | True | True |
| marginal: harder strength | PASS | True | True |
| marginal: harder observation_noise | PASS | True | True |
| marginal: harder samples | PASS | True | True |
| marginal: harder drift | PASS | True | True |
| marginal: harder targets | PASS | True | True |
| marginal: fitted constant-cost screen | PASS | True | True |
| marginal: plain long-stream completion | PASS | True | True |
| marginal: deep-lag 100-distractor completion | PASS | True | True |
| oracle: fitted constant-cost screen | PASS | True | True |
| oracle: plain long-stream completion | PASS | True | True |
| oracle: deep-lag 100-distractor completion | PASS | True | True |
| petta-link-first@7925f8d5: harder distractors | UNMEASURED | True | True |
| petta-link-first@7925f8d5: harder strength | PASS | True | True |
| petta-link-first@7925f8d5: harder observation_noise | PASS | True | True |
| petta-link-first@7925f8d5: harder samples | PASS | True | True |
| petta-link-first@7925f8d5: harder drift | PASS | True | True |
| petta-link-first@7925f8d5: harder targets | UNMEASURED | True | True |
| petta-link-first@7925f8d5: fitted constant-cost screen | UNMEASURED | True | True |
| petta-link-first@7925f8d5: plain long-stream completion | PASS | True | True |
| petta-link-first@7925f8d5: deep-lag 100-distractor completion | UNMEASURED | True | True |
| petta@7925f8d5: harder distractors | UNMEASURED | True | True |
| petta@7925f8d5: harder strength | PASS | True | True |
| petta@7925f8d5: harder observation_noise | PASS | True | True |
| petta@7925f8d5: harder samples | PASS | True | True |
| petta@7925f8d5: harder drift | PASS | True | True |
| petta@7925f8d5: harder targets | UNMEASURED | True | True |
| petta@7925f8d5: fitted constant-cost screen | UNMEASURED | True | True |
| petta@7925f8d5: plain long-stream completion | PASS | True | True |
| petta@7925f8d5: deep-lag 100-distractor completion | UNMEASURED | True | True |

The corrected primary set has **51 PASS, 0 FAIL, 8 UNMEASURED**. These are
reference-system and 7925f8d5 gates; the powered base prediction and recovery
tables above are unchanged. The historical 117/117 claim counted six repeated
negative-control families, including two algebraic identity constructions.
Positive fixtures now pass the same predicates for all six families. Their
repetition across gates is not additional experimental power.

Reference mean cost-growth estimates [95% seed bootstrap CI] are budgeted
**.0099 [−.0077, .0296]**, budgeted-flat **.0470 [.0056, .0926]**, and marginal
**−.0160 [−.0643, .0475]**. All pass the unchanged .5 threshold after averaging
types within each seed. A noisy constant-mean 35µs fixture passes, while a
linear-history-work fixture fails. Selecting the maximum over types had caused
the earlier false positives.

Both PeTTa arms completed **10/10 plain 1024-step streams**. Their mean fitted
growth is **−.0072 [−.0672, .0620]** (link-last) and **.0500 [−.0227, .1312]**
(link-first), a useful positive result. A universal constant-cost claim remains
unmeasured because the distinct 100-distractor/deep-lag cohorts are censored.
Neither those timeouts nor their incomplete seed count now fail the plain
long-stream completion gate. The four reference systems also completed ten
directed deep-lag streams each; their distinct `directed_deep_lags` experiment
label is correctly recognized by the deep-lag completion screen.

The regenerated full report and complete gate evidence are linked in the final
review follow-up below.

![Learning curves](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-final/learning_curves.svg)

The figure shows base/long cohorts. The selective 171b2820 follow-up appears
below at its own difficulty levels.

The earlier 7925f8d5 ChainerGame suite: 188 passed, 25 skipped, 13 subtests passed, with mining live tests enabled. Independent reproducibility verification matched 440 base/long prefixes exactly. Timing and RSS remain machine-dependent.

PeTTa stress censoring prevents a universal constant-cost claim. Partial timing excludes unfinished stalled steps. The withdrawal defect invalidated predictive reads on the original runtime. The unchanged repro confirms its repair on 7925f8d5, whose separately reported predictions use no harness workaround. Numeric formula discovery, higher-order or mixed-polarity conjunction proposals, multi-link conjunction mining, and an auditable relevance-directed total work budget remain separate capability limits.

## Targeted follow-up on 171b2820

The saved-evidence report was committed before these runs. The follow-up uses the unmodified PeTTaChainer commit `171b2820744ff778cadba32213654fe7002f0281`, through the shared interpreter, with the withdrawal and memory changes included. Its results are a separate cohort; none is pooled with 7925f8d5 to manufacture ten-seed power.

Only eleven conditions without completed 7925f8d5 streams were selected: numeric seed 1 in both arms; 100-distractor single-literal link-first seed 7 and both arms for seeds 8–10; and deep-lag single-literal seed 1 in both arms. At most two one-worker sweeps ran concurrently. Children used stable labels, explicit sweep priority, measured reservations plus 20%, and 1800-second limits (3600 seconds for deep lags).

All 11 selected attempts are terminal: **6 completed, 5 timed out, 0 other failures**. The [completion audit](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-completion.json) verifies the exact task set, runtime, priority, labels and measured admission profiles.

**Every new statistical cell is underpowered.** Numeric and deep-lag partials have no full-stream prediction or recovery score. The missing numeric, higher-arity 100-distractor, three-target and link-first entity-holdout coverage in the frozen table remains incomplete; these selective runs do not close that matrix.

| System | Level / kind | Completed seeds | Probability Brier [95% bootstrap CI] | Recovery recall [95% bootstrap CI] | Power |
|---|---|---:|---|---|---|
| petta-link-first@171b2820 | distractors / single | 3 | 0.2786 [0.2778, 0.2797] | 0.0000 [0.0000, 0.0000] | underpowered |
| petta@171b2820 | distractors / single | 3 | 0.2786 [0.2778, 0.2797] | 0.0000 [0.0000, 0.0000] | underpowered |

These **fixed-order distractor scores are invalid as neutral distractor tests**.
Signals were inserted first and systematically excluded by the newest-24-true-facts
window. They remain historical artefact evidence, not the corrected shuffle result.

The intervals describe the available completed seeds only. Timeout-dependent selection and fewer than ten seeds preclude a powered performance claim.

### Per-cell resource evidence

Per-cell measurements; no population confidence interval is claimed. Failed runs retain only lower bounds on full-run peak RSS.

| Level / kind | Seed | Arm | Runtime | Completed steps | Complete | Peak MiB |
|---|---:|---|---|---:|---|---:|
| deep_lags / single | 1 | petta | 171b2820 | 409 | False | 11617.9+ |
| deep_lags / single | 1 | petta-link-first | 171b2820 | 438 | False | 12877.7+ |
| distractors / single | 10 | petta | 171b2820 | 256 | True | 7507.2 |
| distractors / single | 8 | petta | 171b2820 | 256 | True | 6442.5 |
| distractors / single | 9 | petta | 171b2820 | 256 | True | 7629.3 |
| distractors / single | 10 | petta-link-first | 171b2820 | 256 | True | 7511.6 |
| distractors / single | 7 | petta-link-first | 171b2820 | 230 | False | 7175.2+ |
| distractors / single | 8 | petta-link-first | 171b2820 | 256 | True | 6428.5 |
| distractors / single | 9 | petta-link-first | 171b2820 | 256 | True | 7633.3 |
| base / numeric | 1 | petta | 171b2820 | 64 | False | 4287.1+ |
| base / numeric | 1 | petta-link-first | 171b2820 | 64 | False | 4250.6+ |

Comparisons pair the same generator configuration and seed at equal completed-step counts; exact probabilities and outcomes match. Saved input prefixes are also checked where available. Killed runs can lose buffered sample logs, so the JSON states how many input rows were independently verified. A positive reduction means lower RSS. The 4e6b059b comparison also includes the withdrawal fix; 7925f8d5 is the direct pre-memory-fix comparison.

| Level / kind | Seed | Arm | Before | Shared steps | Before MiB | After MiB | Reduction | Changed predictions |
|---|---:|---|---|---:|---:|---:|---:|---:|
| deep_lags / single | 1 | petta | 4e6b059b | 97 | 2303.8 | 1872.8 | 18.7% | 92 |
| deep_lags / single | 1 | petta | 7925f8d5 | 409 | 12615.0 | 11617.9 | 7.9% | 268 |
| deep_lags / single | 1 | petta-link-first | 4e6b059b | 97 | 2299.4 | 1870.9 | 18.6% | 92 |
| deep_lags / single | 1 | petta-link-first | 7925f8d5 | 373 | 10833.5 | 9918.2 | 8.4% | 232 |
| distractors / single | 10 | petta | 4e6b059b | 68 | 2030.0 | 1516.8 | 25.3% | 63 |
| distractors / single | 8 | petta | 4e6b059b | 111 | 2518.1 | 2193.5 | 12.9% | 96 |
| distractors / single | 9 | petta | 4e6b059b | 76 | 2191.1 | 1803.0 | 17.7% | 64 |
| distractors / single | 10 | petta-link-first | 4e6b059b | 70 | 2188.7 | 1613.1 | 26.3% | 65 |
| distractors / single | 7 | petta-link-first | 4e6b059b | 68 | 2181.8 | 1743.8 | 20.1% | 59 |
| distractors / single | 8 | petta-link-first | 4e6b059b | 113 | 2571.5 | 2235.6 | 13.1% | 98 |
| distractors / single | 9 | petta-link-first | 4e6b059b | 76 | 2202.3 | 1814.1 | 17.6% | 64 |
| base / numeric | 1 | petta | 4e6b059b | 64 | 5745.4 | 4287.1 | 25.4% | 62 |
| base / numeric | 1 | petta | 7925f8d5 | 64 | 4620.9 | 4287.1 | 7.2% | 60 |
| base / numeric | 1 | petta-link-first | 4e6b059b | 64 | 5846.5 | 4250.6 | 27.3% | 62 |
| base / numeric | 1 | petta-link-first | 7925f8d5 | 64 | 4547.1 | 4250.6 | 6.5% | 60 |

Matched-history reductions range from **12.9–27.3% versus 4e6b059b** and
**6.5–8.4% versus 7925f8d5**. These workloads do not confirm a 40% reduction.
Numeric processing was slower in the new attempts, despite lower recorded RSS:

| Numeric arm, seed 1 | Shared steps | 7925f8d5 mining ms/step | 171b2820 mining ms/step |
|---|---:|---:|---:|
| petta | 64 | 5423.7 | 21799.1 |
| petta-link-first | 64 | 3622.6 | 19927.9 |

The [machine-readable comparison](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-memory.json) also reports matched-prefix mining time and input-log verification lengths. Original killed attempts sometimes lost buffered sample logs: identical generator configuration, seed, exact probabilities and labels were verified for every compared step, but only available raw input prefixes can be independently checked.

These are individual same-seed measurements, not confidence intervals on a population memory reduction. Peaks for failed runs are lower bounds and omit any increase during the unfinished step. Timings were not measured side by side at identical machine load. The revision comparison includes other upstream changes, notably refined rule truth/live-instance tallies (`7f7ea3d9`), so changed predictions and timing cannot be attributed exclusively to memory retirement.

### Historical validation and remaining limits

The historical [complete results table](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-final/report.md) retains all runtime cohorts separately. The historical aggregate repeated six controls across 117 screens; actual outcomes are {'PASS': 45, 'FAIL': 12, 'UNMEASURED': 60}. The powered base scores remain unchanged; the cost and completion gate conclusions are corrected below. New-runtime gates without ten paired seeds remain UNMEASURED, and incomplete long streams cannot pass the completion screen.

The final ChainerGame suite on 171b2820 passed: **190 passed, 25 skipped, 13 subtests passed**, with mining live tests enabled. See the [test log](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-suite-final.log). The independent 440-prefix reproducibility audit remains valid because no generator or learner code changed in this follow-up.

The benchmark is usable with explicit missing cells and the corrections below. It does not establish constant per-step PeTTa cost, full numeric recovery, ten-seed deep-lag scaling, or complete coverage of every internal directed-miner design choice. Those are remaining measurements or capabilities, not passing results.

All processes from that historical follow-up had exited before the final-review
experiments below began.

## Final review follow-up

The powered base table is unchanged. Its Markdown SHA-256 remains
`0280b046d03223db17fbde9465d5527f54e38cbcfa8c0bc01d0ddb8fde0403ed`.
The [regenerated full report](/nexus/Dev/OpenCog/bench/results/mining-bench/review4-report/report.md)
contains 4457 completed runs, including the six new shuffled distractor runs.
Its provenance independently verifies that all 760 frozen base result rows are
unchanged. Diagnostic numeric and historical-order census probes remain separate
from the powered table.

### Input order and the distractor artefact

`Petta.add_inputs` sorts the input mapping and then shuffles it with an independent
seed-and-period RNG channel. The initial history uses the same policy. The policy
is recorded as `seeded-per-period-v1`; resume and report logic keep it separate
from historical fixed-order evidence. World inputs, outcomes and exact
probabilities are identical to the paired old runs. No proof or miner rule is
filtered or repaired.

| Arm on 171b2820, D100 single, 256 steps | Seeds | Brier [95% CI] | Recall [95% CI] | Precision [95% CI] | Power |
|---|---:|---|---|---|---|
| petta, shuffled | 3 | .1179 [.1089, .1253] | 1 [1, 1] | .556 [.333, 1] | underpowered |
| petta-link-first, shuffled | 3 | .1179 [.1089, .1253] | 1 [1, 1] | .556 [.333, 1] | underpowered |
| either arm, historical fixed order | 3 | .2786 [.2778, .2797] | 0 [0, 0] | 0 [0, 0] | ordering artefact |

The arms have identical predictions for these seeds; they are three independent
seeds, not six. All three shuffled streams recover the planted rule. Two also
retain spurious sufficient rules, so this is not perfect recovery. The paired
improvement and all per-cell resource measurements are in the final experiment
summary linked below. The new maximum RSS is about 5442 MiB, versus 6429–7633 MiB
in the same fixed-order cells.

The newest-24-true-facts window remains a real candidate-access limit. In the
shuffled input rows, F0 falls within that window on 62/131, 50/122 and 61/126
periods when it is true (seeds 8, 9, 10). These are input-order opportunities,
not internal work counters: target facts and other miner state also affect its
view. Fixed order gave the signal no such opportunity. Historical D100 and
deep-lag performance must therefore not be used as neutral signal-selection
measurements; their measured resource use still describes those workloads.

### Gates and corrected miner diagnoses

The corrected primary acceptance set is **51 PASS, 0 FAIL, 8 UNMEASURED**.
Across all historical and new runtime/order cohorts it is **56 PASS, 7 FAIL,
86 UNMEASURED**. The seven residual failures belong to historical cohorts; they
are not a failure of the three corrected reference cost screens. The full report
names every gate and retains its interval, power and control evidence.

Each of the 149 screens rejects its negative fixture and accepts its positive
fixture. This reuses **six control families**, not 149 independent experiments.
The historical 117/117 statement likewise repeated six negative families; two
were identity constructions (`x-x` and zero normalized skill). These checks show
that each gate predicate can pass and fail; they do not prove every scientific
acceptance criterion is complete. In particular, a positive cost fixture is a
known-constant noisy 35µs learner, and the three actual reference learners pass
with the unchanged growth threshold after mean aggregation over types per seed.

The [lifecycle and lag audit](/nexus/Dev/OpenCog/bench/results/mining-bench/review4-diagnoses-final.json)
finds the exact AND2 pair in **all ten seeds of both arms**; eight are retired and
two survive. Thus final recall .20 is a **review/retirement defect**, not missing
pair proposals. Seed 2 proposes the pair at step 1 and drops it at step 7. At the
drop, the pair reads .99875 and F1 reads .66637 on roughly four versus six
observations; their difference is inside the two-standard-error review band.
F1's earlier near-one estimate is evidence of the small-sample problem, but it
is not its value at the drop. The tried set prevents the dropped pair from being
proposed again.

The lag seed-1 prediction hashes are byte-identical between the two arms
(`fad0dba30642565bc9f49f9bec587339fd2f5534679a1473dfab959bf3e53303`).
The miner's **present-first** policy suppresses cross-link proposals once a
same-period hypothesis concludes the target. In this experiment the second arm
separates nothing; it is not evidence that either link ingestion order solves
the lag problem. The positive long-stream result remains: both arms completed
10/10 plain 1024-step streams, with fitted mean growth near zero as reported
above. Deep-lag D100 censoring is a distinct completion gate.

### Numeric feasibility and timing

A live width-8, seed-1, 32-step probe on 171b2820 completed with Brier **.1090**,
behavioural recall **22/54 = .4074**, and precision **22/31 = .7097**. It used
shuffled inputs. This is one short feasibility diagnostic, not a powered
numeric base cell; all 256-step numeric base coverage remains incomplete.
Its mean mining time was 19.226 seconds per step under shared-pool load and is
not compared to the reserved-pool timing experiment.

The probe also exposes an adapter limit: variable-output numeric rules are not
exported as categorical predictions. Of thirteen such raw verdict entries,
twelve are dropped; the only kept one has strength .5, below the .9 recovery
threshold. It therefore does not change this probe's sufficient-rule recall,
but its association calibration is omitted. This export limitation must not be
misreported as proof that PeTTa cannot propose variable numeric outputs. General
arithmetic formula discovery is still unshown.

The matched timing experiment uses fresh processes, width 32, seed 1, 64 steps,
budget 40, the same historical input order, the same shared interpreter and
PeTTa VM, and full-pool CPU reservations at sweep priority. Historical ordering
is explicit only in diagnostic repros; production ingestion remains shuffled.
Its results are recorded separately below when both revisions finish.

### Memory census and validation

Read-only censuses use SWI `predicate_property(number_of_clauses/size)` at two
history lengths. They do not materialize rows, invoke GC, or modify stores.
The census itself is outside the mining timer. Predicate allocation estimates
are not an exact partition of process RSS.

The historical-order D100 seed-9 reproduction reaches **7633.6 MiB** at 256
steps, matching the earlier roughly 7629 MiB peak. At 64 versus 256 steps,
skeleton allocation grows from **667.6 to 4060.3 MiB**, out of **892.5 to
5090.6 MiB** across measured dynamic predicates. The four requested hypothesis
stores together remain below 1 MiB: `hm_seen` has 35→29 rows, `hm_period`
2520→2520, `hm_outcomes` 1→1, and `hm_tried` 171→292. At 256, the largest stores
include `&sk_row` (307463 rows, 801.6 MiB), `&sk_wait` (225608, 536.8 MiB),
`&sk_latest` (283140, 346.0 MiB), and `sk_search_clause` (616274, 292.8 MiB).
The retained skeleton workload, rather than those four miner stores, dominates
the measured allocation. Two points do not establish an asymptotic bound.

The historical-order deep-lag reproduction also completed its requested 409
steps: RSS grows from **1222.4 MiB at 64** to **11619.0 MiB at 409**, closely
reproducing the old 11617.9 MiB measurement. Dynamic predicate allocation grows
from **816.2 to 7718.8 MiB**, of which skeleton stores hold **622.3 to
6146.1 MiB**. The four miner stores again total less than 1 MiB: seen 4→6 rows,
period 2520→2520, outcomes 1→1, tried 146→291. This identifies the dominant
retained stores without attributing every byte of RSS or modifying the engine.

The complete ChainerGame suite passed with mining live tests enabled on the
pinned 171b2820 runtime: **192 passed, 25 skipped, 15 subtests passed**
([log](/nexus/Dev/OpenCog/bench/results/mining-bench/review4-suite-verified.log)).
Focused tests cover seeded ordering, mean cost aggregation, positive and negative
controls, completion separation and runtime/order cohort preservation. No upstream
PeTTaChainer files were modified.

## Commits

- `ebc5afa` through `31a8e38`: reviewer corrections, behavioural probes, immutable evidence replay and powered cohorts.
- `12827e0`: saved-evidence report, committed before the selective follow-up.
- `649dbce`: measured admission profiles, stable labels, explicit sweep priority and runtime-pin checks.
- `0872431`: matched-history memory comparison and seed-clustered ancillary confidence intervals.
- `5e19173`: qualify missing input-log verification and retain matched-prefix processing time.
- `55b9b57`: deterministic input order, calibrated cost and completion screens, positive controls and census.
- `a660b8f`: reproducible review diagnostics and separate shuffled failure cohorts.
- `0bc753b`: directed completion labels and matched numeric timing inputs.
- `8d4c8bd`: complete report provenance and all-seed AND2 retirement audit.
- The earlier documentation commit and branch history are in [the commit record](/nexus/Dev/OpenCog/bench/results/mining-bench/review-v3-171-commits.txt). No push or merge was performed.
