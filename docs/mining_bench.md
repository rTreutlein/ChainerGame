# Planted temporal pattern mining

This is ChainerGame benchmark code. It does not modify PeTTaChainer or implement
the proposed directed miner. It evaluates discovery against known mechanisms,
not only prediction against sampled labels. PeTTaChainer is frozen at
`4e6b059be39537d2c7c8a28e5dc807fb4f3a83c9`.

## Running

From `/nexus/Dev/OpenCog/ChainerGame-mining`:

```bash
unset DISPLAY
ulimit -v 16000000
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export TMPDIR="$PWD/.scratch"
export MININGBENCH_CAPPED=1
mkdir -p "$TMPDIR"
P=/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python
R=/nexus/Dev/OpenCog/bench/results/mining-bench
/nexus/Dev/OpenCog/bench/capped --slots 1 "$P" -m miningbench run \
  --system marginal --samples 256 --seed 1 --out "$R/example"
/nexus/Dev/OpenCog/bench/capped --slots 4 --peak-mb 5000 "$P" -m miningbench sweep \
  --systems oracle marginal petta --separate-kinds --samples 64 --seeds 1 \
  --budget 40 --jobs 4 --timeout 90 --out "$R/matrix"
/nexus/Dev/OpenCog/bench/capped --slots 1 "$P" -m miningbench report "$R/matrix"
```

A sweep's admission slot count must cover `--jobs`. Children run sequentially
within each worker, inheriting the parent's capped scope and address-space limit.
Each child has its own interpreter and a GNU timeout with TERM and a 10-second
KILL grace period. Jobs run in the foreground; the report is written after all
children finish. A failed child leaves a process log, status and partial raw trace;
it is not a completed result. The sweep exits nonzero on any failure.

`--kinds` selects any mixture of families. `--targets` makes that many copies of
each requested family. Without `--separate-kinds`, the entire mixture competes in
one learner. Results from different mixtures should not be silently compared as
if they were identical workloads. `run` takes each difficulty knob directly;
`sweep` uses fixed, published levels below. The system's complete query budget is
shared by all target roots; PeTTa reviews use the API's per-new-hypothesis budget.

## Generator and ground truth

`world.py` uses SHA-256-derived independent RNG streams for each target, label
channel and distractor. Adding distractors or target copies preserves the original
streams. A longer stationary stream preserves every shorter prefix. Drift occurs
at half the requested stream, so drifted streams of different lengths intentionally
have different changepoints. Three unscored input times initialize the lag window.
All predictions precede the new outcome label. Past labels may be learned from;
current labels and the ground-truth probability are inaccessible to non-oracles.

Each Boolean target has four independent fair input predicates, with irrelevant
ones retained for small rules. Inputs are fully observed at every time, including
explicit false observations. Independent fair distractor predicates are added in
the same way. A binding task instead observes one categorical `At(entity)` and
one categorical `Did(entity)` over four entities. Numeric tasks observe an integer
coordinate on a reflecting unit-velocity path in `[0,4]`. These are raw coordinate
facts: the learner is not given direction, wall indicators or formula templates
as extra features.

| Family | Planted mechanism (all unspecified lags are 1) |
|---|---|
| single | `A -> Y` |
| and2 / and3 / and4 | conjunction of 2 / 3 / 4 inputs |
| negation | `A and not B -> Y` |
| lag | `A[t-1] and B[t-2] and C[t-3] -> Y` |
| xor | `A and not B -> Y`; `not A and B -> Y` |
| or | `A and B -> Y`; `C and D -> Y`, including overlapping cases |
| numeric | interior: `x_next = 2*x1-x2`; at either wall: `x_next = x2` |
| binding | `At($x)[t-1] and Did($x)[t-1] -> Y`, with one shared variable |

The current numeric task measures a next-coordinate outcome channel driven by an
exogenous reflecting coordinate stream. Drift changes the outcome law to `x2` in
the interior; it does not change that input stream. This distinction matters when
interpreting it as a simulation of physical motion.

For Boolean target rules with active firing strengths `s_i`, the true outcome
probability is `1 - product(1-s_i)`. Firings are independent; overlapping rules are
not counted as independent *observations*. Label observation noise `eta` flips
the Boolean outcome, giving `q = eta + (1-2*eta)*p`. Numeric rule failure produces
a uniform coordinate, as does numeric observation noise, so the true coordinate
has mass `(1-eta)*s + (1-(1-eta)*s)/5`, with the remaining mass uniform. A categorical
label is sampled from this exact distribution.

Input observation noise is deliberately absent. Corrupting hidden inputs in a
persistent temporal process would require conditioning on past labels to obtain
an exact posterior; treating noisy inputs as independent would give false ground
truth. The implemented observation-noise knob corrupts the *outcome observation
channel*. All probabilities are therefore exact given the available history,
not empirical estimates or Monte Carlo approximations. Soft input evidence is an
instrumentation-contract case, not part of these generated runs.

Drift complements the first input in Boolean rules (XOR becomes XNOR), removes
the `e0` case from entity equality, and changes the numeric interior formula.
`world.json` records both rule regimes, all strengths, domains, configuration and
source provenance. `samples.jsonl` records observed lag histories, phase, label and
exact probability for every evaluated time. It is evaluator data, never an adapter
input. `steps.jsonl` records each prediction, score, cost, target request and miner
event. The stream has a deterministic SHA-256 digest.

## Recovery definition

A learned rule is a target/value conclusion and a conjunction of signed literals;
a literal contains predicate, arguments, lag and polarity. Repeated literals are
removed. Contradictory Boolean conjunctions vanish. For ground Boolean rules,
consensus and absorption compute the Blake cover: all prime implicants of the DNF
per conclusion. This accepts reordering, duplicated rules, absorbed special cases
and logically equivalent DNF decompositions. A 4096-cube safety limit raises an
error instead of silently approximating equivalence.

Variables in a relational conjunction are renamed canonically by trying all
bijections (maximum six variables), preserving equal positions. Independent
existential conjunctions are **not** combined by propositional consensus. Finite
lists of constant-specific rules do not count as a universally generalized rule.
Thus relational recovery is alpha-equivalence of conjunctions, not unrestricted
first-order theorem proving. The current population contains four entities; no
novel-entity generalization score is claimed.

Numeric conclusions use an exact affine-expression normal form with rational
coefficients over `x1`, `x2`, `x3`, so `x1+x1-x2` matches `2*x1-x2`. Region conditions
use named structural cases (`interior`, `wall0`, `wall4`) in ground truth; the
current PeTTa adapter cannot express these formulas. Equivalent piecewise formula
systems outside this template vocabulary require a new normalization adapter.
Memorizing coordinate transitions does not recover a formula.

Precision is matched normalized rules divided by learned normalized rules; recall
is matched divided by planted. Empty learned precision is zero. Both are reported
per family and overall, along with exact rule-set equivalence. Strength estimation
is evaluated by prediction, separately from structural recovery. With stochastic
rules, logically equivalent Boolean structure need not imply identical noisy-OR
rates; probability scoring detects that difference. Recovery curves use the active
regime, so obsolete rules after drift are false positives.

## Prediction and cost

For each target distribution `q` and learner distribution `p`, log loss is
`-sum(q*log(p))` (probabilities floored at `1e-12` only for the logarithm).
Excess log loss subtracts the entropy of `q`. Brier is the mean squared difference
between `p` and `q` over categories; in a Boolean task it is `(p_true-q_true)^2`.
Expected label Brier additionally includes the irreducible term
`mean(q*(1-q))`. The oracle has zero probability Brier and excess log loss, but
has positive ordinary log loss and expected label Brier in noisy worlds.

Scores are prequential, including cold start; eight approximately equal windows
also show cumulative error and structural recovery versus sample count. Scores
are macro-averaged across target/time pairs. Numeric Brier uses a five-category
mean, so absolute binary/numeric Briers have different scales; compare systems
within a family. Timing separates prediction and feedback/review, and reports
end-to-end learner time, early/late quarters and the least-squares time slope.
Input ingestion in the PeTTa adapter occurs during prediction and is included in
end-to-end learner cost. Feedback cost alone is **not** all mining cost.

The constant-cost screen requires at least 64 samples and a late-quarter mean at
most twice the early-quarter mean. This is a measurement screen, not a proof of
O(1). The marginal learner has a fixed-size sufficient-statistics table and does
O(features * targets * categories) work per sample, independent of history length.
It records its update work count. Increasing features or targets may increase cost;
constant per step means constant **in stream length at fixed configuration**.
PeTTa has no public candidate-update counter; its screen uses wall time and makes
no stronger complexity claim.

Peak RSS is Linux process peak memory including the interpreter, learner and
benchmark. Sweeps isolate processes. The harness streams raw records and maintains
constant-space score and cost accumulators plus a bounded number of recovery
snapshots; it does not retain the whole stream in memory. Numeric durations and
RSS depend on machine load, allocator and runtime; a seed reproduces data,
predictions and score arithmetic, not wall-clock or memory measurements.

## Systems and PeTTa encoding

The oracle evaluates the planted mechanism with the active regime. It is an
explicit evaluator-only system. The marginal reference searches all observed
single literals at lags 1–3, including categorical value indicators, and chooses
the empirical least-squared-error stump separately for every target. Both branch
rates use Jeffreys smoothing. Its recovered rules require at least eight total
samples, four samples on that branch, positive-branch uplift over the base rate
and a squared-error gain above 0.005. It cannot emit a conjunction or shared
variable. The benchmark does not install a classical tree learner.

PeTTa uses `set_hypothesis_mining(4,8)`, evidence k=1, predicate time declarations,
`NextPeriod`, observed targets and the current failure-mining path. The query and
review budget in the measured matrix is 40. It receives no planted rules and no
synthetic negated, equality, velocity or lag-feature predicates. Inputs are stored
as explicit true/false facts. `y_t` is encoded as the end-of-period outcome of
context `x_(t-1)`: input layer 0 and outcome layer 1 share a decision-period clock.
The adapter reverses this transformation when extracting rules: same-period input
is event lag 1; a previous-period input is event lag 2.

It predicts the outcome before adding labels. It then adds positives and explicit
negatives for target conclusions outside the previously known live anticipation
domain. The closing link lets observed predicates supply false observations for
missed conclusions inside that domain, including conclusions whose learned rate
is low. This preserves samples for never-successful targets while exercising
anticipation for existing predictions. Reviews return actual kept/dropped rules.
Raw verdicts and truth values remain in the trace. For a CTV, the branch with higher target probability is the positive association
exported for recovery, even when its strength is below 0.5. A higher negative
branch is expanded with De Morgan into positive-target DNF rules. An STV must
exceed the empirical target marginal. This tests association, not majority-class
prediction, and supports weak planted rules. Unparsed verdicts are exposed in
results and must be inspected before trusting recovery.

The time-link repro under `benchmarks/mining/repros/` shows an actual boundary in
the frozen miner: with identical facts, link-first ingestion proposes
`NextPeriod(q,p) and F(entity,q) -> Y(entity,p)` at strength 0.99938, but link-last
ingestion proposes none. `hm-observe!` registers a new link without revisiting
prior endpoints. Observed predicates require link-last closure, so this is relevant
to natural cross-period ingestion. Equal layers in the repro prevent a
contemporaneous rule from masking the missing temporal candidate. The benchmark's
same-period context encoding avoids this issue for event lag 1; it does not repair
or bypass it for deeper lags. Raw repro: `time-link-order.log` in the result root.

```bash
benchmarks/mining/repros/run.sh time-link-order
benchmarks/mining/repros/run.sh xor
benchmarks/mining/repros/run.sh numeric
```

## Measuring the directed-miner design

`trace.py` is an adapter-facing trace contract for the decisions in
`mm2-chainer/docs/directed-pattern-mining.md`. An adapter receives public feature
and target-domain schemas, implements `predict(history)`, `learn(history, labels)`
and `rules()`, and may implement `set_targets(requests)`. It can be supplied to
`runner.run(..., system="custom", adapter=Factory)`. The evaluator never supplies
rules, phase or exact probabilities to that factory.

The next step's requests contain observed surprise, predictive entropy and missing
learned-target coverage as a gap signal. Requests use previous labels/predictions
only. Every third competing target is assigned zero decision relevance; the other
targets have unit relevance. This is a controlled target-selection probe, not a
game with delayed rewards. Recovery and probability scores still cover all targets.
A future chainer adapter can substitute its genuine backward-gap diagnostics.

| Design decision | Instrumentation and measurement |
|---|---|
| Surprise / uncertainty / gap targets, decision relevance | `target` events; requests; irrelevant-target share |
| Residual-guided one-conjunct refinement | `split`: parent, literal, residual case IDs and gain |
| Shared gain/cost queue, lag priors, deeper search | `queue`: chosen priority regret, lag and prior |
| Numeric conditions and formulas, learned constants | `template`: template family, role, used/bound variables, provenance of constant; reject unbound variables/enumeration |
| Pair sketches after single splits stall, XOR cells | `pair`: complete 2x2x2 outcome cells, interaction contrast and premature-pair fraction |
| Separate disjuncts and overlap refinements | canonical DNF recovery; `overlap`: joint support/rate |
| Strength-weighted uncertain conditions | `soft_sample`: positive mass equals strength, negative mass equals complement |
| Anti-unification, equality, pooled evidence | `generalize`: compatible skeletons, shared variables, cap, unique pooled case IDs; binding recovery |
| Specialization preserves exceptions | `specialize`: exception support; numeric wall recovery |
| Penalized predictive usefulness and promotion | `promote`: positive parent-minus-child loss less complexity penalty, confidence, support |
| Retirement and decision irrelevance | `retire`: reason and opportunity count; drift recovery curves |
| Indexable concrete compilation and backfill | `compile`: indexability, unique backfilled cases, duplicate sample count |
| Constant work, beam, bounded generation/sketches | `budget`: updates <= active targets * beam, proposals <= cap; measured cost curves |
| Later predicate invention | `invent`: reuse count and defining condition; optional, no current system implements it |

Raw events remain in `steps.jsonl`; the audit retains bounded aggregates and at
most 32 violation examples. Missing fields or telemetry are reported explicitly.
The current PeTTa API exposes candidates, reviews and withdrawals, not queue
priorities, residual case IDs, compiler backfills or candidate update counts. Its
trace therefore does **not** certify these internal design decisions. No such
absence is counted as a successful constant-cost or promotion check. The fixture
generators exercise the present Boolean/numeric/binding vocabulary; arbitrary
user-declared templates, synthesized expression trees, uncertain input worlds and
invented predicates are contract extension points, not measured capability claims.

## Acceptance interpretation

The fixed checks require zero oracle excess error and full oracle recovery;
an oracle-minus-marginal Brier advantage exceeding 0.01 in every non-single family;
zero marginal XOR recovery; and harder-knob degradation of mean error by >0.002,
or >20% cost growth for distractors/targets. Noise checks use expected label Brier,
because probability estimation can become easier as the generating law approaches
a constant. Thresholds are fixed in `report.py`, not adjusted to measured scores.
The checks report FAIL and UNMEASURED as well as PASS.

More samples is easier; the sample-count difficulty setting uses one quarter as
many samples. Other levels are 100 distractors, strength 0.7, outcome observation
noise 0.2, one midpoint drift, and three simultaneous copies. Base has strength 1,
no noise, no distractors and one target per selected family.

The requested blanket monotonicity does not hold mathematically for all miners and
rules. For fair XOR with a constant-0.5 predictor, observation flips change exact
probabilities from {0,1} to {eta,1-eta}, reducing probability Brier to
`(0.5-eta)^2`; label Brier remains 0.25. Reducing firing strength can also improve
a sparse target's baseline. A miner already at zero recall cannot lose more
recall. We preserve failures of the acceptance checks instead of changing worlds
or scoring thresholds to manufacture monotonic degradation.

## Validation and measured results

The main matrix contains **190 completed runs and 20 timed-out runs**, seed 1,
64 samples (16 for the harder sample-count level), budget 40. All 70 oracle and
70 marginal runs completed. PeTTa completed 50 runs; every 100-distractor and every
three-target run hit the 90-second child timeout. Partial traces are retained and
are not used as full-stream scores. `final/provenance.json` identifies the original
source directory of each copied, immutable raw run.

The oracle has zero probability Brier and excess log loss, and precision/recall 1
in every case. Across **all seven difficulty levels**, every non-single marginal
case is worse than its paired oracle by more than 0.01 Brier; the smallest gap is
0.0281247. The marginal learner's XOR recovery is zero throughout.

Base cases:

| Family | Marginal Brier | PeTTa Brier | PeTTa precision | PeTTa recall |
|---|---:|---:|---:|---:|
| and2 | 0.19030 | 0.20479 | 0.00 | 0.00 |
| and3 | 0.13472 | 0.09164 | 0.00 | 0.00 |
| and4 | 0.05436 | 0.03170 | 0.00 | 0.00 |
| binding | 0.15745 | 0.16778 | 1.00 | 1.00 |
| lag | 0.13948 | 0.14879 | 0.00 | 0.00 |
| negation | 0.14855 | 0.19443 | 0.00 | 0.00 |
| numeric | 0.14047 | 0.14806 | 0.00 | 0.00 |
| or | 0.19101 | 0.20737 | 0.50 | 0.50 |
| single | 0.01261 | 0.26754 | 1.00 | 1.00 |
| xor | 0.29494 | 0.26682 | 0.00 | 0.00 |

Difficulty averages below give each family equal weight. A dash is an incomplete
run, not a missing prediction filled with a favorable default.

| Difficulty | Marginal probability Brier | PeTTa probability Brier | PeTTa expected label Brier | PeTTa completed |
|---|---:|---:|---:|---:|
| base | 0.14639 | 0.17289 | 0.17289 | 10/10 |
| distractors | 0.17183 | — | — | 0/10 |
| strength | 0.08962 | 0.08964 | 0.15043 | 10/10 |
| observation_noise | 0.08354 | 0.08043 | 0.23019 | 10/10 |
| samples | 0.17796 | 0.18791 | 0.18791 | 10/10 |
| drift | 0.18051 | 0.17553 | 0.17553 | 10/10 |
| targets | 0.15957 | — | — | 0/10 |

Acceptance is **not fully satisfied**. In the isolated matrix, all six marginal
knob checks pass, but reducing strength improves PeTTa's mean expected label Brier
by 0.02246, failing monotone degradation. Distractor and target-count accuracy
comparisons for PeTTa are unmeasured because those runs timed out. Resource failure
is observed, but is not substituted for a numeric accuracy degradation.

The independent mixed-world check uses all ten families together, 256 samples,
seeds 1/2/3 and all seven levels: 42 completed oracle/marginal runs. Its strength
check fails too: expected label Brier improves by 0.00475. This is preserved as a
counterexample to blanket monotonicity, not removed from the report. The mixed
PeTTa pilot completed 19 of 64 steps before its separate 180-second timeout;
it is not included in scored comparisons.

End-to-end cost screening passes all 60 eligible oracle and 60 marginal cases.
PeTTa fails in 5 of 40 eligible completed cases. These are **end-to-end** screens,
including predictive queries; they do not isolate the miner's internal CPU time.
The runner now also records `mining_path_seconds` and `mining_cost`: input ingestion
plus feedback/review for PeTTa, excluding predictive queries; all sufficient-table
work for the marginal learner. This is the externally callable mining path,
including fact compilation, not a claim to isolate an internal Prolog predicate.
The main matrix predates this additional timing split; separate 128-step single
and binding cost probes under `cost-probes/` record it without altering scored
matrix results. Feedback cost and total cost remain available in every matrix row.

| Family (128 steps) | Mining path ms/step | Mining late/early | Total ms/step | Total late/early | Peak MiB |
|---|---:|---:|---:|---:|---:|
| binding | 42.475 | 0.39 | 205.317 | 1.47 | 471.2 |
| single | 29.541 | 0.73 | 106.216 | 0.97 | 270.7 |

These probes completed separately at 128 samples. Their source patch against
`a81403b` is retained in `cost-probes/source.patch`; only timing attribution
changed. A screen passing here does not override an earlier failed screen or
prove bounded internal work.

Six independent replays cover oracle, marginal and PeTTa on single and XOR worlds.
Data digests, every prediction, scores, recovery and learned rules match exactly.
See `reproducibility.json`; timing and memory are deliberately excluded. This
verifies these repeats, not every cell of the matrix under arbitrary environments.

Validation: the full ChainerGame suite passed **173 tests, 25 skipped**, including
the benchmark's explicitly enabled live test. The final focused suite is also
recorded in `timing-validation.log`. Pytest and plotting dependencies were installed
only under this worktree's ignored `.test-deps/`; PeTTa's shared environment was
not modified. Use `PYTHONPATH=src:.test-deps` with its shared interpreter to reproduce
that invocation. `MININGBENCH_LIVE_PETTACHAINER=1` enables the new live test; the
existing benchmarks keep their own opt-in gates.

Raw artifacts:

- Main matrix and full table: `/nexus/Dev/OpenCog/bench/results/mining-bench/final/`
- Three-seed mixed-world matrix: `/nexus/Dev/OpenCog/bench/results/mining-bench/mixed/`
- Replay verification: `/nexus/Dev/OpenCog/bench/results/mining-bench/reproducibility.json`
- Learning curves: `final/learning_curves.svg` and `final/learning_curves.png`
- Gaps and commands: `benchmarks/mining/repros/README.md`
- `matrix/` retains the first live extraction, `petta-final/` the corrected
  association-based extraction, and `grid-initial/` the preliminary 128-step grid.
  Do not pool these preliminary/duplicate directories into the final scores.

The exact matcher and probability ground truth have unit coverage for every
Boolean truth table, independent overlapping firings with noise, XOR marginals,
wall reflection, shared entity binding, drift, affine algebra, alpha renaming,
Boolean consensus/absorption, prequential ordering, replay, evidence preservation
and trace-contract violations. Numeric formula discovery, mixed-polarity XOR
proposals, three-/four-input proposals and general lag search remain current-miner
capability gaps. The binding result demonstrates an existing supported form.
The single-literal predictive-read discrepancy and late-link issue have runnable
repros; their upstream implementation is left unchanged.

Limits: the main PeTTa matrix has one seed and short streams; timings share a busy
machine. Inputs are fully observed; noise affects labels. Entity generalization
has no novel-entity holdout. Numeric formulas and matching use a restricted affine
vocabulary. The current API cannot expose all directed-miner internals; absent
audit events remain unmeasured. No classical tree learner, production miner,
chainer patch or guarantee of monotone degradation is supplied.

### Complete system × family × difficulty table


| Difficulty | System | Type | Seeds | Brier | Excess log loss | Precision | Recall | ms/step | Late/early | Peak MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | marginal | and2 | 1 | 0.19030 | 0.54149 | 0.000 | 0.000 | 0.011 | 0.97 | 20.1 |
| base | marginal | and3 | 1 | 0.13472 | 0.39258 | 0.000 | 0.000 | 0.013 | 0.95 | 20.1 |
| base | marginal | and4 | 1 | 0.05436 | 0.21031 | 0.000 | 0.000 | 0.012 | 0.99 | 19.8 |
| base | marginal | binding | 1 | 0.15745 | 0.50756 | 0.000 | 0.000 | 0.020 | 1.06 | 20.1 |
| base | marginal | lag | 1 | 0.13948 | 0.40548 | 0.000 | 0.000 | 0.012 | 0.88 | 20.1 |
| base | marginal | negation | 1 | 0.14855 | 0.42132 | 0.000 | 0.000 | 0.011 | 0.85 | 20.1 |
| base | marginal | numeric | 1 | 0.14047 | 1.22853 | 0.000 | 0.000 | 0.018 | 0.98 | 20.1 |
| base | marginal | or | 1 | 0.19101 | 0.57748 | 0.000 | 0.000 | 0.011 | 0.89 | 19.8 |
| base | marginal | single | 1 | 0.01261 | 0.07215 | 1.000 | 1.000 | 0.012 | 0.95 | 20.1 |
| base | marginal | xor | 1 | 0.29494 | 0.81557 | 0.000 | 0.000 | 0.012 | 0.90 | 19.8 |
| base | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.91 | 20.1 |
| base | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.74 | 20.1 |
| base | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.69 | 20.1 |
| base | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.006 | 1.15 | 20.3 |
| base | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.93 | 20.1 |
| base | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.002 | 0.90 | 19.8 |
| base | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.98 | 20.3 |
| base | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.81 | 20.1 |
| base | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.002 | 0.84 | 20.1 |
| base | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.86 | 20.1 |
| base | petta | and2 | 1 | 0.20479 | 0.63529 | 0.000 | 0.000 | 116.123 | 1.10 | 190.7 |
| base | petta | and3 | 1 | 0.09164 | 0.29190 | 0.000 | 0.000 | 110.744 | 0.94 | 185.8 |
| base | petta | and4 | 1 | 0.03170 | 0.15188 | 0.000 | 0.000 | 82.079 | 0.92 | 172.7 |
| base | petta | binding | 1 | 0.16778 | 0.59178 | 1.000 | 1.000 | 225.359 | 2.37 | 308.0 |
| base | petta | lag | 1 | 0.14879 | 0.52481 | 0.000 | 0.000 | 110.782 | 0.97 | 189.2 |
| base | petta | negation | 1 | 0.19443 | 0.58264 | 0.000 | 0.000 | 102.700 | 0.98 | 186.5 |
| base | petta | numeric | 1 | 0.14806 | 1.44682 | 0.000 | 0.000 | 137.555 | 1.43 | 187.1 |
| base | petta | or | 1 | 0.20737 | 0.67571 | 0.500 | 0.500 | 114.478 | 1.14 | 190.9 |
| base | petta | single | 1 | 0.26754 | 0.76992 | 1.000 | 1.000 | 114.472 | 0.94 | 188.6 |
| base | petta | xor | 1 | 0.26682 | 0.79423 | 0.000 | 0.000 | 121.808 | 0.97 | 193.0 |
| distractors | marginal | and2 | 1 | 0.23507 | 0.67893 | 0.000 | 0.000 | 0.221 | 0.89 | 21.1 |
| distractors | marginal | and3 | 1 | 0.14553 | 0.42736 | 0.000 | 0.000 | 0.198 | 1.07 | 20.9 |
| distractors | marginal | and4 | 1 | 0.06285 | 0.23115 | 0.000 | 0.000 | 0.213 | 1.06 | 21.1 |
| distractors | marginal | binding | 1 | 0.21064 | 0.70022 | 0.000 | 0.000 | 0.218 | 1.20 | 21.1 |
| distractors | marginal | lag | 1 | 0.20007 | 0.57298 | 0.000 | 0.000 | 0.213 | 1.04 | 20.8 |
| distractors | marginal | negation | 1 | 0.15784 | 0.46078 | 0.000 | 0.000 | 0.202 | 1.05 | 21.1 |
| distractors | marginal | numeric | 1 | 0.14845 | 1.34173 | 0.000 | 0.000 | 0.281 | 0.98 | 21.1 |
| distractors | marginal | or | 1 | 0.24319 | 0.70999 | 0.000 | 0.000 | 0.201 | 1.08 | 21.1 |
| distractors | marginal | single | 1 | 0.01261 | 0.07215 | 1.000 | 1.000 | 0.220 | 1.23 | 21.1 |
| distractors | marginal | xor | 1 | 0.30209 | 0.86643 | 0.000 | 0.000 | 0.212 | 1.07 | 21.1 |
| distractors | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 1.02 | 20.8 |
| distractors | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.97 | 20.6 |
| distractors | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 1.02 | 20.8 |
| distractors | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.007 | 0.77 | 20.8 |
| distractors | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.59 | 20.6 |
| distractors | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.71 | 20.6 |
| distractors | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.87 | 20.6 |
| distractors | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.74 | 20.8 |
| distractors | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 1.09 | 20.6 |
| distractors | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.84 | 20.6 |
| drift | marginal | and2 | 1 | 0.19349 | 0.56176 | 0.000 | 0.000 | 0.011 | 0.94 | 19.8 |
| drift | marginal | and3 | 1 | 0.14619 | 0.44134 | 0.000 | 0.000 | 0.012 | 0.90 | 20.1 |
| drift | marginal | and4 | 1 | 0.08059 | 0.28100 | 0.000 | 0.000 | 0.012 | 0.85 | 19.9 |
| drift | marginal | binding | 1 | 0.15745 | 0.50756 | 0.000 | 0.000 | 0.020 | 0.98 | 20.3 |
| drift | marginal | lag | 1 | 0.16111 | 0.47725 | 0.000 | 0.000 | 0.013 | 0.69 | 20.1 |
| drift | marginal | negation | 1 | 0.14455 | 0.41334 | 0.000 | 0.000 | 0.012 | 1.01 | 20.3 |
| drift | marginal | numeric | 1 | 0.15398 | 1.45274 | 0.000 | 0.000 | 0.018 | 0.92 | 20.1 |
| drift | marginal | or | 1 | 0.20429 | 0.60014 | 0.000 | 0.000 | 0.012 | 0.87 | 20.1 |
| drift | marginal | single | 1 | 0.25499 | 0.73201 | 0.000 | 0.000 | 0.011 | 0.90 | 20.1 |
| drift | marginal | xor | 1 | 0.30845 | 0.84300 | 0.000 | 0.000 | 0.012 | 0.89 | 20.1 |
| drift | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 1.35 | 19.8 |
| drift | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.86 | 20.1 |
| drift | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.62 | 19.8 |
| drift | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 1.06 | 20.1 |
| drift | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.75 | 19.8 |
| drift | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.95 | 20.1 |
| drift | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.88 | 20.3 |
| drift | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.93 | 19.8 |
| drift | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.71 | 20.1 |
| drift | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.64 | 20.1 |
| drift | petta | and2 | 1 | 0.20183 | 0.63130 | 0.000 | 0.000 | 124.098 | 1.22 | 190.0 |
| drift | petta | and3 | 1 | 0.15199 | 0.59043 | 0.000 | 0.000 | 118.615 | 1.01 | 188.9 |
| drift | petta | and4 | 1 | 0.06083 | 0.26120 | 0.000 | 0.000 | 99.004 | 1.19 | 183.5 |
| drift | petta | binding | 1 | 0.16778 | 0.59178 | 0.000 | 0.000 | 237.188 | 2.52 | 307.4 |
| drift | petta | lag | 1 | 0.17179 | 0.58187 | 0.000 | 0.000 | 114.231 | 1.08 | 188.3 |
| drift | petta | negation | 1 | 0.14003 | 0.46058 | 0.000 | 0.000 | 104.309 | 1.01 | 186.5 |
| drift | petta | numeric | 1 | 0.14807 | 1.44544 | 0.000 | 0.000 | 155.227 | 1.36 | 186.9 |
| drift | petta | or | 1 | 0.18878 | 0.63750 | 0.000 | 0.000 | 115.370 | 1.13 | 190.1 |
| drift | petta | single | 1 | 0.25701 | 0.74860 | 0.000 | 0.000 | 123.104 | 1.16 | 189.2 |
| drift | petta | xor | 1 | 0.26715 | 0.79500 | 0.000 | 0.000 | 124.025 | 1.08 | 189.7 |
| observation_noise | marginal | and2 | 1 | 0.08305 | 0.18433 | 0.000 | 0.000 | 0.012 | 0.95 | 19.8 |
| observation_noise | marginal | and3 | 1 | 0.09964 | 0.22310 | 0.000 | 0.000 | 0.013 | 0.95 | 20.1 |
| observation_noise | marginal | and4 | 1 | 0.02812 | 0.09336 | 0.000 | 0.000 | 0.012 | 0.93 | 20.1 |
| observation_noise | marginal | binding | 1 | 0.10485 | 0.28224 | 0.000 | 0.000 | 0.019 | 1.11 | 19.8 |
| observation_noise | marginal | lag | 1 | 0.08110 | 0.23042 | 0.000 | 0.000 | 0.012 | 0.90 | 20.1 |
| observation_noise | marginal | negation | 1 | 0.07561 | 0.19980 | 0.000 | 0.000 | 0.012 | 0.90 | 19.8 |
| observation_noise | marginal | numeric | 1 | 0.09797 | 0.84686 | 0.000 | 0.000 | 0.019 | 0.93 | 20.1 |
| observation_noise | marginal | or | 1 | 0.10435 | 0.25377 | 0.000 | 0.000 | 0.012 | 0.89 | 20.1 |
| observation_noise | marginal | single | 1 | 0.01073 | 0.03673 | 1.000 | 1.000 | 0.011 | 0.91 | 19.8 |
| observation_noise | marginal | xor | 1 | 0.14993 | 0.33815 | 0.000 | 0.000 | 0.013 | 0.73 | 20.1 |
| observation_noise | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.92 | 20.1 |
| observation_noise | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.94 | 19.9 |
| observation_noise | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.76 | 19.8 |
| observation_noise | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 1.47 | 20.1 |
| observation_noise | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.83 | 19.8 |
| observation_noise | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.93 | 20.1 |
| observation_noise | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.99 | 19.8 |
| observation_noise | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.77 | 20.1 |
| observation_noise | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.002 | 0.94 | 20.1 |
| observation_noise | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.92 | 20.1 |
| observation_noise | petta | and2 | 1 | 0.08050 | 0.21475 | 1.000 | 1.000 | 123.969 | 1.10 | 190.3 |
| observation_noise | petta | and3 | 1 | 0.07590 | 0.20592 | 0.000 | 0.000 | 136.549 | 1.09 | 186.1 |
| observation_noise | petta | and4 | 1 | 0.02644 | 0.20699 | 0.000 | 0.000 | 119.809 | 1.38 | 185.3 |
| observation_noise | petta | binding | 1 | 0.07171 | 0.27713 | 0.500 | 1.000 | 400.947 | 3.07 | 415.5 |
| observation_noise | petta | lag | 1 | 0.05526 | 0.13666 | 0.000 | 0.000 | 127.363 | 1.16 | 186.1 |
| observation_noise | petta | negation | 1 | 0.07686 | 0.21005 | 0.000 | 0.000 | 129.439 | 1.24 | 186.0 |
| observation_noise | petta | numeric | 1 | 0.09465 | 0.84845 | 0.000 | 0.000 | 162.446 | 1.72 | 187.3 |
| observation_noise | petta | or | 1 | 0.08825 | 0.28400 | 0.000 | 0.000 | 121.989 | 1.22 | 189.3 |
| observation_noise | petta | single | 1 | 0.11154 | 0.27309 | 1.000 | 1.000 | 114.064 | 0.98 | 188.7 |
| observation_noise | petta | xor | 1 | 0.12318 | 0.31970 | 0.000 | 0.000 | 129.849 | 1.06 | 188.9 |
| samples | marginal | and2 | 1 | 0.24224 | 0.66750 | 0.000 | 0.000 | 0.012 | 0.76 | 20.1 |
| samples | marginal | and3 | 1 | 0.21466 | 0.61697 | 0.000 | 0.000 | 0.014 | 0.51 | 19.8 |
| samples | marginal | and4 | 1 | 0.12899 | 0.42124 | 0.000 | 0.000 | 0.012 | 0.83 | 19.9 |
| samples | marginal | binding | 1 | 0.08745 | 0.28943 | 0.000 | 0.000 | 0.021 | 0.71 | 20.1 |
| samples | marginal | lag | 1 | 0.22583 | 0.63730 | 0.000 | 0.000 | 0.012 | 0.78 | 19.9 |
| samples | marginal | negation | 1 | 0.17230 | 0.49836 | 0.000 | 0.000 | 0.012 | 0.77 | 20.1 |
| samples | marginal | numeric | 1 | 0.16632 | 1.60563 | 0.000 | 0.000 | 0.018 | 0.95 | 20.1 |
| samples | marginal | or | 1 | 0.22186 | 0.61423 | 0.000 | 0.000 | 0.013 | 0.60 | 20.1 |
| samples | marginal | single | 1 | 0.04768 | 0.20300 | 1.000 | 1.000 | 0.012 | 0.75 | 20.1 |
| samples | marginal | xor | 1 | 0.27229 | 0.82489 | 0.000 | 0.000 | 0.013 | 0.70 | 19.8 |
| samples | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.51 | 20.1 |
| samples | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.63 | 19.9 |
| samples | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.19 | 20.1 |
| samples | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.68 | 20.1 |
| samples | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.67 | 19.8 |
| samples | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.41 | 19.9 |
| samples | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.73 | 20.1 |
| samples | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.009 | 1.31 | 20.1 |
| samples | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.86 | 20.1 |
| samples | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.77 | 20.1 |
| samples | petta | and2 | 1 | 0.16331 | 0.63379 | 0.000 | 0.000 | 108.178 | 0.61 | 131.7 |
| samples | petta | and3 | 1 | 0.14999 | 0.49181 | 0.000 | 0.000 | 114.140 | 1.28 | 131.8 |
| samples | petta | and4 | 1 | 0.07851 | 0.45303 | 0.000 | 0.000 | 82.955 | 1.44 | 129.2 |
| samples | petta | binding | 1 | 0.09358 | 0.57456 | 0.250 | 1.000 | 86.848 | 11.49 | 148.4 |
| samples | petta | lag | 1 | 0.25381 | 0.93223 | 0.000 | 0.000 | 116.688 | 0.81 | 133.0 |
| samples | petta | negation | 1 | 0.13974 | 0.48563 | 0.000 | 0.000 | 99.879 | 1.06 | 130.2 |
| samples | petta | numeric | 1 | 0.15339 | 1.49554 | 0.000 | 0.000 | 123.485 | 2.38 | 132.1 |
| samples | petta | or | 1 | 0.27509 | 1.05352 | 0.000 | 0.000 | 98.130 | 1.72 | 130.6 |
| samples | petta | single | 1 | 0.32583 | 1.01190 | 0.333 | 1.000 | 120.802 | 0.87 | 134.0 |
| samples | petta | xor | 1 | 0.24587 | 0.94938 | 0.000 | 0.000 | 126.136 | 1.08 | 135.0 |
| strength | marginal | and2 | 1 | 0.12683 | 0.36856 | 0.000 | 0.000 | 0.012 | 0.92 | 20.1 |
| strength | marginal | and3 | 1 | 0.07451 | 0.23208 | 0.000 | 0.000 | 0.012 | 0.98 | 19.8 |
| strength | marginal | and4 | 1 | 0.04092 | 0.17337 | 0.000 | 0.000 | 0.012 | 0.81 | 20.1 |
| strength | marginal | binding | 1 | 0.10031 | 0.36488 | 0.000 | 0.000 | 0.020 | 1.03 | 20.3 |
| strength | marginal | lag | 1 | 0.09271 | 0.34061 | 0.000 | 0.000 | 0.014 | 1.30 | 20.1 |
| strength | marginal | negation | 1 | 0.07445 | 0.21708 | 0.000 | 0.000 | 0.013 | 0.96 | 19.8 |
| strength | marginal | numeric | 1 | 0.08017 | 0.71313 | 0.000 | 0.000 | 0.019 | 1.00 | 20.1 |
| strength | marginal | or | 1 | 0.12221 | 0.35435 | 0.000 | 0.000 | 0.012 | 0.81 | 20.1 |
| strength | marginal | single | 1 | 0.01870 | 0.07968 | 1.000 | 1.000 | 0.012 | 0.99 | 20.1 |
| strength | marginal | xor | 1 | 0.16542 | 0.44841 | 0.000 | 0.000 | 0.012 | 0.91 | 20.4 |
| strength | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.89 | 20.1 |
| strength | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.76 | 19.8 |
| strength | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 1.04 | 19.8 |
| strength | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 1.00 | 20.1 |
| strength | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.95 | 20.1 |
| strength | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.003 | 0.86 | 20.1 |
| strength | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 1.87 | 20.3 |
| strength | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.004 | 0.71 | 20.1 |
| strength | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.002 | 0.91 | 19.9 |
| strength | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.005 | 0.75 | 20.1 |
| strength | petta | and2 | 1 | 0.11016 | 0.39491 | 0.000 | 0.000 | 119.909 | 1.18 | 190.7 |
| strength | petta | and3 | 1 | 0.04146 | 0.18077 | 0.000 | 0.000 | 113.259 | 0.93 | 185.7 |
| strength | petta | and4 | 1 | 0.01701 | 0.09392 | 0.000 | 0.000 | 83.819 | 0.90 | 172.8 |
| strength | petta | binding | 1 | 0.08472 | 0.40289 | 1.000 | 1.000 | 203.058 | 13.72 | 312.8 |
| strength | petta | lag | 1 | 0.06945 | 0.36582 | 0.000 | 0.000 | 121.493 | 1.50 | 187.7 |
| strength | petta | negation | 1 | 0.09218 | 0.30166 | 0.000 | 0.000 | 116.361 | 1.54 | 188.3 |
| strength | petta | numeric | 1 | 0.07459 | 0.67591 | 0.000 | 0.000 | 176.721 | 2.19 | 194.1 |
| strength | petta | or | 1 | 0.11011 | 0.37923 | 0.500 | 0.500 | 113.520 | 1.19 | 190.9 |
| strength | petta | single | 1 | 0.14825 | 0.43824 | 1.000 | 1.000 | 113.709 | 0.95 | 188.8 |
| strength | petta | xor | 1 | 0.14850 | 0.43900 | 0.000 | 0.000 | 129.200 | 1.08 | 189.3 |
| targets | marginal | and2 | 1 | 0.17601 | 0.50285 | 0.000 | 0.000 | 0.078 | 1.15 | 20.1 |
| targets | marginal | and3 | 1 | 0.13228 | 0.40813 | 0.000 | 0.000 | 0.071 | 1.02 | 20.3 |
| targets | marginal | and4 | 1 | 0.05039 | 0.20049 | 0.000 | 0.000 | 0.076 | 1.29 | 20.1 |
| targets | marginal | binding | 1 | 0.23653 | 0.69527 | 0.000 | 0.000 | 0.127 | 1.09 | 20.3 |
| targets | marginal | lag | 1 | 0.16376 | 0.47684 | 0.000 | 0.000 | 0.068 | 1.11 | 20.3 |
| targets | marginal | negation | 1 | 0.15753 | 0.44526 | 0.000 | 0.000 | 0.070 | 1.04 | 20.3 |
| targets | marginal | numeric | 1 | 0.14064 | 1.34413 | 0.000 | 0.000 | 0.127 | 1.07 | 20.3 |
| targets | marginal | or | 1 | 0.22471 | 0.65968 | 0.000 | 0.000 | 0.071 | 0.94 | 20.1 |
| targets | marginal | single | 1 | 0.02146 | 0.09385 | 1.000 | 1.000 | 0.068 | 0.97 | 20.3 |
| targets | marginal | xor | 1 | 0.29245 | 0.82867 | 0.000 | 0.000 | 0.068 | 1.00 | 20.3 |
| targets | oracle | and2 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.006 | 0.88 | 19.8 |
| targets | oracle | and3 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.007 | 0.97 | 20.1 |
| targets | oracle | and4 | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.007 | 0.97 | 20.1 |
| targets | oracle | binding | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.014 | 0.65 | 20.4 |
| targets | oracle | lag | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.007 | 0.89 | 20.1 |
| targets | oracle | negation | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.007 | 0.97 | 20.3 |
| targets | oracle | numeric | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.010 | 0.98 | 20.1 |
| targets | oracle | or | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.011 | 0.78 | 20.3 |
| targets | oracle | single | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.006 | 0.67 | 20.3 |
| targets | oracle | xor | 1 | 0.00000 | 0.00000 | 1.000 | 1.000 | 0.011 | 0.75 | 20.1 |

## Incomplete runs

These runs have no completed score. Partial scores are not substituted for full-stream results.

| Difficulty | System | Type | Seed | Exit | Completed steps |
|---|---|---|---:|---:|---:|
| distractors | petta | and2 | 1 | 124 | 38 |
| distractors | petta | and3 | 1 | 124 | 38 |
| distractors | petta | and4 | 1 | 124 | 41 |
| distractors | petta | binding | 1 | 124 | 41 |
| distractors | petta | lag | 1 | 124 | 35 |
| distractors | petta | negation | 1 | 124 | 38 |
| distractors | petta | numeric | 1 | 124 | 22 |
| distractors | petta | or | 1 | 124 | 40 |
| distractors | petta | single | 1 | 124 | 33 |
| distractors | petta | xor | 1 | 124 | 35 |
| targets | petta | and2 | 1 | 124 | 30 |
| targets | petta | and3 | 1 | 124 | 34 |
| targets | petta | and4 | 1 | 124 | 36 |
| targets | petta | binding | 1 | 124 | 42 |
| targets | petta | lag | 1 | 124 | 30 |
| targets | petta | negation | 1 | 124 | 29 |
| targets | petta | numeric | 1 | 124 | 7 |
| targets | petta | or | 1 | 124 | 33 |
| targets | petta | single | 1 | 124 | 24 |
| targets | petta | xor | 1 | 124 | 26 |

## Acceptance

| Check | Status | Evidence |
|---|---|---|
| oracle exact | PASS | 70 oracle runs |
| oracle beats marginal: and2 | PASS | min paired Brier gap 0.19030 > 0.01 |
| oracle beats marginal: and3 | PASS | min paired Brier gap 0.13472 > 0.01 |
| oracle beats marginal: and4 | PASS | min paired Brier gap 0.05436 > 0.01 |
| oracle beats marginal: binding | PASS | min paired Brier gap 0.15745 > 0.01 |
| oracle beats marginal: lag | PASS | min paired Brier gap 0.13948 > 0.01 |
| oracle beats marginal: negation | PASS | min paired Brier gap 0.14855 > 0.01 |
| oracle beats marginal: numeric | PASS | min paired Brier gap 0.14047 > 0.01 |
| oracle beats marginal: or | PASS | min paired Brier gap 0.19101 > 0.01 |
| oracle beats marginal: xor | PASS | min paired Brier gap 0.29494 > 0.01 |
| marginal XOR recovery fails | PASS | max recall 0.0 |
| marginal: harder distractors | PASS | mean error delta +0.02544; cost ratio 16.84 |
| marginal: harder strength | PASS | mean error delta +0.00402; cost ratio 1.04 |
| marginal: harder observation_noise | PASS | mean error delta +0.08691; cost ratio 1.01 |
| marginal: harder samples | PASS | mean error delta +0.03157; cost ratio 1.06 |
| marginal: harder drift | PASS | mean error delta +0.03412; cost ratio 1.01 |
| marginal: harder targets | PASS | mean error delta +0.01319; cost ratio 6.17 |
| petta: harder distractors | UNMEASURED | no matched sweep |
| petta: harder strength | FAIL | mean error delta -0.02246; cost ratio 1.05 |
| petta: harder observation_noise | PASS | mean error delta +0.05730; cost ratio 1.23 |
| petta: harder samples | PASS | mean error delta +0.01502; cost ratio 0.92 |
| petta: harder drift | PASS | mean error delta +0.00263; cost ratio 1.07 |
| petta: harder targets | UNMEASURED | no matched sweep |
| marginal: constant-cost screen | PASS | 0/60 completed runs exceed late/early ratio 2; empirical screen only |
| oracle: constant-cost screen | PASS | 0/60 completed runs exceed late/early ratio 2; empirical screen only |
| petta: constant-cost screen | FAIL | 5/40 completed runs exceed late/early ratio 2; empirical screen only |

Timing and RSS are measurements, not seed-deterministic values. Late/early is a diagnostic, not a complexity proof.
Brier is squared error against the exact distribution; excess log loss subtracts its entropy.
