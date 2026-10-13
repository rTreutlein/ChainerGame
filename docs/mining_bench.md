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
Raw verdicts and truth values remain in the trace. CTV negative branches above
0.5 are expanded with De Morgan into positive-target DNF rules; weak positive
branches are not mistaken for positive rules. Unparsed verdicts are exposed in
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

Results are populated from the completed matrix below. Raw preliminary probes
remain in separate directories and are not pooled into final tables. All runs use
PeTTa's shared interpreter and leave the frozen repository untouched.
