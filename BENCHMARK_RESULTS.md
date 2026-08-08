# PeTTaChainer benchmark results

## Full BaseRateTriage-v1 budget sweep

Run on 2026-08-04 with:

- ChainerGame `58ce328`
- PeTTaChainer `2f45e50dbcd55efe5e2817650eb8f2ea70eb15b3`
- PeTTa `e038e4dbb587e48fdb9d14990966108d38fde0b3`
- PeTTaChainer virtual-environment Python 3.10.16
- Default v1 fixture: 2,000 initial history cases and 42 total queries
- One independent episode per budget; no repetitions

The PeTTaChainer virtual environment predates StationOps' declared Python 3.11
minimum. The benchmark imported StationOps directly from `src`; semantic tests
also pass under the repository's supported system Python.

| Backward budget | Wall time | Round wall times | Round coverage | Round max absolute error | Normalized score | Regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 17.81 s | 17.27 s, 0.54 s | 0%, 0% | n/a, n/a | 0.0000 | 179.8947 |
| 10 | 216.02 s | 215.19 s, 0.83 s | 100%, 100% | 0.02660, 0.01526 | 1.0000 | 0.0000 |
| 20 | 2,022.76 s | 1,934.45 s, 88.31 s | 97.5%, 0% | 0.02660, n/a | 0.8968 | 18.5739 |
| 50 | 5,871.63 s | 5,660.98 s, 210.65 s | 100%, 100% | 0.02660, 0.01526 | 1.0000 | 0.0000 |

Every positive-budget run performed the same incremental work: 4,126 named
statement additions, 40 removals, 4,122 forward seed facts, and 8,244 forward
steps. Only the per-query backward budget changed.

Budget 10 is the practical point in this sweep. It is the smallest budget with
complete coverage and perfect decisions, and budget 50 produces the same error
and decisions while taking about 27 times longer. Budget 20 demonstrates that
bounded inference is not monotonic: additional search did not reach a better
fixed point and instead lost coverage in this run.

The dominant remaining performance problem is the 40 independent backward
queries in round one. Round two is much cheaper at budget 10 because it reuses
the persistent KB and updates only the public statement delta.

## MM2 live integration

Validated on 2026-08-04 with MM2-Chainer `2df2a80` and its Python 3.13 wheel.
The native fix bounds large fold-proof provenance to MORK-compatible expression
arity. Before that fix, the full StationOps history panicked while refreshing
computed base rates.

- Full v0 conformance: 2,000 history cases, 100 incidents, budget 100; passed in
  34.63 seconds with 100% coverage, zero regret, normalized score 1.0, and
  maximum absolute belief error 0.00335.
- Compact two-round v1: budget 100; passed in 11.96 seconds with 100% coverage,
  zero regret, normalized score 1.0, and round maximum absolute errors 0.00809
  and 0.01164.

The append-only persistent adapter was validated on the full v1 fixture at
budget 10. It retained one engine, added 4,044 statements in round one and only
42 new statements in round two, and removed none. The run completed in 59.47
seconds (30.45 and 29.01 seconds by round) with 100% coverage, zero regret,
normalized score 1.0, and round maximum absolute errors 0.02659 and 0.02777.
The prior rebuild-per-round adapter took 63.56 seconds (31.46 and 32.08
seconds). Persistence therefore saved about 6% overall and 10% in round two;
the remaining round-two cost is native inference and derived-cache refresh,
not statement parsing or KB reconstruction.

MM2-Chainer `21d1e70` batches forward seeds and settles contribution deltas by
identity. With that engine and StationOps forwarding the 42 round-two facts in
one call, the same full v1 budget-10 run completed in 34.45 seconds (31.16 and
3.28 seconds by round). An instrumented repeat attributed 0.03 seconds to
round-two insertion, 2.71 seconds to incremental forward maintenance, and 0.51
seconds to the two warm queries. Coverage remained 100%, regret remained zero,
and the round maximum absolute errors were 0.02659 and 0.02777.

StationOps queries MM2's inverted `SealLeak` belief directly. The
unit-strength `PatchPaysOff` wrapper has the same action-belief semantics, but
the current MM2 backward surface does not compose an inverted proof through
that additional rule in one query.

## StationOps-v2 shared-state induction

Run on 2026-08-06 with MM2-Chainer `1a397fc`, PeTTaChainer `b709d31`, and
PeTTa `e038e4d`. The current ChainerGame change was tested before its final
commit. Both backends used seed 7, six shifts, ten modules, 20 initial resolved
cases per cohort, the mixed knowledge model, a backward budget of 300, and
three-shift reporting windows. The two processes ran concurrently, so wall
times are validation observations rather than a clean speed comparison.

The mixed model supplies a full CTV for coolant pumps, a positive-only STV for
oxygen scrubbers and power converters, and no causal rule for thermal-loop and
ore-feed pumps. Positive-only and undocumented types both use the induced
shared-state inverse for diagnosis. All false observations are positive facts
with `STV 0 1`; only resolved cases enter induction.

| Backend | Coverage early / late | Brier early / late | Log loss early / late | Mean regret early / late | Confirmed cases added | Normalized score | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MM2 | 1.00 / 1.00 | 0.1650 / 0.2570 | 2.1252 / 4.0182 | 5.02 / 65.32 | 9 / 8 | 0.4984 | 80.51 s |
| PeTTaChainer | 1.00 / 1.00 | 0.1652 / 0.1234 | 1.3597 / 0.3445 | 1.69 / 0.00 | 10 / 9 | 0.9882 | 72.40 s |

PeTTaChainer improved on all three quality/decision metrics in this short run.
MM2 retained complete coverage but became less calibrated and made worse
decisions. This was intentionally recorded as a result rather than hidden by a
Python fallback.

The divergence was subsequently traced to a missing MM2 capability, not to
incremental inheritance aggregation. MM2-Chainer `1a397fc` could infer the
shared-state sensor relations, but it rejected the
`WeightedSubsetPosteriorDP` and `WeightedSubsetPosteriorMarginal` Compute
operators. StationOps therefore reported `shortfall_supported: false` and made
later decisions without conditioning on the anonymous production-loss totals.

MM2-Chainer `a97a10b` (with MORK `904e1ba`) added the registered native
operators. A focused live probe produced the exact oracle posterior for a
five-unit loss: pump `0.7772727`, motor `0.2227273`, and valve `0.2227273`.
Repeating the same six-shift configuration with the original uncalibrated
budget of 300 produced:

| Backend revision | Coverage early / late | Brier early / late | Log loss early / late | Mean regret early / late | Confirmed cases added | Normalized score | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MM2 `a97a10b` | 1.00 / 1.00 | 0.1650 / 0.1226 | 2.1252 / 0.3425 | 1.69 / 0.00 | 10 / 9 | 0.9882 | 243.59 s |

This closes the decision-quality gap with the recorded PeTTaChainer run. It is
not a valid speed comparison: MM2 reaches all ten first-shift diagnosis results
at native budget 1, while PeTTaChainer first reaches them at 47. Native steps
have different meanings, and MM2 executes substantially more scheduler work
per requested step.

### Backend-calibrated sequential rerun

The same fixture was rerun sequentially on 2026-08-06 after separating the
diagnosis and shortfall budgets in the v2 runner. MM2 used diagnosis budget 1
and shortfall budget 20. PeTTaChainer used 300 for both because lower tested
settings were not complete or monotonic over the full interactive episode:
47/20 and 50/50 missed two shift-two diagnoses and four shortfall marginals;
100/100 changed the policy and reduced diagnosis coverage further.

| Backend | Diagnosis / shortfall budget | Diagnosis coverage early / late | Shortfall results | Brier early / late | Log loss early / late | Mean regret early / late | Normalized score | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MM2 `a97a10b` | 1 / 20 | 1.00 / 1.00 | 32 / 32 | 0.1650 / 0.1226 | 2.1251 / 0.3425 | 1.69 / 0.00 | 0.9882 | 80.29 s |
| PeTTaChainer `b709d31` | 300 / 300 | 1.00 / 1.00 | 30 / 32 | 0.1652 / 0.1234 | 1.3597 / 0.3445 | 1.69 / 0.00 | 0.9882 | 70.07 s |

Both backends selected the same repairs and reached the same score. MM2
returned every requested diagnosis and shortfall result. PeTTaChainer returned
every diagnosis but missed two shift-six shortfall marginals even at budget
300; those omissions did not change this episode's decisions. The wall times
therefore describe the actual calibrated runs, but should not be read as exact
semantic parity or as equal native work. The main follow-up is why PeTTa's
forward update does not materialize the learned sensor relation and why its
bounded best-first coverage is non-monotonic.

The comparison is an interactive-policy benchmark, not a fixed replay: repairs
change later persistent faults, and each policy selects which cases become
labeled. It therefore measures whole-system learning and control, but it does
not by itself attribute the divergence to one formula. A one-shift parity smoke
test with the same mixed model and 20 initial cases per cohort gave both
backends 10/10 belief coverage; MM2 took 18.22 seconds and PeTTaChainer 12.42
seconds. An earlier ten-shift all-induced MM2 run with 40 initial cases per
cohort took 196.31 seconds, demonstrating why longer exact-induction runs need
query reuse or materialization before they become routine CI benchmarks.

## Declarative action-query budget rerun

Run on 2026-08-07 after moving diagnostic and repair valuation behind the
context-scoped `ActionProposal` query. The tested revisions were MM2-Chainer
`ff0b068`, PeTTaChainer `72d3ed6`, and MORK `6d908e0`. Diagnosis probabilities
are explicit numeric inputs to this action query; MeTTa rules derive the single
physical `Inspect` action with either `Diagnostic` or `Learning` rationale, and
the `Repair` action with `Intervention` rationale.

A four-candidate isolated query showed distinct backend knees. MM2 returned
only learning-supported inspection proposals at budgets 1--30, added all four
repair proposals at 40, and first returned all three diagnostic inspection
proposals at 100. PeTTaChainer returned the complete nine-proposal set at
budget 1; larger budgets did not add answers.

The harder closed-loop fixture used seed 7, six shifts, ten modules, 20 initial
cases per cohort, and the mixed sensor model:

| Backend | Diagnosis / action / shortfall budget | Diagnosis | Shortfall | Empty action queries | Action proposals | Normalized score | Regret | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MM2 | 1 / 40 / 50 | 60 / 60 | 41 / 44 | 0 | 303 | 0.7501 | 117.26 | 70.10 s |
| MM2 | 1 / 100 / 50 | 60 / 60 | 32 / 32 | 0 | 445 | 0.9882 | 5.07 | 134.67 s |
| PeTTaChainer | 300 / 1 / 300 | 60 / 60 | 30 / 32 | 0 | 445 | 0.9882 | 5.07 | 90.70 s |

MM2 action budget 40 was sufficient for a two-shift/four-module smoke test but
failed on the full trajectory because it omitted diagnostic-value proposals.
Budget 100 is therefore the current complete-action calibration for this MM2
fixture. PeTTaChainer needs only action budget 1. These are deliberately
separate from each backend's diagnosis and shortfall budgets. The increased
wall times relative to earlier runs are real: action selection now performs up
to three fresh chainer searches per shift instead of calculating utilities in
Python.

## PeTTaChainer shared multi-root diagnosis

Run on 2026-08-07 with PeTTaChainer `2dcfae4`. StationOps now submits all
incident diagnosis roots through `query_many`, whose expansion budget is shared
by the entire batch. The previous adapter called `query` once per incident, so
its nominal budget was multiplied by ten in this fixture.

In the isolated first shift, the old sequential adapter returned all ten
diagnoses at budget 47 in 3.30 seconds. With one honest shared allowance, the
multi-root adapter returned 2/10 at budgets 40--100, 3/10 at 300, 6/10 at 400,
9/10 at 470, and 10/10 at 500. The roots have little exact overlap: the eight
learned sensor relations differ by cohort, equipment type, or alarm state. The
main immediate saving therefore comes from compiling and running one arena,
not from collapsing identical roots.

The six-shift, ten-module, seed-7 closed-loop comparison used action budget 1
and shortfall budget 300:

| Diagnosis mode / budget | Diagnosis | Shortfall | Normalized score | Regret | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential / 300 (`72d3ed6`) | 60 / 60 | 30 / 32 | 0.9882 | 5.07 | 90.70 s |
| Multi-root / 500 | 58 / 60 | 49 / 51 | 0.9666 | 15.07 | 29.46 s |
| Multi-root / 600 | 57 / 60 | 30 / 32 | 0.9882 | 5.07 | 33.63 s |
| Multi-root / 1000 | 58 / 60 | 32 / 32 | 0.9882 | 5.07 | 55.80 s |

Multi-root search therefore reduces end-to-end runtime substantially while
exposing a useful partial-search quality curve. Budget 600 reproduces the
recorded policy and score in about 37% of the prior wall time despite three
missing diagnosis proofs. More budget improves aggregate proof coverage but
does not monotonically complete every diagnosis in this closed-loop run. This
result motivated sharing the member-inheritance producers beneath distinct
roots rather than returning to per-root budgets or eagerly saturating the KB.

### Shared member-inheritance producer follow-up

PeTTaChainer `0877a94` shares one open `Member` producer for each distinct
concept requested by compatible inheritance folds. This is effective when
several relation pairs reuse a concept. The StationOps first-shift fixture does
not: its eight learned relations contain sixteen distinct concepts because the
cohort, equipment type, and observed alarm state are part of each concept.
Consequently, no producer is reused and the additional producer/subscriber
goals add search overhead.

The isolated complete-coverage knee moved from budget 500 on `2dcfae4` to 600
on `0877a94`. At budget 470 the returned count changed from 9/10 to 8/10; at
500 it changed from 10/10 to 9/10. Warm budget-600 time was essentially
unchanged at 3.87 seconds versus 3.98 seconds before the optimization.

The same six-shift closed-loop fixture produced:

| PeTTaChainer revision / diagnosis budget | Diagnosis | Shortfall | Normalized score | Regret | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2dcfae4` / 600 | 57 / 60 | 30 / 32 | 0.9882 | 5.07 | 33.63 s |
| `0877a94` / 600 | 58 / 60 | 49 / 51 | 0.9666 | 15.07 | 37.40 s |
| `0877a94` / 800 | 56 / 60 | 30 / 32 | 0.9882 | 5.07 | 47.14 s |

Budget 800 recovers the previous policy and score but remains slower than the
pre-optimization multi-root run. The next optimization should therefore either
share the outer object/domain enumeration across several distinct concepts or
avoid installing producer/subscriber machinery when a batch contains no reused
concepts. A per-concept producer alone does not help this workload.

PeTTaChainer `62b56a4` implements the second option: shared producers are now
installed only for concepts with at least two inheritance-pair consumers. The
StationOps disjoint roots retain their original direct expansion path. The
isolated curve returned to 9/10 at budget 470 and 10/10 at 500; the budget-500
truth metrics exactly matched `2dcfae4`. Warm budget-600 time was 3.89 seconds.

The six-shift budget-600 rerun returned 57/60 diagnoses and 30/32 shortfall
marginals, with normalized score 0.9882, regret 5.07, and wall time 34.90
seconds. The coverage, decisions, score, and regret exactly recover the
pre-regression multi-root result; its 1.27-second wall-time difference from the
earlier 33.63-second observation is small residual overhead or run variance.
The calibrated PeTTaChainer diagnosis budget is therefore restored to 600.
Shared producers remain available for overlapping concept batches, while a
future improvement for this disjoint workload would need to share the outer
object/domain enumeration itself.

## Latest MM2 budget-degradation stress test

The following 2026-08-06 results predate declarative action queries: utilities
were still calculated in Python, so their wall times exclude the action-search
cost measured in the newer section above.

Run on 2026-08-06 with MM2-Chainer `ff0b068` and MORK `dd23929`. MM2 now shares
one native search allowance across Compute waves and batches ground inheritance
queries. The six-shift fixture used seed 7, ten modules, 20 initial cases per
cohort, the mixed sensor model, diagnosis budget 1, and one independent episode
per shortfall budget. Because this is a closed-loop benchmark, each budget can
choose different repairs and encounter a different later state.

| Shortfall budget | Returned / requested | Coverage | Normalized score | Regret | Wall time | Shortfall transitions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 7 / 55 | 0.1273 | 0.8087 | 90.07 | 5.85 s | 985,719 |
| 20 | 7 / 55 | 0.1273 | 0.8087 | 90.07 | 6.70 s | 1,608,729 |
| 30 | 7 / 55 | 0.1273 | 0.8087 | 90.07 | 8.02 s | 2,724,983 |
| 40 | 24 / 55 | 0.4364 | 0.8511 | 70.07 | 9.15 s | 3,798,194 |
| 50 | 32 / 32 | 1.0000 | 0.9882 | 5.07 | 8.91 s | 3,636,037 |
| 75 | 32 / 32 | 1.0000 | 0.9882 | 5.07 | 11.48 s | 6,800,470 |
| 100 | 32 / 32 | 1.0000 | 0.9882 | 5.07 | 12.95 s | 8,331,150 |
| 150 | 32 / 32 | 1.0000 | 0.9882 | 5.07 | 17.33 s | 13,124,121 |

Diagnosis coverage was 100% throughout this seed-7 curve. Its useful shortfall
boundary is sharp, not gradual: budgets 10 through 30 return the same seven
marginals despite doing increasingly more native work; 40 is partial; and 50 is
the first complete point. Above 50, quality no longer improves while work and
wall time continue to rise. The requested count also changes from 55 to 32 once
budget 50 produces better repairs. That is not a denominator error: low-budget
policies leave more faults unresolved, so later aggregate losses have more
candidate causes.

Two additional seeds show why this is a degradation curve rather than a global
calibration constant:

| Seed | Shortfall budget | Diagnosis coverage | Shortfall coverage | Normalized score | Regret |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 20 | 0.9000 | 18 / 74 (0.2432) | 0.0000 | 260.00 |
| 11 | 40 | 0.9000 | 38 / 74 (0.5135) | 0.2500 | 195.00 |
| 11 | 50 | 0.9000 | 22 / 24 (0.9167) | 0.2767 | 166.51 |
| 19 | 20 | 0.9000 | 0 / 8 (0.0000) | 0.7540 | 50.05 |
| 19 | 40 | 0.9000 | 0 / 8 (0.0000) | 0.7540 | 50.05 |
| 19 | 50 | 0.9000 | 8 / 8 (1.0000) | 0.7540 | 50.05 |

Increasing seed 11's diagnosis budget from 1 to 2 and then 10 left coverage at
exactly 54/60 and did not change its policy; budget 10 merely increased native
diagnosis transitions from about 0.74 million to 1.66 million. Those omissions
therefore reflect unavailable public proofs for that generated history, not a
too-small diagnosis search budget. The stress report deliberately keeps proof
coverage separate from decision score so this distinction remains visible.

A larger seed-7 workload used 20 modules, 12 shifts, 40 initial cases per
cohort, diagnosis budget 1, and the same mixed model. At shortfall budget 25 it
completed in 155.60 seconds with 240/240 diagnosis results but only 38/726
shortfall marginals (5.23% coverage), normalized score 0.8212, and regret
212.56. Diagnosis consumed 25,653,217 native transitions; shortfall conditioning
consumed 103,292,631. Per-shift shortfall demand grew from 16 candidates in
shift two to 117 in shift twelve, and the last five shifts each took roughly
20--24 seconds. This is the intended compounding degradation signal: incomplete
reasoning causes weaker maintenance decisions, which create a harder next
state.

The same large workload at shortfall budget 50 did not complete one episode
within a 180-second external cap. Consequently there is no score for that point;
it is recorded as a timeout, not as zero coverage. The small-fixture budget knee
therefore cannot be treated as a scale-independent calibration. Multi-seed
curves should use the streaming stress command so completed points remain
available when a later budget/workload combination times out.

### Immutable-result cache follow-up

ChainerGame now caches successful shortfall marginals by immutable event
revision in both MM2 and PeTTaChainer adapters. A focused live MM2 probe reduced
an unchanged three-marginal repeat from 0.247 seconds to 0.00013 seconds: engine
queries fell from three to zero and all three values were cache hits. Missing
answers are not negatively cached because another bounded call can continue
MM2's retained partial proof state.

The seed-7 six-shift decisions and coverage remained identical at shortfall
budgets 20, 40, and 50. At budget 50, shortfall transitions decreased from
3,636,037 to 3,504,451 (3.6%); wall time changed from 8.91 to 8.65 seconds.
Budgets 20 and 40 performed essentially the same native work as before because
their unfinished queries still had to continue.

The 20-module, 12-shift budget-25 workload completed in 155.83 seconds, with the
same 38/726 answers and decision metrics as before, but reported zero reusable
cache hits. Completed evidence caused its source event to be revised or retired
before the next shift, while unresolved marginals remained unfinished searches.
This isolates the next scaling problem: preserving/resuming the partial DP work
and reusing it across closely related event revisions, rather than merely
remembering completed top-level answers.

## Dependency-graph diagnosis baseline

Run on 2026-08-07 with MM2-Chainer `cf9d31c`, PeTTaChainer `2dcfae4`, and
MORK `6d908e0`. The new default seed-7 fixture used one shift, ten modules,
twenty initial cases per cohort, and the mixed sensor model. Each repeated
five-module DAG has four edges and a maximum causal path of three dependency
edges between the initiating seal leak and the final ore-feed alarm.

| Backend / budget | Returned | Mean absolute error vs joint oracle | Wall time |
| --- | ---: | ---: | ---: |
| MM2 / 1 | 10 / 10 | 0.11407 | 1.42 s |
| MM2 / 2 | 10 / 10 | 0.11407 | 1.42 s |
| MM2 / 10 | 10 / 10 | 0.11443 | 1.53 s |
| PeTTaChainer / 100 | 2 / 10 | partial | 5.38 s |
| PeTTaChainer / 600 | 10 / 10 | 0.11069 | 3.18 s |
| PeTTaChainer / 1200 | 10 / 10 | 0.11122 | 8.00 s |

The sampled state had a local leak at M09 and a propagated symptom at M10;
both modules alarmed. The exact bounded-component oracle assigned M09 a
0.94119 leak probability after jointly conditioning the train. MM2 returned
0.50725 and PeTTaChainer 0.50724, essentially the old local-sensor result.
Both chainers therefore retain their prior proof-coverage behavior but do not
yet combine correlated downstream evidence into the local root posterior. This
is the intended first graph benchmark gap: returning all ten local diagnoses is
not the same as solving the causal diagnosis.

The five-shift closed-loop PeTTaChainer run at diagnosis budget 600 and action
budget 1 completed in 24.22 seconds. Per-shift diagnosis coverage was
`1.0, 0.8, 0.8, 0.8, 0.8`; normalized score was 0.93849 with 60.20 expected
decision regret. It recovered 165 production through two root-cause repairs
and also spent three repairs on downstream symptoms. The exact reference
controller happened to make the same physical root/symptom repairs and produced
the same 1,255 total production on this seed, despite its 1.0 normalized score.
This is why the graph report keeps posterior error, expected regret, actual
production recovery, and symptom repairs as separate measurements.

A closed-loop MM2 timing was attempted with both action budgets 100 and 1. A
single current-context `ActionProposal` query did not complete within a minute
even with `--independent-modules`, while the isolated graph diagnosis above
completed in 1.45 seconds. The run was stopped and no closed-loop score is
reported. This isolates the stall to the action path in the current MM2
checkout rather than attributing it to the new causal graph.

### Explicit problem-state OR follow-up

The dependency model now distinguishes a module's local fault from a propagated
problem. A local leak produces `LocalProblemCause`; an upstream `Problem`
produces `ProblemDependency`; an existential cause fold builds the literal
`(Or LocalProblemCause ProblemDependency)`; and that disjunction produces the
module's `Problem` and alarm. Fully calibrated graph diagnoses now query the
context-scoped local cause, avoiding the former direct `SealLeak` shortcut.

On the same seed-7 fixture with fully calibrated sensors, the exact joint oracle
assigned M09 a local-leak probability of 0.94119 and M10 a probability of
0.28271. The live results were:

| Backend / budget | Returned | M09 | M10 | Mean absolute error | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint oracle | 10 / 10 | 0.94119 | 0.28271 | 0 | reference |
| MM2 / 100 | 10 / 10 | 0.73052 | 0.44532 | 0.05292 | 3.49 s |
| PeTTaChainer / 600 | 9 / 10 | 0.73052 | 0.45454 | partial | 6.61 s |
| PeTTaChainer / 2000 | 10 / 10 | 0.73052 | 0.45454 | 0.09487 | 6.71 s |

This exposed two distinct remaining inference issues. MM2 can prove
`Problem(shift-01, M09)` as a root query, but cannot use that aggregate proof as
the premise needed to prove `ProblemDependency(shift-01, M10)`. Increasing the
MM2 budget did more work without changing M09's result. PeTTaChainer can compose
that premise and prove the downstream dependency, but a grounded
`LocalProblemCause(shift-01, M09)` query still returns only its direct local
evidence rather than fusing the inverted downstream observation. Thus the
explicit state model improves the marginal and makes the missing proof paths
observable, but neither backend yet reaches the joint causal posterior.

### Prior/evidence knowledge-model correction

Run on 2026-08-08 with PeTTaChainer `e3961f7` and the installed MM2 binding from
`cf9d31c`. PeTTaChainer had just added backward support for pure top-level OR
premises. The StationOps history generator previously violated its
own causal model: live alarms were sampled from total module unavailability,
but historical `Problem` labels were copied from local seal-leak labels. An
intact downstream module affected by an upstream fault was therefore taught as
`Problem = false` even though its alarm was generated from `Problem = true`.

Initial history is now generated as resolved station snapshots. Each row keeps
the local leak, propagated problem, and alarm as separate public labels. The
rule graph likewise separates forward prior messages from backward diagnostic
evidence, so a local prior cannot be returned as if it were an alarm-conditioned
answer. Structural `Problem = OR(local, dependency)` equality is represented by
two deterministic implications. A probe using only the forward OR implication
returned no PeTTaChainer proof at budget 20,000 because Bayesian implication
inversion still needs a base rate for the OR term; the explicit reverse
definition supplies logical equality rather than introducing an adapter phase.

The full unit suite passed 69 tests with six optional live tests skipped. A
one-shift, ten-module, all-calibrated seed-7 smoke run used 40 initial cases per
cohort and no action queries:

| Backend / diagnosis budget | Returned | Wall time |
| --- | ---: | ---: |
| MM2 / 100 | 2 / 10 | 5.56 s |
| PeTTaChainer / 600 | 2 / 10 | 9.97 s |

Both engines returned the two root-module diagnoses with matching strengths
(M03 approximately `0.055785`, M08 `0`). A focused PeTTaChainer budget-20,000
query did traverse M10's alarm, project its dependency evidence, propagate that
evidence to M09, merge it with M09's own alarm, and finally project M09's local
cause. It returned M09 `0.66675`; the exact joint oracle returned `0.97674` for
this newly sampled history. The semantics are now coherent and the missing
quality/coverage at practical budgets is exposed as a chainer search and
approximation problem rather than being hidden by incorrect training labels or
a Python-managed two-phase base-rate transfer.
