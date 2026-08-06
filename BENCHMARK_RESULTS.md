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
