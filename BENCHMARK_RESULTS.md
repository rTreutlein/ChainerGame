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

StationOps queries MM2's inverted `SealLeak` belief directly. The
unit-strength `PatchPaysOff` wrapper has the same action-belief semantics, but
the current MM2 backward surface does not compose an inverted proof through
that additional rule in one query.
