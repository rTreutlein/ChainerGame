# StationOps: BaseRateTriage-v0

A deterministic, abstract, turn-based benchmark for cohort-conditioned
maintenance triage. Two cohorts have different empirical seal-leak rates but
share the same pressure-alarm sensor model. The game asks a reasoner for every
candidate `PatchPaysOff` goal at the same budget, then independently evaluates
expected utility and chooses the best feasible repairs.

## Install and run

Python 3.11+ and the standard library are sufficient for the reference backend:

```sh
python -m pip install -e .
python -m stationops.cli run --backend reference --seed 7 --budget 100
python -m stationops.cli sweep --backend reference --seed 7 --budgets 0,1,10,100
python -m stationops.cli generate --seed 7 > episode.metta
python -m unittest discover -s tests -v
```

Every `run`/`sweep` output line is one JSON object. Wall time is observational;
all semantic fields are deterministic. The default episode contains exactly
100 current incidents and 10 repair slots.

## Semantics

Each cohort has a coherent, explicit labeled contingency table: both positive
facts and `(Not ...)` facts are emitted, while absence means unknown. The exact
oracle estimates `P(leak|cohort)` from those rows and applies Bayes' rule using
`P(alarm|leak)=0.90` and `P(alarm|no leak)=0.12`. With default histories,
positive-alarm posteriors are about 0.283 (old) and 0.036 (new). Repair has
incremental utility `p*100 - (1-p)*10 - 5`, hence threshold `15/110 ≈ 0.136`.
Allocation sorts positive incremental utilities descending and incident IDs
ascending, then takes at most the configured slot count.

The generated MeTTa uses ordinary cohort-bearing predicates, `STV`, `CTV`,
`Implication`, `Premises`, `Conclusions`, and `Not`. Current observations are
crisp; sensor uncertainty occurs only in the causal CTV. Benchmark truth,
scoring, and Bayes calculations do not depend on a reasoner or proof strings.

## MM2 adapter and future backends

MM2-Chainer is not vendored or located by a machine-specific default. Build its
Python binding under Python 3.13 as documented upstream, then make the installed
package importable normally, set `MM2_CHAINER_PYTHONPATH`, or pass
`--mm2-path /path/to/python/site-packages`. For example:

```sh
MM2_CHAINER_PYTHONPATH=/path/to/site-packages \
  python -m stationops.cli run --backend mm2 --budget 100
```

The adapter loads all generated statements, configures cohort-specific MM2 base
rates, and queries every payoff goal. MM2 currently does not expose execution
counters or a distinct calibrated posterior statistic for this fixture through
its Python API, so counters are explicitly `null` and the reported decision
belief is the independently specified empirical-Bayes value. The integration
test skips with an actionable reason when the binding is unavailable.

`ReasonerBackend` in `stationops.backends` is the complete seam. A future PeTTa
adapter can implement `infer` without changing simulation, oracle, policy, or
scoring code.

