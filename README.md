# StationOps: BaseRateTriage

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
`belief_error` reports coverage, missing results, mean absolute error, and
maximum absolute error against the scoring-only oracle.

BaseRateTriage-v0 remains the default. BaseRateTriage-v1 is an explicit,
deterministic multi-round benchmark: every round is inferred and scored from
the history available before that round, then its private resolutions are
revealed and appended for subsequent rounds. The included `prior-shift`
fixture moves the new-cohort positive-alarm decision from defer to repair while
an old-cohort control remains invariant.

```sh
# One complete v1 episode; budget is constant per round.
python -m stationops.cli run --benchmark v1 --backend reference --budget 100

# Each output line independently reruns the identical full episode.
python -m stationops.cli sweep --benchmark v1 --backend reference --budgets 0,1,100

# Emit each round's public MeTTa in order, with earlier resolutions only in later rounds.
python -m stationops.cli generate --benchmark v1 > episode-v1.metta

# Play the same fixture and scoring rules interactively.
python -m stationops.cli play --benchmark v1 --repair-slots 10 > human-result.json
```

Human play displays only the current visible incident IDs, cohort/alarm state,
repair slots, and history-derived priors. Enter comma-separated IDs, a blank
line to defer all, or `quit`. Invalid, duplicate, and over-capacity choices are
rejected. Choices are locked and scored before that round's private resolutions
are revealed. Prompts and reveals go to stderr; stdout contains the same
JSON-compatible per-round and aggregate schema with `backend` set to `human`.
EOF/quit scores the current round as all-defer, reveals it, and returns a
partial result with `status` set to `quit`.

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
direct-sided `Implication`, and `Not`. Current observations are
crisp; sensor uncertainty occurs only in the causal CTV. Benchmark truth,
scoring, and Bayes calculations do not depend on a reasoner or proof strings.
The shared sensor and payoff rules are emitted once per concrete cohort so
inverse population folds remain cohort-conditioned instead of pooling histories
across cohorts or registering overlapping open cache interests.

## Reasoner backends

MM2-Chainer is not vendored or located by a machine-specific default. Build its
Python binding under Python 3.13 as documented upstream, then make the installed
package importable normally, set `MM2_CHAINER_PYTHONPATH`, or pass
`--mm2-path /path/to/python/site-packages`. For example:

```sh
MM2_CHAINER_PYTHONPATH=/path/to/site-packages \
  python -m stationops.cli run --backend mm2 --budget 100
```

The adapter keeps one MM2 engine for the episode. Each round adds only newly
named statements; knowledge from earlier rounds remains in the append-only KB.
A repeated name with different content is rejected instead of retracting the
old statement. An incident's alarm retains the same statement name when its
outcome becomes known, preventing duplicate evidence during that transition.
The adapter updates the derived cohort base-rate cache and
queries each `SealLeak` action belief directly. `PatchPaysOff` is a
unit-strength identity consequence of `SealLeak`, but current MM2 coverage does
not compose an inverted proof through that additional wrapper in one backward
query. The strongest returned MM2 STV strength is the backend's action-belief
value. Missing proofs—including zero or insufficient budgets—produce no belief
and therefore no repair. Oracle Bayes beliefs remain separate and are used only
to score chosen actions. MM2 does not expose execution counters through this
Python API, so those counters are explicitly `null`. Live conformance uses the
same maximum absolute belief error of 0.05 as PeTTaChainer. The integration test
skips with an actionable reason when the binding is unavailable; controlled
engine tests always verify that wrong or empty MM2 results change or remove
decisions.

PeTTaChainer is an explicitly selected peer backend; StationOps never switches
to it automatically when MM2 is unavailable. Install PeTTaChainer and its
commit-locked PeTTa dependency as documented upstream. Make the
`pettachainer` package importable normally, set `PETTACHAINER_PYTHONPATH` to
the PeTTaChainer checkout/package parent, or pass `--pettachainer-path`:

```sh
PETTACHAINER_PYTHONPATH=/path/to/PeTTaChainer \
  python -m stationops.cli run --backend pettachainer --budget 10

python -m stationops.cli run --backend pettachainer \
  --pettachainer-path /path/to/PeTTaChainer --budget 10
```

The adapter creates one isolated PeTTaChainer knowledge base per episode and
retains it across that episode's rounds. Each round's public view contributes
only newly named statements; disappeared statements remain as earlier
knowledge, and a repeated name with different content is rejected. Rules are
added before facts. Newly added facts are selected in batches of 100 and receive
two bounded forward agenda steps per seed before the grounded `PatchPaysOff`
queries run in incident order. This updates provisional base-rate caches while
keeping inference explicitly finite and the KB append-only.

The query budget maps to backward PeTTaChainer steps and does not include this
reported forward work. Round counters include `statements_added`,
`statements_removed`, `forward_seed_facts`, and `forward_steps`; `engine_steps`
remains `null` because the API does not expose total internal execution steps.
`statements_removed` remains zero under the append-only adapter contract.
The strongest returned proof STV supplies each action belief; a missing proof
remains a missing belief, not numeric zero. Oracle beliefs remain scoring-only.

`PETTACHAINER_PYTHONPATH` is consumed by the StationOps adapter; Python itself
does not interpret that variable. To validate canonical source checkouts by
direct import, put both PeTTaChainer and PeTTa's `python` directory on
`PYTHONPATH` (along with StationOps `src`):

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/path/to/PeTTaChainer:/path/to/PeTTa/python:src \
python -c 'from pettachainer import PeTTaChainer; PeTTaChainer(); print(PeTTaChainer)'
```

Constructing the handler is part of availability validation because it loads
PeTTa's Janus/SWI-Prolog runtime. Missing dependencies such as `janus_swi` are
reported by StationOps as actionable `BackendUnavailable` errors rather than
as successful conformance. A live conformance run requires the handler
construction above to succeed; skipped integration tests are not evidence.

`ReasonerBackend` in `stationops.backends` remains the complete backend seam.

Measured PeTTaChainer accuracy/runtime curves are recorded in
[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).
