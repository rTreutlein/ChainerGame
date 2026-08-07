# StationOps

StationOps now has two complementary surfaces:

- **StationOps-v2** is a hidden-state, multi-shift maintenance simulation for
  humans and reasoners. It has a browser dashboard, diagnostic and repair
  actions, scarce credits and parts, production consequences, and a maintenance
  record containing only facts that were actually discovered.
- **BaseRateTriage-v0/v1** remain deterministic conformance benchmarks for
  isolating base-rate, inversion, and incremental-KB behavior.

## Play the station simulation

Start the local dashboard from the repository root:

```sh
python -m pip install -e .
stationops game
```

Then open <http://127.0.0.1:8765>. The server binds only to localhost by
default. A source-tree run without installation is also supported:

```sh
PYTHONPATH=src python -m stationops.cli game
```

Each shift shows only the current modules, their cohort/manufacturer, pressure
sensor state, criticality, and production value at risk. That last value is a
knowable impact—the production lost if the module leaks—not a disclosed failure
probability. New-fault and sensor behavior is shared by visible cohort and
equipment type, not arbitrary per-module constants. This lets a resolved
coolant-pump case inform later coolant pumps. Failure hazards remain hidden;
the dashboard reveals only the configured full or partial sensor documentation.
It does not reveal
the simulator's hidden fault state or a calculated posterior. Inspections
consume credits and diagnostic slots. Repairs consume credits, repair slots,
and seal kits. Committing a shift applies maintenance decisions and production
losses. Inspection or repair can add a confirmed case to the compact maintenance
log, while an undiagnosed production failure reports only the aggregate loss,
does not identify the responsible module, and does not become labeled evidence.
An unrepaired leak persists into later shifts; a diagnosed leak remains visibly
known until it is repaired.

The default episode lasts five shifts. Credits, parts, production, score, and
learned maintenance evidence carry through the episode. Useful controls are:

```sh
stationops game --seed 9 --shifts 8 --modules 12
stationops game --diagnostic-slots 1 --repair-slots 1 --credits 20
stationops game --sensor-knowledge induced --shifts 30
```

## Test reasoners in the same simulation

Automated runs use the exact same `GameSession`, hidden outcomes, resource
rules, and learning boundary as the browser game:

```sh
stationops run --benchmark v2 --backend reference --budget 100
stationops run --benchmark v2 --backend mm2 --budget 1 \
  --action-budget 100 --shortfall-budget 50
stationops run --benchmark v2 --backend pettachainer --budget 300 \
  --action-budget 1 --shortfall-budget 300
stationops sweep --benchmark v2 --backend reference --budgets 0,1,10,100
```

V2 keeps diagnosis, action, and aggregate-loss budgets separate because their
rules have different search depths and a native step does not represent
equivalent work in MM2 and PeTTaChainer. The values above are current
six-shift fixture-specific calibration points, not cross-backend units or safe
defaults for larger workloads. If `--action-budget` is omitted it defaults to
`--budget` for compatibility, which is not the calibrated choice above.

Use the dedicated stress sweep to measure how gracefully a backend loses proof
coverage and decision quality when its budget is insufficient:

```sh
stationops stress --backend mm2 --budgets 1 \
  --action-budgets 100 \
  --shortfall-budgets 10,20,30,40,50,75,100 \
  --seeds 7,11,19 --shifts 6 --modules 10 \
  --initial-history-per-cohort 20 --stream

# A deliberately hard closed-loop workload, protected by an external cap.
timeout 180s stationops stress --backend mm2 --budgets 1 \
  --action-budgets 100 \
  --shortfall-budgets 25 --seeds 7 --shifts 12 --modules 20 \
  --initial-history-per-cohort 40 --learning-window 4 --stream
```

`--stream` emits one `stress-run` JSON object as soon as each point completes,
so earlier results survive if a later point becomes intractable. The final
`stress-summary` contains the same points together plus a `budget_curve` with
cross-seed mean, minimum, and maximum coverage, score, regret, and wall time.
Each point reports diagnosis, action-query, and aggregate-shortfall work,
score, regret, logical versus actually executed shortfall queries, cache hits,
per-shift deterioration, and, when exposed by the backend, native steps,
transitions, and unifications. Runs are intentionally interactive: an
under-budget decision can leave faults unresolved, which increases the number
of ambiguous candidates—and therefore the workload—in later shifts.

StationOps-v2 defaults to a mixed information model:

- coolant pumps have a calibrated full CTV;
- oxygen scrubbers and power converters have only a positive-path STV;
- thermal-loop and ore-feed pumps have no supplied causal rule.

The dashboard shows the corresponding calibrated, partial, or uncharacterized
sensor label, including exactly those numeric rates that the logic receives.

Every resolved case is also encoded as a shared state subject, for example
`(Inheritance (State shift-03-M04) (SealLeak new thermal-loop-pump))` and a
corresponding `PressureAlarm` or `PressureNormal` observation. False labels use
complemented-strength positive facts (`STV 0 1`); absence remains unknown.
Only inspected or repaired outcomes enter this table. An unresolved current
alarm is deliberately excluded, since inserting an observation without its
leak label would dilute the learned conditional.

For positive-only and undocumented types the controller queries an induced
`PressureAlarm/PressureNormal -> SealLeak` inheritance relation. The
positive-path STV remains useful in its known forward direction; the learned
relation supplies the inverse needed for diagnosis. If a learned relation has
no proof yet, otherwise-unused diagnostic capacity explores the least-sampled
feature group so learning cannot permanently starve itself.

Long runs expose windowed learning metrics and the public data behind them:

```sh
stationops run --benchmark v2 --backend mm2 --budget 1 \
  --action-budget 100 --shortfall-budget 50 \
  --sensor-knowledge mixed --shifts 30 --learning-window 5
stationops run --benchmark v2 --backend pettachainer --budget 300 \
  --action-budget 1 --shortfall-budget 300 \
  --sensor-knowledge induced --shifts 30 --learning-window 5
```

The JSON contains per-window Brier score, log loss, coverage, regret, confirmed
case growth, and station score; before/after resolved-model tables; the public
knowledge regime; and a clearly labeled scoring-only hidden model. Because
repairs affect later hidden state and inspections select which labels become
public, improvement is not guaranteed or monotonic. Compare several metrics
and fixed seeds rather than interpreting one late window as convergence.

The minimal MM2 induction probe can be run directly:

```sh
/path/to/mm2-chainer query --kb stationLearningKb --steps 1 \
  --add examples/inductive_sensor_statements.metta \
  '(Inheritance (PressureAlarm old coolant-pump) (SealLeak old coolant-pump))'
```

The standard controller first queries every visible incident with the same
diagnosis budget. It combines each returned belief with public evidence: time
since a module was last inspected or serviced and exact aggregate production
shortfalls. When a shortfall can be produced by several combinations of module
impacts, the controller conditions the probabilities over those combinations
rather than reading hidden fault identities.

The resulting probabilities are then asserted under one immutable decision
context and the chainer receives a single open query:

```metta
(ActionProposal decision-s03-step00 $action $utility $confidence $rationale)
```

Generic MeTTa rules derive the physical `(Inspect $incident)` and
`(Repair $incident)` actions. The separate rationale is `Diagnostic`,
`Learning`, or `Intervention`; learning and diagnosis never masquerade as
different simulator operations.
`RepairValue` is expected avoided production minus unnecessary-repair loss and
repair cost. `InspectionValue` is the incremental value of a perfect observation
before the best immediate repair/defer choice. A learning probe is ranked from
the number of resolved samples and production at risk. Python only applies the
action lifecycle and physical constraints: observations precede interventions,
and credits, diagnostic slots, repair slots, and seal kits remain simulator
state.

After every inspection its public result is inserted into a fresh context as a
0/1 leak probability and the same open query runs again. The action workspace
contains only that current context, so obsolete decision steps and earlier
shifts cannot consume search budget or influence the answer. The longer-lived
diagnosis KB still retains resolved history for induction. Ordinary rule
premises cannot bind a proof STV's strength as a numeric term in both chainers,
so the context-scoped `LeakProbability` fact is the explicit adapter boundary;
valuation after that boundary is performed by the chainer.

PeTTaChainer performs this conditioning with `WeightedSubsetPosteriorDP`, which
merges configurations by reachable production loss into reusable prefix and
postfix tables rather than enumerating fault sets.
`WeightedSubsetPosteriorMarginal` projects a module probability from those
tables for a diagnostic decision. StationOps sends the same rules and queries
to MM2. Current MM2 builds provide registered native implementations of both
operators; older builds report `shortfall_supported: false` instead of silently
substituting the Python reference result.

Both chainer adapters memoize successful marginals by the immutable shortfall
revision hash. An unchanged revision therefore returns prior results without an
engine query. Missing marginals are deliberately retried: bounded MM2 calls can
retain partial proof state and finish on a later shift. Discovering a candidate's
condition changes the revision hash, correctly invalidating the old result.

The self-contained MeTTa experiment `examples/shortfall_foldall_vs_dp.metta`
compares an exhaustive `FoldAll` over complete explanations with an ordinary-rule
sparse DP chain, including fresh-query search-budget probes and partial
best-first explanation results:

```sh
petta examples/shortfall_foldall_vs_dp.metta \
  | rg 'ExactResult|BudgetProbe|ExplanationBudgetProbe|should'
```

Each JSON result reports per-shift visible state, returned beliefs, coverage,
absolute error against the exact empirical reference, Brier score, log loss,
diagnostic priorities, the complete context-by-context `action_trace`, raw and
public-evidence-adjusted beliefs, chosen and oracle repairs, decision regret,
actual production outcomes, backend counters, and aggregate station score. A
zero-budget reasoner receives no inferred
beliefs and therefore schedules no evidence-driven action. It may still use an
otherwise-idle diagnostic slot to explore a type whose inverse relation must be
learned; only a confirmed inspection or an already-known fault can then cause a
repair.

## Base-rate conformance benchmarks

A deterministic, abstract, turn-based benchmark for cohort-conditioned
maintenance triage. Two cohorts have different empirical seal-leak rates but
share the same pressure-alarm sensor model. The game asks a reasoner for every
candidate `SealLeak` action belief at the same budget, then independently
evaluates expected utility and chooses the best feasible repairs.

### Install and run

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

### Semantics

Each cohort has a coherent, explicit labeled contingency table: positive and
complemented-strength (`STV 0 1`) facts are emitted, while absence means
unknown. The exact oracle estimates `P(leak|cohort)` from those rows and
applies Bayes' rule using
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

### Reasoner backends

MM2-Chainer is not vendored or located by a machine-specific default. Build its
Python binding under Python 3.13 as documented upstream, then make the installed
package importable normally, set `MM2_CHAINER_PYTHONPATH`, or pass
`--mm2-path /path/to/python/site-packages`. For example:

```sh
MM2_CHAINER_PYTHONPATH=/path/to/site-packages \
  python -m stationops.cli run --backend mm2 --budget 100
```

The adapter keeps one MM2 diagnosis engine for the episode. Each round adds only newly
named statements; knowledge from earlier rounds remains in the append-only KB.
A repeated name with different content is rejected instead of retracting the
old statement. An incident's alarm retains the same statement name when its
outcome becomes known, preventing duplicate evidence during that transition.
The first backward query performs MM2's complete cold cache bootstrap. Later
rounds batch all newly added fact seeds into one bounded forward call, allowing
MM2 to update only dirty base-rate contributions before backward inference.
The adapter updates the explicit cohort-prior cache and
queries each `SealLeak` action belief directly. `PatchPaysOff` is a
unit-strength identity consequence of `SealLeak`, but current MM2 coverage does
not compose an inverted proof through that additional wrapper in one backward
query. The strongest returned MM2 STV strength is the backend's action-belief
value. Missing proofs—including zero or insufficient budgets—produce no belief
and therefore no repair. Oracle Bayes beliefs remain separate and are used only
to score chosen actions. Recent MM2 builds expose a snapshot of the last native
execution. StationOps records its steps, transitions, and unifications
separately for diagnosis and shortfall conditioning; older bindings leave those
fields `null`. Live conformance uses the same maximum absolute belief error of
0.05 as PeTTaChainer. The integration test skips with an actionable reason when
the binding is unavailable; controlled engine tests always verify that wrong or
empty MM2 results change or remove decisions.

Action valuation uses a separate MM2 engine containing only the current
decision context. It is rebuilt after an observation, preventing irrelevant
past action facts from making a fixed budget deteriorate over time. Action
counters report the open queries, proposals, inserted context statements,
forward seeds, and native execution snapshot separately from diagnosis and
shortfall work.

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

The adapter creates one isolated PeTTaChainer diagnosis knowledge base per episode and
retains it across that episode's rounds. Each round's public view contributes
only newly named statements; disappeared statements remain as earlier
knowledge, and a repeated name with different content is rejected. Rules are
added before facts. Newly added facts are selected in batches of 100 and receive
two bounded forward agenda steps per seed before grounded `SealLeak` or learned
inheritance-relation queries run in incident order. This updates provisional
base-rate caches while keeping inference explicitly finite and the KB
append-only.

The query budget maps to backward PeTTaChainer steps and does not include this
reported forward work. Round counters include `statements_added`,
`statements_removed`, `forward_seed_facts`, and `forward_steps`; `engine_steps`
remains `null` because the API does not expose total internal execution steps.
`statements_removed` remains zero under the append-only adapter contract.
The strongest returned proof STV supplies each action belief; a missing proof
remains a missing belief, not numeric zero. Oracle beliefs remain scoring-only.
Each action query runs in a separate current-context PeTTaChainer workspace for
the same relevance and bounded-search semantics as MM2; observation results are
carried forward explicitly by the next context rather than retaining stale
action facts.

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
