# NARS as a comparison reasoner

Status: October 2026. `supplynet run --backend nars` (stages 1–3) and
`stationops run --backend nars --replay …` (temporal model, beliefs only),
both through OpenNARS for Applications (ONA). Code: `src/supplynet/nars.py`,
`src/stationops/nars.py`; runs: `benchmarks/nars/`; results:
`benchmark-runs/nars-2026-10-09/report.md` (raw runs: `/nexus/Dev/OpenCog/bench/results/nars-2026-10-09/`).

## Verdict

NARS can play: it accepts every statement shape SupplyNet uses and answers
nearly every belief query (coverage 0.94–0.98). It does not play well. Its
answers sit at the history base rates on every SupplyNet stage (error to the
exact posterior 0.21–0.24, base rates 0.18–0.24, PeTTaChainer 0.000–0.008).
On StationOps replay they are worse than PeTTaChainer's (Brier 0.233 against
0.061). They take 50–200 times PeTTaChainer's wall time, and more inference
cycles do not help. The
cause is not the translation alone: on a six-shipment toy that ONA handles in
under a second, the storm belief does not move from its base rate when every
shipment is late (exact 1.000, ONA 0.300). See "Why the answers stay at the
base rates".

## Which NARS

**ONA v0.9.3** (tag, commit dc4efd0, May 2025; built with its `build.sh` in
`/nexus/Dev/OpenCog/ONA`, executable `NAR`). Chosen because it is the
maintained NARS (Patrick Hammer), C, builds in a minute, has a stdin shell
with eternal (declarative) beliefs, NAL-6 variables, negation and
questions, and is deterministic on these inputs (identical output for
identical input; the backend's cache relies on it).

- **PyNARS** (OpenNARS 3 in Python) and **OpenNARS 3.x** (Java) implement
  the same NAL truth functions (deduction f1·f2, abduction without base rates,
  revision by evidence weight), so the evidence-combination results below
  would not change. OpenNARS answers questions by backward inference, which
  ONA does not do for declarative questions, so it might reach the second
  abduction hop that ONA drops; it is also slower and heavier. Not tried.

## Translation (MeTTa → Narsese)

| MeTTa | Narsese | Loss |
|---|---|---|
| `(Pred a b)` | `<(a * b) --> pred>`; one argument: `<a --> pred>` | none; `-` becomes `_` in atoms |
| `(Not x)` | `(! x)` | ONA derives `x` from `(! x)`, never `(! x)` from `x` |
| `(And x y z)` | `((x && y) && z)` | none (ONA reads `&&` as binary and **silently drops** a third part of a flat `(&&, x, y, z)`) |
| CTV `A → B`, P(B\|A)=p, P(B\|¬A)=q | `<A ==> B> %p%`, `<A ==> (! B)> %1-p%`, `<¬A ==> B> %q%`, `<¬A ==> (! B)> %1-q%` | NARS has no conditional semantics: deduction gives f = f_A·p with confidence f_A·p·c², not the mixture P(A)p + (1−P(A))q. Implications of frequency 0 are left out: every ONA truth function gives their conclusions confidence 0 |
| ¬A for `(And (NextPeriod $p $t) X)` | `(&& next (! X))` | none: the period link is a given, so the negative branch means ¬X |
| `Or` antecedent, negated `And` of several parts | one rule per part (De Morgan) | exact only for strengths 0 and 1, the only ones that occur (asserted) |
| fact `(STV s c)` | `x %s;c%` if s ≥ 0.5, else `(! x) %1-s;c%` | none for the game's certain facts |
| confidence 1 | 0.99 (ONA's `MAX_CONFIDENCE`) | certain facts can still be revised |
| labelled history | per statement, `<<$t --> period> ==> x($t)> %k/n; n/(n+1)%` and the same for `(! x($t))` | this is what NARS induction plus revision over the n labelled periods would sum to; the adapter computes it because ONA would not form it under its resource bounds. NARS gets the base rate as a count, PeTTaChainer gets the raw facts |
| StationOps `(ForAll ($cohort) (WithPrior P (Implication A B)))` | `P ⇒ A`, `¬P ⇒ ¬A` with certainty, plus the sensor CTV | an open sensor rate (`(STV s 1)` or `(STV 0.5 0.5)`) is induction over resolved units (k/n, n/(n+1)) |
| StationOps shortfall rule (`FoldAllTruth`, `Compute`) | — | not expressible; left out with its facts |
| query `(Pred a t)` | question `<(a * t) --> pred>?` | answer: the concept's eternal belief; `Answer: None.` is unanswered |

**Answers as probabilities.** A belief is the answer's frequency (default),
or its expectation c(f − 0.5) + 0.5 (`--nars-reading expectation`). The
frequency is NARS's estimate of the proportion of positive evidence; the
expectation adds the k = 1 ignorance prior and so pulls weakly supported
answers to 0.5. Coverage is the share of questions with an answer other than
`None`.

**What each round's KB holds.** ONA keeps at most 255 atoms and 4096
concepts, so it cannot hold the whole history as facts. Each round starts a
fresh `NAR shell` with the rules, the base rates and the facts of the
unresolved periods plus the last resolved one (StationOps: the units of the
last 15 shifts). The exact reference uses the same information; PeTTaChainer
holds everything. This is a relevance filter in NARS's favour.

**Budget.** A round costs one ONA cycle per input statement (ONA runs a cycle
on each input) plus `budget × --nars-cycles-per-step` cycles before the
questions. Runs use 0 extra cycles and 1 cycle per budget step (100 per
round).

## Cost

ONA's per-cycle cost is set by `Decision_Anticipate`
(`DECLARATIVE_IMPLICATIONS_CYCLE_PROCESS`, on by default). Each cycle it
matches every declarative implication against every concept, about 2·10⁶
unifications once the concept table is full (gprof: 99.6% of the time). The
table is full after the rules and about 20 facts: one statement creates tens
of concepts through NAL-4 image and product rules, e.g. `(storm /1 t5)` or
`<south <-> north>`. So a 220-statement stage-1 round takes about 40 s of
input processing alone, at 1.2 GB per process.

With the sweep switched off (a scratch build, not used for the results), a
round takes 0.7 s. ONA then answers only from base-rate deductions, leaves
some queries unanswered, and further cycles change nothing.

## Why the answers stay at the base rates

A toy in the stage-1 shape: one region, three routes, two shipments each,
the rates of the game, a base rate of 9/30. ONA's answer for the storm
(frequency/confidence) against the exact posterior:

| shipments late | exact | ONA, 0 cycles | 100 cycles | 1000 cycles |
|---|---|---|---|---|
| 0/6 | 0.004 | 0.300 | 0.386 | 0.386 |
| 1/6 | 0.032 | 0.300 | 0.471 | 0.471 |
| 3/6 | 0.789 | 0.300 | 0.520 | 0.520 |
| 6/6 | 1.000 | 0.300 | 0.300 | 0.300 |

With routes inspected instead (one hop), k of 3 blocked: exact 0.004 / 0.33 /
0.984 / 0.9995; ONA 0.457 / 0.556 / 0.625 / 0.675.

Three mechanisms, each visible in the trace:

1. **Abductive conclusions get almost no priority.** The first hop works
   (blocked from late rises to 0.78 for 6/6), but those conclusions are
   derived with priority 0.007–0.02. They are never selected again, so the
   second hop, to the storm, is never drawn. The selections go to structural
   by-products (`<s3_2 <-> s3_1>`, image terms) instead. After about 100
   selections the event queue is empty, so 1000 cycles give the same answers
   as 100. Asking the questions before the cycles (question priming) changes
   nothing.
2. **Abduction ignores P(B|¬A) and the base rate.** Abduction gives A the
   frequency of the observation with confidence w2c(p·c²), whatever the rate
   of B without A. Revision then averages the views by evidence weight, while
   Bayes multiplies likelihood ratios. Evidence against the cause flows
   through `(! storm)` concepts that rarely get selected. So clear routes
   raise the storm belief (0.457 > 0.3), and the one-hop answers are
   compressed into 0.46–0.68.
3. **Bounded memory makes answers order-dependent.** With the concept table
   full, a queried concept can be evicted and recreated; its eternal belief
   is then whatever was derived last, e.g. 0.324 → 0.996 for one storm within
   10 cycles.

## Results

ChainerGame `nars-backend` (off `supply-network` 9cfb585), PeTTaChainer master
999745f5, ONA v0.9.3. Means over seeds 1–4. SupplyNet: 3 regions, 30 history
periods, 30 rounds, budget 100, evidence k 5. StationOps: the bench suite's
temporal replay (`bench/recordings/fv_s<seed>.jsonl`, 20 shifts, 10 modules,
mixed sensors, budget 50, shortfall budget 25). NARS runs used 0 extra
cycles (input processing only, one cycle per statement). The expectation
reading reuses the same ONA outputs (cached), so its seconds are not a cost.
Other agents' benchmarks shared the machine during all runs, so seconds are
indicative. Error is the mean absolute difference from the exact posterior;
an unanswered query counts as 0.5 in SupplyNet and is left out of StationOps'
Brier (StationOps reports coverage separately).

Rows: `NARS, 0 cycles` is the headline; `expectation` re-reads its answers;
`100 cycles/round` ran stages 1 and 2 only (it was worse on both and doubles
the time; stage 3 would have taken several more hours). In StationOps the
reference is the game's oracle and the error is measured against it.
Reasoner-only regret is the regret of repairs allocated from the reasoner's
own beliefs; the replayed actions are the same for every reasoner.

### SupplyNet stage 1 (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| exact posterior | 4 | 0.000 | 0.027 | 0.102 | 1.00 | 0.0 |
| history base rates | 4 | 0.239 | 0.134 | 0.435 | 1.00 | 0.0 |
| pettachainer | 4 | 0.008 | 0.028 | 0.106 | 1.00 | 5.9 |
| NARS, 0 cycles | 4 | 0.242 | 0.139 | 0.455 | 0.97 | 1152.2 |
| NARS, expectation | 4 | 0.286 | 0.144 | 0.460 | 0.97 | 0.1 |
| NARS, 100 cycles/round | 4 | 0.256 | 0.161 | 0.544 | 0.98 | 2815.4 |

Exact posterior's own Brier 0.027, log loss 0.102.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | storm |
|---|---|---|
| exact posterior | 0.000 [1.00] | 0.000 [1.00] |
| history base rates | 0.233 [1.00] | 0.254 [1.00] |
| pettachainer | 0.002 [1.00] | 0.021 [1.00] |
| NARS, 0 cycles | 0.234 [0.97] | 0.259 [0.99] |
| NARS, expectation | 0.283 [0.97] | 0.292 [0.99] |
| NARS, 100 cycles/round | 0.255 [0.98] | 0.260 [0.99] |

### SupplyNet stage 2 (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| exact posterior | 4 | 0.000 | 0.154 | 0.477 | 1.00 | 0.0 |
| history base rates | 4 | 0.183 | 0.213 | 0.615 | 1.00 | 0.0 |
| pettachainer | 4 | 0.000 | 0.154 | 0.477 | 1.00 | 12.1 |
| NARS, 0 cycles | 4 | 0.196 | 0.223 | 0.699 | 0.94 | 1318.5 |
| NARS, expectation | 4 | 0.207 | 0.220 | 0.630 | 0.94 | 0.1 |
| NARS, 100 cycles/round | 4 | 0.238 | 0.245 | 0.714 | 0.66 | 2836.0 |

Exact posterior's own Brier 0.154, log loss 0.477.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | previous_storm | storm |
|---|---|---|---|
| exact posterior | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| history base rates | 0.120 [1.00] | 0.277 [1.00] | 0.225 [1.00] |
| pettachainer | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| NARS, 0 cycles | 0.145 [0.92] | 0.273 [0.98] | 0.231 [0.94] |
| NARS, expectation | 0.172 [0.92] | 0.271 [0.98] | 0.220 [0.94] |
| NARS, 100 cycles/round | 0.218 [0.54] | 0.281 [0.89] | 0.237 [0.67] |

### SupplyNet stage 3 timed (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| exact posterior | 4 | 0.000 | 0.140 | 0.429 | 1.00 | 0.7 |
| history base rates | 4 | 0.229 | 0.226 | 0.649 | 1.00 | 0.0 |
| pettachainer | 4 | 0.003 | 0.140 | 0.431 | 1.00 | 16.9 |
| NARS, 0 cycles | 4 | 0.214 | 0.227 | 0.688 | 0.98 | 2554.3 |
| NARS, expectation | 4 | 0.218 | 0.218 | 0.623 | 0.98 | 0.1 |

Exact posterior's own Brier 0.140, log loss 0.429.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | degraded | previous_storm | producing | producing_unstocked | storm |
|---|---|---|---|---|---|---|
| exact posterior | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| history base rates | 0.131 [1.00] | 0.235 [1.00] | 0.289 [1.00] | 0.399 [1.00] | 0.316 [1.00] | 0.236 [1.00] |
| pettachainer | 0.001 [1.00] | 0.007 [1.00] | 0.002 [1.00] | 0.011 [1.00] | 0.006 [1.00] | 0.001 [1.00] |
| NARS, 0 cycles | 0.133 [1.00] | 0.213 [1.00] | 0.286 [0.88] | 0.352 [1.00] | 0.258 [0.99] | 0.231 [0.98] |
| NARS, expectation | 0.145 [1.00] | 0.219 [1.00] | 0.282 [0.88] | 0.327 [1.00] | 0.276 [0.99] | 0.223 [0.98] |

### SupplyNet stage 3 untimed (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| exact posterior | 4 | 0.000 | 0.135 | 0.414 | 1.00 | 0.7 |
| history base rates | 4 | 0.214 | 0.212 | 0.616 | 1.00 | 0.0 |
| pettachainer | 4 | 0.002 | 0.136 | 0.416 | 1.00 | 17.5 |
| NARS, 0 cycles | 4 | 0.206 | 0.217 | 0.698 | 0.98 | 2284.0 |
| NARS, expectation | 4 | 0.212 | 0.208 | 0.602 | 0.98 | 0.1 |

Exact posterior's own Brier 0.135, log loss 0.414.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | degraded | previous_storm | producing | producing_unstocked | storm |
|---|---|---|---|---|---|---|
| exact posterior | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| history base rates | 0.131 [1.00] | 0.225 [1.00] | 0.288 [1.00] | 0.420 [1.00] | 0.212 [1.00] | 0.236 [1.00] |
| pettachainer | 0.001 [1.00] | 0.004 [1.00] | 0.000 [1.00] | 0.008 [1.00] | 0.004 [1.00] | 0.000 [1.00] |
| NARS, 0 cycles | 0.138 [1.00] | 0.208 [1.00] | 0.298 [0.92] | 0.395 [1.00] | 0.156 [1.00] | 0.232 [0.98] |
| NARS, expectation | 0.147 [1.00] | 0.215 [1.00] | 0.281 [0.92] | 0.357 [1.00] | 0.216 [1.00] | 0.222 [0.98] |

### StationOps temporal replay (seeds 1-4, 20 shifts, budget 50)

| reasoner | seeds | error to oracle | Brier | log loss | coverage | reasoner-only regret | seconds per run |
|---|---|---|---|---|---|---|---|
| exact posterior | 4 | 0.000 | 0.082 | 0.760 | 1.00 | 327.5 | 0.0 |
| pettachainer | 4 | 0.117 | 0.061 | 0.211 | 1.00 | 366.1 | 16.2 |
| nars | 4 | 0.229 | 0.233 | 1.195 | 0.95 | 747.4 | 830.1 |

Summary:

- **SupplyNet.** NARS is within ±0.02 of the base rates' error on every
  stage. It is slightly better on stage 3's production queries, where the
  certain cycle rules give forward deductions from observed facts, e.g.
  untimed production without stock 0.156 against 0.212. PeTTaChainer is
  within 0.01 of exact everywhere.
- **StationOps.** NARS's answers are polarized: in seed 1, 95 of 190 are
  below 0.05 and 31 above 0.85, mostly with confidence 0.99. Certain rules
  (`LeakPredicted ⇒ SealLeak`, `shiftEnd`) turn a deduced prior into a
  near-certain belief. Brier 0.233 and reasoner-only regret 747, against
  PeTTaChainer's 0.061 and 366.
- **Readings.** Expectation instead of frequency trades error for log loss:
  it pulls answers to 0.5. Neither reading changes the picture.

## How fair the comparison is

- **Budget.** No NARS budget matches PeTTaChainer's wall time: reading one
  round's statements already takes 40–90 s (stage 1–3), against 0.2–0.6 s for
  a PeTTaChainer round. NARS was run at its minimum (0 extra cycles) and at
  100 extra cycles per round, one per game budget step. The extra cycles made
  stage 1 worse, and the toy shows the event queue empty after about 100
  selections. A larger budget is not expected to help.
- **In NARS's favour:** base rates and open sensor rates arrive as evidence
  counts, not raw facts; each round holds only the facts the exact posterior
  uses; certain truth values become ONA's maximum confidence 0.99.
- **Against NARS:** the game's semantics are conditional probabilities, which
  NAL does not have (no P(B|¬A) in abduction, revision instead of Bayes'
  rule), and its rules are given rather than learned, which is not where NARS
  is strongest. ONA is a sensorimotor system first. Its declarative inference
  is tuned for small KBs, and its attention spends cycles on structural
  NAL-4 terms these KBs never ask for. A NARS expert might choose another
  term representation (e.g. `<t5 --> [storm_north]>`) or ONA settings
  (concept table size, `DECLARATIVE_IMPLICATIONS_CYCLE_PROCESS`); neither
  changes the truth functions behind mechanism 2.
- **Untested:** OpenNARS 3 / PyNARS question-driven backward inference.

## Other comparison reasoners

| reasoner | fit to the game | effort |
|---|---|---|
| **ProbLog** (`pip install problog`) | CTV rules are probabilistic clauses (`0.8::blocked(r1,T) :- storm(north,T).` plus a clause for ¬storm), facts are evidence, queries are `query/1`; exact inference by knowledge compilation; least-fixed-point semantics settle stage 3's untimed loop as the game does | low: a MeTTa → ProbLog translator of the same size as this one; stages 1–3 and StationOps beliefs (the shortfall fold would need a custom predicate). The strongest candidate: exact, and a known baseline in the PLN literature |
| **pgmpy** (Bayesian network) | the game's KB unrolled over the window is a BN; variable elimination is exact | medium: the unrolling is per stage, i.e. a second reference implementation rather than a reasoner given the KB |
| **Hyperon PLN / lib_pln** | already tried (`feature/libpln-backend`): no base rates, inversion keeps the implication's strength, coverage 0–2 of 8 | done; not useful |
| **LLM baseline** | the MeTTa KB and the queries as text, probabilities as answers | low to build, costly to run (hundreds of queries per run); answers are not reproducible and calibration is the open question; useful as a "general reasoner" reference point |
