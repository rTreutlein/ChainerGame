# ProbLog as the exact comparison reasoner

Status: October 2026. `supplynet run --backend problog` (stages 1–3 and
scale) and `stationops run --backend problog --replay … --temporal-model`
(beliefs only). Code: `src/supplynet/problog_backend.py`,
`src/stationops/problog_backend.py`; runs: `benchmarks/problog/`; results:
`/nexus/Dev/OpenCog/bench/results/problog/`.

## Verdict

The translation is exact. ProbLog matches the exact reference to within
1e-13 on every SupplyNet stage, including the untimed production loop, and on
stage scale whenever it finishes. The cost is the result:

- **Small problems (stages 1–3, scale s):** exact at 0.05–0.3 s per round,
  faster than PeTTaChainer (0.16–0.48 s per round). PeTTaChainer's own
  error there is 0.000–0.008, and in stage 1 almost all of it comes from
  learning the storm base rate from 30 periods. ProbLog given the same
  learned rates instead of the true ones has error 0.0071; PeTTaChainer has
  0.0077.
- **Scale:** the cost of exact compilation grows exponentially as the
  unresolved window fills (size m: 0.5, 1.1, 6, 45 s for rounds 1–4, then
  out of memory at 8 GB and timeouts at 120 s). It is also unpredictable in
  size: l (532 hidden nodes per period) finishes in 60 s per round, while m
  (159) and xl (1446) do not. PeTTaChainer answers every query at every size
  in 0.2–27 s per round, within 0.004–0.037 of the exact posterior. When
  ProbLog cannot finish a round it has no answer at all, and its error is
  0.21–0.26.
- **SDD fails first:** with ProbLog's default vtree the SDD compiler
  exhausts 8 GB by round 5 already at size s (22 hidden nodes per period),
  where d-DNNF needs 0.6 s.
- **StationOps replay (temporal model):** exact joint conditioning on every
  alarm and shortfall gives Brier 0.058 and log loss 0.199, against
  PeTTaChainer's 0.061 and 0.212, and less reasoner-only regret (331 against
  366). Its cost grows with the shift count: up to 28 s per shift on seed 3,
  where PeTTaChainer stays under 1.5 s.

## Which ProbLog

**ProbLog 2.3.0** (PyPI), with **PySDD 1.0.6** for the SDD compiler and the
`dsharp` d-DNNF compiler that the ProbLog wheel bundles. Pinned in
`benchmarks/problog/requirements.txt`. It lives in its own venv, so no other
backend needs it:

    cd ChainerGame-problog
    uv venv --python 3.11 .venv-problog
    uv pip install --python .venv-problog/bin/python -r benchmarks/problog/requirements.txt pytest

ProbLog is imported only inside the backends (`problog_backend.load` and the
forked child); the rest of ChainerGame runs without it, and
`tests/test_problog.py` skips when it is missing.

## Translation (MeTTa → ProbLog)

The backend parses the same MeTTa statements PeTTaChainer receives
(`metta.rules`, `observation_facts`, `resolution_facts`; `scale` for stage
scale) and writes one program per round.

| MeTTa | ProbLog | Exact? |
|---|---|---|
| `(Storm north t5)` | `storm('north', 't5')` | yes |
| CTV `(Implication A B) (CTV (STV p c) (STV q c))` | `ant_i(Vars) :- A.` `p::b :- live(T), ant_i(Vars).` `q::b :- live(T), \+ant_i(Vars).` | yes: the two bodies exclude each other, so P(B\|A) = p and P(B\|¬A) = q (no noisy-OR). Every head has exactly one rule (asserted). A branch of strength 0 is left out, one of strength 1 is a plain clause |
| `(And x y (Not z))` | `x, y, \+z` (negations last, once their variables are bound) | yes |
| `(Or x y)` antecedent | one `ant_i` clause per part | yes: logical OR, ProbLog sums over the overlap correctly |
| `(Not x)` | `\+x` | yes (stratified: no negation inside the cycle) |
| variables of A beyond B's, e.g. `$p` in `(And (NextPeriod $p $t) (Storm north $p))` | existential inside `ant_i(Vt)` | yes |
| persistence rule over periods | the CTV above, with `nextperiod('t4', 't5')` facts | yes |
| untimed production loop (`complete_predicates`) | the same certain rules, a positive cycle | yes: ProbLog's least fixpoint is the chainer's complete-predicate reading (nothing runs without stock) |
| observation of a statement some rule derives (`Late`, inspected `Blocked`/`Degraded`, reported `Producing`, `Alarm`) | `evidence(late('s1-1', 't5'), true).` | yes |
| observation of a statement no rule derives (`NextPeriod`, `StockedFuel`, `InputArrived`) | a fact when true, absent when false (closed world; unknown predicates fail) | yes |
| belief query `(: $prf (Storm north t5) $tv)` | `query(storm('north', 't5')).` | answer = the marginal |
| certain confidence `c = 1` | ignored: every rate is a point probability | — |

**Rates and history.** Rates are given: every CTV carries its rate, and
ProbLog uses it as stated. The labelled history is there for PeTTaChainer to
learn base rates from; ProbLog reads it only for the last resolved period.
One rate the MeTTa does not state: in stage 1 storms have no rule (periods
are independent), so the backend states each region's storm rate from the
game's `Rates` (`given_priors`, `r::storm('north', T) :- live(T).`), as the
exact reference does. Stages 2, 3 and scale need no such rate: every hidden
statement has a rule, and the chain starts from the last resolved period.

**What a round's program holds.** A period is `live` from its observation
until it resolves, and every rule holds only for live periods
(`live(T)` in each clause). The last resolved period enters through its
labels and reported production as plain facts, so it d-separates the
window from all older periods; older periods are left out. That is exactly
the information the reference's forward-backward uses, so the answers are
the same posterior, not an approximation of it. Stated as evidence instead,
the whole history would be ground and compiled every round, at a cost
growing with the history and no change in any answer.

**Engines.** Each round is ground (`problog.engine.ground`, unknown
predicates fail), compiled and evaluated in a forked child: d-DNNF through
`dsharp` (`--problog-engine ddnnf`, the default) or SDD through PySDD
(`sdd`; `fsdd` and `sddx` were tried as well). The parent waits
`--problog-timeout` seconds (default 60; the scale runs use 120) and then
kills the child's process group, `dsharp` included; a child that runs out of
memory is recorded as failed. Either way the round's keys stay unanswered
(scored as 0.5). A round's seconds are the wall time of the whole child:
grounding, compilation and evaluation, plus the fork (about 5 ms).

## StationOps temporal replay

`stationops run --backend problog --temporal-model --replay …`, beliefs only
(as for NARS: decisions are expected-value queries, which replay does not
ask). Each shift's view is merged into the statements seen so far, and one
program answers `(SealLeak cohort type unit)` for every incident:

| MeTTa | ProbLog | Exact? |
|---|---|---|
| `persist-<group>`: `(And (NextState $previous $unit) (SealLeakAtShiftEnd … $previous))` → `LeakPredicted`, CTV (1, hazard) | `leakpredicted(G, U) :- live(G, U), leaked(G, U).` `h::leakpredicted(G, U) :- live(G, U), \+leaked(G, U).` | yes |
| `fresh-<group>`: `Serviced` → `LeakPredicted`, CTV (hazard, hazard) | left out | yes: a serviced unit has no `NextState`, so `persist` already gives it the hazard; both rules together would be a noisy-OR, 1 − (1 − h)² |
| `(ForAll ($cohort) (WithPrior P (Implication SealLeak Alarm)))` | `sealleak(C, T, U) :- live(C, T, U), leakpredicted(C, T, U).` and the alarm CTV | yes |
| sensor rate left open: `(STV s 1)` (only P(alarm\|leak)) or `(STV 0.5 0.5)` | Laplace estimate (k + 1)/(n + 2) over the labelled units of the type | an estimate, the same data PeTTaChainer learns from |
| noise sensors (`NoiseN`, equal rates) | their CTV and evidence | yes (likelihood ratio 1) |
| `shiftEnd` | `sealleakatshiftend(C, T, U) :- sealleak(C, T, U).` | yes |
| `shortfallPosterior` (`FoldAllTruth`, `Compute WeightedSubsetPosteriorDP`) | `evidence(leaksum([u(C, T, U, Impact), …], Loss), true).`: the impacts of the leaking candidates sum to the loss | exact, and stronger than the rule it replaces: every shortfall is conditioned jointly with every alarm and every other shortfall, not one shift at a time over the reasoner's own beliefs |
| labelled unit (`SealLeak` fact) | a fact (true) or absent; not `live`, so its alarm is not evidence | yes: its label d-separates its past |
| `patchGoal`, `Consequence` chains | left out | yes: never asked, never observed |

StationOps' "error to oracle" is against the non-temporal reference (cohort
base rate and the current alarm), not an exact posterior of the temporal
model; Brier, log loss and regret are against the truth.

## Validation against the exact reference

Every belief key of every round, ProbLog against `ReferenceBackend` (the
forward-backward posterior under the true rates), seeds 1–4 and 30 rounds:

| stage | keys per seed | largest absolute difference, d-DNNF | SDD |
|---|---|---|---|
| 1 | 272–341 | 3.2e-15 | 2.0e-14 |
| 2 | 343–381 | 4.6e-15 | 1.9e-13 |
| 3 timed | 568–618 | 7.4e-15 | 2.5e-13 |
| 3 untimed (cyclic) | 568–618 | 7.4e-15 | 2.3e-13 |

Stage scale: mean absolute error 1e-15 over all 10 rounds of sizes s and l
(seeds 1, 2), and over every round that m and xl finish. The cyclic untimed
cell checks the least-fixpoint reading: production without stock is 0 in
ProbLog as in the reference. `tests/test_problog.py` checks stages 1–3 with
both compilers, scale s, the timeout path, and the StationOps translation
(persistence, an alarm and a shortfall) against brute-force enumeration.

The stage-1 base-rate check (not in the grid): ProbLog with each region's
storm rate estimated from the 30 labelled periods (k/n, or (k + 1)/(n + 2))
instead of given. Error to exact 0.0071 (0.0072), Brier 0.0278. PeTTaChainer
has error 0.0077 and Brier 0.0282.

## Results

ChainerGame `problog-backend` (off `supply-network` 9cfb585), PeTTaChainer
master dd53deb6, ProbLog 2.3.0. SupplyNet stages: 3 regions, 30 history
periods, 30 rounds, budget 100, evidence k 5. Scale: 20 history periods, 10
rounds, window 8; ProbLog 120 s per round under an 8 GB address-space cap,
2 runs at a time; PeTTaChainer from `supplynet.sweep` (seeds 1–2, xl seed
1). StationOps: the bench suite's temporal replay
(`bench/recordings/fv_s<seed>.jsonl`, 20 shifts, 10 modules, mixed sensors,
budget 50). Other agents' benchmarks shared the machine, so seconds are
indicative. Raw JSON: `/nexus/Dev/OpenCog/bench/results/problog/`.

The ProbLog column of the PeTTaChainer table counts a timed-out round at
its 120 s. "Peak MB" is the largest answered round (the parent, the forked
child and dsharp); the failed rounds reached the 8 GB cap.

### SupplyNet stages 1-3 (seeds 1-4, 30 rounds, budget 100)

Error is the mean absolute difference from the exact posterior; seconds are inference per round (PeTTaChainer: query_many; ProbLog: ground + compile + evaluate in a forked child).

| stage | reasoner | error to exact | Brier | log loss | coverage | s per round | max s per round |
|---|---|---|---|---|---|---|---|
| stage 1 | reference | 0.0000 | 0.0272 | 0.1017 | 1.00 | 0.000 |  |
| stage 1 | prior | 0.2392 | 0.1337 | 0.4346 | 1.00 | 0.000 |  |
| stage 1 | pettachainer | 0.0077 | 0.0282 | 0.1063 | 1.00 | 0.156 |  |
| stage 1 | problog | 0.0000 | 0.0272 | 0.1017 | 1.00 | 0.123 | 0.152 |
| stage 1 | problog-sdd | 0.0000 | 0.0272 | 0.1017 | 1.00 | 0.047 | 0.074 |
| stage 2 | reference | 0.0000 | 0.1540 | 0.4767 | 1.00 | 0.000 |  |
| stage 2 | prior | 0.1833 | 0.2130 | 0.6148 | 1.00 | 0.000 |  |
| stage 2 | pettachainer | 0.0000 | 0.1540 | 0.4767 | 1.00 | 0.342 |  |
| stage 2 | problog | 0.0000 | 0.1540 | 0.4767 | 1.00 | 0.154 | 0.182 |
| stage 2 | problog-sdd | 0.0000 | 0.1540 | 0.4767 | 1.00 | 0.095 | 0.356 |
| stage 3 timed | reference | 0.0000 | 0.1398 | 0.4286 | 1.00 | 0.024 |  |
| stage 3 timed | prior | 0.2286 | 0.2260 | 0.6493 | 1.00 | 0.000 |  |
| stage 3 timed | pettachainer | 0.0035 | 0.1403 | 0.4295 | 1.00 | 0.461 |  |
| stage 3 timed | problog | 0.0000 | 0.1398 | 0.4286 | 1.00 | 0.285 | 1.216 |
| stage 3 timed | problog-sdd | 0.0000 | 0.1398 | 0.4286 | 1.00 | 0.174 | 0.926 |
| stage 3 untimed | reference | 0.0000 | 0.1350 | 0.4143 | 1.00 | 0.024 |  |
| stage 3 untimed | prior | 0.2137 | 0.2116 | 0.6165 | 1.00 | 0.000 |  |
| stage 3 untimed | pettachainer | 0.0018 | 0.1357 | 0.4161 | 1.00 | 0.481 |  |
| stage 3 untimed | problog | 0.0000 | 0.1350 | 0.4143 | 1.00 | 0.195 | 0.266 |
| stage 3 untimed | problog-sdd | 0.0000 | 0.1350 | 0.4143 | 1.00 | 0.141 | 0.916 |

### SupplyNet stage scale (10 rounds, window 8, 20 history periods)

ProbLog per round (all seeds' rounds pooled): rounds answered / timed out (120 s) / failed (out of memory under an 8 GB address-space cap), the median and the largest answered round, peak memory of an answered round, and the error to the exact posterior over all queries (an unanswered query counts as 0.5).

| size | engine | hidden / observed per period | queries per round | answered / timeout / failed | median s per round | max s per round | peak MB | error to exact | coverage |
|---|---|---|---|---|---|---|---|---|---|
| s | ddnnf | 22 / 30 | 14 | 20 / 0 / 0 | 0.30 | 0.64 | 30 | 0.000 | 1.00 |
| m | ddnnf | 159 / 360 | 64 | 8 / 10 / 2 | 3.49 | 46.08 | 712 | 0.258 | 0.39 |
| l | ddnnf | 532 / 648 | 125 | 20 / 0 / 0 | 21.26 | 63.19 | 205 | 0.000 | 1.00 |
| xl | ddnnf | 1446 / 3024 | 188 | 10 / 10 / 0 | 37.20 | 115.00 | 394 | 0.209 | 0.49 |
| s | sdd | 22 / 30 | 14 | 9 / 3 / 8 | 0.10 | 37.62 | 4655 | 0.173 | 0.43 |

ProbLog (d-DNNF) seconds per round by round, mean over seeds (t = timeout, f = failed; the window fills over rounds 1-8):

| size | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|
| s | 0.1 | 0.1 | 0.2 | 0.2 | 0.3 | 0.3 | 0.4 | 0.6 | 0.6 | 0.6 |
| m | 0.5 | 1.1 | 6.2 | 45.4 |  (ff) |  (tt) |  (tt) |  (tt) |  (tt) |  (tt) |
| l | 2.6 | 2.3 | 6.1 | 12.3 | 17.8 | 24.7 | 37.8 | 59.6 | 58.6 | 61.1 |
| xl | 7.7 | 12.9 | 37.2 | 72.9 | 113.8 |  (tt) |  (tt) |  (tt) |  (tt) |  (tt) |

PeTTaChainer against ProbLog per size: error to exact (coverage) and seconds per round, mean over seeds.

| size | PeTTaChainer 1 steps/query | PeTTaChainer 4 steps/query | PeTTaChainer 16 steps/query | ProbLog d-DNNF | local | base rates |
|---|---|---|---|---|---|---|
| s | 0.033 (1.00), 0.19 s, 0.2 GB | 0.033 (1.00), 0.38 s, 0.2 GB | 0.029 (1.00), 0.91 s, 0.3 GB | 0.000 (1.00), 0.35 s | 0.114 | 0.174 |
| m | 0.007 (1.00), 1.60 s, 0.6 GB | 0.004 (1.00), 2.69 s, 1.0 GB | 0.005 (1.00), 7.45 s, 1.7 GB | 0.258 (0.39), 71.27 s | 0.089 | 0.223 |
| l | 0.030 (1.00), 1.84 s, 0.7 GB | 0.005 (1.00), 5.67 s, 2.1 GB | 0.008 (1.00), 15.17 s, 4.1 GB | 0.000 (1.00), 28.31 s | 0.100 | 0.172 |
| xl | 0.037 (1.00), 4.38 s, 1.5 GB | 0.013 (1.00), 15.72 s, 4.5 GB | 0.007 (1.00), 27.17 s, 5.1 GB | 0.209 (0.49), 84.54 s | 0.113 | 0.196 |

**Where the blow-up comes from.** The window couples its periods through
the front, so each round's circuit covers up to 8 periods of a whole zone.
dsharp's cost depends on how its decomposition meets that structure and the
evidence, not on node counts. One zone of size m (4 regions, seed 1), rounds
1–6: 0.2, 0.3, 1.7, 3.9, 33 s, then a timeout at 60 s. With fanout 2
instead of 3 it runs out of memory in round 6. With tiers 3 and fanout 2 it
times out in round 6. With 6 regions, tiers 3 and fanout 2 (an l zone,
larger) it takes 0.4–2.6 s per round. The SDD runs at size s fail with
`exit 1` (out of memory, 4.7 GB peak in an answered round) from round 5 or
6.

### StationOps temporal replay (seeds 1-4, 20 shifts, budget 50)

| reasoner | seeds | error to oracle | Brier | log loss | coverage | reasoner-only regret | s per shift | max s per shift |
|---|---|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.082 | 0.760 | 1.00 | 327.5 | 0.00 | 0.00 |
| pettachainer | 4 | 0.117 | 0.061 | 0.212 | 1.00 | 366.1 | 0.44 | 1.48 |
| problog | 4 | 0.113 | 0.058 | 0.199 | 1.00 | 330.6 | 2.33 | 27.99 |


ProbLog's seconds per shift grow with the game, since every unlabelled
unit's chain and every shortfall stay in the program. Shifts 1, 10 and 20:
seed 1 0.1 / 0.3 / 0.5 s, seed 2 0.1 / 1.1 / 2.9 s, seed 3 0.1 / 0.9 /
23 s (28 s at shift 18), seed 4 0.1 / 0.5 / 1.3 s. Every shift was answered.

## Takeaways for the presentation

1. **Same inputs, same answers.** ProbLog is a faithful exact reading of the
   MeTTa KB: CTV rules map to two mutually exclusive probabilistic clauses,
   And/Or/Not to logic, persistence to period-linked rules, and the cyclic
   production loop to ProbLog's least fixpoint. It reproduces the hand-written
   exact posterior to 1e-13. ProbLog is the correctness yardstick, not the
   reference code.
2. **On the small stages PeTTaChainer is at the exact answer.** The error is
   0.000 (stage 2), 0.002–0.004 (stage 3) and 0.008 (stage 1). In stage 1
   that is the cost of learning base rates: exact inference with the same
   learned rates gets 0.007. The Brier scores agree to the third decimal.
3. **Exact inference does not scale with the problem; PeTTaChainer's cost
   does.** On stage scale a period's evidence couples every other period
   through the front. ProbLog's per-round cost grows roughly ×5 per added
   window period at size m (0.5 → 1 → 6 → 45 s, then it fails). At m and xl
   it answers 39–49 % of the queries in 120 s per round and 8 GB; the error
   is 0.21–0.26, worse than the base rates. PeTTaChainer answers all of
   them at every size, 0.004–0.037 from exact, in 0.2–27 s per round.
4. **Knowledge-compilation cost is not predictable from size.** Size l
   (3.3× m's hidden nodes) compiles in 60 s per round while m runs out of
   memory. One zone of m with 4 regions blows up; the same zone with 6
   regions finishes in 3 s. SDD, the other compiler, fails already at size s.
   An exact engine's cost depends on its compiler's decomposition heuristic
   meeting the problem's evidence. That is an argument for bounded anytime
   inference, whose cost is set by its budget.
5. **Where exact inference is affordable, it is still the quality ceiling.**
   In StationOps the joint conditioning on all shortfalls gives a lower
   Brier (0.058 against 0.061) and lower regret (331 against 366) than
   PeTTaChainer's per-shift DP. Its cost per shift grows with the game
   (to 28 s), whereas PeTTaChainer's stays under 1.5 s.

## Rerun

    cd ChainerGame-problog
    out=/nexus/Dev/OpenCog/bench/results/problog
    JOBS=4 benchmarks/problog/run_grid.sh $out/stages "problog problog-sdd reference prior pettachainer" "1 2 3 4"
    JOBS=2 TIMEOUT=120 benchmarks/problog/run_scale.sh $out/scale "s m l xl" "1 2" ddnnf
    JOBS=2 TIMEOUT=120 benchmarks/problog/run_scale.sh $out/scale "s" "1 2" sdd
    /nexus/Dev/OpenCog/bench/capped bash -c "PYTHONPATH=src /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python -m supplynet.sweep run \
      --pettachainer-path /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer --out $out/scale_pettachainer.json \
      --sizes s,m,l,xl --seeds 1,2 --seeds-for xl=1 --steps-per-query 1,4,16 --jobs 2"
    JOBS=4 benchmarks/problog/run_stationops.sh $out/stationops "reference pettachainer problog" "1 2 3 4"
    python3 benchmarks/problog/report.py $out
