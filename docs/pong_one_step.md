# Pong with one-step rules, goal chaining and anticipation

`onabench.pong`'s PLN agent, October 2026. It replaces the credit-window
agent (supply-network 4b0cf2a), which copied each scored ball's outcome onto
the 6 decisions before it and was given explicit misses.

## Encoding

One time point per decision, `d1, d2, ...`, linked by `(NextPeriod d d+1)`.
A decision states, in this order:

1. the outcome of the step that led to the time point: `(Hit d)` if the ball
   was hit and, with explicit misses, `(Miss d)` if it was missed;
2. the observation, as a positive-only condition `(At x d)` and as an event
   `(Ball x d)`, with x right, left or equal (Pong's level bat, for which ONA
   gets no input, is equal);
3. the link `(NextPeriod d-1 d)`;

and then, after deciding, `(Did op d)` with op left, right, (Pong2) stop or
nothing.

All the rules are refined, stated at the start at strength 0.5 with the
evidence of 0.01 observations, k = 1:

- world model, one step, for every x, op and y:
  `(And (NextPeriod $t $u) (At x $t) (Did op $t)) -> (Ball y $u)`
  (27 rules in Pong, 36 in Pong2);
- goal rules: `(And (NextPeriod $t $u) (At y $t)) -> (Hit $u)` and, with
  explicit misses, the failure rules `... -> (Miss $u)`.

Failure rules exist in two forms. With explicit misses, `(At right $t) ->
(Miss $u)` learns where the ball is missed. Without them, a goal rule whose
hit does not come learns a low rate. Every world rule also learns where its
predicted position does not come.

## Anticipation

Ball, Hit and Miss are PeTTaChainer observed predicates
(`set_observed_predicate`, PeTTaChainer `docs/metta/observed_predicates.md`).
The link into a time point stores false each statement there that a rule
predicts and that was not observed: the positions the ball is not at, and
the outcomes that did not happen. A rule whose prediction did not come thus
gets a negative sample. Without explicit misses, no miss is ever stated:
the agent sees only the hits, as ONA does.

## Decision

One query reads the truths of the observed position's world rules and the
goal rules (12 to 18 `RuleTruth` goals; budget only at a rule's first read,
since the instance tallies keep truths current). The agent then chains the
one-step rules by asynchronous value iteration:

    Q_g(op) = sum_y P(y | x, op) (E(g | y) + gamma V_g(y)),   V(x) <- Q(best op)

Here P(y | x, op) is the world rules' strengths normalized over y, and E is
the goal rule's truth expectation c (f - 0.5) + 0.5, so a state seen rarely
counts near 0.5 and is explored, as ONA reads desire. gamma is 0.9. The best
op maximizes Q_Hit - Q_Miss (Q_Hit without misses). Its desire, the hit
share Q_Hit / (Q_Hit + Q_Miss) (Q_Hit without misses), feeds ONA's babbling
rule: probability 0.2 of a random op unless the desire exceeds 0.55. The cost
per decision is constant (18 truth reads and a few sums).

Rejected:

- One step of lookahead on immediate values (V = 0): Pong stuck at a wall
  (one move does not reach a better state), 0.06 and 0.12 on two seeds at
  2000 steps without misses.
- Strengths instead of expectations for E: a state that missed its first
  chances read 0 and was never revisited (the same traps).
- An Assuming query per op, `(Assuming (Did op d) (Hit d+1))`: 1 to 12 s a
  query, and the same answer for every op (`benchmarks/ona/repros/
  assuming_lookahead.py`). The revision of the three goal rules' views is no
  mixture over the next position, either.
- gamma, seeds 1-3 at 2000 steps (mean hit ratio, pong / pong2; on the
  WIP certain-rows runtime, before the At split, which leaves decisions
  unchanged):

  | gamma | misses | anticipation |
  |---|---|---|
  | 0.8 | 0.609 / 0.973 | 0.514 / 0.857 |
  | 0.9 | 0.581 / 0.961 | 0.589 / 0.980 |
  | 0.95 | 0.585 / 0.948 | 0.611 / 0.928 |

## Chainer gaps found (repros under `benchmarks/ona/repros`)

- **Closed world at past time points** (`closed_world_past_events.py`):
  built as observed predicates on PeTTaChainer branch miner-failures.
- **Miner and failures** (`mined_never_rewarded.py`): built on the same
  branch.
- **Refined-rule rows at certain statements** (`one_step_rule_growth.py`):
  with the observation as the condition (Ball instead of At), the refined
  rules' completions onto it built a row for every past time point, each
  reading the rule's truth, which every sample changes, and joins waited on
  that truth for good. On 2307ca50 a decision took 0.84 s over the first 100
  decisions and 3.8 s over the third 100 (At: 33 to 37 ms). A prototype that
  stops such joins and does not wait on certain premises is on the WIP
  branch certain-rows. It moved accuracy in 23 of 27 compare.sh cases, so
  it is not for landing. The agent avoids the case with At.
- **A key expanded while unknown keeps its started joins once a certain fact
  arrives**: 250 waiting joins a decision when the goal rules read the
  outcome before the step played. The agent's ordering avoids it.
- **Forgetting breaks refined tallies**: `set_forgetting(20, 1)` with an
  epoch per decision halves memory, but a world rule's truth read (0.005,
  c 0.50) instead of (0.944, c 0.997).
- **Instance sampling depends on conjunct order**: with `(Did op $t)` first
  in the world rules' antecedent, no instance was ever sampled (the truths
  stayed at the prior).

## Results

Seeds 11-20, 10001 steps, PeTTaChainer 2307ca50 (the one-step agent with
the harness stating the false facts; the chainer's observed predicates
decide identically, checked over 300 steps of each game and mode). The
credit-window agent is supply-network 4b0cf2a rerun on the same runtime, and
ONA is on the port. Raw results and the full tables (learning curves, per
seed) are in `/nexus/Dev/OpenCog/bench/results/pong-onestep/` (`tables.md`).

| game | agent | hit ratio (sd) | ms per decision (mean, p95) | 50 ms deadline: hit ratio, late steps |
|---|---|---|---|---|
| pong | ONA | 0.807 (0.035) | 0.7, 1.0 | |
| pong | credit window, explicit misses | 0.642 (0.174) | 12.7, 25.0 | 0.753, 37 |
| pong | one-step, explicit misses | 0.729 (0.051) | 49.3, 79.8 | 0.555, 1938 |
| pong | one-step, anticipation only | 0.647 (0.032) | 43.9, 74.8 | 0.614, 1050 |
| pong2 | ONA | 0.830 (0.230) | 0.9, 1.1 | |
| pong2 | credit window, explicit misses | 0.980 (0.026) | 14.0, 27.2 | 0.974, 97 |
| pong2 | one-step, explicit misses | 0.997 (0.004) | 57.0, 106.2 | 0.847, 1926 |
| pong2 | one-step, anticipation only | 0.991 (0.004) | 42.8, 71.2 | 0.812, 2175 |

Learning curves (hit ratio per window of steps): in Pong the one-step agent
starts slower than ONA (0.38 to 0.41 against 0.42 in the first 1000 steps)
and reaches 0.81 (misses) and 0.69 (anticipation) in the last 2000, against
ONA's 0.86. In Pong2 both one-step agents reach 1.000 after 1000 steps; ONA
plateaus at 0.85 (three seeds near 0.5).

Cost grows with the history in every PLN agent (the history-growth work on
PeTTaChainer is separate): the one-step agent goes from about 30 to 35 ms in
the first 1000 decisions to 57 to 78 ms in the last 2000. The credit-window
agent goes from 10 to 19 ms. At a 50 ms deadline, the one-step agent's
decisions overrun more and more steps as the run goes on, and its hit ratio
falls over the run (Pong2 with anticipation: 1.000 in steps 1000-2000, 0.65
in the last 2000).
