# SupplyNet: a supply network with feedback

Status: stages 1 to 3 and stage "scale" built (`src/supplynet`, see the
Stage 1, 2, 3 and scale results), October 2026.

## Why another benchmark

StationOps is one causal layer: a hidden leak per module, a sensor, an
outcome, priors per cohort, and incidents that are almost independent. Since
decisions became `Assuming` queries in the belief knowledge base
(`single_kb_actions.md`), it runs flat over 60 shifts and leaves PeTTaChainer
with little that is structurally new to answer.

SupplyNet keeps what StationOps measures (beliefs against an exact posterior,
decisions against an oracle, quality per unit of compute) and adds the
structures it lacks:

| Structure | What it asks of the reasoner |
|---|---|
| common causes | a regional disruption explains several late shipments: explaining away, correlated beliefs, evidence that must not be counted twice |
| delays | a disruption shows downstream only after transit time: beliefs about the past from observations now |
| cycles | plants that supply each other's inputs: derivations that must not support themselves |
| interacting interventions | actions that compete for capacity and change each other's effects: joint rather than independent choices |
| lookahead | an action whose value shows only a period later: assumptions over two periods |
| unreliable reports | supplier statements of unknown reliability: trust learned from history |

Each structure can be turned on alone, so a failure points at one capability.

## World

A small network of **sites** in **regions**, linked by **routes**, run for a
number of **periods**.

- **Sites.** Mines produce a raw good. Plants turn inputs into an output and
  need power. A power plant needs fuel from a mine. Warehouses hold stock.
  Markets demand goods each period at a known price.
- **Routes** carry goods between sites; a route has a transit time of 1–3
  periods and a region.
- **Hidden state per period:**
  - `Storm region`: a regional disruption, persisting with probability
    `p_persist` and starting with probability `p_start` per region class.
  - `Degraded site`: a producing site produces at reduced capacity; persists until
    repaired; starts with a per-class hazard (as `SealLeak` in StationOps).
  - A route is **blocked** when its region has a storm, with a route-class
    probability (the common cause); otherwise open.
- **Flows.** Each period, every site produces from what arrived and what it
  holds, ships along open routes and receives what was shipped one transit
  time earlier. Fixed shipping plans; the player changes them only through
  actions. Quantities are small integers so the simulator is exact.
- **The cycle.** Mine → (fuel) → power plant → (power) → mine and plants.
  Power is not stored: a power plant without fuel stops the mine that fuels
  it, unless the fuel warehouse has stock. Whether the loop runs is a
  property of the whole cycle, not of one site.

Default size: 3 regions, 2 mines, 1 power plant, 3 plants, 2 warehouses,
2 markets, about 14 routes; 30 periods. Knobs: number of sites and regions,
transit times, storm persistence, report reliability, horizon.

## What the reasoner sees

The reasoner gets what an operator would:

- **Structure:** sites, routes, regions, transit times, production recipes.
- **Rules with known rates** where an operator would know them (recipes,
  that a storm blocks routes in its region), and **without rates** where
  they must be learned (storm start and persistence per region class,
  degradation hazards, block probability per route class).
- **History:** resolved past periods, as StationOps' initial snapshots.
- **Each period:**
  - shipments arrived and their lateness (exact);
  - own warehouse stock and own production (exact);
  - weather reports per region (noisy);
  - supplier reports, "our site is fine" or "we are degraded", whose
    reliability per supplier is unknown and learned from inspected outcomes.
- **Inspections** reveal a site's or region's true state, as in StationOps.

The simulator's hidden state is never shown; it is used only for scoring.

## Decisions

Each period the player has a budget of credits and a few action slots.

| Action | Effect | Interaction |
|---|---|---|
| `Inspect site/region` | reveals its state now | costs a slot; value is information |
| `Repair site` | clears `Degraded`; the site produces nothing this period | short-term loss for later gain |
| `Reroute flow route-a route-b` | moves a flow to another route from now on | adds load to route-b; useless if route-b's region is also stormy |
| `Stockpile site good n` | ships extra to a warehouse | uses route capacity now; protects the cycle later |
| `Expedite shipment` | halves remaining transit | costs credits |

Actions compete for slots and credits and change each other's value:
rerouting two flows onto one route overloads it, and stockpiling fuel makes
repairing the power plant cheap. The reference policy chooses jointly.

## Representation

The same conventions as StationOps' temporal model:

- **Periods** are individuals linked by `(NextPeriod $previous $now)`.
- **Persistence and onset:**

      (: persist-storm-north
         (Implication (And (NextPeriod $p $t) (Storm north $p)) (Storm north $t))
         (CTV (STV 1 1) (STV <start rate> 1)))

  with rates learned from history when not given (rule refinement, as for
  StationOps' hazards).
- **Common cause:**

      (Implication (And (Storm $region $t) (InRegion $route $region)) (Blocked $route $t))

  with a rate per route class; `(Late $shipment)` follows from `Blocked`
  over the periods the shipment was in transit, so a late shipment now is
  evidence about a storm several periods ago.
- **Production and the cycle:**

      (Implication (And (Powered $site $t) (InputArrived $site $t) (Not (Degraded $site $t)))
                   (Producing $site $t))
      (Implication (Or (Producing power-plant $t) (StockedPower $t)) (Powered $site $t))

  The cycle passes through time (fuel produced at `t` arrives at `t+1`), so
  each period's derivation is acyclic. Within a period, the mine and power
  plant support each other only through stock. Stating that loop without the
  time index is the cycle test (stage 3).
- **Reports:** `(Reports supplier-a (Degraded site-a) $t)` with a learned
  `(Implication (Reports $supplier $claim $t) $claim)` per supplier, fitted
  from inspected cases like StationOps' induced sensors.
- **Decisions** are `Assuming` queries, as in StationOps:

      (: $prf (Assuming (And (Action t7 (Reroute f3 r9)) <period facts>)
                        (PeriodValue t8 $value))
         $tv)

  `PeriodValue` folds outcome amounts over their truth values
  (`FoldAllTruth`), as `ExpectedValue` does now. **Lookahead** assumes a pair
  of actions over two periods: `(Assuming (And (Action t7 a) (Action t8 b)) …)`.
  The game asks for a small set of candidate pairs, not all of them.

## Oracle and scoring

The hidden state per period is small: one bit per region and one per
producing site (mines, power plant, plants). The default size has 3 + 6
bits, so exact forward filtering over the joint state is cheap (512 states,
factored transitions, observations conditionally independent given the
state).

- **Belief quality:** Brier score and log loss of the reasoner's
  `Storm`/`Degraded`/`Blocked` beliefs against the exact posterior, and
  against the truth.
- **Decision quality:** the reference policy uses the exact posterior and the
  true dynamics, and chooses jointly by expected value over a one-period
  horizon (two with lookahead), searching the actions exhaustively. Regret
  is reported per period.
- **Outcome:** profit, meaning demand served times price, minus costs. Also
  the periods the fuel cycle was stalled.
- **Compute:** time per period, split into belief queries, decision queries
  and forward runs. Peak memory.
- **Hypotheses (stage 7):**
  - each candidate's truth over time, against its true conditional
    probability, which is known to the simulator;
  - the period at which true candidates become usable and spurious ones
    stop being confirmed;
  - the share of search spent on rejected candidates;
  - the decision gain over the same game without the candidates.

As in StationOps, a run streams one JSON object per period, and a replay
mode applies a recorded run's actions so only the beliefs differ.

## Stages

Each stage is a playable benchmark with its oracle and tests before the next
starts.

1. **Common causes, no time.** One period, storms and blocked routes, late
   shipments, inspections. Tests explaining away and evidence overlap:
   - two late shipments from one region raise the storm belief once per
     shipment, not once per path through the storm;
   - an inspected clear route lowers the others' block beliefs.
2. **Time and delays.** Periods, persistence, transit times. Beliefs about
   past storms from late arrivals now; filtering against the exact
   posterior.
3. **The cycle.** Mine–power–plant loop with stock. Production beliefs
   through the loop. The same loop stated without the time index tests
   self-support exclusion: with no stock, the loop must not prove itself
   running.
4. **Interacting decisions.** Rerouting, stockpiling, repair with downtime;
   joint reference policy; `Assuming` decisions with shared capacity.
5. **Lookahead.** Two-period action pairs; repair-now versus later.
6. **Unreliable reports.** Supplier statements, learned reliability.
7. **Hypotheses.** The game states candidate implications with a weak prior
   (PeTTaChainer `docs/metta/hypothesis_rules.md`) and the chainer confirms or
   rejects them from the data as it arrives. Candidates are a mix of:
   - **true rules**, such as `(Storm north $t) → (Blocked r2 $t)`, where the
     rule's rate is left for the data;
   - **spurious ones** that hold only through a common cause, such as
     `(Late $s r1) → (Late $s r2)` for two routes in one region;
   - **noise:** unrelated pairs.

   Observational data confirms the spurious ones as well as the true ones.
   Interventions (rerouting, expediting) and inspections break them: a
   rerouted shipment's lateness no longer follows the old route's region.
   Generating candidates stays outside the chainer, so a pattern miner can
   replace the game's candidate list later without changing the scoring.

## What would count as success

- **Beliefs:** Brier within a set margin of the exact posterior, at a
  compute cost per period that stays flat over 30 periods.
- **Decisions:** regret against the reference policy that shrinks with
  budget, at each stage.
- **Each structure** has a stage where PeTTaChainer's weakness, if it has
  one, shows as a gap to the oracle we can name. Expected candidates:
  - double counting through a shared cause (stage 1);
  - cyclic self-support (stage 3);
  - decision cost growing with the number of candidate action pairs
    (stage 5).

## Open questions

- **Separate package or inside StationOps?** A sibling package in this
  repository (`src/supplynet`) sharing the backend adapters seems cleaner
  than a StationOps mode.
- **Candidate pairs for lookahead.** Who proposes them: the game (top single
  actions extended by one), or the chainer (an open query for the second
  action under an assumed first)? The second tests more but may be
  expensive.
- **Continuous quantities.** Stock and flow amounts are small integers so
  the oracle stays exact; a larger version would need an approximate oracle.
- **MM2.** Whether the MM2 backend takes part from stage 1, as in StationOps,
  or only once stages are stable.

## Stage 1 results

`supplynet run --backend reference|prior|pettachainer`: 3 regions, 10 routes,
17 shipments; 30 labelled periods of history; 20 rounds, each observing
every shipment's lateness and a quarter of the routes by inspection. The
rates are given as certain CTV rules: `(Storm r $p) → (Blocked route $p)` and
`(Blocked route $p) → (Late shipment $p)`. Base rates come from the history,
and each round joins it once scored.

Mean over seeds 1–4:

| reasoner | Brier | log loss | error to exact posterior | seconds per run |
|---|---|---|---|---|
| exact posterior | 0.028 | 0.105 | 0 | 0 |
| history base rates only | 0.136 | 0.444 | 0.242 | 0 |
| PeTTaChainer, budget 20 | 0.113 | 0.368 | 0.151 | 3.4 |
| PeTTaChainer, budget 100 | 0.113 | 0.368 | 0.151 | 14.7 |
| PeTTaChainer, budget 400 | 0.113 | 0.368 | 0.151 | 15.7 |

PeTTaChainer answers every belief, but barely beats the base rates, and
more budget changes nothing: it finds all the evidence and combines it
wrongly.

**Example.** Seed 7, round 19: 6 of the east region's 9 shipments are late.
The exact posterior for an east storm is 0.996; PeTTaChainer gives 0.148,
with confidence 0.997. The proof inverts each shipment's lateness to its
route and each route to the storm, then merges the pieces by revision.
Revision averages strengths weighted by confidence. Conditionally
independent evidence about one hidden cause should instead multiply its
likelihood ratios, so on-time shipments dilute late ones towards the base
rate. The confidence, meanwhile, adds up as if the pieces were independent
observations of the storm itself.

This is the opposite of the double counting stage 1 was designed to catch:
evidence is under-combined, not over-counted.

**After factoring plain inversions** (PeTTaChainer master, prior-factored
merge extended to population priors): plain-implication inversions now combine
over their antecedent's base rate by Bayes' rule instead of revising, at both
levels (shipments → route, routes → storm). Mean over seeds 1–4, budget
100: Brier 0.065, log loss 0.238, error to the exact posterior 0.081 (from
0.113, 0.368, 0.151). What remains: a route's own belief revises its
shipment-factored view with the forward view from the storm, and both
contain the route's prior, so it is counted twice.

**After factoring a derived view in the base rate's place** (PeTTaChainer
735f2c3d): a route's forward view from the storm and its shipment
inversions combine by Bayes' rule too. Mean over seeds 1–4, budget 100:
Brier 0.034, log loss 0.153, error to the exact posterior 0.019 (exact
posterior Brier 0.028), at unchanged cost. Every answer is a
`factored-revision`. The remaining gap comes from degenerate inversions: an
inversion recovers P(late | open) from the base rates, which can be
inconsistent in small histories (clipped to 0), and a near-certain update
then decides a factored merge whatever its confidence.


## Stage 2 as built

`supplynet run --stage 2` (stage 1 stays the default). Stage 1 is the special
case of the same code with transit 0 and no persistence; its reference and
prior numbers are unchanged.

- **World.** Stage 1's network, plus a transit time of 1–3 periods per route.
  A storm continues with `p_persist` = 0.7 and starts with the region kind's
  stage-1 rate (coastal 0.3, inland 0.1); the first history period is drawn
  from the stationary rate. Blocks and lateness as in stage 1, given the
  period's storm; every route's shipments depart every period, and a
  shipment's lateness is decided by its route's block in the departure period.
- **Observations.** In period t the operator sees the lateness of the
  shipments departed at t − L on each route with transit L, and the
  inspected routes of period t (a quarter of them). Nothing departing at t has
  arrived, so the current storm is seen only through inspections and
  persistence.
- **Resolution.** A period is resolved (its storms and blocks labelled) once
  all its shipments have arrived, i.e. at the end of round d + Lmax for the
  network's longest transit Lmax. The history periods are resolved before the
  first round, including the lateness of their shipments still in transit at
  its end. The reference uses the same information.
- **Queries per round.** Every region's storm now and in the previous period
  (when that period is unresolved, i.e. from round 2 on), and every
  uninspected route's block now.
- **Exact reference** (`world.Knowledge`). Per region, enumerate the storm
  sequences of the unresolved window (Lmax + 1 ≤ 4 periods, at most 16
  sequences), summing each route-period's block out given that period's storm.
  The distribution carried into the window is the last resolved period's
  storm, a point mass, because resolution labels it; before any period it is
  the stationary rate. A test compares it with brute-force enumeration over
  all storm and block states (1 region, 2 routes, 6 periods), before and
  after resolving the first period.
- **MeTTa.** Periods are individuals linked by `(NextPeriod p t)` facts, stated
  with each period's observations. Persistence per region:

      (: persist-north
         (Implication (And (NextPeriod $p $t) (Storm north $p)) (Storm north $t))
         (CTV (STV 0.7 1) (STV <start rate> 1)))

  The persistence rate is given, unlike the design's `(STV 1 1)` with a
  learned rate. Block rules as in stage 1; lateness is keyed by departure
  period, `(Late s1-1 t5)`, and stated when the shipment arrives, rather than
  derived from blocks over the transit periods. Base rates come from the
  resolved periods, which join the knowledge base as they resolve.
- **Scoring.** As stage 1, plus a split per query kind (`storm`,
  `previous_storm`, `blocked`) in every round record and the summary.

## Stage 2 results

3 regions, 7–9 routes; 30 labelled history periods, 30 rounds, inspection
rate 0.25; PeTTaChainer master 735f2c3d, evidence k 5. Error is the mean
absolute difference from the exact posterior.

| reasoner | seed | Brier | log loss | error | error: storm now | error: previous storm | error: block now | seconds |
|---|---|---|---|---|---|---|---|---|
| exact posterior | mean 1–4 | 0.154 | 0.477 | 0 | 0 | 0 | 0 | 0 |
| history base rates only | 1 | 0.179 | 0.541 | 0.151 | 0.187 | 0.246 | 0.094 | 0 |
| | 2 | 0.226 | 0.643 | 0.219 | 0.286 | 0.358 | 0.129 | 0 |
| | 3 | 0.250 | 0.698 | 0.218 | 0.267 | 0.277 | 0.162 | 0 |
| | 4 | 0.197 | 0.578 | 0.146 | 0.159 | 0.227 | 0.096 | 0 |
| | **mean** | **0.213** | **0.615** | **0.183** | **0.225** | **0.277** | **0.120** | 0 |
| PeTTaChainer, budget 100 | 1 | 0.182 | 0.545 | 0.153 | 0.193 | 0.258 | 0.089 | 17.6 |
| | 2 | 0.199 | 0.586 | 0.177 | 0.225 | 0.300 | 0.103 | 18.2 |
| | 3 | 0.225 | 0.638 | 0.179 | 0.221 | 0.241 | 0.126 | 18.6 |
| | 4 | 0.197 | 0.579 | 0.139 | 0.163 | 0.239 | 0.074 | 13.1 |
| | **mean** | **0.201** | **0.587** | **0.162** | **0.200** | **0.260** | **0.098** | 16.8 |

Exact-posterior Brier by kind (mean): storm now 0.172, previous storm 0.141,
block now 0.151. PeTTaChainer answers every query (coverage 1.0) and its
cost per round stays flat (0.4–0.8 s), but it is only a little better than the
base rates, and worst on the previous period's storm, the query that needs
late arrivals.

Budget (seeds 1 and 2, mean):

| budget | Brier | error | storm now | previous storm | block now | seconds |
|---|---|---|---|---|---|---|
| 20 | 0.185 | 0.153 | 0.192 | 0.267 | 0.087 | 8.5 |
| 100 | 0.191 | 0.165 | 0.209 | 0.279 | 0.096 | 17.9 |
| 400 | 0.191 | 0.169 | 0.216 | 0.284 | 0.099 | 26.7 |

More budget makes it slightly worse: the search finds the same views and
costs more.

**Diagnosis: the persistence view shuts out the window's evidence.** Seed 2,
round 26 (window t23–t26; north stormed in the last resolved period). North
has a late arrival from t25 and route r1 inspected blocked at t26:

| query | exact | PeTTaChainer |
|---|---|---|
| Storm north t25 | 0.950 | 0.532 |
| Storm north t26 | 0.963 | 0.513 |
| Storm east t25 | 0.029 | 0.468 |
| Storm south t26 | 0.043 | 0.487 |

Every storm answer in the window is the persistence chain alone, e.g.

    (by persist-north (conjunction next-t25 (by persist-north (conjunction next-t24
      (by persist-north (conjunction next-t23 (merge/revision storm-north-t22 …)))))))
    (STV 0.532 0.99994)

i.e. 0.7 → 0.58 → 0.532 from the resolved storm, the exact prediction with
no evidence; east and south from no storm give 0.3 → 0.42 → 0.468. No
inversion from the window's late shipments or inspected routes takes part.

A one-region toy (40 history periods with no storm in the last, then period
w1 with all three shipments late; exact P(Storm w1) = 0.99) isolates it:

| variant | Storm north w1 | proof |
|---|---|---|
| persistence rule, w1 linked | 0.300 (conf 0.9999) | the persistence chain alone |
| same, route r1 also inspected blocked at w1 | 0.300 (conf 0.9999) | the persistence chain alone |
| persistence rule, w1 not linked (no `NextPeriod` for w1) | 0.988 (conf 0.80) | `factored-revision cpu` over both block inversions |
| no persistence rule | 0.988 | same |

So the evidence views are derived and dropped in the merge with the
persistence view, even when the evidence is an inspected fact. The
persistence view's proof explains why: each resolved period's certain storm
label is revised with the inversions of that period's block facts
(`(merge/revision storm-north-h39 ((inverted block-r1 …) blocked-r1-h39) …)`),
and the chain recurses through every history period. Its evidence set
therefore contains the region's block rules, and every block inversion about
the window uses the same rules. Under the prior-factored merge
(`docs/metta/prior_factored_merge.md`) views that share evidence, rules
included, are not factored; the overlap rule keeps one view, and keeps the
persistence view, whose confidence (0.9999, from certain rules and facts) is
the higher. When the history holds only storm labels, so that the persistence
view contains no block rule, the merge does factor the persistence view in
the base rate's place: `(factored-revision (by persist-north …) (prior (STV
0.3 0.9999)) (block inversions))`. That variant's value is not meaningful,
since without block and lateness history the inversions are wrong even
without persistence.

Two causes, both in PeTTaChainer:

1. A certain stored fact (a resolved storm label) is still revised with
   derived views, so it carries their evidence (the block rules) into
   everything derived from it, and the forward chain descends through the
   whole history instead of stopping at the label.
2. Sharing a rule with given, certain rates counts as shared evidence, so the
   persistence prior and the window's likelihoods are treated as overlapping
   and one is discarded instead of multiplied.

Either fix alone is expected to let the window's inversions update the persistence
prediction, which is the forward filter the exact reference computes. The
remaining gap would then be backward smoothing: the previous storm also needs
the current period's inspections through the persistence rule's inversion.

**Stage 2 after the merge fixes** (PeTTaChainer c2fdc232: a certain stored
fact ends its key's merge; rules certain in every branch carry no rule
evidence): mean over seeds 1–4, 30 rounds, budget 100: Brier 0.155
against exact 0.154 (master 0.202), error to the exact posterior 0.010
(0.166); by kind, storm 0.0012, previous storm 0.039, block 0.0007.
About 16% slower on stage 2 (answers hold every factored part); StationOps
unchanged in results and cost. What remains is smoothing: the previous
period's storm takes the next period's evidence through an antecedent
completion of the persistence rule, which names no prior and so is revised,
not factored.


## Stage 3 as built

`supplynet run --stage 3 --cycle timed|untimed` (timed by default). Stage 3
is stage 2's network, with its storms, routes, transits and observations
unchanged, plus a production cell; the cell is appended to the network after
it is drawn, so a seed's stage-2 network is kept. Stages 1 and 2 give the
same reference and prior results as before (to 1e-16: the exact posterior is
now computed by forward-backward instead of enumerating the window's storm
sequences).

- **Cell.** A mine, a power plant and two plants (`plant-a`, `plant-b`). The
  mine's fuel travels to the power plant over a new route `rf` in a random
  region, blocked by that region's storms like any route; it carries no
  shipments of its own. Each plant takes its input from one shipment of the
  network, drawn at random: the input is available in period t when that
  shipment, departed t − L, is not late. Power is not stored. The fuel
  warehouse holds stock in a period with probability 0.4, independently of
  production: it is supplied from outside and observed, so it acts only as the
  loop's bootstrap. (The design's integer stock levels, drawn down and
  refilled by the mine, are left out: a stock level the mine refills would
  itself be evidence about the fuel route, which the chainer's rules do not
  state.)
- **Degradation.** `Degraded site` per period, a chain like the storms: it
  persists with 0.75 (otherwise the site is repaired) and starts with the site
  kind's hazard (mine 0.08, power plant 0.05, plant 0.1); the first history
  period is drawn from the stationary rate.
- **Production** in period t is the *least* fixed point of the rules, iterated
  from nothing producing (`world.production`): the power plant runs when it is
  fuelled and not degraded; it is fuelled by stock, or by the mine's fuel over
  an open fuel route; the mine and each plant run when the power plant runs,
  their input arrived (plants) and they are not degraded. Nothing produces by
  itself: without stock, fuel comes only from the mine.
  - *timed:* the fuel route's transit is 1, and the fuel arriving at t is what
    the mine produced at t − 1, if `rf` was open at t − 1. Each period's rules
    are acyclic; the loop passes through time, so once running it sustains
    itself without stock until a degradation or a block stops it, and stock
    restarts it.
  - *untimed:* the transit is 0 and the mine's fuel arrives within the period.
    The rules are a loop, mine → fuel → power plant → mine, and its least
    fixed point is that the power plant runs if and only if there is stock and
    it is not degraded; without stock nothing runs, although everything
    running is also a fixed point.
- **Observations per round,** besides stage 2's: the inspected sites'
  degradation (each site with the inspection rate, all in the history), the
  fuel stock and the plants' inputs (exact), and the previous period's
  production of every site (exact, reported at the period's end). The fuel
  arriving at the power plant is not reported: it is seen only through the
  power plant's output. In the timed cell a power plant that ran without stock
  is evidence that `rf` was open in the previous period, and so about the
  storm there.
- **Queries per round,** besides stage 2's (which now include `Blocked rf`):
  `Degraded site` now for every uninspected site, and `Producing site` now for
  every site, asked before the period's production is reported. Scoring
  splits production by the period's stock: `producing` (stocked) and
  `producing_unstocked`. In the untimed cell the exact answer for
  `producing_unstocked` is 0, so its error is the self-support a reasoner
  attributes to the loop. Every kind also reports its coverage.
- **Resolution** as in stage 2, at the end of round d + Lmax; a resolved
  period also labels its degradations and `rf`'s block.
- **Exact reference** (`world.Knowledge`). Per region, forward-backward over
  the window; a region's state is its storm, and in the cell's region also
  `rf`'s block and the four degradations (2^6 states). The other routes'
  blocks are summed out per period given the storm, as before. A window
  period's reported production is a 0/1 factor between consecutive states: it
  is determined by the period's state, the previous state's `rf` block and the
  mine's previous production, which is reported (or labelled, for the period
  before the window). The current production comes from the last pairwise
  marginal. Tests compare it with brute-force enumeration over every storm,
  block and degradation of 3 periods (1 region, 1 plant, both cycles), with
  production computed period by period from the hidden state, before and after
  resolving the first period; and check that without stock the untimed loop
  never runs and its exact production is 0.
- **MeTTa.** Degradation persistence per site as for storms; the cycle as
  certain rules (`(CTV (STV 1 1) (STV 0 1))`):

      (Implication (And (Fuelled power-plant $t) (Not (Degraded power-plant $t))) (Producing power-plant $t))
      (Implication (Or (StockedFuel $t) (FuelArrived $t)) (Fuelled power-plant $t))
      (Implication (And (Producing power-plant $t) (Not (Degraded mine $t))) (Producing mine $t))
      (Implication (And (Producing power-plant $t) (InputArrived plant-a $t) (Not (Degraded plant-a $t)))
                   (Producing plant-a $t))

  The two variants differ in one rule:

      timed:   (Implication (And (NextPeriod $p $t) (Producing mine $p) (Not (Blocked rf $p))) (FuelArrived $t))
      untimed: (Implication (And (Producing mine $t) (Not (Blocked rf $t))) (FuelArrived $t))

  `Powered` is not a predicate of its own: power is not stored, so a site is
  powered exactly when the power plant produces. Observations add
  `(StockedFuel t)`, `(InputArrived plant t)`, inspected `(Degraded site t)`
  and the reported `(Producing site t-1)` as certain facts.


## Stage 3 results

Seeds 1–4, 3 regions, 30 labelled history periods, 30 rounds, inspection
rate 0.25, budget 100; PeTTaChainer master c2fdc232, evidence k 5. Error is
the mean absolute difference from the exact posterior; an unanswered query
counts as 0.5. In brackets: the share of queries answered.

**Timed cycle**

| reasoner | seed | Brier | log loss | error | storm | previous storm | block | degraded | producing (stock) | producing (no stock) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact posterior | mean | 0.140 | 0.429 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.2 |
| history base rates | mean | 0.226 | 0.649 | 0.228 | 0.236 | 0.289 | 0.133 | 0.236 | 0.401 | 0.317 | 0 |
| PeTTaChainer | 1 | 0.133 | 0.413 | 0.022 | 0.001 | 0.050 | 0.001 | 0.069 | 0.063 | 0.006 | 36.4 |
| | 2 | 0.148 | 0.444 | 0.025 | 0.003 | 0.045 | 0.002 | 0.063 | 0.083 | 0.025 | 36.9 |
| | 3 | 0.167 | 0.487 | 0.022 | 0.001 | 0.046 | 0.001 | 0.063 | 0.027 | 0.031 | 36.5 |
| | 4 | 0.136 | 0.432 | 0.028 | 0.002 | 0.028 | 0.001 | 0.078 | 0.075 | 0.036 | 24.0 |
| | **mean** | **0.146** | **0.444** | **0.024** | 0.002 | 0.042 | 0.001 | 0.068 | 0.059 | 0.025 | 33.5 |

Coverage 1.0 throughout. Exact-posterior Brier by kind: storm 0.172,
previous storm 0.136, block 0.169, degraded 0.145, producing 0.101,
producing without stock 0.042; PeTTaChainer's: 0.172, 0.142, 0.169, 0.170,
0.121, 0.043.

**Untimed cycle**

| reasoner | seed | Brier | log loss | error | storm | previous storm | block | degraded | producing (stock) | producing (no stock) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact posterior | mean | 0.135 | 0.414 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.5 |
| history base rates | mean | 0.212 | 0.617 | 0.214 | 0.236 | 0.288 | 0.133 | 0.225 | 0.422 | 0.209 | 0 |
| PeTTaChainer | 1 | 0.173 | 0.519 | 0.107 | 0.001 | 0.049 | 0.001 | 0.068 | 0.394 [0] | 0.500 [0] | 33.6 |
| | 2 | 0.180 | 0.532 | 0.103 | 0.001 | 0.034 | 0.003 | 0.050 | 0.357 [0] | 0.500 [0] | 33.8 |
| | 3 | 0.200 | 0.578 | 0.104 | 0.000 | 0.043 | 0.000 | 0.055 | 0.373 [0] | 0.500 [0] | 31.8 |
| | 4 | 0.176 | 0.537 | 0.109 | 0.001 | 0.028 | 0.002 | 0.047 | 0.354 [0] | 0.500 [0] | 19.0 |
| | **mean** | **0.182** | **0.541** | **0.106** | 0.001 | 0.039 | 0.001 | 0.055 | 0.371 [0] | 0.500 [0] | 29.5 |

Coverage 0.80: every query except production is answered, and no production
query is (480 queries). Exact-posterior Brier: producing 0.103, without stock 0.

**Self-support.** PeTTaChainer never proves the untimed loop running, with or
without stock; it proves nothing about production at all. A toy KB isolates
why (no history, certain facts):

| KB | query | answer |
|---|---|---|
| `(A t)` true, `(Or (A $t) (B $t)) → (C $t)`, no statement about B | `C t` | none |
| same, `(B t)` stated false | `C t` | 1.0 |
| same, `(B t)` derived from a fact by a rule | `C t` | 1.0 |
| loop `(Or (S $t) (M $t)) → (P $t)`, `(And (P $t) (Not (Deg $t))) → (M $t)`, S true | `P t`, `M t` | none |
| same, S false | `P t`, `M t` | none |
| the Or split into `S → P` and `M → P`, S true | `P t`, `M t` | 1.0, 1.0 |
| same, S false | `P t`, `M t` | 0, 1e-6 |

An `Or` is proved only when every disjunct has a proof, however certain the
other one is. Cycle exclusion is sound: `FuelArrived t` can be proved only
through `Producing mine t`, `Producing power-plant t`, `Fuelled t` and back
to the `Or` itself, so it has no proof. The `Or` is then unprovable, and so
is everything after it, even when the stock alone settles it. The design's
failure mode, a loop supporting itself, does not occur; the opposite one
does: the least fixed point's reading that an underivable loop is not
running is never drawn, and the stock case is lost too.

With the `fuelled` rule stated as two certain rules (`StockedFuel → Fuelled`,
`FuelArrived → Fuelled`; a scratch variant, not in the code: each rule's
`(STV 0 1)` negative branch is false of the world), the untimed cell is
answered fully. Error over seeds 1–4 is 0.015–0.021; production without
stock has error 0.000 (no self-support), and production with stock has
error 0.021–0.083, as in the timed cell.

**Diagnosis of the largest timed gap: production reports do not reach
degradation.** Seed 1, round 6. In t5 the power plant ran and the mine did not
(both reported as certain facts), so the mine was degraded in t5. The exact
P(Degraded mine t6) is 0.75, the persistence rate. PeTTaChainer:

| query | exact | PeTTaChainer |
|---|---|---|
| Degraded mine t6 | 0.750 | 0.134 |
| Producing mine t6 | 0.250 | 0.866 |
| Degraded plant-b t6 | 0.750 | 0.253 |
| Producing plant-b t6 | 0.250 | 0.747 |

    (Degraded mine t5): (by persist-degraded-mine (conjunction next-t5 degraded-mine-t4)) (STV 0.08 0.9999)
    (Degraded mine t6): (by persist-degraded-mine (conjunction next-t6 (by persist-degraded-mine …))) (STV 0.1336 0.9999)

The only view of the degradation is the persistence chain from the last
inspection (t4, not degraded). No proof uses the reported production, which
could only enter through an antecedent completion of
`runs-mine: (And (Producing power-plant $t) (Not (Degraded mine $t))) →
(Producing mine $t)`. The production answers are right given that belief:
`Producing mine t6` is `runs-mine` over a power plant that certainly runs
(stocked) and the persistence belief. A toy with only `runs-mine` and 40
fully labelled instances gives no answer for the degradation of an instance
with the power plant on and the mine off, nor with a positive `Healthy`
child in place of the `Not`: the completion produces no result here. It
could not settle this case even if it applied, because its negative branch
P(M | not K) averages over every way K = (power plant ∧ mine) fails, not over
"power plant on, mine off". The same gap is the degraded error of both cells
(0.068 timed, 0.055 untimed) and most of the timed production error. The
previous-storm error (0.04) is stage 2's smoothing gap.

Stages 1 and 2 give their earlier reference and prior results for seed 1
(30 rounds) to 1e-16.

**Stage 3 after connective settling** (PeTTaChainer 07e829a7: a certain stored
part settles an Or to true or an And to false without proofs of the other
parts). Untimed, seeds 1–4: production queries answered 100% (were 0%),
production error 0.052 (0.369), overall error 0.062 (0.106), coverage 0.91.
Without stock, production stays unanswered where no proof exists outside the
loop (answered false only when a degraded site breaks it): concluding
"not running" from the absence of a proof would need closed-world reasoning,
which the chainer does not do. Timed and stages 1–2 unchanged.

**Stage 3 with complete predicates** (PeTTaChainer branch
complete-predicates fb6a6456, against master 28473546, run concurrently).
With a cell, the backend declares `Producing`, `Fuelled` and `FuelArrived`
complete (`metta.complete_predicates`): their rules are every way they become
true, so a statement whose only support runs through its own loop is false,
and what it gets from outside the loop is kept (the chainer's
`docs/metta/complete_predicates.md`). Untimed, seeds 1–4:

| seed | error | coverage | producing (no stock): error [answered] | seconds |
|---|---|---|---|---|
| 1 | 0.055 → 0.021 | 0.93 → 1.00 | 0.270 [0.46] → 0.000 [1.00] | 21.6 → 23.4 |
| 2 | 0.067 → 0.017 | 0.90 → 1.00 | 0.341 [0.32] → 0.000 [1.00] | 22.9 → 24.0 |
| 3 | 0.052 → 0.015 | 0.93 → 1.00 | 0.312 [0.38] → 0.000 [1.00] | 21.7 → 22.0 |
| 4 | 0.075 → 0.015 | 0.88 → 1.00 | 0.399 [0.20] → 0.000 [1.00] | 13.5 → 15.4 |
| mean | **0.062 → 0.017** | **0.91 → 1.00** | **0.331 → 0.000** | +6% |

Brier 0.163 → 0.140 (exact posterior 0.135); production with stock is
unchanged (error 0.052), and the loop is now answered not running without
stock, as the least fixed point says. The declarations change nothing where
no loop exists: timed seed 1 (error 0.0217, Brier 0.1332) and stage 2 seed 1
(error 0.0082, Brier 0.1486) are identical to master, within 1% of its time.


## Stage "scale" as built

`supplynet run --stage scale --size s|m|l|xl` (`src/supplynet/scale.py`), and
the grid `python -m supplynet.sweep run|report` (`src/supplynet/sweep.py`).

**Why.** In stages 1 to 3 every query's relevant evidence is small and local:
regions are independent and only a few periods matter, so about 2 expansions
per query reach everything, and per-query cost is flat in KB size. Those
stages cannot show how a reasoner degrades when it cannot reach all the
evidence. Stage scale makes each query depend on hundreds to thousands of
observations spread over many regions and periods, while the exact
posterior stays linear in the network's size.

**World.** Zones, each with a weather **front**: a two-state chain over
periods that persists with 0.9 and starts with 0.05 (stationary 1/3). Per
period, a zone is a tree hanging from its front:

    Front z ─┬─ Storm g (region; coastal 0.7/0.05, inland 0.5/0.02 given front / no front)
             │    ├─ Alarm a (weather sensor; 0.6/0.25 given storm / none)    × sensors
             │    └─ Blocked k (route; exposed 0.8/0.03, sheltered 0.4/0.03)  × routes
             │         └─ Delayed d (depot; 0.85/0.05) × fanout, for tiers − 1 levels
             │              └─ Late s (shipment; 0.9/0.05) × fanout
             └─ … regions

Zones are independent. Of the ingredients the brief offered, the stage uses:

- *correlated storms across neighbouring regions:* the regions of a zone
  share its front, so a region's late shipments are evidence about its
  neighbours;
- *multi-tier supply chains:* a block reaches a shipment over `tiers` hops
  (route → depots → shipment), each a noisy transmission;
- *long persistence:* the front persists with 0.9 over a window of 8
  unresolved periods, so evidence from every window period matters to every
  other;
- *many weak reports:* alarms (likelihood ratio 2.4 when on, 0.53 when off)
  and noisy late shipments, several per route.

What was left out keeps the structure a chain across time × a tree across
space: storm persistence of its own (a grid of regions × periods, treewidth
the number of regions in a zone), and shipments over several legs in
different regions (loops between regions). With the front as the only
temporal link, the exact posterior is a forward-backward pass over a 2-state
chain per zone.

**Sizes** (`scale.SIZES`; knobs `--zones --regions --routes --tiers --fanout
--sensors` override). Statements are counted over the default 20 history
periods and 10 rounds; stage 2's default run (3 regions, 30 + 20 periods) has
about 1.6k.

| size | zones × regions × routes | tiers, fanout | sensors | hidden / observed nodes per period | queries per round | statements |
|---|---|---|---|---|---|---|
| s | 1 × 3 × 2 | 2, 2 | 2 | 22 / 30 | 14 | 1.4k (1×) |
| m | 3 × 4 × 3 | 2, 3 | 3 | 159 / 360 | 65 | 14k (9×) |
| l | 4 × 6 × 3 | 3, 2 | 3 | 532 / 648 | 125 | 32k (20×) |
| xl | 6 × 6 × 3 | 3, 3 | 3 | 1446 / 3024 | 188 | 122k (75×) |

**Observations.** Each route has a transit of 1–3 periods. In period t the
operator sees the alarms of t and the lateness of the shipments that left
t − L on each route of transit L; a period's own shipments are not seen in
it. History periods (20 by default) are observed fully and labelled.

**Resolution.** A period is labelled (front, storms, blocks, delays) 8 rounds
after it is observed (`--window`), so the unresolved window holds up to 8
periods, every one of them informative about the others through the front.

**Queries per round.** Every front, storm and block now (`front`, `storm`,
`blocked`), and every front and storm of the oldest window period
(`past_front`, `past_storm`), which the window's later evidence smooths. A
round's budget is `--steps-per-query` times its number of queries (or
`--budget`), shared by the round's queries as in the other stages.

**Exact reference** (`scale.Knowledge`). Per zone and window period, upward
messages sum each node's subtree evidence into its parent (normalized per
node, so thousands of observations do not underflow), giving the front's
likelihood; forward-backward over the front's chain gives its marginals,
carried in from the last resolved front (a point mass) or the stationary
rate; a downward pass gives each storm's and block's posterior. Cost is
linear in nodes × window: under 0.1 s per round at size xl. Tests compare it
with brute-force enumeration (two regions under one front over 3 periods,
and a chain of depots below one block), before and after resolving the
first period.

**Baselines.**

- `reference`: the exact posterior, error 0 by definition.
- `local` (`Knowledge(local=True)`): exact given only the query's
  neighbourhood: its own period and the stationary front, the whole zone of
  that period for a front, the query's region of that period for a storm or
  a block. It shows what is reachable without the chain across time and the
  regions around a query. A test checks that it equals the exact posterior
  for a single region observed in a single period.
- `prior`: each statement's frequency in the labelled periods.

**MeTTa.** One certain CTV implication per node from its parent, and the
front's persistence:

    (: persist-z1 (Implication (And (NextPeriod $p $t) (Front z1 $p)) (Front z1 $t)) (CTV (STV 0.9 1) (STV 0.05 1)))
    (: storm-g1 (Implication (Front z1 $t) (Storm g1 $t)) (CTV (STV 0.7 1) (STV 0.05 1)))
    (: blocked-k1 (Implication (Storm g1 $t) (Blocked k1 $t)) (CTV (STV 0.8 1) (STV 0.03 1)))
    (: delayed-d1-1 (Implication (Blocked k1 $t) (Delayed d1-1 $t)) (CTV (STV 0.85 1) (STV 0.05 1)))
    (: late-s1-1 (Implication (Delayed d1-1 $t) (Late s1-1 $t)) (CTV (STV 0.9 1) (STV 0.05 1)))
    (: alarm-a1-1 (Implication (Storm g1 $t) (Alarm a1-1 $t)) (CTV (STV 0.6 1) (STV 0.25 1)))

Observations are certain facts, `(Late s1-1 t5)` keyed by departure period
and `(Alarm a1-1 t5)`, with `(NextPeriod t4 t5)`; resolution adds the labels.
The backends take the stage's statements and knowledge class
(`PeTTaChainerBackend(path, k, scale)`, `ReferenceBackend(scale.Knowledge)`).

**Measurements.** `python -m supplynet.sweep run` runs, in parallel, every
size × seed × steps per query (0.25, 0.5, 1, 2, 4, 8, 16) for PeTTaChainer
and each baseline once per size and seed. Each run is a CLI subprocess under
`ulimit -v 16000000`, started only while 8 GB of memory is available. Per
run it records error to the exact posterior, coverage, Brier against the
exact posterior's, ms per query, setup seconds, statements and peak RSS. It
rewrites its JSON as runs finish; `sweep report` prints the curve tables and
an ASCII plot. Default grid: sizes s, m, l, seeds 1–2; size xl, seed 1.


## Stage "scale" results

PeTTaChainer master d8bf4fcc (frozen at `bench/chainers/master-d8bf4fcc`),
evidence k 5; 20 history periods, 10 rounds, window 8; seeds 1–2 (xl: seed
1). The whole grid (70 runs, 12 in parallel) took 10 min 48 s, bounded by xl
at 16 steps per query (11.5 min alone). Raw JSON:
`bench/results/supplynet_scale/grid_master-d8bf4fcc.json`.

Error to the exact posterior (coverage in brackets; an unanswered query
counts as 0.5):

| steps/query | s | m | l | xl |
|---|---|---|---|---|
| 0.25 | 0.362 (0.00) | 0.421 (0.00) | 0.389 (0.00) | 0.372 (0.00) |
| 0.5 | 0.362 (0.00) | 0.421 (0.00) | 0.389 (0.00) | 0.372 (0.00) |
| 1 | 0.083 (1.00) | 0.023 (1.00) | 0.069 (1.00) | 0.042 (1.00) |
| 2 | 0.083 (1.00) | 0.023 (1.00) | 0.069 (1.00) | 0.042 (1.00) |
| 4 | 0.052 (1.00) | 0.022 (1.00) | 0.070 (1.00) | 0.039 (1.00) |
| 8 | 0.022 (1.00) | 0.016 (1.00) | 0.061 (0.99) | 0.039 (1.00) |
| 16 | 0.028 (1.00) | 0.012 (1.00) | 0.021 (1.00) | 0.026 (1.00) |
| local | 0.114 | 0.089 | 0.100 | 0.113 |
| history base rates | 0.174 | 0.223 | 0.172 | 0.196 |

Brier (exact posterior's in brackets) and cost:

| size | Brier at 1 / 4 / 16 steps | local | base rates | ms per query at 1 / 4 / 16 | peak MB at 1 / 4 / 16 | setup s |
|---|---|---|---|---|---|---|
| s (0.170) | 0.191 / 0.188 / 0.180 | 0.178 | 0.240 | 17 / 48 / 164 | 205 / 239 / 569 | 0.6–1 |
| m (0.053) | 0.053 / 0.053 / 0.056 | 0.082 | 0.130 | 11 / 36 / 412 | 333 / 785 / 4324 | 6 |
| l (0.072) | 0.099 / 0.099 / 0.075 | 0.092 | 0.135 | 14 / 26 / 242 | 661 / 1174 / 6433 | 17 |
| xl (0.092) | 0.105 / 0.104 / 0.101 | 0.124 | 0.164 | 20 / 41 / 310 | 1188 / 2223 / 10314 | 88 |

Above the cliff the chainer beats the local reasoner at every size, and at m
it matches the exact Brier from 1 step per query: the front's chain and the
shared arena let one search serve many roots. The curve is not an anytime
curve, though: a cliff, then steps.

**Weakness 1: the budget cliff.** Below one step per query nothing is
answered, at every size, and the error (0.36–0.42) is twice the base rates'.
Example: size s, seed 1, 0.5 steps per query: 7 steps for 14 roots, coverage
0 in every round. The fix in progress (a cheap estimate for every root first)
should turn the 0.25 and 0.5 rows into at least the base-rate row.

**Weakness 2: answers do not take in evidence that arrives later; late
shipments, the strongest evidence, are reached only at a high budget.** A
period's shipments arrive 1–3 rounds after it, so the exact posterior of a
past storm moves a lot over the window. The chainer's answer to a statement
asked again is, at low budget, the one it gave first. At size m, seed 1, 1
step per query, 117 of 135 re-asked fronts and storms come back unchanged
(error 0.064), and `past_storm` is the only kind where the chainer is no
better than local (l, 4 steps: 0.127 against 0.114). Examples, all from
proofs that use alarms and the front chain only, no `Late` fact:

| run | query | rounds | exact | local | PeTTaChainer |
|---|---|---|---|---|---|
| m, seed 1, 1 step | `Storm g12 t1` | 3 → 4–8 | 0.291 → 0.918 | 0.062 → 0.637 | 0.598 in every round |
| m, seed 1, 1 step | `Storm g10 t3` | 10 | 0.963 | 0.873 | 0.390 |
| l, seed 1, 4 steps | `Storm g21 t3` | 10 | 0.007 | 0.037 | 0.800 |
| s, seed 1, 1 step | `Storm g1 t1` | 4–8 | 0.46–0.49 | 0.846 | 0.018 |

`Storm g10 t3` at round 10 is

    (factored-revision (by storm-g10 (factored-revision (by persist-z3 (conjunction next-t3 …front-z3-h20…))
        (prior …) ((inverted storm-g9 …) (factored-revision … alarm-a9-1-t1 …)) …))
      (prior …) (((inverted alarm-a10-1 …) alarm-a10-1-t3) …))
    (STV 0.390 0.813)

the front chained from the last history label through the window's alarms,
then the region's own alarms. The region's late shipments of t3 (arrived in
rounds 4–6, through `blocked-k → delayed-d → late-s`, three inversions down)
take no part. The search spends its budget on the shallow alarm inversions
first, and a root answered once is not revisited when new facts below it
arrive.

**Weakness 3: more budget buys little until it buys too much.** 1 and 2 steps
per query give identical answers at every size (all 645 answers of m, seed 1);
l and xl are flat from 1 to 8 steps (0.069 → 0.061, 0.042 → 0.039), then 16
steps improve the error (l 0.021, xl 0.026) at 10–20 times the cost per query
and 5–10 GB of memory (xl: 310 ms per query, 10.3 GB peak; memory grows with
the round's budget, not the KB). A scalable reasoner would improve at every
step at constant cost; here quality comes in jumps, and the cost per step
rises with the budget.

**Rerun against another build:**

    cd ChainerGame-scale; unset DISPLAY
    PYTHONPATH=src /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python -m supplynet.sweep run \
      --pettachainer-path /nexus/Dev/OpenCog/bench/chainers/<build> \
      --out /nexus/Dev/OpenCog/bench/results/supplynet_scale/grid_<build>.json --jobs 12
    PYTHONPATH=src python3 -m supplynet.sweep report /nexus/Dev/OpenCog/bench/results/supplynet_scale/grid_<build>.json
