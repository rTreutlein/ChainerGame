# SupplyNet: a supply network with feedback

Status: stages 1 to 5 and stage "scale" built (`src/supplynet`, see the
Stage 1, 2, 3, 4, 5 and scale results), October 2026. Stage 5 is a small,
fully inspected planning benchmark, with limits below.

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
  expensive. Stage 5 measures both; see "Stage 5 results". On the frozen
  chainer the nested open query returns no candidates, so the game supplies
  the working proposal policy.
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


## Stage 4 as built

`supplynet run --stage 4` (`src/supplynet/decisions.py`, the decision step in
`game.run_game`). Stage 4 is stage 3's timed cell, with its storms, routes,
observations and production rules, plus a decision each round. Stages 1 to 3
give the same reference and prior results as before (to 1e-12; the exact
posterior's forward-backward now contracts one variable at a time, below).

- **World.** Stage 3's network for the seed, with two changes drawn after it:
  the fuel route `rf` is exposed, and a sheltered alternate route `ra` in the
  same region carries the mine's fuel when the operator reroutes it. `--plants
  N` sets the number of plants (default 2, stage 3's).
- **Actions** (`world.Action`), decided each round for the current period,
  after its observation and the reasoner's beliefs:

  | action | effect |
  |---|---|
  | stockpile | the fuel warehouse holds stock this period (its only source; the history's stock is the former operator's, drawn with 0.4 as in stage 3) |
  | reroute | the mine's fuel leaves this period over `ra` instead of `rf` (it arrives next period, blocked by `ra`'s block now) |
  | repair site | the site produces nothing this period (downtime) and is healthy at its end: its degradation next period starts with the site's hazard instead of persisting |

  Each action takes one unit of a shared capacity (`--capacity`, default 2:
  two crews or trucks per period), so the default game has 6 single actions
  and 22 feasible joint actions (1 + 6 + 15).
- **Utility** (`decisions.utility`, values in `decisions.Values`): the period's
  production (mine 2, power plant 1, each plant 4), the fuel sent on for the
  next period when the mine runs and the route taken is open now (8), and
  each repaired site's degradation cleared (mine 10, power plant 20, plant 8),
  less the costs (stockpile 2, reroute 1, repair 1). The cleared degradation
  is a terminal value standing for the production a repair saves later; stage
  5 replaces it with the next period itself.
- **Why the actions interact.** Through the cycle: a repaired power plant
  stops everything, so a stockpile or a reroute beside it is wasted; a
  repaired mine sends no fuel, so a reroute beside it is wasted; a stockpile
  that restarts the mine makes a reroute worth something, which alone is worth
  nothing; and repairs taken together lose less production than one at a
  time. Over seeds 1–6 the exact posterior with the singles ranked by their
  own gains (`greedy`) loses 1.5 per round to the joint choice, 13% of what
  doing nothing loses (11.4). The values were set for that: with stage 3's
  production values alone the interaction cost 0.1 per round.
- **Oracle and reference policy.** The world stays exact: production is the
  least fixed point of the rules under the action, and the next period's
  degradations follow the repairs. `world.Knowledge` takes each period's
  action (`act`) and filters exactly over it: a repair changes that site's
  transition out of the period, the route taken decides which block the
  arriving fuel met, the stock is the action's. The reference policy is the
  feasible joint action of highest expected utility under the exact posterior
  (`decisions.expected_utilities`, `best`), enumerated (22 options by
  default; ties go to fewer units). A test checks every option's expected
  utility and the posterior after an action against enumeration over every
  storm, block and degradation.
- **Scoring.** Per round: regret, the exact expected utility of the reference
  policy's choice less that of the reasoner's (0 or more), agreement with
  the reference choice, the realized utility, and the decision's seconds.
  Beliefs are scored as in stage 3 (storm now and before, blocks, degraded),
  without production, which now depends on the action; they are asked before
  the decision.
- **Exact filter, faster.** With the alternate route and up to 5 plants the
  cell region has up to 2^10 states, too many for the pairwise sums stage 3
  used. A state is now a bit mask, the transition is a product over
  variables (a block does not depend on its previous value), and the report
  factor reads the previous state through one bit (the block the arriving
  fuel met), so a forward or backward step is one 2-by-2 contraction per
  variable.

**Reasoners.**

- `reference`: the exact posterior and the reference policy (regret 0).
- `greedy`: the exact posterior, but the single actions are ranked by their
  own gain over doing nothing and taken while they gain and capacity lasts.
  Its regret is what ignoring the interaction costs.
- `prior`: every hidden variable at its history frequency, independent; the
  best joint action under that (`decisions.factored_utilities`).
- `problog`: exact, like its other stages. One more compilation per round
  values every joint action: each single action is a probabilistic fact of
  0.5 that the period's action statements follow, a joint action is the
  conjunction of its singles taken and the others not, and P(outcome | joint)
  = P(joint ∧ outcome) / P(joint). The action facts are roots no evidence
  depends on, so this is exact given the evidence.
- `pettachainer` (and `mm2`, below): decisions in the knowledge base that
  holds the beliefs, as in StationOps (`single_kb_actions.md`).
- NARS takes no part. Its backend translates beliefs; ONA's operations take
  one action towards a goal and have no expected utility over a joint action
  under shared capacity to compare with the reference policy.

**PeTTaChainer's decisions.** The knowledge base gains, once, outcome rules
named `(no_inverse ...)` (`metta.decision_rules`), each outcome carrying its
amount from a stored value fact:

    (Implication (And (OutputValue $site $v) (Producing $site $t)) (Outcome $t (Output $site) $v))
    (Implication (And (FuelValue $v) (FuelVia $r $t) (Producing mine $t) (Not (Blocked $r $t))) (Outcome $t FuelSent $v))
    (Implication (And (RepairValue $site $v) (Repairing $site $t) (Degraded $site $t)) (Outcome $t (Cleared $site) $v))

A joint action's value is one `Assuming` query that assumes its facts and
asks for its outcomes (`metta.decision_query`):

    (: $prf (Assuming (And (FuelVia ra t7) (StockedFuel t7) (Repairing mine t7) (InService power-plant t7) …)
                      (Outcome t7 $o $v)) $tv)

Its expected utility is the sum of the outcomes' amounts weighted by their
strength, less the action's cost (certain, so the client subtracts it). The
taken action joins the knowledge base as facts (`metta.action_facts`).

**Proposals.** `--proposer extend` (default) asks doing nothing and every
single action (1 + n queries for n single actions), then extends the `--beam`
best options of the last layer (2) by one more single action each, until the
capacity is used: beam·(n − 1) more queries per extra unit at most (minus
duplicates). At the default size that is 16 queries for 22 options; with 9
single actions and capacity 3, 38 queries for 130 options. `--proposer
exhaustive` asks every feasible option. The roots of one layer are one
`query_many_refs` call with `--budget` steps per root, so they share the
search of what their assumptions leave unchanged.

**Chainer gaps and the harness's workarounds** (PeTTaChainer master
2307ca50; repros in `benchmarks/supplynet_stage4/repro/`):

1. **An `Assuming` context reads no complete statement as false.**
   `complete-statement?` (`runtime/backward/skeleton.metta`) accepts the
   main context only, so in a decision context a stock or a repair that is
   not assumed has no value, and every rule reading it has no answer:
   `(Assuming (Decide t) (P t))` with `S`, `P` complete and `(Or (S $t) (F
   $t)) → (P $t)` answers nothing (the main context answers `(P t)` from F
   alone; `repro/assuming_complete.metta`). A one-line change (accepting `context-world?` contexts, which see
   the world) fixed it and its MeTTa test passed with `verify.sh`; it was not
   kept, since PeTTaChainer is frozen (`repro/assuming_complete.patch`).
   *Workaround* (labelled in `metta._cycle_rules`): the chainer's rules read
   only positive action facts, one per branch. A site runs while `(InService
   site t)`; the power plant is fuelled by `StockedFuel → Fuelled` and
   `(And (Unstocked $t) (FuelArrived $t)) → Fuelled`, exactly one of the two
   facts stated or assumed per period, so the two rules never give a view
   together. ProbLog and the reference keep the semantic rules (`Or`, `Not
   Repairing`).
2. **`Not` of an unstated complete statement has no answer outside a rule
   concluding a complete predicate** (`repro/not_complete.py`; by design per
   `complete_predicates.md`, but the root case `(Not (R t1))` has no answer
   either). The degradation's persistence reads `(Not (Repairing site $p))`;
   the harness states every past repair true or false.
3. **Later decision contexts lose answers.** With the production rules
   invertible, seed 1's round-1 decision contexts (all 22) value every option
   exactly at 200 steps, and in round 2 every stockpile context answers no
   production outcome at all: the stockpile's value is −2.0 (its cost)
   against an exact 10.6, at 200 and at 1,000 steps alike
   (`repro/round2_contexts.py STEPS`, run from the ChainerGame root). While gap 1 was patched locally, the effect showed on a
   single dumped knowledge base too: the same context answered at 50 steps
   the first time, and not the second and third time at 200 (it did at
   1,000; with 15 history periods instead of 30 it did at 200). Neither
   collector (`deps-gc!`, `stored-refs-gc-step!`) was the cause.
   *Workaround* (labelled in `metta._cycle_rules`): the chainer's production
   rules are `(no_inverse ...)`. Their antecedent completions make the
   degradations and the repairs affected in every context, so each context
   derived the degradation chains again; without them a context reads those
   in the main context, later rounds' contexts answer (86 of the 88 option
   values of seed 1's first four rounds exact at 200 steps, one context per
   option), and beliefs
   are no worse: with a fixed do-nothing policy (seed 1, 30 rounds, budget
   200) the error to the exact posterior is 0.0027 without the completions
   and 0.0062 with them (degraded 0.014 and 0.032), at a third of the time
   (14.5 s against 46.3 s). In stage 3 the completions gave production
   reports no way into the degradations either (Stage 3 results).
4. **An in-KB `FoldAllTruth` over the outcomes misses members, and more
   budget can cover fewer.** The first design folded the outcomes in the KB,
   `(DecisionValue t $g)`, as StationOps does. On the round-1 knowledge base
   (`repro/fold_repro.py STEPS`, a fresh knowledge base per budget) the fold
   for stockpile and repair of the mine covers nothing at 200 steps
   (`partial-fold 0 1`, confidence 0), 1 of 5 members at 400 (value 7.5,
   confidence 0.2), nothing again at 1,000 and 1 of 2 at 2,000 (7.5, 0.5), while the outcome query at the
   same budgets answers every outcome, summing to the exact 12.82.
   `partial_folds.md` expects coverage to grow with the budget. *Workaround*:
   the client sums the outcomes (above).


## Stage 4 results

Seeds 1–4, 3 regions, 2 plants, capacity 2 (22 options), 30 labelled history
periods, 30 rounds, inspection rate 0.25; PeTTaChainer master 2307ca50
(shared venv), evidence k 5, with the harness's workarounds 1–3; ProbLog
ddnnf, 60 s per compilation. Regret is per round, in the utility's units
(doing nothing loses 11.4 per round over seeds 1–6). Raw JSON:
`bench/results/supplynet-stage4/`; `benchmarks/supplynet_stage4/sweep.sh`
reruns it and `report.py` prints these tables.

| reasoner | regret | agreement | realized utility | Brier | error to exact | s/round beliefs | s/round decision | queries/round |
|---|---|---|---|---|---|---|---|---|
| reference | 0 | 1.00 | 12.30 | 0.147 | 0 | 0 | 0.01 | – |
| ProbLog | 0 | 1.00 | 12.30 | 0.147 | 0 | 0.30 | 22.7 | – |
| beam (exact values, extend proposer) | 0 | 1.00 | 12.30 | 0.147 | 0 | 0 | 0.01 | – |
| greedy (exact values, singles alone) | 1.22 | 0.67 | 10.42 | 0.146 | 0 | 0 | 0.01 | – |
| history base rates | 7.61 | 0.06 | 4.11 | 0.211 | 0.188 | 0 | 0.01 | – |
| PeTTaChainer, budget 50 | 2.33 | 0.33 | 11.14 | 0.153 | 0.018 | 0.35 | 4.5 | 16 |
| PeTTaChainer, budget 100 | 1.74 | 0.45 | 11.19 | 0.154 | 0.019 | 0.39 | 7.6 | 16 |
| PeTTaChainer, budget 200 | 1.29 | 0.59 | 11.51 | 0.152 | 0.019 | 0.42 | 11.5 | 16 |
| PeTTaChainer, budget 400 | 0.98 | 0.66 | 11.69 | 0.153 | 0.018 | 0.45 | 12.3 | 16 |
| PeTTaChainer, budget 200, every option | 1.52 | 0.55 | 11.43 | 0.152 | 0.018 | 0.44 | 17.0 | 22 |

Per seed:

| reasoner | regret s1 | s2 | s3 | s4 | error to exact s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|---|
| reference | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| problog | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| beam | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| greedy | 0.80 | 0.61 | 1.87 | 1.60 | 0.000 | 0.000 | 0.000 | 0.000 |
| prior | 8.17 | 6.27 | 7.65 | 8.35 | 0.189 | 0.180 | 0.194 | 0.188 |
| pettachainer-extend-b50 | 1.93 | 1.76 | 1.98 | 3.66 | 0.019 | 0.018 | 0.017 | 0.020 |
| pettachainer-extend-b100 | 2.07 | 1.42 | 1.25 | 2.21 | 0.022 | 0.018 | 0.019 | 0.019 |
| pettachainer-extend-b200 | 2.01 | 1.03 | 1.17 | 0.94 | 0.021 | 0.017 | 0.017 | 0.019 |
| pettachainer-extend-b400 | 1.57 | 0.66 | 0.86 | 0.84 | 0.022 | 0.017 | 0.015 | 0.019 |
| pettachainer-exhaustive-b200 | 2.24 | 1.22 | 1.32 | 1.31 | 0.021 | 0.017 | 0.018 | 0.018 |

Brier differs between reasoners partly because their actions make different
worlds; the error to the exact posterior compares beliefs.

- **Regret falls with budget, beliefs do not change.** The chainer's error to
  the exact posterior stays 0.018–0.019 at every budget, while its regret
  halves from 50 to 400 steps (2.33 → 0.98), below greedy-independent's
  1.22 on average at 400 (each seed is its own world: lower on 2 of 4). What more budget buys is the decision
  contexts' outcomes: at 50 steps a context misses some of them. The budget
  is per decision context (16 per round), so 400 steps is 6,400 per round
  for decisions, against 400 for the round's beliefs.
- **The proposer costs nothing here.** With exact values the extend
  proposer (`beam`) chooses the reference's action in every round of every
  seed and size measured (regret 0, up to 130 options), and the chainer asking every option does no
  better than asking 16 (1.52 against 1.29 at 200 steps).
- **The interaction is visible.** Ignoring it (greedy) loses 1.22 per round
  with exact beliefs: a stockpile picked alone when the reroute with it would
  have saved the fuel, a reroute or a stockpile beside a repair of the mine
  or the power plant that makes it worthless, repairs taken one at a time.
- **Where the chainer's regret comes from: degradations.** Seed 1, budget
  200 (a run before the layers were batched, 30 rounds, regret 1.78): 15 of
  30 rounds lose, and in every one the reference repairs a site the chainer
  leaves alone. In 13 of them the chainer's degradation beliefs are 0.11–0.29
  from the exact posterior: the exact posterior reads a plant that
  did not produce while the power plant ran and its input arrived as almost
  certainly degraded, and the chainer does not (stage 3's gap: production
  reports do not reach degradation, see Stage 3 results). With the cell
  idle (a fixed do-nothing policy), so that production says nothing, the
  same chainer's degradation error is 0.014.

**Decision cost against the number of options** (seed 1, 10 rounds, budget
200; n single actions = 4 + plants):

| plants | singles | capacity | options | proposer | queries/round | answered/round | s/round decision | regret |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 2 | 22 | reference | – | – | 0.01 | 0.000 |
| 2 | 6 | 3 | 42 | reference | – | – | 0.01 | 0.000 |
| 3 | 7 | 2 | 29 | reference | – | – | 0.01 | 0.000 |
| 3 | 7 | 3 | 64 | reference | – | – | 0.01 | 0.000 |
| 4 | 8 | 2 | 37 | reference | – | – | 0.04 | 0.000 |
| 4 | 8 | 3 | 93 | reference | – | – | 0.08 | 0.000 |
| 5 | 9 | 2 | 46 | reference | – | – | 0.05 | 0.000 |
| 5 | 9 | 3 | 130 | reference | – | – | 0.12 | 0.000 |
| 2 | 6 | 2 | 22 | beam | – | – | 0.00 | 0.000 |
| 2 | 6 | 3 | 42 | beam | – | – | 0.01 | 0.000 |
| 3 | 7 | 2 | 29 | beam | – | – | 0.01 | 0.000 |
| 3 | 7 | 3 | 64 | beam | – | – | 0.01 | 0.000 |
| 4 | 8 | 2 | 37 | beam | – | – | 0.03 | 0.000 |
| 4 | 8 | 3 | 93 | beam | – | – | 0.07 | 0.000 |
| 5 | 9 | 2 | 46 | beam | – | – | 0.05 | 0.000 |
| 5 | 9 | 3 | 130 | beam | – | – | 0.09 | 0.000 |
| 2 | 6 | 2 | 22 | problog | – | – | 10.79 | 0.000 |
| 2 | 6 | 3 | 42 | problog | – | – | 23.78 | 0.160 |
| 3 | 7 | 2 | 29 | problog | – | – | 29.39 | 3.627 |
| 3 | 7 | 3 | 64 | problog | – | – | 40.76 | 3.725 |
| 4 | 8 | 2 | 37 | problog | – | – | 49.40 | 7.920 |
| 4 | 8 | 3 | 93 | problog | – | – | 54.87 | 7.276 |
| 5 | 9 | 2 | 46 | problog | – | – | 46.05 | 9.711 |
| 5 | 9 | 3 | 130 | problog | – | – | 58.70 | 13.957 |
| 2 | 6 | 2 | 22 | extend | 16.0 | 16.0 | 9.63 | 0.097 |
| 2 | 6 | 3 | 42 | extend | 23.0 | 23.0 | 12.27 | 1.529 |
| 3 | 7 | 2 | 29 | extend | 19.0 | 19.0 | 8.26 | 1.370 |
| 3 | 7 | 3 | 64 | extend | 28.0 | 28.0 | 13.46 | 1.835 |
| 4 | 8 | 2 | 37 | extend | 22.0 | 21.6 | 11.67 | 0.773 |
| 4 | 8 | 3 | 93 | extend | 33.0 | 32.6 | 21.72 | 1.419 |
| 5 | 9 | 2 | 46 | extend | 25.0 | 25.0 | 15.61 | 0.066 |
| 5 | 9 | 3 | 130 | extend | 38.0 | 38.0 | 23.53 | 0.032 |
| 2 | 6 | 2 | 22 | exhaustive | 22.0 | 22.0 | 15.14 | 1.618 |
| 2 | 6 | 3 | 42 | exhaustive | 42.0 | 42.0 | 28.26 | 1.729 |
| 3 | 7 | 2 | 29 | exhaustive | 29.0 | 29.0 | 14.70 | 1.370 |
| 3 | 7 | 3 | 64 | exhaustive | 64.0 | 64.0 | 33.86 | 1.835 |
| 4 | 8 | 2 | 37 | exhaustive | 37.0 | 36.6 | 26.23 | 0.773 |
| 4 | 8 | 3 | 93 | exhaustive | 93.0 | 92.6 | 102.17 | 1.419 |
| 5 | 9 | 2 | 46 | exhaustive | 46.0 | 45.8 | 31.55 | 0.066 |
| 5 | 9 | 3 | 130 | exhaustive | 130.0 | 129.8 | 134.04 | 0.032 |

- **The chainer's cost per round is linear in the contexts it asks**, at
  about 0.4–1.1 s each (budget 200, batched by layer); exhaustively that is
  every option, 22 to 130, and 102 s per round at 93 options, 134 s at 130. The extend
  proposer asks 1 + n contexts, then 2(n − 1) − 1 more for a second unit and
  2(n − 2) − 1 for a third (16 to 38 here), so its cost grows linearly in n
  and the capacity instead of combinatorially, and its regret was never
  higher than asking every option's. This is the weakness the design
  expected; bounded proposals contain it at these sizes. A complement whose
  parts both rank outside the beam alone would be missed; here the reroute,
  worthless alone, is found by extending the stockpile, which ranks first.
- **ProbLog's exact decisions stop being affordable first.** One circuit
  values every option, but compiling it grows with the joint actions as
  conjunctions of n action facts: its slowest round already takes 46–60 s at
  the default size; at capacity 3 with 6 singles 1 of 10 rounds passes the
  60 s timeout, with 7 singles 3 to 6 of 10, with 8 or 9 singles 7 to 9 of
  10, and a timed-out round does nothing (regret 0.16, 3.6–3.7, 7.3–14.0). Raising the timeout trades time for exactness; the reference's
  enumeration over the exact filter stays under 0.15 s per round.

**MM2** (`--backend mm2`, `mm2_chainer.petta.PeTTaChainer` from
`NL2PLN_Project/mm2-chainer` a227553): no stage 4 numbers. It answers stage 1
(5 history periods: error 0.009), but on stage 2 a round's belief queries take
0.15 s, then 0.78 s, then more than 290 s in round 3, once three unresolved
periods are linked by persistence; on stage 4 (2 history periods) a round's
beliefs take 7.4 s. Its decisions cannot run: it has no `Assuming` queries
(every decision context answers nothing; mm2-chainer
`docs/pettachainer-parity.md` lists them as missing) and no
`set_complete_predicate` (the harness records it in `unsupported` and goes
on).

**Limits.** Rates are given (no learned rules in stage 4). The horizon is one
period with a terminal value for a cleared degradation; stage 5 replaces it
with the next period. NARS takes no part. The chainer's numbers include the
harness's three workarounds (positive action facts, non-invertible
production rules, the client's sum of outcomes); without them the chainer
decides on nothing (gap 1) or on contexts that lose their answers (gaps 3, 4).


## Stage 5 as built

`supplynet run --stage 5` uses `src/supplynet/lookahead.py` and
`lookahead_worker.py`. This stage is benchmark work only. It isolates
planning from the partial-observation problems of stages 1–4:

- One coastal region, the exposed `rf` and sheltered `ra` fuel routes,
  transit one, a mine, power plant, and one plant by default. Plant inputs
  are guaranteed contracts. `--plants 0..3` and `--capacity 1..3` keep the
  enumerations finite; the main grid uses one plant and capacity two.
- The operator observes the full current state: storm, degradations,
  both route blocks, and usable fuel arriving from its preceding action.
  The next observation reveals the same variables. Thus enumeration over
  next states **is** enumeration over next observations, with a point-mass
  posterior after each observation. This is not a clairvoyant reference
  compared with an agent receiving noisy observations. It does not test
  stage 4's filtering or delayed plant-input evidence.
- Production is the existing `world.production` least fixed point, under
  the existing joint `Action` and capacity constraint. Storm persistence,
  degradation persistence/onset, repair downtime and clearance, stock and
  rerouting use the stage 4 rates and mechanics. Repairing a site resets
  its next degradation probability to its onset hazard. The mine's fuel
  actually arrives next period if it produced now and its selected route
  was open now; it is not an independent marginal.
- One decision per physical period, for 30 rounds after 30 history
  periods. Only the first action of a plan is enacted. Every policy
  replans at the next fully observed state. History is used for the
  frozen history-frequency prior baseline; the other planners know the
  rates.

**Horizon and reference.** Two periods, without discounting:

```
Q(s,a) = production(s,a) - cost(a)
       + sum_s' P(s' | s,a) max_b [production(s',b) - cost(b) + inventory(s',b)]
```

The reference enumerates every feasible first action, every next state,
and every feasible second action. At the default size there are 5 single
actions, 16 joint options, 256 fixed pairs, and 64 next states per first
action (storm + two route blocks + three degradations; incoming fuel is
determined by the first action). The two blocks share their storm. The
tests compare this expectimax and every fixed pair with a separately
written exhaustive product calculation, and check repair hazards and fuel
transitions.

Stage 4's cleared-degradation terminal reward is entirely removed from
stage 5's objective. Fuel sent in the **first** period is consumed in the
second and never credited separately. Fuel left after the second period
retains stage 4's value 8 as terminal inventory. Immediate output values
and action costs remain stage 4's. A repair can cost current output and
credits, and pay through actual next-period production. Deferring it
preserves current resources and allows a later decision after onset or
natural recovery. A repair in the plan's final period cannot benefit
production before the horizon ends; such plans are dominated. Repair
later is a choice made at the next receding-horizon decision, not an
artificial final-period repair bonus.

Regret is `max_a Q(s,a) - Q(s,chosen)`. `lookahead_gain` compares the chosen
action with the myopic action **in the same visited state**, using Q;
`utility` is realized immediate output minus costs. Different policies
visit different states, so their average realized rewards are not paired
counterfactual returns. `reference_seconds` reports the reference's own
transition/pair/feedback enumeration per round; its policy decision time
includes that enumeration. Scoring-reference time is excluded from the
other policies' decision times.

**PLN decisions and pair proposals.** Each round creates a bounded
two-period snapshot KB in a fresh child process under the admitted parent
job. The process exits after the decision, discarding all its contexts and
KB. Cold process and library setup are included in seconds per round;
`worker_setup_seconds` and `timing()` deltas additionally separate setup,
inference and overhead. No historical or hypothetical KB grows across
rounds. The rates and inspected current state are facts/rules, not values
calculated by the reference.

The working pair query flattens the two assumptions into one context:

```metta
(: $p (Assuming (And (StockedFuel now) (InService mine now) …
                     (Unstocked later) (InService mine later) …)
                (Outcome $period $outcome $amount)) $tv)
```

Current production determines later `FuelArrived`, and the first action's
`Repairing`/`InService` branch determines later degradation. The client
sums distinct outcome amounts weighted by their truth strengths and
subtracts the costs of both actions. No `FoldAllTruth` is used. Exclusive
positive action branches and `(no_inverse …)` rules retain the stage 4
workarounds. The two alternative sufficient causes of `Fuelled` use positive
STV implications: a CTV zero-valued negative branch on each alternative
would incorrectly deny fuelling when the other cause holds. Because the
state is inspected, each site's next-degradation
rule can directly select its known current state and read either positive
action branch, avoiding an unstated `Not Repairing` in a context.
An explicit zero-valued `Empty` outcome makes a pair's value available
even when all production is off. Consequently `decision_answered` alone
does not establish outcome completeness; the raw records compare every
queried pair's numeric value with the exact fixed-pair value as well.

- **Game proposer (`extend`, default).** Ask no action and all singles,
  with both no action and stock as second-period seed continuations.
  The stock continuation lets repair-now candidates pay even when their
  downtime sends no fuel. Extend the top two singles by one legal action,
  then extend the best two first actions in time with legal second
  actions, interleaved between the two first actions. Every seed and
  extension consumes `--pair-cap` (default 32). A batch uses one
  `query_many_refs` call with `budget × roots` total search steps.
- **Chainer proposer (`open`).** The game supplies only the legal second
  action vocabulary, as rules of the form
  `(Implication (Assuming <second facts> (Outcome $t $o $v))
  (SecondOutcome action-id $t $o $v))`. Under each of the two top first
  singles, ask `(SecondOutcome $second $t $o $v)` openly. The legal
  vocabulary is truncated so seed pairs plus first-by-second branches
  stay within the pair cap. The total potential context bound is that
  cap plus at most the beam's outer contexts. If no second action is
  returned, use the best seed pair and record `open_answered=0`.
- **Exhaustive pair control.** Ask fixed pairs in the action enumeration
  order, stopping at the pair cap. At 256 this covers all default pairs;
  at smaller caps the order can exclude useful first actions.

All proposals are fixed pairs chosen before seeing the next observation.
The reference adapts its second action to that observation. `openloop`
is the best exact fixed pair, and `beam` uses exact values with the game's
bounded proposer, to distinguish that information gap from proposal and
chainer errors. `feedback_value_gap` records the reference's value minus
the best fixed pair's value; it need not translate into first-action
regret, since every planner replans after enacting its first action.

**Other systems.** `myopic` applies stage 4's exact joint one-period policy
(including its original repair proxy) at every visited state. `greedy`
combines positive single-action gains from that same policy. `prior`
uses independent Laplace-smoothed frequencies from the 30 history
periods, ignores current observations, evaluates fixed pairs under those
frequencies and the known dynamics, and caches its resulting policy.
ProbLog independently represents the shared next storm, its two route
blocks, and each site's repaired/unrepaired transition, queries the joint
next-state probabilities, then maximizes utilities in Python after that
state is observed. It receives no reference probabilities. A timed-out
compilation falls back to no action and is explicitly counted. NARS is
skipped for the stage 4 reason: no comparable joint expected-utility
planner. MM2 is attempted through `--backend mm2` and its existing adapter.

**Frozen-chainer gaps.** PeTTaChainer stays at `888379a2`; no edits to it
or its shared venv. Repros are in `benchmarks/supplynet_stage5/repro/`:

1. `nested_open.py`: a joint `(Assuming (And (A now) (B later)) (P later))`
   succeeds with strength 1. The equivalent nested `Assuming` returns no
   answer at 400 steps. A rule reading the inner `Assuming` to expose an
   open `(Choice $b)` also returns no answer. The game-pair query above
   avoids nesting; the open proposer measures this unsupported path rather
   than silently replacing it with game-generated second actions.
2. `mm2_assuming.py`: direct versus assumed zero-value outcome, run with
   MM2's Python 3.13 environment. The installed extension is CPython 3.13,
   so importing it from PeTTa's shared Python 3.10 environment fails before
   inference. The matching-interpreter behavior and missing API settings
   are recorded with the results.

An **encoding defect in the initial harness**, not a chainer gap, is kept
as `repro/branch_semantics.py`: the inherited alternative fuel rules with
CTV zero-valued negative branches gave 7.142544 for a pair whose exact
value is 16.285088. Each negative branch contradicted the other rule's
sufficient cause, halving certain production. Using positive STV
implications fixes it (16.285088 at both 50 and 400 steps).
`fuel_between_periods.py BUDGET` verifies this fuel path, freshly and after
proposal probes. Draft measurements using the contradictory encoding are
archived under `results/supplynet-stage5/draft-ctv-alternatives/` and are
excluded from the final tables. No reference values enter the PLN KB.

**Running and validation.** `benchmarks/supplynet_stage5/sweep.py` covers
budgets 50, 100, 200, 400, seeds 1–4, 30 rounds, and both proposers; it
also measures pair caps 16–256 over five rounds and exact/ProbLog sizes
with 0, 2 and 3 plants. All jobs use `capped --slots 1`, DISPLAY unset,
`ulimit -v 16000000`, and an outer timeout. ProbLog gets 15 seconds per
compilation in the sweep. Streams, summaries, exit status, command/source
fingerprints and repro outputs go to
`/nexus/Dev/OpenCog/bench/results/supplynet-stage5/`. Resume skips only
successful runs with the same arguments and source fingerprint.

```bash
unset DISPLAY
ulimit -v 16000000
/nexus/Dev/OpenCog/bench/capped --slots 4 --peak-mb 128 \
  python3 benchmarks/supplynet_stage5/sweep.py --jobs 4 --group all
python3 benchmarks/supplynet_stage5/report.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

The suite's stage 5 live tests use the existing gates
`SUPPLYNET_LIVE_PETTACHAINER=1` and `SUPPLYNET_LIVE_PROBLOG=1`; execute
them through capped using the corresponding installed environment.

## Stage 5 results

Seeds 1–4, 30 history periods and 30 rounds, default one-plant inspected
cell, capacity two, 16 joint options and 64 next states. PeTTaChainer's
package tree is `0458a175b52c38e7758845e6801ef43daa65ce07`, exactly the
tree at `888379a2`. The shared checkout advanced independently to
`4e6b059b` during the work, changing workflow docs/scripts only; no
PeTTaChainer package or shared-venv files were changed for this benchmark.
`environment.json` records the Python versions, component identity and
MM2 extension hash. The measured SupplyNet source SHA-256 is
`61aa820e753fb012982a5b44a866a7eb4b6457bce6d0b4969244ea13588e5efa`,
implemented by commit `8e91721`.

Mean over the four seeds (seconds include fresh-worker startup for PLN/MM2):

| policy | regret/round | gain over myopic at visited states | realized reward/round | decision s/round | candidate pair branches/round |
|---|---|---|---|---|---|
| exact feedback reference | 0 | 0 | 4.942 | 0.039 | all 256 pairs |
| myopic, stage 4 joint policy | 0 | 0 | 4.942 | 0.000065 | – |
| exact fixed pair | 0 | 0 | 4.942 | 0.033 | 256 |
| exact values, game proposer | 0 | 0 | 4.942 | 0.033 | 32 |
| greedy, stage 4 singles | 0.089 | −0.089 | 4.875 | 0.000088 | – |
| history prior | 4.517 | −4.493 | 1.300 | 0.136 | – |
| ProbLog exact feedback | 0 | 0 | 4.942 | 2.255 | exact next-state circuit |
| PLN game, budget 50 | 0 | 0 | 4.942 | 3.815 | 32 |
| PLN game, budget 100 | 0 | 0 | 4.942 | 3.778 | 32 |
| PLN game, budget 200 | 0 | 0 | 4.942 | 3.806 | 32 |
| PLN game, budget 400 | 0 | 0 | 4.942 | 3.777 | 32 |
| PLN open, budget 200, seed-pair fallback | 0.478 | −0.468 | 4.542 | 2.054 | 32 potential branches; 12 valued seeds |
| MM2, no-action fallback, **zero answers** | 6.095 | −6.070 | 0.008 | 0.100 | 12 attempted seed pairs |

The reference cost is its own enumeration, rather than the almost-free
lookup of a choice after scoring already calculated it. All four ProbLog
runs answered all 30 rounds without a timeout; their Q values agree with
the independently enumerated reference to floating-point precision.
Myopic and the exact feedback reference chose identical actions in all
120 main rounds, so **the measured lookahead gain is zero**. The exact
fixed pair and game proposer also choose those same first actions. This
is not evidence that a general two-period partially observed problem is
solved: the small inspected cell has very few opportunities to change a
myopic choice, as the structural probe below shows.

The PLN game proposer scores its queried fixed pairs correctly, with all
four tested budgets already above its accuracy threshold. More budget
does not improve this cell. The open proposer returns **no second-action
answers in any of the 120 rounds**. Its lower time pays for 12 scored
seed pairs plus two unsuccessful open roots; it is not a cheaper
successful planner. Its fallback's regret per seed is 0.357, 0.182,
0.599, 0.774. The game therefore proposes the working pairs on this
frozen chainer; the open-query alternative requires support for nested
`Assuming` goals/premises before its cost can be assessed as a planner.

At budget 200, game decisions average 3.794 s in the first ten rounds
and 3.816 s in the last ten; per round the worker setup is 0.863 s,
chainer inference 1.424 s and chainer overhead 1.459 s. These diagnostics
overlap during setup, so they should not be summed as exclusive phases.
Contexts and inference cost do not accumulate with history. Timings come
from one shared-machine sweep, with other admitted jobs running; small
differences between budgets are not performance conclusions.

MM2 uses its existing CPython 3.13 extension, SHA-256
`64592c18b7e149070958bb1c7f2b47aa5fec33b29f229874d41d1d851b511b0f`.
With its matching interpreter all four rollouts finish, but every
decision query is unanswered, so the table is the no-action fallback,
not supported two-period reasoning. The repro proves a direct outcome
fact, then fails to retrieve the same fact under `Assuming`. Missing
adapter settings are `set_complete_predicate` (Producing, Fuelled,
FuelArrived) and `set_positive_only_predicate` (the eight structural/action
predicates listed in `lookahead_worker.py`). No MM2 source or extension
was rebuilt. NARS remains skipped for the stage 4 reason.

The structural probe `repro/myopic_states.py` enumerates all 128 current
states of the default cell (the 64 stochastic bits configurations, with
each incoming-fuel value). Stage 4's proxy policy is suboptimal in only
two of them, by 0.5776: a healthy mine and power plant, degraded plant,
blocked main route, open alternate and no incoming fuel, under either
storm value. With two crews it stocks and reroutes; the two-period policy
stocks and repairs the plant. These are uniform structural counts, not
visitation probabilities. The main seeds' actual lookahead gain must be
reported separately; this small cell can be too forgiving of the myopic
proxy even though the timing tradeoff exists.

**Cost against candidate pairs** (seed 1, five rounds, budget 200; all
default fixed pairs total 256):

| pair cap | game pairs actually queried | game s/round | game regret | exhaustive-prefix s/round | exhaustive-prefix regret |
|---|---|---|---|---|---|
| 16 | 16 | 2.296 | 0.578 | 2.319 | 5.957 |
| 32 | 32 | 3.700 | 0 | 3.703 | 5.357 |
| 64 | 54 | 5.526 | 0 | 6.453 | 1.460 |
| 128 | 54 | 5.585 | 0 | 12.798 | 0.578 |
| 256 | 54 | 5.619 | 0 | 25.512 | 0 |

Exhaustive-prefix queries exactly the cap's number of distinct pairs.
Its time rises from 2.32 s for 16 pairs to 25.51 s for 256, approximately
linear in pair count after setup. Asking every pair therefore grows
quadratically in the number of joint actions. At 256 the full fixed-pair
planner gains no first-action quality here over the 32-pair game proposer,
while costing about seven times as much. The game heuristic stops at 54
pairs at this size even if given a larger cap. Reducing its cap to 16 does
hurt: 0.578 regret, showing the proposal-quality tradeoff.

The open experiment ranges from 16 potential pair branches/18 contexts
at cap 16 to 44 branches/46 contexts at caps 64–256. It remains around
2.0–2.1 s/round and 0.578 regret, with zero second-action answers at every
cap. Those are bounds on the attempted open experiment; they do not
measure the cost of a working open proposer. The lower exhaustive caps
are enumeration prefixes, not competitive proposal policies.

**Exact-system size cost** (seed 1, five rounds; ProbLog ddnnf, 15 s
per-compilation limit):

| plants | joint options | possible pairs | next states/first | reference s/round | ProbLog s/round | ProbLog answered / timeout rounds |
|---|---|---|---|---|---|---|
| 0 | 11 | 121 | 32 | 0.007 | 0.407 | 5 / 0 |
| 1, main four-seed mean | 16 | 256 | 64 | 0.039 | 2.255 | 120 / 0 |
| 2 | 22 | 484 | 128 | 0.143 | 15.101 | 0 / 5 |
| 3 | 29 | 841 | 256 | 0.474 | 15.123 | 0 / 5 |

The larger ProbLog rows are all timeouts and use the no-action fallback
(regret 9.919 and 11.694 respectively), not exact choices. The reference
stays exact at those sizes. No solver failure was hidden as a timeout or
an exact result. All 73 sweep processes finished, with the same SupplyNet
source fingerprint; solver-level timeouts are separate from process exit
status. Raw streams, summaries and `report.md` are under
`/nexus/Dev/OpenCog/bench/results/supplynet-stage5/`.

**Validation and limits.** `PYTHONPATH=src .venv/bin/python -m pytest -q
tests`: 158 passed, 29 skipped. The stage 5 live PLN test also passed,
checking actual pair values (including certain production fuelling the
second period) at budgets 50 and 400; the live ProbLog test passed against
the exact reference. Across every round of the four-seed PLN budget grid,
the largest queried-pair value error was below `4e-9`.

This is a planning/cost benchmark with full inspection, guaranteed plant
inputs and horizon-end fuel inventory. It does not establish lookahead
gains in stage 4's partially observed delayed network, or support for
nested assumptions. The reference adapts its second choice to the next
observation while fixed-pair agents replan only after committing their
first choice; a mean reference-versus-best-fixed-plan value gap remains
even where their first choices agree. No terminal repair bonus hides a
third period. Longer horizons, larger PLN cells and successful open
proposals remain unmeasured. No push or merge is part of this work.

## Learned rules

`supplynet run --learned-rules` (stages 1–3) hands every reasoner the rules'
structure with a weak prior in place of their rates, which each learns from
the labelled periods; the stage 3 production rules stay given. The encodings
for PLN, ProbLog and NARS, the exact posterior under counted rates (the
ceiling for any learner) and the results: `docs/supplynet_learned_rules.md`.


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
