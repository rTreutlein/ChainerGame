# SupplyNet: a supply network with feedback

Status: stage 1 built (`src/supplynet`, see Stage 1 results), October 2026.

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

