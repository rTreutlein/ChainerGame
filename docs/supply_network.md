# SupplyNet: a supply network with feedback

Status: design, October 2026. Nothing built.

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
