import itertools
import os
import random
import unittest

from supplynet import metta
from supplynet.backends import PeTTaChainerBackend, PriorBackend, ReferenceBackend
from supplynet.game import GameConfig, run_game, stage_rates
from supplynet.world import (
    MINE,
    POWER,
    Cell,
    Knowledge,
    Network,
    Observation,
    Rates,
    Route,
    generate_network,
    observe,
    production,
    sample_period,
)


def brute_force_posterior(network, rates, before, periods, observations):
    """Enumerates every storm, block and degradation of every region, route
    and site over ``periods``, with production computed from them period by
    period. ``before`` holds the period preceding them: P(on) of each storm
    ("Storm", region) and degradation ("Degraded", site), and, with a cell, the
    fuel route's block ("Blocked", route) and the mine's production
    ("Producing", mine)."""
    cell = network.cell
    sites = cell.sites if cell else {}
    late = {key: value for o in observations for key, value in o.late.items()}
    inspected = {(subject, o.period): value for o in observations for subject, value in (o.inspected | o.degraded).items()}
    reported = {(site, o.previous): value for o in observations for site, value in o.reported.items()}
    by_period = {o.period: o for o in observations}
    chained = [("Storm", region) for region in network.regions] + [("Degraded", site) for site in sites]
    variables = [(var, period) for period in periods for var in chained + [("Blocked", route.name) for route in network.routes]]
    routes = {route.name: route for route in network.routes}
    totals = {}
    norm = 0.0
    for values in itertools.product((True, False), repeat=len(variables)):
        state = dict(zip(variables, values))
        weight = 1.0
        previous = dict(before)
        for period in periods:
            for predicate, subject in chained:
                kind = network.regions[subject] if predicate == "Storm" else sites[subject]
                rate = rates.storm_rate if predicate == "Storm" else rates.degraded_rate
                p = rate(kind, float(previous[(predicate, subject)]))
                weight *= p if state[((predicate, subject), period)] else 1 - p
            for route in network.routes:
                blocked = state[(("Blocked", route.name), period)]
                if inspected.get((route.name, period), blocked) != blocked:
                    weight = 0.0
                p = rates.block_rate(route.kind, state[(("Storm", route.region), period)])
                weight *= p if blocked else 1 - p
                p_late = rates.late_given_blocked if blocked else rates.late_given_open
                for shipment in route.shipments:
                    if (shipment, period) in late:
                        weight *= p_late if late[(shipment, period)] else 1 - p_late
            if cell:
                degraded = {site: state[(("Degraded", site), period)] for site in sites}
                if any(inspected.get((site, period), value) != value for site, value in degraded.items()):
                    weight = 0.0
                fuel = cell.fuel_route.name
                fuel_blocked = previous[("Blocked", fuel)] if cell.timed else state[(("Blocked", fuel), period)]
                observation = by_period[period]
                produced = production(cell, degraded, observation.stocked, observation.inputs, fuel_blocked, previous[("Producing", MINE)])
                if (MINE, period) in reported and any(reported[(site, period)] != on for site, on in produced.items()):
                    weight = 0.0
                for site, on in produced.items():
                    state[(("Producing", site), period)] = on
            if weight == 0.0:
                break
            previous = {var: state[(var, period)] for var in chained} | (
                {("Blocked", cell.fuel_route.name): state[(("Blocked", cell.fuel_route.name), period)], ("Producing", MINE): produced[MINE]}
                if cell
                else {}
            )
        if weight == 0.0:
            continue
        norm += weight
        for ((predicate, subject), period), value in state.items():
            if (subject, period) in inspected or (predicate == "Producing" and period != periods[-1]):
                continue
            totals[(predicate, subject, period)] = totals.get((predicate, subject, period), 0.0) + weight * value
    return {key: value / norm for key, value in totals.items()}


def linked_run(network, rates, rng, count, inspect_rate):
    """``count`` linked periods, each observed with the shipments arriving in
    it (departures from the first period on) and the previous period's
    production."""
    periods, observations = [], []
    for index in range(count):
        period = sample_period(network, rates, rng, f"t{index}", periods)
        periods.append(period)
        arrivals = {
            (shipment, periods[index - route.transit].name): periods[index - route.transit].late[shipment]
            for route in network.routes
            if index - route.transit >= 0
            for shipment in route.shipments
        }
        observations.append(observe(network, period, rng, inspect_rate, arrivals, periods[-2].producing if index else {}))
    return periods, observations


def cell_network(cycle):
    """One region, one route carrying a plant's input, and the fuel route."""
    fuel = Route("rf", "north", "exposed", (), 1 if cycle == "timed" else 0)
    return Network({"north": "coastal"}, (Route("r1", "north", "sheltered", ("s1-1",), 1), fuel), Cell(fuel, {"plant-a": "s1-1"}))


class WorldTests(unittest.TestCase):
    def assert_matches_brute_force(self, knowledge, brute):
        exact = knowledge.posterior()
        self.assertTrue(exact)
        for key, value in exact.items():
            self.assertAlmostEqual(value, brute[key], places=9, msg=key)

    def test_stage1_exact_posterior_matches_brute_force(self):
        rates = Rates()
        rng = random.Random(3)
        network = generate_network(rng, 2)
        for index in range(20):
            period = sample_period(network, rates, rng, f"t{index}", [])
            observation = observe(network, period, rng, 0.3, {(s, period.name): v for s, v in period.late.items()})
            knowledge = Knowledge(network, rates)
            knowledge.observe(observation)
            before = {("Storm", region): rates.stationary(kind) for region, kind in network.regions.items()}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, [period.name], [observation]))

    def test_stage2_exact_posterior_matches_brute_force(self):
        rates = stage_rates(2)
        network = Network(
            {"north": "coastal"},
            (Route("r1", "north", "exposed", ("s1-1", "s1-2"), 1), Route("r2", "north", "sheltered", ("s2-1",), 3)),
        )
        for seed in range(3):
            periods, observations = linked_run(network, rates, random.Random(seed), 6, 0.3)
            names = [period.name for period in periods]
            knowledge = Knowledge(network, rates)
            for observation in observations:
                knowledge.observe(observation)
            before = {("Storm", "north"): rates.stationary("coastal")}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names, observations))
            # Resolving the first period conditions the rest on its storm.
            knowledge.resolve(periods[0])
            before = {("Storm", "north"): periods[0].storms["north"]}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names[1:], observations))

    def test_stage3_exact_posterior_matches_brute_force(self):
        rates = stage_rates(3)
        for cycle in ("timed", "untimed"):
            network = cell_network(cycle)
            for seed in range(2):
                periods, observations = linked_run(network, rates, random.Random(seed), 3, 0.3)
                names = [period.name for period in periods]
                knowledge = Knowledge(network, rates)
                for observation in observations:
                    knowledge.observe(observation)
                before = {("Storm", "north"): rates.stationary("coastal"), ("Blocked", "rf"): False, ("Producing", MINE): False}
                before |= {("Degraded", site): rates.degraded_stationary(kind) for site, kind in network.cell.sites.items()}
                self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names, observations))
                # Resolving the first period carries its labels, the fuel
                # route's block and the mine's production included.
                knowledge.resolve(periods[0])
                first = periods[0]
                before = {("Storm", "north"): first.storms["north"], ("Blocked", "rf"): first.blocked["rf"], ("Producing", MINE): first.producing[MINE]}
                before |= {("Degraded", site): value for site, value in first.degraded.items()}
                self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names[1:], observations))

    def test_production_is_the_least_fixed_point(self):
        cell = cell_network("untimed").cell
        healthy = dict.fromkeys(cell.sites, False)
        supplied = {"plant-a": True}
        # Without stock, everything running is a fixed point of the untimed
        # rules too; the least one has nothing running.
        self.assertEqual(production(cell, healthy, False, supplied, False, True), dict.fromkeys(cell.sites, False))
        self.assertEqual(production(cell, healthy, True, supplied, False, False), dict.fromkeys(cell.sites, True))
        self.assertEqual(production(cell, healthy | {POWER: True}, True, supplied, False, False), dict.fromkeys(cell.sites, False))
        self.assertEqual(
            production(cell, healthy | {MINE: True}, True, {"plant-a": False}, False, False),
            {MINE: False, POWER: True, "plant-a": False},
        )
        # Timed, last period's fuel keeps the loop running over an open route.
        timed = cell_network("timed").cell
        self.assertEqual(production(timed, healthy, False, supplied, False, True), dict.fromkeys(cell.sites, True))
        self.assertEqual(production(timed, healthy, False, supplied, True, True), dict.fromkeys(cell.sites, False))
        self.assertEqual(production(timed, healthy, False, supplied, False, False), dict.fromkeys(cell.sites, False))

    def test_without_stock_the_untimed_loop_does_not_run(self):
        rates = stage_rates(3)
        network = cell_network("untimed")
        periods, observations = linked_run(network, rates, random.Random(5), 30, 0.3)
        knowledge = Knowledge(network, rates)
        for period, observation in zip(periods, observations):
            knowledge.observe(observation)
            posterior = knowledge.posterior()
            if not period.stocked:
                self.assertFalse(any(period.producing.values()))
                self.assertTrue(all(posterior[("Producing", site, period.name)] == 0.0 for site in network.cell.sites))
            knowledge.resolve(period)
        self.assertTrue(any(period.producing[MINE] for period in periods))

    def test_a_late_shipment_raises_its_region_and_an_inspected_clear_route_lowers_it(self):
        rates = Rates()
        network = generate_network(random.Random(1), 1)
        region = next(iter(network.regions))
        quiet = {(s, "t"): False for route in network.routes for s in route.shipments}
        first = network.routes[0]
        late = quiet | {(first.shipments[0], "t"): True}

        def storm(late, inspected):
            knowledge = Knowledge(network, rates)
            knowledge.observe(Observation("t", None, late, inspected))
            return knowledge.posterior()[("Storm", region, "t")]

        base, raised, cleared = storm(quiet, {}), storm(late, {}), storm(late, {first.name: False})
        self.assertGreater(raised, base)
        self.assertLess(cleared, raised)

    def test_a_late_arrival_raises_its_departure_storm_and_the_storm_after_it(self):
        rates = stage_rates(2)
        network = Network({"north": "coastal"}, (Route("r1", "north", "exposed", ("s1-1",), 1),))

        def posterior(arrivals):
            knowledge = Knowledge(network, rates)
            knowledge.observe(Observation("t1", None, {}, {}))
            knowledge.observe(Observation("t2", "t1", arrivals, {}))
            return knowledge.posterior()

        quiet, late = posterior({("s1-1", "t1"): False}), posterior({("s1-1", "t1"): True})
        self.assertGreater(late[("Storm", "north", "t1")], quiet[("Storm", "north", "t1")])
        self.assertGreater(late[("Storm", "north", "t2")], quiet[("Storm", "north", "t2")])


class GameTests(unittest.TestCase):
    def test_reference_matches_the_posterior_and_beats_the_prior(self):
        kinds = {"storm", "blocked"}
        for stage, cycle, extra in (
            (1, "timed", set()),
            (2, "timed", {"previous_storm"}),
            (3, "timed", {"previous_storm", "degraded", "producing", "producing_unstocked"}),
            (3, "untimed", {"previous_storm", "degraded", "producing", "producing_unstocked"}),
        ):
            config = GameConfig(seed=7, rounds=10, stage=stage, cycle=cycle)
            reference = run_game(config, ReferenceBackend())
            prior = run_game(config, PriorBackend())
            self.assertEqual(reference["posterior_error"], 0.0)
            self.assertLess(reference["brier"], prior["brier"])
            self.assertEqual(reference["coverage"], 1.0)
            self.assertEqual(set(reference["kinds"]), kinds | extra)
            for kind in extra & {"degraded", "producing"}:
                self.assertLess(reference["kinds"][kind]["brier"], prior["kinds"][kind]["brier"])

    def test_runs_are_deterministic(self):
        for stage, cycle in ((1, "timed"), (2, "timed"), (3, "timed"), (3, "untimed")):
            config = GameConfig(seed=11, rounds=5, stage=stage, cycle=cycle)
            first, second = (
                {field: value for field, value in run_game(config, ReferenceBackend()).items() if "seconds" not in field}
                for _ in range(2)
            )
            self.assertEqual(first, second)
        # Stage 3 adds the cell after stage 2's network, which it keeps.
        stage2, stage3 = generate_network(random.Random(4), 3, 3), generate_network(random.Random(4), 3, 3, "timed")
        self.assertEqual(stage3.routes[:-1], stage2.routes)

    def test_statements(self):
        rates = stage_rates(2)
        network = generate_network(random.Random(2), 1, 3)
        route = network.routes[0]
        region, kind = next(iter(network.regions.items()))
        lines = metta.rules(network, rates)
        self.assertIn(
            f"(: block-{route.name} (Implication (Storm {route.region} $p) (Blocked {route.name} $p)) "
            f"(CTV (STV {rates.block_given_storm[route.kind]:.6g} 1) (STV 0.03 1)))",
            lines,
        )
        self.assertIn(
            f"(: persist-{region} (Implication (And (NextPeriod $p $t) (Storm {region} $p)) (Storm {region} $t)) "
            f"(CTV (STV 0.7 1) (STV {rates.storm[kind]:.6g} 1)))",
            lines,
        )
        self.assertFalse(any("persist" in line for line in metta.rules(network, Rates())))
        observation = Observation("t3", "t2", {("s1-1", "t1"): True}, {})
        self.assertEqual(
            metta.observation_facts(observation),
            ["(: next-t3 (NextPeriod t2 t3) (STV 1 1))", "(: late-s1-1-t1 (Late s1-1 t1) (STV 1 1))"],
        )
        self.assertEqual(metta.query(("Storm", "north", "t3")), "(: $prf (Storm north t3) $tv)")

    def test_cycle_statements(self):
        rates = stage_rates(3)
        timed, untimed = (metta.rules(cell_network(cycle), rates) for cycle in ("timed", "untimed"))
        self.assertEqual([line for line in timed if line not in untimed], [
            "(: fuel-arrives (Implication (And (NextPeriod $p $t) (Producing mine $p) (Not (Blocked rf $p))) (FuelArrived $t)) "
            "(CTV (STV 1 1) (STV 0 1)))"
        ])
        self.assertIn(
            "(: fuel-arrives (Implication (And (Producing mine $t) (Not (Blocked rf $t))) (FuelArrived $t)) (CTV (STV 1 1) (STV 0 1)))",
            untimed,
        )
        self.assertIn(
            "(: persist-degraded-mine (Implication (And (NextPeriod $p $t) (Degraded mine $p)) (Degraded mine $t)) "
            "(CTV (STV 0.75 1) (STV 0.08 1)))",
            timed,
        )
        self.assertIn(
            "(: runs-plant-a (Implication (And (Producing power-plant $t) (InputArrived plant-a $t) (Not (Degraded plant-a $t))) "
            "(Producing plant-a $t)) (CTV (STV 1 1) (STV 0 1)))",
            timed,
        )
        observation = Observation("t3", "t2", {}, {}, {"mine": True}, False, {"plant-a": True}, {"mine": False})
        self.assertEqual(
            metta.observation_facts(observation),
            [
                "(: next-t3 (NextPeriod t2 t3) (STV 1 1))",
                "(: degraded-mine-t3 (Degraded mine t3) (STV 1 1))",
                "(: stocked-t3 (StockedFuel t3) (STV 0 1))",
                "(: input-plant-a-t3 (InputArrived plant-a t3) (STV 1 1))",
                "(: producing-mine-t2 (Producing mine t2) (STV 0 1))",
            ],
        )

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_answers_every_belief(self):
        for stage in (1, 2, 3):
            backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
            summary = run_game(GameConfig(seed=7, rounds=3, budget=50, stage=stage), backend)
            self.assertEqual(summary["coverage"], 1.0)

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_does_not_run_the_untimed_loop_without_stock(self):
        # Seed 1's first rounds have no fuel stock, so the exact production is 0.
        backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
        summary = run_game(GameConfig(seed=1, rounds=3, budget=50, stage=3, cycle="untimed"), backend)
        unstocked = summary["kinds"]["producing_unstocked"]
        self.assertEqual(unstocked["posterior_brier"], 0.0)
        self.assertLessEqual(unstocked["posterior_error"], 0.5)


if __name__ == "__main__":
    unittest.main()
