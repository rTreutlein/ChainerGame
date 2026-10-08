import itertools
import os
import random
import unittest

from supplynet import metta
from supplynet.backends import PeTTaChainerBackend, PriorBackend, ReferenceBackend
from supplynet.game import GameConfig, run_game, stage_rates
from supplynet.world import Knowledge, Network, Observation, Rates, Route, generate_network, observe, sample_period


def brute_force_posterior(network, rates, before, periods, late, inspected):
    """Enumerates every storm and block assignment of every region and route
    over ``periods``; ``before`` is P(storm) of the period preceding them."""
    regions = list(network.regions)
    storm_vars = [(region, period) for period in periods for region in regions]
    block_vars = [(route, period) for period in periods for route in network.routes]
    totals = {}
    norm = 0.0
    for storm_values in itertools.product((True, False), repeat=len(storm_vars)):
        storm = dict(zip(storm_vars, storm_values))
        prior = 1.0
        for region in regions:
            previous = before[region]
            for period in periods:
                p = rates.storm_rate(network.regions[region], previous)
                prior *= p if storm[(region, period)] else 1 - p
                previous = float(storm[(region, period)])
        for block_values in itertools.product((True, False), repeat=len(block_vars)):
            weight = prior
            for (route, period), blocked in zip(block_vars, block_values):
                if inspected.get((route.name, period), blocked) != blocked:
                    weight = 0.0
                    break
                p = rates.block_given_storm[route.kind] if storm[(route.region, period)] else rates.block_without_storm
                weight *= p if blocked else 1 - p
                p_late = rates.late_given_blocked if blocked else rates.late_given_open
                for shipment in route.shipments:
                    if (shipment, period) in late:
                        weight *= p_late if late[(shipment, period)] else 1 - p_late
            if weight == 0.0:
                continue
            norm += weight
            for (region, period), value in storm.items():
                totals[("Storm", region, period)] = totals.get(("Storm", region, period), 0.0) + weight * value
            for (route, period), blocked in zip(block_vars, block_values):
                key = ("Blocked", route.name, period)
                totals[key] = totals.get(key, 0.0) + weight * blocked
    return {key: value / norm for key, value in totals.items()}


def linked_run(network, rates, rng, count, inspect_rate):
    """``count`` linked periods, each observed with the shipments arriving in
    it (departures from the first period on)."""
    periods, observations = [], []
    for index in range(count):
        period = sample_period(network, rates, rng, f"t{index}", periods[-1] if periods else None)
        periods.append(period)
        arrivals = {
            (shipment, periods[index - route.transit].name): periods[index - route.transit].late[shipment]
            for route in network.routes
            if index - route.transit >= 0
            for shipment in route.shipments
        }
        observations.append(observe(network, period, rng, inspect_rate, arrivals))
    return periods, observations


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
            period = sample_period(network, rates, rng, f"t{index}", None)
            observation = observe(network, period, rng, 0.3, {(s, period.name): v for s, v in period.late.items()})
            knowledge = Knowledge(network, rates)
            knowledge.observe(observation)
            inspected = {(route, period.name): v for route, v in observation.inspected.items()}
            before = {region: rates.stationary(kind) for region, kind in network.regions.items()}
            self.assert_matches_brute_force(
                knowledge, brute_force_posterior(network, rates, before, [period.name], observation.late, inspected)
            )

    def test_stage2_exact_posterior_matches_brute_force(self):
        rates = stage_rates(2)
        network = Network(
            {"north": "coastal"},
            (Route("r1", "north", "exposed", ("s1-1", "s1-2"), 1), Route("r2", "north", "sheltered", ("s2-1",), 3)),
        )
        for seed in range(3):
            periods, observations = linked_run(network, rates, random.Random(seed), 6, 0.3)
            late = {key: value for o in observations for key, value in o.late.items()}
            inspected = {(route, o.period): value for o in observations for route, value in o.inspected.items()}
            names = [period.name for period in periods]
            knowledge = Knowledge(network, rates)
            for observation in observations:
                knowledge.observe(observation)
            self.assert_matches_brute_force(
                knowledge, brute_force_posterior(network, rates, {"north": rates.stationary("coastal")}, names, late, inspected)
            )
            # Resolving the first period conditions the rest on its storm.
            knowledge.resolve(periods[0])
            before = {"north": float(periods[0].storms["north"])}
            self.assert_matches_brute_force(
                knowledge, brute_force_posterior(network, rates, before, names[1:], late, inspected)
            )

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
        for stage in (1, 2):
            config = GameConfig(seed=7, rounds=10, stage=stage)
            reference = run_game(config, ReferenceBackend())
            prior = run_game(config, PriorBackend())
            self.assertEqual(reference["posterior_error"], 0.0)
            self.assertLess(reference["brier"], prior["brier"])
            self.assertEqual(reference["coverage"], 1.0)
            self.assertEqual(set(reference["kinds"]), {"storm", "blocked"} | ({"previous_storm"} if stage == 2 else set()))

    def test_runs_are_deterministic(self):
        for stage in (1, 2):
            config = GameConfig(seed=11, rounds=5, stage=stage)
            self.assertEqual(run_game(config, ReferenceBackend()), run_game(config, ReferenceBackend()))

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

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_answers_every_belief(self):
        for stage in (1, 2):
            backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
            summary = run_game(GameConfig(seed=7, rounds=3, budget=50, stage=stage), backend)
            self.assertEqual(summary["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
