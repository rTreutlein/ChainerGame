import itertools
import os
import random
import unittest

from supplynet import metta
from supplynet.backends import PeTTaChainerBackend, PriorBackend, ReferenceBackend
from supplynet.game import GameConfig, run_game
from supplynet.world import Observation, Rates, exact_posterior, generate_network, observe, sample_period


def brute_force_posterior(network, rates, observation):
    """Enumerates every storm and block assignment of the network."""
    regions = list(network.regions)
    routes = list(network.routes)
    totals = {}
    norm = 0.0
    for storms in itertools.product((True, False), repeat=len(regions)):
        storm = dict(zip(regions, storms))
        for blocks in itertools.product((True, False), repeat=len(routes)):
            weight = 1.0
            for region in regions:
                p = rates.storm[network.regions[region]]
                weight *= p if storm[region] else 1 - p
            for route, blocked in zip(routes, blocks):
                if route.name in observation.inspected and observation.inspected[route.name] != blocked:
                    weight = 0.0
                    break
                p = rates.block_given_storm[route.kind] if storm[route.region] else rates.block_without_storm
                weight *= p if blocked else 1 - p
                p_late = rates.late_given_blocked if blocked else rates.late_given_open
                for shipment in route.shipments:
                    weight *= p_late if observation.late[shipment] else 1 - p_late
            if weight == 0.0:
                continue
            norm += weight
            for region in regions:
                if storm[region]:
                    totals[("Storm", region)] = totals.get(("Storm", region), 0.0) + weight
            for route, blocked in zip(routes, blocks):
                if blocked:
                    totals[("Blocked", route.name)] = totals.get(("Blocked", route.name), 0.0) + weight
    return {key: value / norm for key, value in totals.items()}


class WorldTests(unittest.TestCase):
    def test_exact_posterior_matches_brute_force(self):
        rates = Rates()
        rng = random.Random(3)
        network = generate_network(rng, 2)
        for index in range(20):
            period = sample_period(network, rates, rng, f"t{index}")
            observation = observe(network, period, rng, 0.3)
            exact = exact_posterior(network, rates, observation)
            brute = brute_force_posterior(network, rates, observation)
            for key, value in exact.items():
                self.assertAlmostEqual(value, brute.get(key, 0.0), places=9, msg=key)

    def test_a_late_shipment_raises_its_region_and_an_inspected_clear_route_lowers_it(self):
        rates = Rates()
        network = generate_network(random.Random(1), 1)
        region = next(iter(network.regions))
        quiet = {s: False for route in network.routes for s in route.shipments}
        first = network.routes[0]
        late = dict(quiet, **{first.shipments[0]: True})
        base = exact_posterior(network, rates, Observation("t", quiet, {}))[("Storm", region)]
        raised = exact_posterior(network, rates, Observation("t", late, {}))[("Storm", region)]
        cleared = exact_posterior(network, rates, Observation("t", late, {first.name: False}))[("Storm", region)]
        self.assertGreater(raised, base)
        self.assertLess(cleared, raised)


class GameTests(unittest.TestCase):
    def test_reference_matches_the_posterior_and_beats_the_prior(self):
        config = GameConfig(seed=7, rounds=10)
        reference = run_game(config, ReferenceBackend())
        prior = run_game(config, PriorBackend())
        self.assertEqual(reference["posterior_error"], 0.0)
        self.assertLess(reference["brier"], prior["brier"])
        self.assertEqual(reference["coverage"], 1.0)

    def test_runs_are_deterministic(self):
        config = GameConfig(seed=11, rounds=5)
        self.assertEqual(run_game(config, PriorBackend()), run_game(config, PriorBackend()))

    def test_statements(self):
        rates = Rates()
        network = generate_network(random.Random(2), 1)
        route = network.routes[0]
        lines = metta.rules(network, rates)
        self.assertIn(
            f"(: block-{route.name} (Implication (Storm {route.region} $p) (Blocked {route.name} $p)) "
            f"(CTV (STV {rates.block_given_storm[route.kind]:.6g} 1) (STV 0.03 1)))",
            lines,
        )
        self.assertEqual(metta.query(("Storm", "north"), "t3"), "(: $prf (Storm north t3) $tv)")

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_answers_every_belief(self):
        backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
        summary = run_game(GameConfig(seed=7, rounds=3, budget=50), backend)
        self.assertEqual(summary["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
