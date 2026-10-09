import itertools
import random
import unittest

from supplynet import scale, sweep
from supplynet.backends import PriorBackend, ReferenceBackend
from supplynet.scale import GameConfig, Knowledge, Observation, Rates, Size, generate_network, run_game, sample_period


def brute_force_posterior(network, rates, before, periods, seen):
    """Enumerates every hidden front and node of every period; ``before`` is
    P(front true) in the period preceding ``periods``, and an observed node
    weighs its parent by its likelihood."""
    hidden = network.hidden()
    variables = [(var, period) for period in periods for var in hidden]
    totals = dict.fromkeys(((*var, period) for var, period in variables), 0.0)
    norm = 0.0
    for values in itertools.product((False, True), repeat=len(variables)):
        state = dict(zip(variables, values))
        weight = 1.0
        for index, period in enumerate(periods):
            for front in network.fronts:
                previous = state[(front, periods[index - 1])] if index else before[front]
                p = rates.front_rate(float(previous))
                weight *= p if state[(front, period)] else 1 - p
            for var, node in network.nodes.items():
                value = state.get((var, period), seen.get((*var, period)))
                if value is None:
                    continue
                p = node.p(state[(node.parent, period)])
                weight *= p if value else 1 - p
        norm += weight
        for (var, period), value in state.items():
            totals[(*var, period)] += weight * value
    return {key: total / norm for key, total in totals.items()}


TINY = Size(zones=1, regions=2, routes=1, tiers=2, fanout=1, sensors=1)


def run(network, rates, rng, count):
    periods, observations = [], []
    for index in range(count):
        periods.append(sample_period(network, rates, rng, f"t{index}", periods[-1] if periods else None))
        seen = {}
        for var, node in network.nodes.items():
            if node.seen is not None and index - node.seen >= 0 and rng.random() < 0.8:
                departed = periods[index - node.seen]
                seen[(*var, departed.name)] = departed.values[var]
        observations.append(Observation(periods[-1].name, periods[-1].previous, seen))
    return periods, observations


class ScaleWorldTests(unittest.TestCase):
    def test_sizes_shape_the_network(self):
        rates = Rates()
        network = generate_network(random.Random(1), Size(zones=2, regions=3, routes=2, tiers=3, fanout=2, sensors=4), rates)
        count = {}
        for predicate, _ in [*network.fronts, *network.nodes]:
            count[predicate] = count.get(predicate, 0) + 1
        self.assertEqual(count, {"Front": 2, "Storm": 6, "Alarm": 24, "Blocked": 12, "Delayed": 12 * 6, "Late": 12 * 8})
        self.assertEqual(network.nodes[("Late", "s1-8")].parent, ("Delayed", "d1-6"))
        self.assertEqual(network.nodes[("Delayed", "d1-3")].parent, ("Delayed", "d1-1"))
        self.assertEqual(network.nodes[("Delayed", "d1-1")].parent, ("Blocked", "k1"))
        self.assertEqual(network.nodes[("Blocked", "k3")].parent, ("Storm", "g2"))
        self.assertEqual(network.nodes[("Storm", "g4")].parent, ("Front", "z2"))

    def test_exact_posterior_matches_brute_force(self):
        rates = Rates()
        # Two regions under one front, and a chain of depots below one block.
        shapes = (Size(1, 2, 1, 1, 2, 1), Size(1, 1, 1, 3, 1, 1))
        for seed, shape in itertools.product(range(2), shapes):
            rng = random.Random(seed)
            network = generate_network(rng, shape, rates)
            periods, observations = run(network, rates, rng, 3)
            seen = {key: value for o in observations for key, value in o.seen.items()}
            knowledge = Knowledge(network, rates)
            for observation in observations:
                knowledge.observe(observation)
            before = {front: rates.front_stationary for front in network.fronts}
            brute = brute_force_posterior(network, rates, before, [p.name for p in periods], seen)
            exact = knowledge.posterior()
            self.assertEqual(set(exact), set(brute))
            for key, value in exact.items():
                self.assertAlmostEqual(value, brute[key], places=9, msg=key)
            # Resolving the first period carries its front in as a point mass.
            knowledge.resolve(periods[0])
            before = {front: float(periods[0].values[front]) for front in network.fronts}
            brute = brute_force_posterior(network, rates, before, [p.name for p in periods[1:]], seen)
            for key, value in knowledge.posterior().items():
                self.assertAlmostEqual(value, brute[key], places=9, msg=key)

    def test_local_is_exact_for_a_lone_region_in_a_lone_period(self):
        rates = Rates()
        lone = Size(zones=1, regions=1, routes=2, tiers=2, fanout=2, sensors=2)
        network = generate_network(random.Random(4), lone, rates)
        periods, observations = run(network, rates, random.Random(5), 4)
        exact, local = Knowledge(network, rates), Knowledge(network, rates, local=True)
        for knowledge in (exact, local):
            knowledge.observe(observations[-1])
        for key, value in exact.posterior().items():
            self.assertAlmostEqual(local.posterior()[key], value, places=12, msg=key)
        # With more periods and regions, local ignores evidence the exact
        # posterior uses.
        network = generate_network(random.Random(4), TINY, rates)
        periods, observations = run(network, rates, random.Random(5), 4)
        exact, local = Knowledge(network, rates), Knowledge(network, rates, local=True)
        for observation in observations:
            exact.observe(observation)
            local.observe(observation)
        self.assertNotAlmostEqual(exact.posterior()[("Front", "z1", "t1")], local.posterior()[("Front", "z1", "t1")])

    def test_a_late_shipment_raises_its_front_in_every_window_period(self):
        rates = Rates()
        network = generate_network(random.Random(0), TINY, rates)
        shipment = next(var for var in network.nodes if var[0] == "Late")

        def front(late):
            knowledge = Knowledge(network, rates)
            for index in range(4):
                seen = {(*shipment, "t0"): late} if index == 3 else {}
                knowledge.observe(Observation(f"t{index}", f"t{index - 1}" if index else None, seen))
            return knowledge.posterior()

        quiet, late = front(False), front(True)
        for period in ("t0", "t1", "t2", "t3"):
            self.assertGreater(late[("Front", "z1", period)], quiet[("Front", "z1", period)])
            self.assertGreater(late[("Storm", "g2", period)], quiet[("Storm", "g2", period)])

    def test_statements(self):
        rates = Rates()
        network = generate_network(random.Random(1), TINY, rates)
        lines = scale.rules(network, rates)
        self.assertEqual(
            lines[0],
            "(: persist-z1 (Implication (And (NextPeriod $p $t) (Front z1 $p)) (Front z1 $t)) (CTV (STV 0.9 1) (STV 0.05 1)))",
        )
        self.assertIn("(: delayed-d1-1 (Implication (Blocked k1 $t) (Delayed d1-1 $t)) (CTV (STV 0.85 1) (STV 0.05 1)))", lines)
        self.assertIn("(: late-s1-1 (Implication (Delayed d1-1 $t) (Late s1-1 $t)) (CTV (STV 0.9 1) (STV 0.05 1)))", lines)
        self.assertIn("(: alarm-a1-1 (Implication (Storm g1 $t) (Alarm a1-1 $t)) (CTV (STV 0.6 1) (STV 0.25 1)))", lines)
        self.assertEqual(len(lines), 1 + len(network.nodes))
        observation = Observation("t3", "t2", {("Late", "s1-1", "t1"): True, ("Alarm", "a1-1", "t3"): False})
        self.assertEqual(
            scale.observation_facts(observation),
            [
                "(: next-t3 (NextPeriod t2 t3) (STV 1 1))",
                "(: late-s1-1-t1 (Late s1-1 t1) (STV 1 1))",
                "(: alarm-a1-1-t3 (Alarm a1-1 t3) (STV 0 1))",
            ],
        )
        period = sample_period(network, rates, random.Random(2), "t3", None)
        facts = scale.resolution_facts(period, observation)
        self.assertEqual(len(facts), len(network.hidden()))
        self.assertIn(f"(: front-z1-t3 (Front z1 t3) (STV {int(period.values[('Front', 'z1')])} 1))", facts)


class ScaleGameTests(unittest.TestCase):
    def config(self, **changes):
        return GameConfig(**{"seed": 3, "size": Size(1, 3, 2, 2, 2, 2), "history": 10, "rounds": 10, **changes})

    def test_reference_is_exact_and_beats_local_and_prior(self):
        reference = run_game(self.config(), ReferenceBackend(Knowledge))
        local = run_game(self.config(), ReferenceBackend(lambda network, rates: Knowledge(network, rates, local=True), "local"))
        prior = run_game(self.config(), PriorBackend())
        self.assertEqual(reference["posterior_error"], 0.0)
        self.assertEqual(reference["coverage"], 1.0)
        self.assertEqual(set(reference["kinds"]), {"front", "storm", "blocked", "past_front", "past_storm"})
        self.assertLess(reference["brier"], local["brier"])
        self.assertLess(local["posterior_error"], prior["posterior_error"])

    def test_queries_and_budget(self):
        summary = run_game(self.config(steps_per_query=0.5), PriorBackend())
        # Every front, storm and block now; from round 2 the oldest window
        # period's front and storm too.
        self.assertEqual(summary["queries_per_round"], (10 + 9 * 14) / 10)
        records = []
        run_game(self.config(steps_per_query=0.5), PriorBackend(), on_round=records.append)
        self.assertEqual([r["budget"] for r in records[:2]], [5, 7])
        run_game(self.config(budget=9), PriorBackend(), on_round=records.append)
        self.assertEqual(records[-1]["budget"], 9)

    def test_runs_are_deterministic(self):
        first, second = (
            {k: v for k, v in run_game(self.config(), ReferenceBackend(Knowledge)).items() if "seconds" not in k and k != "peak_rss_mb"}
            for _ in range(2)
        )
        self.assertEqual(first, second)


class SweepTests(unittest.TestCase):
    def test_report_tabulates_error_by_steps_per_query(self):
        def record(backend, size, spq, error, coverage=1.0):
            return {
                "backend": backend, "size": size, "seed": 1, "steps_per_query": spq, "posterior_error": error,
                "coverage": coverage, "brier": 0.2, "posterior_brier": 0.1, "ms_per_query": 10.0, "setup_seconds": 1.0,
                "peak_rss_mb": 100.0, "queries_per_round": 10.0, "setup_statements": 800, "statements": 1000,
                "hidden_per_period": 22, "observed_per_period": 30, **vars(scale.SIZES[size]),
            }

        records = [
            record("pettachainer", "s", 0.5, 0.2, 0.1),
            record("pettachainer", "s", 1.0, 0.1),
            record("local", "s", None, 0.15),
            record("prior", "s", None, 0.3),
            record("reference", "s", None, 0.0),
            {"backend": "pettachainer", "size": "m", "seed": 1, "steps_per_query": 1.0, "failed": "timeout"},
        ]
        text = sweep.report(records)
        self.assertIn("| 0.5 | 0.200 (0.10) |", text)
        self.assertIn("| 1 | 0.100 (1.00) |", text)
        self.assertIn("| local | 0.150 |", text)
        self.assertIn("| prior | 0.300 |", text)
        self.assertIn("| s | 0.5 | 0.200 | 0.10 | 0.200 | 0.100 | 10.0 | 1.0 | 100 |", text)
        self.assertIn("Failed: pettachainer m seed 1 spq 1.0 (timeout)", text)

if __name__ == "__main__":
    unittest.main()
