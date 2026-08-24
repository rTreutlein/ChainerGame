import unittest
from types import SimpleNamespace

from stationops.backends import MM2Backend, PeTTaChainerBackend
from stationops.config import Config
from stationops.shortfall import (
    event_atom,
    generate_shortfall_statements,
    oracle_shortfall_marginals,
)


def example_event():
    return {
        "shift": 1,
        "remaining_loss": 5,
        "candidates": {
            "pump": {"impact": 5, "prior": 0.02},
            "valve": {"impact": 3, "prior": 0.10},
            "motor": {"impact": 2, "prior": 0.05},
        },
    }


class ShortfallConditioningTests(unittest.TestCase):
    def test_oracle_preserves_alternative_fault_sets_before_projection(self):
        marginals = oracle_shortfall_marginals(example_event())
        self.assertAlmostEqual(marginals["pump"], 0.7772727272727272)
        self.assertAlmostEqual(marginals["valve"], 0.22272727272727275)
        self.assertAlmostEqual(marginals["motor"], 0.22272727272727275)

    def test_source_uses_compact_posterior_dp_and_immutable_events(self):
        event = example_event()
        source = generate_shortfall_statements([event])
        self.assertIn("Compute WeightedSubsetPosteriorDP", source)
        self.assertIn("Compute WeightedSubsetPosteriorMarginal", source)
        self.assertNotIn("Compute WeightedSubsetSumInverse", source)
        self.assertIn("(WeightedCandidate pump 5 0.02)", source)
        self.assertIn(f"(ObservedProductionLoss {event_atom(event)} 5)", source)

        revised = example_event()
        revised["candidates"].pop("motor")
        self.assertNotEqual(event_atom(event), event_atom(revised))

    def test_pettachainer_projects_probabilities_from_reasoner_results(self):
        class Handler:
            def __init__(self):
                self.atoms = []
                self.queries = []

            def add_atoms_no_check(self, atoms):
                self.atoms.extend(atoms)

            def query(self, query, steps, timeout_sec):
                self.queries.append((query, steps))
                unit = next(name for name in ("motor", "pump", "valve") if name in query)
                values = {"motor": 0.2, "pump": 0.8, "valve": 0.2}
                event = event_atom(example_event())
                return [
                    f"(: proof (ShortfallMarginal {event} {unit} {values[unit]}) "
                    "(STV 1 1))"
                ]

        handler = Handler()
        backend = PeTTaChainerBackend(
            Config(), module=SimpleNamespace(PeTTaChainer=lambda: handler)
        )
        marginals, counters = backend.condition_shortfalls([example_event()], 20)

        self.assertEqual(marginals, {1: {"motor": 0.2, "pump": 0.8, "valve": 0.2}})
        self.assertTrue(counters["shortfall_supported"])
        self.assertTrue(
            any("WeightedSubsetPosteriorDP" in atom for atom in handler.atoms)
        )

        repeated, repeated_counters = backend.condition_shortfalls(
            [example_event()], 20
        )
        self.assertEqual(repeated, marginals)
        self.assertEqual(len(handler.queries), 3)
        self.assertEqual(repeated_counters["shortfall_cache_hits"], 3)
        self.assertEqual(repeated_counters["shortfall_engine_queries"], 0)
        self.assertEqual(repeated_counters["shortfall_engine_steps"], 0)

        revised = example_event()
        revised["candidates"].pop("motor")
        backend.condition_shortfalls([revised], 20)
        self.assertEqual(len(handler.queries), 5)

    def test_mm2_caches_answers_but_retries_unfinished_queries(self):
        event_name = event_atom(example_event())

        class Engine:
            instances = []

            def __init__(self):
                self.calls = []
                self.__class__.instances.append(self)

            def add_many(self, *args):
                pass

            def query_many(self, kb, queries, budget):
                self.calls.append((list(queries), budget))
                results = []
                for tag, _ in queries:
                    unit = tag.split("|", 1)[1]
                    if budget >= 30 or unit == "pump":
                        values = {"motor": 0.2, "pump": 0.8, "valve": 0.2}
                        proof = {
                            "term": f"(ShortfallMarginal {event_name} {unit} "
                            f"{values[unit]})"
                        }
                        results.append((tag, [proof]))
                    else:
                        results.append((tag, []))
                return results

        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))
        first, first_counters = backend.condition_shortfalls([example_event()], 20)
        self.assertEqual(first, {1: {"pump": 0.8}})
        self.assertEqual(first_counters["shortfall_engine_queries"], 3)

        repeated, repeated_counters = backend.condition_shortfalls(
            [example_event()], 20
        )
        self.assertEqual(repeated, first)
        self.assertEqual(repeated_counters["shortfall_cache_hits"], 1)
        self.assertEqual(repeated_counters["shortfall_engine_queries"], 2)

        expanded, expanded_counters = backend.condition_shortfalls(
            [example_event()], 30
        )
        self.assertEqual(
            expanded,
            {1: {"motor": 0.2, "pump": 0.8, "valve": 0.2}},
        )
        self.assertEqual(expanded_counters["shortfall_cache_hits"], 1)
        self.assertEqual(expanded_counters["shortfall_engine_queries"], 2)
        self.assertEqual(
            [len(call[0]) for call in backend._engine.calls],
            [3, 2, 2],
        )

    def test_mm2_reports_missing_compute_operators_without_mutating_main_engine(self):
        class Engine:
            instances = []

            def __init__(self):
                self.added = []
                self.__class__.instances.append(self)

            def add_many(self, kb, statements):
                if kb == "stationops-shortfall-probe":
                    raise RuntimeError("unsupported Compute WeightedSubsetPosteriorDP")
                self.added.append((kb, statements))

        backend = MM2Backend(Config(), module=SimpleNamespace(Engine=Engine))
        marginals, counters = backend.condition_shortfalls([example_event()], 20)

        self.assertEqual(marginals, {})
        self.assertFalse(counters["shortfall_supported"])
        self.assertIn(
            "WeightedSubsetPosteriorDP",
            counters["shortfall_unsupported_reason"],
        )
        self.assertEqual(Engine.instances[0].added, [])


if __name__ == "__main__":
    unittest.main()
