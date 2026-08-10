"""Repro: a partial materializing batch corrupts later warm query results.

Run from the ChainerGame checkout with PeTTaChainer's virtual environment:

    PYTHONPATH=src:/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer \
      /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
      examples/pettachainer_query_materialization_repro.py

The first call materializes all ten proof trees.  After an intentionally
partial second call, the third call should still recover all ten materialized
roots.  At PeTTaChainer 41e8847 it instead loses roots and can fail with:

    Type mismatch: got 'no-tv' but expected 'STVType'
"""

from stationops.backends import PeTTaChainerBackend
from stationops.game import GameConfig, GameSession, sensor_knowledge


config = GameConfig(shifts=1, modules=10, sensor_knowledge="mixed")
session = GameSession(config)
backend = PeTTaChainerBackend(
    session.logic_config,
    python_path="/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer",
    sensor_knowledge=sensor_knowledge(config),
)
backend._handler = backend._new_handler()
backend._reconcile(session.logic_statements())

goals = []
for incident in session.incidents:
    mode = backend.sensor_knowledge[incident.equipment_type]
    if mode == "full":
        proposition = (
            f"(LocalProblem {incident.problem_context} {incident.module_id} "
            f"{incident.cohort} {incident.equipment_type} {incident.id})"
        )
    else:
        observation = "PressureAlarm" if incident.alarm else "PressureNormal"
        proposition = (
            f"(Inheritance ({observation} {incident.cohort} "
            f"{incident.equipment_type}) "
            f"(SealLeak {incident.cohort} {incident.equipment_type}))"
        )
    goals.append(f"(: $proof {proposition} $tv)")


def materialize(budget):
    rows = backend._handler.query_many_materialization(goals, steps=budget)
    counts = [len(row) for row in rows]
    print(f"budget={budget}: {counts}", flush=True)
    return counts


assert materialize(2000) == [1] * 10
materialize(10)
assert materialize(100) == [1] * 10
