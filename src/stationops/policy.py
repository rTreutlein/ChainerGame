from .config import Config
from .models import Incident
from .oracle import repair_increment


def allocate(incidents: list[Incident], beliefs: dict[str, float], config: Config) -> dict[str, str]:
    candidates = sorted(
        (
            (repair_increment(beliefs[x.id], config), x.id)
            for x in incidents
            if x.id in beliefs and repair_increment(beliefs[x.id], config) > 0
        ),
        key=lambda row: (-row[0], row[1]),
    )
    repaired = {case_id for _, case_id in candidates[: config.repair_slots]}
    return {x.id: ("repair" if x.id in repaired else "defer") for x in incidents}

