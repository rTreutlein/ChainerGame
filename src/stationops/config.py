from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 7
    history_size: int = 1000
    incidents: int = 100
    repair_slots: int = 10
    old_prior: float = 0.05
    new_prior: float = 0.005
    sensitivity: float = 0.90
    false_positive_rate: float = 0.12
    avoided_loss: float = 100.0
    unnecessary_penalty: float = 10.0
    action_cost: float = 5.0
    irrelevant_statements: int = 0

    @property
    def repair_threshold(self) -> float:
        return (self.unnecessary_penalty + self.action_cost) / (
            self.avoided_loss + self.unnecessary_penalty
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"repair_threshold": self.repair_threshold}
