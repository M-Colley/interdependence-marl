"""Training algorithms and curriculum design for Interdependence-MARL."""

from interdependence_marl.training.mappo import TrainConfig, train_interdependent_mappo
from interdependence_marl.training.upd import AbilityUPD

__all__ = ["AbilityUPD", "TrainConfig", "train_interdependent_mappo"]
