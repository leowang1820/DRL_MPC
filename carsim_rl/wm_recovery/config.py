"""Experiment settings; no changes to the original paper's vehicle/tire model."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from dynamics7dof.model import TireFit, VehicleParameters


@dataclass(frozen=True)
class WorldModelConfig:
    prediction_dt_s: float = .01
    history_steps: int = 10
    physics_substep_s: float = .0005
    ensemble_size: int = 5
    hidden_width: int = 64
    rollout_steps: int = 5
    rollout_loss_weight: float = 1.
    seed: int = 42

    def __post_init__(self):
        for name in ("prediction_dt_s", "physics_substep_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("history_steps", "hidden_width", "rollout_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ensemble_size != 5:
            raise ValueError("The confirmed first-stage ensemble has five members")
        if not math.isfinite(self.rollout_loss_weight) or self.rollout_loss_weight < 0:
            raise ValueError("rollout_loss_weight must be finite and nonnegative")
        ratio = self.prediction_dt_s / self.physics_substep_s
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9) or ratio < 1:
            raise ValueError("prediction_dt_s must be an integer multiple of physics_substep_s")

    @property
    def input_width(self):
        return 7 + 6 + 13 * self.history_steps + 2

    @property
    def substeps(self):
        return round(self.prediction_dt_s / self.physics_substep_s)


def physics_contract():
    root = Path(__file__).resolve().parents[1] / "dynamics7dof"
    return {
        "vehicle": asdict(VehicleParameters()), "tire_fit": asdict(TireFit()), "nominal_mu": .8,
        "physics_sha256": hashlib.sha256((root / "model.py").read_bytes()).hexdigest(),
        "torch_physics_sha256": hashlib.sha256((root / "learning" / "torch_physics.py").read_bytes()).hexdigest(),
        "residual_definition": "discrete_next_state_increment",
        "history_definition": "past_state_and_actual_input_oldest_first_excluding_current",
        "load_memory": "explicit_nominal_force_derived_ax_ay_carried_between_steps",
        "road_friction_is_not_a_network_feature": True,
    }


def contract_digest():
    return hashlib.sha256(json.dumps(physics_contract(), sort_keys=True).encode()).hexdigest()
