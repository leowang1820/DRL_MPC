"""Forward-driving, flat-road reference port of Unified_7DOF_Predictor.

Preserves the user's fitted tire curves, combined-slip weights, empirical
lateral damping, lagged acceleration/load update, and integration clamps.
These are deliberate *reference* semantics, not physical validation. In
particular this is f(x, u, mu, load_memory), NOT a closed f(x, u).

Frenet propagation is outside this physical seven-state module. No CarSim
DLL, SAC network, controller, or learned correction is loaded here.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


WHEEL_ORDER = ("FL", "FR", "RL", "RR")
STATE_NAMES = (
    "vx_m_s", "vy_m_s", "yaw_rate_rad_s",
    "omega_fl_rad_s", "omega_fr_rad_s", "omega_rl_rad_s", "omega_rr_rad_s",
)
EPS = np.finfo(np.float64).eps


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    # Accept ordinary vectors and MATLAB column/row vectors, not matrices.
    if array.shape not in ((size,), (size, 1), (1, size)):
        raise ValueError(f"{name} must contain {size} elements as a vector")
    array = array.reshape(size).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _mu_vector(mu):
    array = np.asarray(mu, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(4, float(array))
    else:
        array = _vector(array, 4, "mu")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("mu must be finite and nonnegative; specify the scenario value")
    return array


@dataclass(frozen=True)
class VehicleParameters:
    """SI units; explicit user values, not inferred from the active CarSim Run."""

    mass_kg: float = 2026.0
    lf_m: float = 1.265
    lr_m: float = 3.16 - 1.265
    track_front_m: float = 1.605
    track_rear_m: float = 1.605
    cg_height_m: float = 0.59
    yaw_inertia_kg_m2: float = 4095.0
    wheel_inertia_kg_m2: float = 2.0
    wheel_radius_m: float = 0.353
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def wheelbase_m(self):
        return self.lf_m + self.lr_m


@dataclass(frozen=True)
class TireFit:
    """Original coefficients: Fz in kN, alpha in deg, slip as a ratio.

    camber_fit is the user's fitted constant. It is not a live camber sensor
    input and is intentionally NOT converted from degrees to radians.
    """

    longitudinal: Tuple[float, ...] = (
        1.3994, -0.0095, 1.0541, 1.0394, 22.0678,
        0.012, -0.0006, 0.0123, 0.6401,
    )
    lateral: Tuple[float, ...] = (
        2.2194, -0.0096, 1.0559, -2.9296,
        39.0998, 1.4627, 0.0002, 1.0081,
    )
    camber_fit: float = 1.9554

    def __post_init__(self):
        _vector(self.longitudinal, 9, "longitudinal coefficients")
        _vector(self.lateral, 8, "lateral coefficients")
        if not math.isfinite(self.camber_fit):
            raise ValueError("camber_fit must be finite")
        if self.lateral[4] == 0.0:
            raise ValueError("lateral b4 must be nonzero")


@dataclass(frozen=True)
class LoadMemory:
    """Previous force-derived ax/ay, matching MATLAB ax_cur/ay_cur.

    These are not dvx/dt and dvy/dt. The original empirical lateral damping
    is not included in ay. Keep that distinction in later data alignment.
    """

    ax_m_s2: float
    ay_m_s2: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.ax_m_s2, self.ay_m_s2)):
            raise ValueError("load memory must be finite")


@dataclass(frozen=True)
class Controls:
    """Actual front-wheel angles [rad], signed NET wheel torques [N*m]."""

    front_steer_rad: np.ndarray
    wheel_net_torque_nm: np.ndarray

    def __post_init__(self):
        angles = _vector(self.front_steer_rad, 2, "front_steer_rad")
        torques = _vector(self.wheel_net_torque_nm, 4, "wheel_net_torque_nm")
        if np.any(np.abs(angles) >= math.pi / 2):
            raise ValueError("front-wheel steer must be inside (-pi/2, pi/2) radians")
        angles.setflags(write=False)
        torques.setflags(write=False)
        object.__setattr__(self, "front_steer_rad", angles)
        object.__setattr__(self, "wheel_net_torque_nm", torques)


@dataclass(frozen=True)
class ForceEvaluation:
    derivative: np.ndarray
    fx_tire_n: np.ndarray
    fy_tire_n: np.ndarray
    fz_n: np.ndarray
    raw_fz_n: np.ndarray
    kappa: np.ndarray
    alpha_rad: np.ndarray
    sum_fx_body_n: float
    sum_fy_body_n: float
    yaw_moment_nm: float
    next_load_memory: LoadMemory
    flags: Tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    state: np.ndarray
    load_memory: LoadMemory
    last_evaluation: ForceEvaluation
    last_force_state: np.ndarray
    horizon_s: float
    substep_s: float
    force_sample_time_s: float
    flags: Tuple[str, ...]


class Reference7DOF:
    """Unvalidated reference model: preserve first, compare before correcting."""

    def __init__(self, vehicle=None, tire=None):
        self.vehicle = vehicle if vehicle is not None else VehicleParameters()
        self.tire = tire if tire is not None else TireFit()

    def tire_forces(self, kappa, alpha_rad, fz_n, *, mu):
        """Return Fx/Fy in tire axes, retaining MATLAB combined-slip behavior."""
        slip = _vector(kappa, 4, "kappa")
        alpha = _vector(alpha_rad, 4, "alpha_rad")
        fz = _vector(fz_n, 4, "fz_n")
        if np.any(fz < 0.0):
            raise ValueError("fz_n must be nonnegative")
        friction = _mu_vector(mu)
        z = fz / 1000.0
        a = self.tire.longitudinal
        b = self.tire.lateral
        cx = a[0]
        dx = friction * (a[1] * z**2 + a[2] * z)
        bx = ((a[3] * z**2 + a[4] * z) * np.exp(-a[5] * z)) / (cx * dx + EPS)
        ex = a[6] * z**2 + a[7] * z + a[8]
        cy = b[0]
        dy = friction * (b[1] * z**2 + b[2] * z)
        by = (b[3] * np.sin(2 * np.arctan(z / b[4]))
              * (1 - b[5] * self.tire.camber_fit)) / (cy * dy + EPS)
        ey = b[6] * z + b[7]
        if np.any(dx < 0.0) or np.any(dy < 0.0):
            raise ValueError("Tire fit produces negative peak force; load is outside this fit")
        bxk = bx * slip
        bya = by * np.degrees(alpha)
        fx0 = 1000 * dx * np.sin(cx * np.arctan(bxk - ex * (bxk - np.arctan(bxk))))
        fy0 = -1000 * dy * np.sin(cy * np.arctan(bya - ey * (bya - np.arctan(bya))))
        denominator = np.maximum(np.abs(1 + slip), 1e-5)
        sx = np.abs(slip / denominator)
        sy = np.abs(np.tan(alpha) / denominator)
        norm = np.maximum(np.hypot(sx, sy), EPS)
        fx = sx / norm * fx0
        fy = sy / norm * fy0
        scale = np.minimum(1.0, friction * fz / (np.hypot(fx, fy) + EPS))
        fx, fy = fx * scale, fy * scale
        if not np.all(np.isfinite([fx, fy])):
            raise FloatingPointError("Non-finite tire force")
        return fx, fy

    def evaluate(self, state, controls, *, mu, load_memory):
        """Instantaneous derivative at state, conditioned on load memory."""
        x = _vector(state, 7, "state")
        if not isinstance(controls, Controls) or not isinstance(load_memory, LoadMemory):
            raise TypeError("Controls and explicit LoadMemory are required")
        friction = _mu_vector(mu)
        v = self.vehicle
        vx, vy, yaw_rate = x[:3]
        omega = x[3:]
        wheel_x = np.array([v.lf_m, v.lf_m, -v.lr_m, -v.lr_m])
        wheel_y = np.array([v.track_front_m/2, -v.track_front_m/2,
                            v.track_rear_m/2, -v.track_rear_m/2])
        delta = np.r_[controls.front_steer_rad, 0.0, 0.0]
        cos_delta, sin_delta = np.cos(delta), np.sin(delta)
        front_load = (v.mass_kg * v.gravity_m_s2 * v.lr_m
                      - v.mass_kg * load_memory.ax_m_s2 * v.cg_height_m) / v.wheelbase_m
        rear_load = (v.mass_kg * v.gravity_m_s2 * v.lf_m
                     + v.mass_kg * load_memory.ax_m_s2 * v.cg_height_m) / v.wheelbase_m
        front_transfer = (v.mass_kg * load_memory.ay_m_s2 * v.cg_height_m * v.lr_m
                          / (v.wheelbase_m * v.track_front_m))
        rear_transfer = (v.mass_kg * load_memory.ay_m_s2 * v.cg_height_m * v.lf_m
                         / (v.wheelbase_m * v.track_rear_m))
        raw_fz = np.array([front_load/2-front_transfer, front_load/2+front_transfer,
                           rear_load/2-rear_transfer, rear_load/2+rear_transfer])
        fz = np.maximum(raw_fz, 100.0)  # Legacy floor; flags expose nonphysical loads.
        vx_body = vx - yaw_rate * wheel_y
        vy_body = vy + yaw_rate * wheel_x
        vx_wheel = vx_body * cos_delta + vy_body * sin_delta
        slip = (omega * v.wheel_radius_m - vx_wheel) / np.maximum(np.abs(vx_wheel), 0.1)
        alpha = np.arctan2(vy_body, np.maximum(vx_body, 0.1)) - delta
        fx, fy = self.tire_forces(slip, alpha, fz, mu=friction)
        fx_body = fx * cos_delta - fy * sin_delta
        fy_body = fx * sin_delta + fy * cos_delta
        sum_fx, sum_fy = float(fx_body.sum()), float(fy_body.sum())
        moment = float(np.sum(wheel_x * fy_body - wheel_y * fx_body))
        tire_ax, tire_ay = sum_fx / v.mass_kg, sum_fy / v.mass_kg
        damping = 100.0 * np.exp(-(vy / 1.0)**2)
        derivative = np.r_[
            tire_ax + yaw_rate * vy,
            tire_ay - damping / v.mass_kg * vy - yaw_rate * vx,
            moment / v.yaw_inertia_kg_m2,
            (controls.wheel_net_torque_nm - fx * v.wheel_radius_m) / v.wheel_inertia_kg_m2,
        ]
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("Non-finite state derivative")
        flags = []
        if np.any(raw_fz < 100.0):
            flags.append("legacy_load_floor_active")
        if np.any(raw_fz <= 0.0):
            flags.append("contact_loss_outside_model_domain")
        if np.any(vx_body < 0.1) or np.any(np.abs(vx_wheel) < 0.1):
            flags.append("low_speed_regularization_active")
        if vx <= 0.0 or np.any(vx_wheel <= 0.0):
            flags.append("forward_driving_domain_exceeded")
        return ForceEvaluation(
            derivative, fx, fy, fz, raw_fz, slip, alpha, sum_fx, sum_fy, moment,
            LoadMemory(tire_ax, tire_ay), tuple(flags),
        )

    def predict(self, state, controls, *, mu, load_memory, horizon_s, substeps=100):
        """Euler reference step. Inputs and mu are held constant across the horizon.

        Force outputs are intentionally timestamped at horizon-dt (matching
        the MATLAB loop); they must not be paired with the terminal state as
        if they had been recomputed there. No silent acceleration reset.
        """
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("horizon_s must be finite and positive")
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        x = _vector(state, 7, "state")
        friction = _mu_vector(mu)
        dt = horizon_s / substeps
        memory = load_memory
        flags = set()
        for _ in range(substeps):
            last_force_state = x.copy()
            evaluation = self.evaluate(x, controls, mu=friction, load_memory=memory)
            flags.update(evaluation.flags)
            x = x + dt * evaluation.derivative
            if not np.all(np.isfinite(x)):
                raise FloatingPointError("Non-finite integrated state")
            if x[0] < 0.1:
                flags.add("legacy_vx_clamp_active")
            if np.any(x[3:] < -1.0):
                flags.add("legacy_wheel_speed_clamp_active")
            x[0] = max(x[0], 0.1)
            x[3:] = np.maximum(x[3:], -1.0)
            memory = evaluation.next_load_memory
        return Prediction(x, memory, evaluation, last_force_state, float(horizon_s),
                          float(dt), float((substeps - 1) * dt), tuple(sorted(flags)))
