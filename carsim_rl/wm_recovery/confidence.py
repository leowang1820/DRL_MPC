"""Causal ensemble/residual confidence statistics, independent of any Risk net."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GateSettings:
    window_steps: int = 10
    exceed_steps: int = 3
    recover_steps: int = 10
    quantile: float = .99

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (self.window_steps, self.exceed_steps, self.recover_steps)):
            raise ValueError("Gate windows and persistence must be positive integers")
        if not 0 < self.quantile < 1:
            raise ValueError("Calibration quantile must lie in (0,1)")


def rolling_mean(values, width):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(width - 1, len(values)):
        result[i] = values[i - width + 1:i + 1].mean()
    return result


def calibrate(episode_metrics, settings, residual_scale):
    """Use separate complete nominal validation episodes, never unknown tests."""
    variance, error = [], []
    for metrics in episode_metrics:
        if not np.all(metrics["valid"]):
            raise ValueError("Calibration contains invalid predictions; do not hide them with filtering")
        u = np.asarray(metrics["uncertainty"])
        e = np.linalg.norm(np.asarray(metrics["error"]) / residual_scale, axis=1)
        u, e = rolling_mean(u, settings.window_steps), rolling_mean(e, settings.window_steps)
        good = np.isfinite(u) & np.isfinite(e)
        variance.extend(u[good])
        error.extend(e[good])
    if len(variance) < max(20, settings.window_steps):
        raise ValueError("Too few nominal validation windows to calibrate thresholds")
    return {"settings": asdict(settings), "uncertainty_threshold": max(float(np.quantile(variance, settings.quantile)), 1e-12),
            "residual_threshold": max(float(np.quantile(error, settings.quantile)), 1e-12),
            "residual_scale": np.asarray(residual_scale).tolist(), "calibration_windows": len(variance),
            "meaning": "empirical nominal validation quantiles; not calibrated failure probabilities"}


class ConfidenceMonitor:
    """Observe completed transitions; warmup is unknown, never trusted by default.

    A residual is only available when the next observation arrives. Output from
    observe() therefore applies at t+1; it cannot be used to claim a t decision.
    """

    def __init__(self, calibration, mode="joint"):
        if mode not in ("joint", "ensemble_only", "residual_only"):
            raise ValueError("Unknown confidence mode")
        self.mode = mode
        self.settings = GateSettings(**calibration["settings"])
        self.u_limit = float(calibration["uncertainty_threshold"])
        self.e_limit = float(calibration["residual_threshold"])
        self.scale = np.asarray(calibration["residual_scale"], dtype=float)
        if (self.scale.shape != (7,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)
                or not np.isfinite([self.u_limit, self.e_limit]).all() or min(self.u_limit, self.e_limit) <= 0):
            raise ValueError("Invalid confidence calibration")
        self.u, self.e = deque(maxlen=self.settings.window_steps), deque(maxlen=self.settings.window_steps)
        self.bad_count = self.good_count = 0
        self.state = "warming_up"

    def observe(self, variance, predicted_mean, observed_next, *, prediction_valid=True):
        prediction = np.asarray(predicted_mean, dtype=float)
        observation = np.asarray(observed_next, dtype=float)
        valid = (prediction_valid and prediction.shape == (7,) and observation.shape == (7,)
                 and np.isfinite(prediction).all() and np.isfinite(observation).all()
                 and np.isfinite(variance) and variance >= 0)
        if not valid:
            self.u.clear()
            self.e.clear()
            self.state, self.good_count, self.bad_count = "untrusted", 0, 0
            return {"state": self.state, "trusted": False, "reason": "invalid_prediction_or_observation"}
        self.u.append(float(variance))
        self.e.append(float(np.linalg.norm((observation - prediction) / self.scale)))
        if len(self.u) < self.settings.window_steps:
            return {"state": self.state, "trusted": False, "reason": "insufficient_window"}
        u_mean, e_mean = float(np.mean(self.u)), float(np.mean(self.e))
        bad = ((self.mode != "residual_only" and u_mean > self.u_limit)
               or (self.mode != "ensemble_only" and e_mean > self.e_limit))
        self.bad_count = self.bad_count + 1 if bad else 0
        self.good_count = 0 if bad else self.good_count + 1
        if self.bad_count >= self.settings.exceed_steps:
            self.state = "untrusted"
        elif self.good_count >= self.settings.recover_steps:
            self.state = "trusted"
        return {"state": self.state, "trusted": self.state == "trusted", "window_uncertainty": u_mean,
                "window_residual": e_mean, "uncertainty_exceeded": u_mean > self.u_limit,
                "residual_exceeded": e_mean > self.e_limit, "reason": "window_statistics"}
