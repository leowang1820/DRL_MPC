"""Augmented derivative/integrator: original physics + bounded gated residual."""

from dataclasses import dataclass
import math

import numpy as np

from ..model import Reference7DOF
from .data import feature_vector


@dataclass
class AugmentedPrediction:
    state: np.ndarray
    load_memory: object
    flags: tuple
    uncertainty_mean: float
    fallback_fraction: float
    clipped_fraction: float


class Learned7DOF:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.physics = Reference7DOF()

    def evaluate(self, state, controls, *, mu, load_memory):
        base = self.physics.evaluate(state, controls, mu=mu, load_memory=load_memory)
        prediction = self.ensemble.predict(feature_vector(state, controls, mu, load_memory))
        correction = prediction['correction'][0].copy()
        if base.flags:
            correction[:] = 0
        return base, base.derivative+correction, prediction

    def predict(self, state, controls, *, mu, load_memory, horizon_s, substeps=2):
        if not np.isfinite(horizon_s) or horizon_s <= 0 or not isinstance(substeps, int) or substeps < 1:
            raise ValueError('Positive horizon and integer substeps required')
        x = np.asarray(state, dtype=float).reshape(7).copy()
        dt = horizon_s/substeps
        memory, flags, uncertainty, fallback, clipping = load_memory, set(), [], [], []
        for _ in range(substeps):
            base, derivative, result = self.evaluate(x, controls, mu=mu, load_memory=memory)
            flags.update(base.flags)
            x += dt*derivative
            if not np.all(np.isfinite(x)):
                raise FloatingPointError('Nonfinite augmented integration')
            if x[0] < .1:
                flags.add('legacy_vx_clamp_active')
            if np.any(x[3:] < -1):
                flags.add('legacy_wheel_speed_clamp_active')
            x[0], x[3:] = max(x[0], .1), np.maximum(x[3:], -1)
            # Force-derived nominal memory: never substitute learned acceleration
            # as a tire force without a separate physical force decomposition.
            memory = base.next_load_memory
            uncertainty.append(float(result['uncertainty'][0]))
            fallback.append(bool(result['rejected'][0]) or bool(base.flags))
            clipping.append(bool(result['clipped'][0]))
        return AugmentedPrediction(x, memory, tuple(sorted(flags)), float(np.mean(uncertainty)),
                                   float(np.mean(fallback)), float(np.mean(clipping)))
