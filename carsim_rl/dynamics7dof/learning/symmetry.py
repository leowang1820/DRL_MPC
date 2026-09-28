"""Left/right reflection for a symmetric vehicle on a flat, uniform road.

This is an explicit modelling assumption, NOT valid for split-mu, banking,
crosswind or unmodelled left/right vehicle asymmetry. Never negate wheel spin
or wheel torque: reflect wheel identity, not forward rotation direction.
"""

import numpy as np


X_PERM = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 13, 14, 15]
X_SIGN = np.array([1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1])
Y_PERM = [0, 1, 2, 4, 3, 6, 5]
Y_SIGN = np.array([1, -1, -1, 1, 1, 1, 1])
MODES = ('none', 'mirror')


def mirror_features(x):
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[-1] != 16:
        raise ValueError('Reflection needs 16 SI features')
    return x[..., X_PERM]*X_SIGN


def mirror_targets(y):
    y = np.asarray(y)
    if y.ndim < 1 or y.shape[-1] != 7:
        raise ValueError('Reflection needs 7 SI derivatives')
    return y[..., Y_PERM]*Y_SIGN


def fit_population(x, y, symmetry):
    """Mirror FIT rows only for statistics; do not double-count recorded samples."""
    if symmetry not in MODES:
        raise ValueError('Unknown residual symmetry mode')
    if symmetry == 'mirror':
        return np.concatenate([x, mirror_features(x)]), np.concatenate([y, mirror_targets(y)])
    return x, y
