"""Confirmed command semantics; SI internally, native degrees at the VS boundary."""

from dataclasses import dataclass
import math

import numpy as np


WHEELS = ("fl", "fr", "rl", "rr")
COMMAND_NAMES = tuple(f"external_torque_{w}_nm" for w in WHEELS) + ("front_steer_rad",)
DIRECT_IMPORT_NAMES = (
    "IMP_MYUSM_L1", "IMP_MYUSM_R1", "IMP_MYUSM_L2", "IMP_MYUSM_R2",
    "IMP_STEER_L1", "IMP_STEER_R1",
)
APPLIED_INPUT_NAMES = (
    "actual_front_steer_fl_rad", "actual_front_steer_fr_rad",
    *(f"actual_external_torque_{w}_nm" for w in WHEELS),
)


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class Command:
    """Four signed external wheel moments, excluding friction brakes, plus one angle.

    No motor dynamics, saturation, friction braking, steering ratio, or Ackermann
    geometry is inferred. Hardware limits must be specified by a future runner.
    """

    wheel_torque_nm: tuple[float, float, float, float]
    front_steer_rad: float

    def __post_init__(self):
        torque = finite_array(self.wheel_torque_nm, (4,), "wheel_torque_nm")
        steer = float(self.front_steer_rad)
        if not math.isfinite(steer) or abs(steer) >= math.pi / 2:
            raise ValueError("front_steer_rad must lie strictly inside (-pi/2, pi/2)")
        object.__setattr__(self, "wheel_torque_nm", tuple(map(float, torque)))
        object.__setattr__(self, "front_steer_rad", steer)

    @classmethod
    def from_vector(cls, values):
        values = finite_array(values, (5,), "command")
        return cls(tuple(values[:4]), float(values[4]))

    def as_vector(self):
        return np.array((*self.wheel_torque_nm, self.front_steer_rad))

    def to_imports(self, names):
        """Map to the six direct-wheel channels, in the verified import order.

        This function verifies channel identity, not the physical steering model
        or its ride/roll/compliance contributions. Those require a CarSim test.
        """
        names = tuple(name.upper() for name in names)
        if len(names) != 6 or set(names) != set(DIRECT_IMPORT_NAMES):
            raise ValueError("Expected four IMP_MYUSM and two direct front-wheel steer imports")
        angle_deg = math.degrees(self.front_steer_rad)
        values = dict(zip(DIRECT_IMPORT_NAMES, (*self.wheel_torque_nm, angle_deg, angle_deg)))
        return np.array([values[name] for name in names], dtype=np.float64)
