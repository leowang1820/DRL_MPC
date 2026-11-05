
"""控制指令定义及其与 CarSim 输入通道之间的转换。"""

from dataclasses import dataclass
import math

import numpy as np


# 四个车轮的排列顺序
WHEELS = ("fl", "fr", "rl", "rr")

# Python 内部控制指令名称
# 前四维为四个车轮的外加转矩，最后一维为前轮转角
COMMAND_NAMES = (
    tuple(f"external_torque_{w}_nm" for w in WHEELS)
    + ("front_steer_rad",)
)

# CarSim Import 通道名称及其排列顺序
DIRECT_IMPORT_NAMES = (
    "IMP_MYUSM_L1",
    "IMP_MYUSM_R1",
    "IMP_MYUSM_L2",
    "IMP_MYUSM_R2",
    "IMP_STEER_L1",
    "IMP_STEER_R1",
)

# 物理模型使用的实际输入名称
# 左右前轮转角保持一致，单位 rad
# 后四个输入为四个车轮实际外加转矩，单位 N·m
APPLIED_INPUT_NAMES = (
    "actual_front_steer_fl_rad",
    "actual_front_steer_fr_rad",
    *(f"actual_external_torque_{w}_nm" for w in WHEELS),
)


def finite_array(value, shape, name):
    """检查数组维度和数值有效性，并转换为 float64 类型。"""

    array = np.asarray(value, dtype=np.float64)

    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} 的维度必须为 {shape}，且所有元素必须是有限数值"
        )

    return array.copy()


@dataclass(frozen=True)
class Command:
    """定义四轮外加转矩和统一前轮转角组成的控制指令。

    四个车轮转矩的单位为 N·m。
    前轮转角的单位为 rad。

    左右前轮使用相同的转角。
    不包含摩擦制动、电机动态和执行器饱和。
    """

    wheel_torque_nm: tuple[float, float, float, float]
    front_steer_rad: float

    def __post_init__(self):
        """检查控制指令中的转矩和转角是否有效。"""

        torque = finite_array(
            self.wheel_torque_nm,
            (4,),
            "wheel_torque_nm"
        )

        steer = float(self.front_steer_rad)

        if not math.isfinite(steer) or abs(steer) >= math.pi / 2:
            raise ValueError(
                "front_steer_rad 必须位于 (-pi/2, pi/2) 范围内"
            )

        object.__setattr__(
            self,
            "wheel_torque_nm",
            tuple(map(float, torque))
        )

        object.__setattr__(
            self,
            "front_steer_rad",
            steer
        )

    @classmethod
    def from_vector(cls, values):
        """将五维数组转换为控制指令对象。"""

        values = finite_array(values, (5,), "command")

        return cls(
            tuple(values[:4]),
            float(values[4])
        )

    def as_vector(self):
        """将控制指令对象转换为五维数组。"""

        return np.array(
            (*self.wheel_torque_nm, self.front_steer_rad),
            dtype=np.float64
        )

    def to_imports(self, names):
        """将 Python 控制指令转换为 CarSim 的六个 Import 输入。

        Python 和 7DOF 模型均使用前轮转角，单位为 rad。

        左右前轮转角保持一致。
        CarSim 转向输入暂按 deg 转换。

        四个车轮转矩保持原有数值和单位。
        """

        # 统一通道名称的大小写
        names = tuple(name.upper() for name in names)

        # 检查 CarSim 输入通道的数量和名称
        if len(names) != 6 or set(names) != set(DIRECT_IMPORT_NAMES):
            raise ValueError(
                "CarSim Import 必须包含四个 IMP_MYUSM 车轮转矩通道，"
                "以及 IMP_STEER_L1 和 IMP_STEER_R1 两个前轮转角通道"
            )

        # 将前轮转角从 rad 转换为 deg
        angle_deg = math.degrees(self.front_steer_rad)

        # 建立 CarSim 通道名称与输入数值的对应关系
        values = dict(
            zip(
                DIRECT_IMPORT_NAMES,
                (
                    *self.wheel_torque_nm,
                    angle_deg,
                    angle_deg
                )
            )
        )

        # 按照实际 CarSim Import 通道顺序生成输入数组
        return np.array(
            [values[name] for name in names],
            dtype=np.float64
        )