# -*- coding: utf-8 -*-
"""统一的 CarSim 场景配置：只定义场景和控制指令，不启动 CarSim。

使用方式：
    from wm_recovery.scenario_config import SCENARIOS, command_at

返回控制指令顺序始终是：
    [左前转矩, 右前转矩, 左后转矩, 右后转矩, 共同前轮转角(rad)]
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# World Model 和现有候选数据契约都使用 10 ms 采样周期。
DT_S = 0.01

# 当前 CarSim 生成的仿真配置文件位置。
DEFAULT_SIM_PATH = (
    r"E:\modelcodeE\carsim2022"
    r"\CarSim2022.1_Data\simfile.sim"
)


@dataclass(frozen=True)
class Scenario:
    """一条轨迹的实验设置。

    name: 轨迹名称，对应输出目录名
    split: train / validation / test
    condition_id: 唯一工况标识，防止训练和测试混用
    mode: "three_phase" 或 "sine"
    其他参数：正弦转矩、差动转矩、转角的幅值与频率
    """

    name: str
    split: str
    condition_id: str
    duration_s: float = 6.0
    mode: str = "sine"
    torque_nm: float = 0.0
    torque_hz: float = 0.0
    difference_nm: float = 0.0
    difference_hz: float = 0.0
    steer_deg: float = 0.0
    steer_hz: float = 0.0


# 原来的 train1 三阶段激励常数。
# 这组参数只用于模式 "three_phase"。
TRAIN1_PHASES = {
    "longitudinal_torque_nm": 40.0,
    "longitudinal_torque_hz": 0.55,
    "steering_deg": 0.35,
    "steering_hz": 0.45,
    "mixed_torque_nm": 25.0,
    "mixed_torque_hz": 0.60,
    "mixed_steer_deg": 0.25,
    "mixed_steer_hz": 0.50,
}


# 统一定义四条不同轨迹。
# 你以后主要修改这里，就可以改变 CarSim 的控制激励。
SCENARIOS = (
    Scenario(
        name="train1",
        split="train",
        condition_id="scenario_train1_three_phase",
        mode="three_phase",
    ),
    Scenario(
        name="train2",
        split="train",
        condition_id="scenario_train2_sine_049",
        mode="sine",
        torque_nm=30.0,
        torque_hz=0.49,
        difference_nm=9.0,
        difference_hz=0.23,
        steer_deg=0.26,
        steer_hz=0.38,
    ),
    Scenario(
        name="validation",
        split="validation",
        condition_id="scenario_validation_sine_071",
        mode="sine",
        torque_nm=25.0,
        torque_hz=0.71,
        difference_nm=7.0,
        difference_hz=0.41,
        steer_deg=0.31,
        steer_hz=0.52,
    ),
    Scenario(
        name="test",
        split="test",
        condition_id="scenario_test_sine_034",
        mode="sine",
        torque_nm=32.0,
        torque_hz=0.34,
        difference_nm=11.0,
        difference_hz=0.57,
        steer_deg=0.40,
        steer_hz=0.63,
    ),
)


def validate_scenarios(scenarios=SCENARIOS):
    """检查场景定义，禁止无意中混淆训练、验证和测试轨迹。"""
    scenes = tuple(scenarios)
    if len(scenes) != 4:
        raise ValueError("当前实验要求恰好四条轨迹")
    if {scene.name for scene in scenes} != {
        "train1", "train2", "validation", "test"
    }:
        raise ValueError("轨迹名称必须为 train1/train2/validation/test")
    if len({scene.condition_id for scene in scenes}) != 4:
        raise ValueError("四条轨迹的 condition_id 必须各不相同")
    for scene in scenes:
        expected_split = (
            "train" if scene.name.startswith("train") else scene.name
        )
        if scene.split != expected_split:
            raise ValueError(f"{scene.name} 的 split 设置错误")
        if scene.mode not in {"three_phase", "sine"}:
            raise ValueError(f"{scene.name} 未知的激励模式：{scene.mode}")
        if not math.isfinite(scene.duration_s) or scene.duration_s < 3.0:
            raise ValueError("当前实验每条轨迹至少采集 3 秒")
        samples = round(scene.duration_s / DT_S)
        if not math.isclose(samples * DT_S, scene.duration_s,
                            rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("轨迹时长必须是 0.01 秒的整数倍")
        for key in (
            "torque_nm", "torque_hz", "difference_nm",
            "difference_hz", "steer_deg", "steer_hz"
        ):
            value = getattr(scene, key)
            if not math.isfinite(value):
                raise ValueError(f"{scene.name}.{key} 不能是 NaN 或无穷大")
        if not scene.condition_id.strip():
            raise ValueError("condition_id 不能为空")
    return scenes


def command_at(scene: Scenario, t: float):
    """根据场景和当前仿真时间，计算五维 Python 控制指令。

    只负责计算控制；不会启动仿真。
    四轮转矩：N·m；前轮共同转角：rad。
    """
    if not math.isfinite(t) or t < 0 or t >= scene.duration_s:
        raise ValueError("控制时间 t 必须位于 [0, duration_s)")

    if scene.mode == "three_phase":
        # 重现旧 validate_carsim_long.py 的三阶段训练激励。
        segment = scene.duration_s / 3.0
        q = t % segment
        smooth = math.sin(math.pi * q / segment) ** 2
        if t < segment:
            torque = (
                TRAIN1_PHASES["longitudinal_torque_nm"]
                * smooth
                * math.sin(2 * math.pi *
                           TRAIN1_PHASES["longitudinal_torque_hz"] * q)
            )
            return [torque, torque, torque, torque, 0.0]
        if t < 2 * segment:
            steer = (
                TRAIN1_PHASES["steering_deg"]
                * smooth
                * math.sin(2 * math.pi *
                           TRAIN1_PHASES["steering_hz"] * q)
            )
            return [0.0, 0.0, 0.0, 0.0, math.radians(steer)]
        torque = (
            TRAIN1_PHASES["mixed_torque_nm"]
            * smooth
            * math.sin(2 * math.pi *
                       TRAIN1_PHASES["mixed_torque_hz"] * q)
        )
        steer = (
            TRAIN1_PHASES["mixed_steer_deg"]
            * smooth
            * math.sin(2 * math.pi *
                       TRAIN1_PHASES["mixed_steer_hz"] * q)
        )
        return [torque, torque, torque, torque, math.radians(steer)]

    if scene.mode == "sine":
        # 让指令在仿真开始和结束时逐渐回到零。
        smooth = math.sin(math.pi * t / scene.duration_s) ** 2

        # base：四轮基础转矩，diff：左右差动转矩。
        base = (
            scene.torque_nm
            * math.sin(2 * math.pi * scene.torque_hz * t)
            * smooth
        )
        diff = (
            scene.difference_nm
            * math.sin(2 * math.pi * scene.difference_hz * t)
            * smooth
        )
        steer_deg = (
            scene.steer_deg
            * math.sin(2 * math.pi * scene.steer_hz * t)
            * smooth
        )

        return [
            base - diff,             # 左前轮
            base + diff,             # 右前轮
            0.9 * base - 0.7 * diff, # 左后轮
            0.9 * base + 0.7 * diff, # 右后轮
            math.radians(steer_deg), # 前轮共同转角，rad
        ]

    raise ValueError(f"未知控制激励：{scene.mode}")
