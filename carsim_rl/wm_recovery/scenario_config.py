# -*- coding: utf-8 -*-
"""CarSim 极限操纵动力学场景配置（标准启发版）。

本文件只定义：
    1. 各实验轨迹的名称与用途；
    2. 方向、强度、时间和控制波形；
    3. 通过 medium / strong / limit 三级横向激励管理场景强度；
    4. 根据已标定的 0.3g 前轮指令参考角 A03 计算控制指令。

不会创建 CarSim Run，不会改变车辆参数、道路附着或初始车速。

重要边界：
- FMVSS 126 定义的是 *方向盘角度*，当前 CarSim 接口发送的是 *前轮角度*；
  所以 sine-with-dwell 仅沿用 0.7 Hz、500 ms 驻留及逐级增幅思想，
  不称为 FMVSS 合规试验。
- ISO 3888-2 是闭环规定路径的避障试验，本文件的 dlc_proxy 是开环双脉冲，
  不等于真正的 ISO 3888-2 试验。
- 峰值是否达到轮胎附着极限，必须通过 CarSim 实测状态检查；输入角度不能保证。
- 当前名义 7DOF 的 mu 固定为 0.8；本文件不会在 Python 中“假装”修改 CarSim 路面 mu。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

DT_S = 0.01
NOMINAL_MU = 0.8
G_MPS2 = 9.80665
WHEELBASE_M = 3.16
MAX_ROADWHEEL_DEG = 25.0       # 直接前轮角指令上限，单位 deg；不是方向盘转角

# ================================================================
# 统一管理横向激励强度
#
# lat_mu_fraction 只是生成转向指令的「名义横向加速度目标倍率」：
#   target_ay_g = lat_mu_fraction * NOMINAL_MU
#
# 例如 NOMINAL_MU=0.8 时：
#   medium -> 0.75 * 0.8 = 0.60 g
#   strong -> 0.90 * 0.8 = 0.72 g
#   limit  -> 1.05 * 0.8 = 0.84 g
#
# 这些数值不改变 CarSim 真实路面附着系数，也不保证实测 ay 达标。
# 修改不同等级的强度只需要修改这里，而不必逐一修改 SCENARIOS。
# ================================================================
LAT_MU_LEVELS = {
    "medium": 0.75,   # 中强度侧向激励：名义 0.60g
    "strong": 0.85,   # 强侧向激励：名义 0.72g
    "limit": 1.00,    # 探索性近极限激励：名义 0.84g，可能超出附着能力
}

DEFAULT_SIM_PATH = (
    r"E:\modelcodeE\carsim2022"
    r"\CarSim2022.1_Data\simfile.sim"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    split: str                      # train / validation / test
    condition_id: str
    mode: str                       # sis / circle / step / chirp / swd / dlc_proxy / brake_turn / torque
    duration_s: float = 6.0
    sign: int = 1                   # +1 为正转向、-1 为负转向
    lat_mu_fraction: float = 0.0    # 三级激励对应的名义倍率，非 CarSim 真实附着系数
    swd_a03_factor: float = 0.0     # Sine-with-dwell 幅值 / 实测 A03
    torque_peak_nm: float = 0.0     # N*m，四轮外加转矩的幅值
    diff_peak_nm: float = 0.0       # N*m，左右差动外加转矩
    note: str = ""                  # 标准来源只是启发，不代表完成标准试验


# 12 条训练 + 3 条验证 + 3 条测试。
# 非 SWD、非纯转矩场景：从 LAT_MU_LEVELS 选择 medium/strong/limit；
# SWD 场景使用 swd_a03_factor，纯转矩场景使用 torque_peak_nm。
# 先在 *同一 CarSim Run、同一车辆、mu=0.8* 上扩大操纵包络。
# 后续如需 40/60/80 km/h，应分别建立/生成对应 simfile，
# 逐批采集，而不是仅改 scenario.name。
SCENARIOS = (
    Scenario("train_sis_left", "train", "sis_left_01", "sis",
             sign=1, lat_mu_fraction=LAT_MU_LEVELS["limit"], note="ISO 4138/SIS 思路：逐步增大正向前轮转角"),
    Scenario("train_sis_right", "train", "sis_right_01", "sis",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["limit"], note="逐步增大反向前轮转角"),
    Scenario("train_circle_left", "train", "circle_left_01", "circle",
             sign=1, lat_mu_fraction=LAT_MU_LEVELS["medium"], note="ISO 4138 启发：近极限持续转向"),
    Scenario("train_circle_right", "train", "circle_right_01", "circle",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["medium"]),
    Scenario("train_step_left", "train", "step_left_01", "step",
             sign=1, lat_mu_fraction=LAT_MU_LEVELS["strong"], note="ISO 7401 启发：快速转向瞬态"),
    Scenario("train_step_right", "train", "step_right_01", "step",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["strong"]),
    Scenario("train_chirp", "train", "chirp_01", "chirp",
             lat_mu_fraction=LAT_MU_LEVELS["strong"], note="ISO 7401 启发：频率扫掠"),
    Scenario("train_swd_left", "train", "swd_left_04a", "swd",
             sign=1, swd_a03_factor=3.0, note="FMVSS 126 思路：0.7 Hz+500ms dwell"),
    Scenario("train_swd_right", "train", "swd_right_05a", "swd",
             sign=-1, swd_a03_factor=4.0),
    Scenario("train_dlc_proxy", "train", "dlc_proxy_left_01", "dlc_proxy",
             sign=1, lat_mu_fraction=LAT_MU_LEVELS["strong"], note="开环双脉冲，非 ISO 3888-2 合规路径试验"),
    Scenario("train_brake_turn_left", "train", "brake_turn_left_01", "brake_turn",
             sign=1, lat_mu_fraction=LAT_MU_LEVELS["medium"], torque_peak_nm=200,
             diff_peak_nm=100, note="ISO 7975 启发：转向与负轮端转矩联合"),
    Scenario("train_torque_sweep", "train", "torque_sweep_01", "torque",
             torque_peak_nm=1300, note="纵向滑移激励；非标准制动踏板试验"),

    Scenario("validation_circle_right", "validation", "val_circle_right_01", "circle",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["strong"]),
    Scenario("validation_swd_left", "validation", "val_swd_left_046a", "swd",
             sign=1, swd_a03_factor=3.6),
    Scenario("validation_brake_turn", "validation", "val_brake_right_01", "brake_turn",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["strong"], torque_peak_nm=200, diff_peak_nm=90),

    Scenario("test_swd_right", "test", "test_swd_right_065a", "swd",
             sign=-1, swd_a03_factor=4.5),
    Scenario("test_dlc_proxy_right", "test", "test_dlc_proxy_right_01", "dlc_proxy",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["limit"]),
    Scenario("test_brake_turn_right", "test", "test_brake_right_01", "brake_turn",
             sign=-1, lat_mu_fraction=LAT_MU_LEVELS["limit"], torque_peak_nm=200,
             diff_peak_nm=100),
)


def validate_scenarios(scenarios=SCENARIOS):
    """检查集合和设置，防止误把验证/测试轨迹作为训练数据。"""
    scenes = tuple(scenarios)
    if len(scenes) < 4:
        raise ValueError("至少需要 2 条训练、1 条验证、1 条测试")
    if len({s.name for s in scenes}) != len(scenes):
        raise ValueError("轨迹名称不能重复")
    if len({s.condition_id for s in scenes}) != len(scenes):
        raise ValueError("condition_id 不能重复")
    splits = [s.split for s in scenes]
    if splits.count("train") < 2 or not {"validation", "test"} <= set(splits):
        raise ValueError("必须有至少 2 条 train，至少 1 条 validation 和 test")
    for scene in scenes:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scene.name):
            raise ValueError(f"不合法的轨迹名：{scene.name}")
        if scene.split not in {"train", "validation", "test"}:
            raise ValueError(f"未知 split：{scene.split}")
        if not scene.condition_id.strip():
            raise ValueError(f"{scene.name} 的 condition_id 不能为空")
        if scene.mode not in {
            "sis", "circle", "step", "chirp", "swd",
            "dlc_proxy", "brake_turn", "torque"
        }:
            raise ValueError(f"{scene.name} 的 mode 不受支持")
        if scene.sign not in (-1, 1):
            raise ValueError("sign 必须为 +1 或 -1")
        if not math.isfinite(scene.duration_s) or scene.duration_s < 6.0:
            raise ValueError("当前设计的极限工况需要至少 6 秒")
        steps = round(scene.duration_s / DT_S)
        if not math.isclose(steps * DT_S, scene.duration_s,
                            rel_tol=0, abs_tol=1e-10):
            raise ValueError("时长必须是 0.01 s 的整数倍")
        for field in ("lat_mu_fraction", "swd_a03_factor",
                      "torque_peak_nm", "diff_peak_nm"):
            value = getattr(scene, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{scene.name}.{field} 必须是非负有限值")
        if scene.lat_mu_fraction > 1.2:
            raise ValueError("lat_mu_fraction 超出当前建议的扫描范围")
        if scene.swd_a03_factor > 6.5:
            raise ValueError("Sine-with-dwell 扫描倍率暂设最大 6.5")
        if scene.torque_peak_nm > 1500 or scene.diff_peak_nm > 200:
            raise ValueError("超过当前模拟试验的转矩指令硬上限")
    return scenes


def roadwheel_delta_for_ay(target_ay_g, speed_kmh):
    """仅用于校准扫描的初始估计：tan(delta)≈L*ay/v²。

    不是极限车辆动力学的真实响应模型。
    """
    v = float(speed_kmh) / 3.6
    if not math.isfinite(v) or v < 8.0:
        raise ValueError("车速过低，无法使用当前转向标定")
    return math.degrees(
        math.atan(WHEELBASE_M * float(target_ay_g) * G_MPS2 / (v * v))
    )


def smoothstep(x):
    """从 0 平滑变化到 1，避免理想瞬时无穷大转向角速度。"""
    y = max(0.0, min(1.0, float(x)))
    return y * y * (3.0 - 2.0 * y)


def pulse(t, start, end, ramp_s=0.15):
    """平滑矩形脉冲：窗口外为 0，窗口内最大为 1。"""
    if not (start <= t <= end):
        return 0.0
    return smoothstep((t - start) / ramp_s) * smoothstep((end - t) / ramp_s)


def steering_amplitude_deg(scene, a03_deg):
    """根据 CarSim 标定的 0.3g 前轮*指令角*给出本场景幅值。

    A03 是在仿真中达到约 0.3g 时发送的前轮指令角，
    既不是方向盘角，也未必等于实测左右前轮的实际角。
    """
    a03 = float(a03_deg)
    if not math.isfinite(a03) or a03 <= 0:
        raise ValueError("0.3g 参考角 A03 必须是正数")
    if scene.mode == "swd":
        # SWD 的强度单独由 swd_a03_factor 管理，和三级 lat_mu_fraction 无关。
        # 平滑限幅不会在有限幅值下突然将多条不同指令截为同一个 25°。
        # 其实际输出幅值通常低于名义 factor*A03。
        target = scene.swd_a03_factor * a03
        return MAX_ROADWHEEL_DEG * math.tanh(target / MAX_ROADWHEEL_DEG)

    # 其他转向场景：把名义目标 ay/g 转换为实测 A03 的倍率。
    # 注意这个线性缩放只是激励设计方法，不是车辆非线性响应预测。
    target_g = scene.lat_mu_fraction * NOMINAL_MU
    target = a03 * (target_g / 0.3)
    return min(MAX_ROADWHEEL_DEG, target)


def command_at(scene: Scenario, t: float, a03_deg: float):
    """某时刻的五维指令：[T_FL,T_FR,T_RL,T_RR,delta_f_rad]。

    转矩为 N*m，delta_f_rad 为直接前轮转角，单位 rad。
    """
    if not math.isfinite(t) or not 0 <= t < scene.duration_s:
        raise ValueError("t 必须在 [0,duration_s) 内")

    amp = steering_amplitude_deg(scene, a03_deg)
    sign = scene.sign
    steer_deg = 0.0
    base_nm = 0.0
    diff_nm = 0.0

    if scene.mode == "sis":
        # 缓增转向，到达目标幅值后保持一段时间，再平滑收回。
        steer_deg = sign * amp * (
            smoothstep((t - 0.5) / 4.0) * smoothstep((5.8 - t) / 0.5)
        )

    elif scene.mode == "circle":
        # 近似稳态圆周：较慢建立固定前轮角，再逐渐恢复。
        steer_deg = sign * amp * pulse(t, 0.6, 5.1, ramp_s=1.0)

    elif scene.mode == "step":
        # 快速转向阶跃（有限斜率）：刺激瞬态侧偏与横摆响应。
        steer_deg = sign * amp * pulse(t, 1.0, 3.7, ramp_s=0.18)

    elif scene.mode == "chirp":
        # 频率从 0.2 Hz 扫到 1.4 Hz，双向转向。
        active = 4.5
        z = t - 0.8
        if 0 <= z <= active:
            f0, f1 = 0.2, 1.4
            phase = 2 * math.pi * (
                f0 * z + 0.5 * (f1 - f0) / active * z * z
            )
            steer_deg = sign * amp * math.sin(phase) * (
                math.sin(math.pi * z / active) ** 2
            )

    elif scene.mode == "swd":
        # FMVSS 126 波形思想：0.7 Hz 正弦、第二峰值驻留 500 ms。
        # 此处是直接前轮角、非方向盘角，且不复制法规的环境/标定条件。
        freq, dwell_s, onset_s = 0.7, 0.5, 1.0
        x = t - onset_s
        dwell_start = 0.75 / freq  # 正弦波第二个峰值（负峰）
        active_end = 2.0 / freq + dwell_s
        if 0 <= x <= active_end:
            if x < dwell_start:
                shape = math.sin(2.0 * math.pi * freq * x)
            elif x < dwell_start + dwell_s:
                shape = -1.0
            else:
                shape = math.sin(
                    2.0 * math.pi * freq * (x - dwell_s)
                )
            steer_deg = sign * amp * shape

    elif scene.mode == "dlc_proxy":
        # 两个正负相反的光滑脉冲，只仿照转向方向变化，不跟踪 ISO 3888-2 场地路径。
        steer_deg = sign * amp * (
            pulse(t, 0.9, 2.5, ramp_s=0.65)
            - pulse(t, 2.5, 4.2, ramp_s=0.65)
        )

    elif scene.mode == "brake_turn":
        # 持续转向 + 四轮负外加转矩 + 小幅左右差动。
        # 这不是制动踏板的标准等价输入，不验证实际摩擦制动是否生效。
        steer_deg = sign * amp * pulse(t, 0.6, 4.7, ramp_s=0.6)
        base_nm = -scene.torque_peak_nm * pulse(t, 1.8, 3.8, 0.25)
        diff_nm = (
            sign * scene.diff_peak_nm * pulse(t, 2.0, 3.3, 0.25)
        )

    elif scene.mode == "torque":
        # 分阶段正/负轮端转矩激励，让车轮角速度与滑移可能进入非线性区。
        # 不保证达到轮胎极限；是否达到需在采集后检查轮胎滑移。
        peak = scene.torque_peak_nm
        base_nm = (
            0.35 * peak * pulse(t, 0.3, 1.2, 0.12)
            + 0.70 * peak * pulse(t, 1.3, 2.2, 0.12)
            + peak * pulse(t, 2.3, 3.0, 0.12)
            - 0.50 * peak * pulse(t, 3.3, 4.0, 0.12)
            - peak * pulse(t, 4.1, 4.8, 0.12)
        )

    # 四轮外加转矩，允许正负，但不宣称是净驱动/摩擦制动转矩。
    command = [
        base_nm - diff_nm,
        base_nm + diff_nm,
        0.9 * base_nm - 0.7 * diff_nm,
        0.9 * base_nm + 0.7 * diff_nm,
        math.radians(steer_deg),
    ]
    if not all(math.isfinite(x) for x in command):
        raise ValueError("出现无效控制指令")
    return command


def audit_scene_control_sequences(calibration, scenes=SCENARIOS):
    """采集前逐条检查 10 ms 五维控制数组，拒绝重复输入。

    注意：这是控制序列预检查，不能代替后续对真实 CarSim 原始
    轨迹、候选文件 SHA256 的防重复审计。
    """
    import hashlib
    import struct

    scene_list = tuple(scenes)
    fingerprints = {}
    details = []
    for scene in scene_list:
        direction = "positive" if scene.sign > 0 else "negative"
        a03 = float(calibration[direction]["reference_roadwheel_angle_deg"])
        steps = round(scene.duration_s / DT_S)
        h = hashlib.sha256()
        for i in range(steps):
            command = command_at(scene, i * DT_S, a03)
            # 固定字节序和浮点格式；不因名字、split 不同而改变指纹。
            h.update(struct.pack("<5d", *command))
        signature = h.hexdigest()
        if signature in fingerprints:
            previous = fingerprints[signature]
            raise ValueError(
                f"检测到重复的五维控制序列：{previous} 与 {scene.name}；"
                "请先调整转角限幅/激励参数，禁止让重复轨迹进入不同数据分区"
            )
        fingerprints[signature] = scene.name
        details.append({
            "scene": scene.name,
            "split": scene.split,
            "mode": scene.mode,
            "sign": scene.sign,
            "a03_deg": a03,
            "actual_requested_peak_steer_deg": steering_amplitude_deg(scene, a03),
            "control_sha256": signature,
        })

    # 对同方向 Sine-with-dwell 跨分区的幅值做额外检查。
    # 避免虽然不完全一样，但幅值过于接近而失去测试意义。
    for i, a in enumerate(details):
        if a["mode"] != "swd":
            continue
        for b in details[i + 1:]:
            if b["mode"] == "swd" and a["sign"] == b["sign"]:
                min_separation = max(0.05, 0.10 * max(a["a03_deg"], b["a03_deg"]))
                diff = abs(a["actual_requested_peak_steer_deg"] -
                           b["actual_requested_peak_steer_deg"])
                if diff < min_separation:
                    raise ValueError(
                        f"{a['scene']} 和 {b['scene']} 的转角仅相差 {diff:.3f}deg，"
                        f"小于预设最低间隔 {min_separation:.3f}deg；"
                        "请调整 SWD 倍率以保证测试工况明显区分"
                    )

    return {
        "format": "wm_scenario_command_uniqueness_preflight_v1",
        "scenes_checked": len(details),
        "all_unique": True,
        "note": "仅确认给定 A03 下控制序列不重复；仍需真实轨迹指纹审计",
        "commands": details,
    }


def describe_scenario(scene):
    info = asdict(scene)
    info["target_ay_g_nominal"] = round(
        scene.lat_mu_fraction * NOMINAL_MU, 3
    ) if scene.lat_mu_fraction else None
    info["is_official_compliance_test"] = False
    return info
