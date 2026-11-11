# -*- coding: utf-8 -*-
"""按标准思想设计的 CarSim 极限操纵激励：标定、采集、覆盖率审计、训练。

执行方式（在项目根目录）：
  python -m wm_recovery.scenario_runner plan
  python -m wm_recovery.scenario_runner capture --output artifacts/wm_recovery/handling_limit_01
  $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"
  python -m wm_recovery.scenario_runner train-existing --workspace \
    artifacts/wm_recovery/handling_limit_01/workspace \
    --output artifacts/wm_recovery/handling_limit_01/model_run_01 --epochs 1

注意：
- 本程序不修改 simfile、车速或道路 mu，仅使用已配置 CarSim Run。
- 首先独立运行正/负向约 0.3g 前轮转向标定，再按标定结果生成 18 条轨迹。
- 标准启发 != 法规/ISO 合规；轮胎接近极限需实际后验验证。
- 只保存真实采集结果，不伪造 ready_for_wm_training=True。
- 输出目录必须不存在，严禁覆盖原始实验。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .scenario_config import (
    DEFAULT_SIM_PATH,
    DT_S,
    G_MPS2,
    MAX_ROADWHEEL_DEG,
    NOMINAL_MU,
    SCENARIOS,
    Scenario,
    command_at,
    describe_scenario,
    roadwheel_delta_for_ay,
    smoothstep,
    steering_amplitude_deg,
    validate_scenarios,
)

EXPERIMENT_ENV = "WM_RECOVERY_EXPERIMENTAL_CARSIM"
CALIBRATION_FORMAT = "wm_recovery_roadwheel_approx_a03_v1"
COVERAGE_FORMAT = "wm_recovery_limit_envelope_coverage_v1"
SPEED_MIN_KMH = 35.0
# 多初始车速批量实验：允许到 180 km/h，给初始化读数留 1 km/h 余量。
# 只是软件允许范围，并不意味着高速极限试验已通过有效性验证。
SPEED_MAX_KMH = 181.0


def _atomic_json(path, value):
    """新文件先写临时文件，写完再重命名；不覆盖既有结果。"""
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2,
                      allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temporary.rename(target)
    except BaseException:
        # 失败时不覆盖原文件；删除由本函数创建的不完整临时文件。
        temporary.unlink(missing_ok=True)
        raise


def print_plan():
    """只显示实验计划，不加载 CarSim DLL、不写入磁盘。"""
    scenes = validate_scenarios()
    print("前置步骤：分别用正/负缓慢转向测得约 0.3g 的参考前轮角 A03")
    print("当前只改变控制激励；不同车速/路面必须使用实际配置好的 CarSim Run。")
    for scene in scenes:
        print(json.dumps(
            describe_scenario(scene),
            ensure_ascii=False, indent=2, allow_nan=False
        ))
    print("场景数量：", len(scenes))
    print("注意：Sine with dwell 为直接前轮角度近似波形，并非 FMVSS 126 合规试验。")


def _check_solver_duration(bridge, duration_s):
    """检查 CarSim Run 能否覆盖本次采集时长。"""
    env = getattr(bridge, "_env", None)
    config = getattr(env, "config", {})
    if isinstance(config, dict) and config.get("t_stop") is not None:
        stop = float(config["t_stop"])
        needed = float(bridge.latest.time_s) + duration_s
        if needed > stop + 1e-8:
            raise RuntimeError(
                f"CarSim TSTOP={stop}s 小于需要的结束时间 {needed}s；"
                "请在 CarSim Run 中增加仿真时间并重新生成 simfile"
            )


def _read_native(snapshot):
    """从 CarSimSnapshot 提取本项目已核对过的三个原生通道。

    Vx：km/h；Ax/Ay：g；Steer_L1/R1：deg；AVz：deg/s。
    """
    data = snapshot.data
    return {
        "vx_kmh": float(data.Vx),
        "ax_g": float(data.Ax),
        "ay_g": float(data.Ay),
        "yaw_deg_s": float(data.AVz),
        "steer_l_deg": float(data.Steer_L1),
        "steer_r_deg": float(data.Steer_R1),
    }


def _require_speed(speed_kmh):
    """初始车速太低时无法通过当前直接转向波形有效激励极限侧向动力学。"""
    if not math.isfinite(speed_kmh):
        raise ValueError("CarSim 初始纵向车速不是有限数")
    if not SPEED_MIN_KMH <= abs(speed_kmh) <= SPEED_MAX_KMH:
        raise ValueError(
            f"当前 CarSim 初始车速={speed_kmh:.2f} km/h；"
            f"本场景包络要求 {SPEED_MIN_KMH:.0f}~{SPEED_MAX_KMH:.0f} km/h。"
            "请修改实际 CarSim Run 初始条件，不是只改 Python 的标签"
        )
    if speed_kmh < 0:
        raise ValueError("当前场景只针对正向行驶")


def _open_bridge(sim_path, bridge_type=None):
    if bridge_type is None:
        from .carsim_bridge import CarSimBridge
        bridge_type = CarSimBridge
    return bridge_type(sim_path, prediction_dt_s=DT_S)


def _calibrate_direction(sim_path, sign, root, bridge_type=None):
    """通过独立 CarSim 仿真找出达到约 0.3g 所需的直接前轮角。

    这是工程近似（有限斜率、速度不闭环保持），并非 FMVSS 正式 SIS；
    记录的是此 CarSim Run 的实际响应，不能跨不同车辆和车速复用。
    """
    direction = "positive" if sign > 0 else "negative"
    duration_s = 6.0
    count = round(duration_s / DT_S)

    print(f"\n标定 {direction}：查找约 0.3g 的直接前轮参考转角")
    with _open_bridge(sim_path, bridge_type) as bridge:
        _check_solver_duration(bridge, duration_s)
        initial = _read_native(bridge.latest)
        speed_kmh = initial["vx_kmh"]
        _require_speed(speed_kmh)

        # 用运动学估计，只作为标定缓增指令的步幅参考。
        estimate = roadwheel_delta_for_ay(0.3, speed_kmh)
        max_deg = min(MAX_ROADWHEEL_DEG,
                      max(3.0, 3.0 * estimate))
        samples = []
        controls = []
        speeds = []
        times = []

        for i in range(count):
            t = i * DT_S
            # 约 4.6 秒内由 0 平滑增加到选定前轮角；保持 0.9s。
            angle_deg = sign * max_deg * smoothstep(
                (t - 0.35) / 4.6
            )
            result = bridge.step([0.0, 0.0, 0.0, 0.0,
                                  math.radians(angle_deg)])
            data = _read_native(result)
            # 侧向轴符号依 CarSim 车辆坐标约定；只用绝对响应定幅。
            samples.append(
                abs(data["ay_g"] - initial["ay_g"])
            )
            controls.append(abs(angle_deg))
            speeds.append(data["vx_kmh"])
            times.append(result.time_s)

        # 保留标定轨迹以便检查，并在后续程序中读取。
        outdir = root / f"calibration_{direction}"
        bridge.save_raw(outdir)

    # 要求至少连续 0.1s 测到接近或超过 0.3g 的响应。
    running = 10
    crossing = None
    for i in range(running - 1, len(samples)):
        window = samples[i - running + 1:i + 1]
        if min(window) >= 0.295:
            crossing = i
            break
    if crossing is None:
        peak = max(samples) if samples else float("nan")
        raise RuntimeError(
            f"{direction} 缓增转向未达到稳定的约 0.3g 响应，"
            f"最大正向横向加速度增量={peak:.3f}g；"
            "请检查 CarSim 初始车速、转向执行和车辆模型，"
            "不要直接把更大的角度当成已达极限"
        )

    return {
        "direction": direction,
        "initial_speed_kmh": speed_kmh,
        "speed_at_reference_kmh": speeds[crossing],
        "reference_roadwheel_angle_deg": controls[crossing],
        "time_at_reference_s": times[crossing],
        "measured_ay_increment_g": samples[crossing],
        "max_calibration_steer_deg": max_deg,
        "steady_check_samples": running,
        "standard_compliance": False,
    }


def calibrate_steer(sim_path, root, bridge_type=None):
    """在输出目录下生成双向 0.3g 标定，并保存原始标定记录。"""
    root = Path(root)
    positive = _calibrate_direction(
        sim_path, 1, root, bridge_type
    )
    negative = _calibrate_direction(
        sim_path, -1, root, bridge_type
    )
    if abs(
        positive["initial_speed_kmh"] -
        negative["initial_speed_kmh"]
    ) > 5.0:
        raise ValueError("左右转向标定的 CarSim 初始车速差过大")

    result = {
        "format": CALIBRATION_FORMAT,
        "sim_path": str(Path(sim_path).expanduser().resolve()),
        "positive": positive,
        "negative": negative,
        "meaning": (
            "仅用于本 CarSim Run 的直接前轮转角幅值参考；"
            "有限斜率的近似 0.3g 标定，不等于法规转向盘 A"
        ),
        "fmvss_126_compliance": False,
    }
    _atomic_json(root / "steer_calibration.json", result)
    print("\n0.3g 前轮角参考：")
    print("  正向:", round(
        positive["reference_roadwheel_angle_deg"], 4
    ), "deg")
    print("  负向:", round(
        negative["reference_roadwheel_angle_deg"], 4
    ), "deg")
    return result


def capture_one(scene: Scenario, sim_path, root, calibration,
                bridge_type=None, candidate_creator=None):
    """一条轨迹：逐步控制 CarSim → 保存原始数据 → 转换候选数据。

    返回候选数据路径、实测激励覆盖率报告。
    """
    if candidate_creator is None:
        from .carsim_candidate import create_candidate
        candidate_creator = create_candidate

    direction = "positive" if scene.sign > 0 else "negative"
    reference = calibration[direction]
    a03_deg = float(reference["reference_roadwheel_angle_deg"])
    root = Path(root)
    raw_dir = root / ("raw_" + scene.name)
    candidate_dir = root / ("candidate_" + scene.name)
    count = round(scene.duration_s / DT_S)
    print(f"\n采集 {scene.name}：{count} 个控制区间 "
          f"(参考前轮角 A03={a03_deg:.3f}°)")

    with _open_bridge(sim_path, bridge_type) as bridge:
        _check_solver_duration(bridge, scene.duration_s)
        initial = _read_native(bridge.latest)
        _require_speed(initial["vx_kmh"])
        # 避免在不同初始车速的 Run 中复用同一标定参考。
        if abs(
            initial["vx_kmh"] - float(reference["initial_speed_kmh"])
        ) > 5.0:
            raise ValueError(
                f"{scene.name} 的初速与转角标定相差超过 5 km/h，"
                "请对当前 CarSim Run 重新标定"
            )

        observed = []
        for i in range(count):
            t = i * DT_S
            cmd = command_at(scene, t, a03_deg)
            snapshot = bridge.step(cmd)
            observed.append(_read_native(snapshot))
            if (i + 1) % 200 == 0 or i + 1 == count:
                print(
                    f"  {scene.name}: {(i+1)*DT_S:.2f}/{scene.duration_s:.2f}s"
                )
            if getattr(bridge, "_ended", False) and i + 1 < count:
                raise RuntimeError(
                    f"{scene.name}：CarSim Solver 提前结束"
                )
        bridge.save_raw(raw_dir)

    if len(observed) != count:
        raise RuntimeError("CarSim 有缺失的采样帧")
    if any(not all(math.isfinite(v) for v in item.values())
           for item in observed):
        raise RuntimeError("CarSim 状态含 NaN 或无穷大")

    # 使用真实 CarSim 输出而不是命令幅值判断激励是否达到目标。
    ay_abs = [
        abs(item["ay_g"] - initial["ay_g"])
        for item in observed
    ]
    ay_peak = max(ay_abs)
    combined_peak = max(
        math.hypot(item["ax_g"] - initial["ax_g"],
                   item["ay_g"] - initial["ay_g"])
        for item in observed
    )
    near_threshold = 0.75 * NOMINAL_MU
    limit_threshold = 0.90 * NOMINAL_MU
    near_count = sum(y >= near_threshold for y in ay_abs)
    strong_count = sum(y >= limit_threshold for y in ay_abs)
    min_speed = min(item["vx_kmh"] for item in observed)

    report = {
        "scenario": scene.name,
        "split": scene.split,
        "mode": scene.mode,
        "sign": scene.sign,
        "condition_id": scene.condition_id,
        "configured_mu_nominal": NOMINAL_MU,
        "target_ay_g_nominal": scene.lat_mu_fraction * NOMINAL_MU,
        "max_requested_steer_deg": steering_amplitude_deg(
            scene, a03_deg
        ),
        "calibrated_a03_roadwheel_deg": a03_deg,
        "peak_abs_ay_g": ay_peak,
        "peak_combined_ax_ay_g": combined_peak,
        "samples_ay_ge_0p75_mu": near_count,
        "duration_ay_ge_0p75_mu_s": near_count * DT_S,
        "samples_ay_ge_0p90_mu": strong_count,
        "duration_ay_ge_0p90_mu_s": strong_count * DT_S,
        "peak_actual_steer_l_deg": max(
            abs(s["steer_l_deg"]) for s in observed
        ),
        "peak_actual_steer_r_deg": max(
            abs(s["steer_r_deg"]) for s in observed
        ),
        "initial_speed_kmh": initial["vx_kmh"],
        "minimum_speed_kmh": min_speed,
        "maximum_speed_kmh": max(
            item["vx_kmh"] for item in observed
        ),
        "peak_abs_yaw_deg_s": max(
            abs(s["yaw_deg_s"]) for s in observed
        ),
        # 不把一次孤立尖峰认定为“激励覆盖”。
        "near_limit_proxy_reached": near_count * DT_S >= 0.10,
        "strong_limit_proxy_reached": strong_count * DT_S >= 0.05,
        "note": (
            "只以实测 CG 加速度判断此数据是否覆盖高横向响应；"
            "尚未证明各轮轮胎饱和、达到真实附着系数极限或法规测试合规"
        ),
    }

    candidate_report = candidate_creator(raw_dir, candidate_dir)
    if candidate_report.get("torque_echo_matches") is not True:
        raise RuntimeError(
            f"{scene.name}: 四轮外加转矩回显未匹配，"
            "不允许将此轨迹加入 World Model 训练集"
        )

    # 附加指标：CG 侧偏角 beta 和四轮纵向滑移相关的 CarSim 原生通道。
    # beta 由候选数据的 vx、vy 计算；KappL 原生单位需在 CarSim 字典中核实。
    # 不把 KappL 原生值直接解释成已验证的轮胎利用率。
    report["peak_abs_beta_deg"] = None
    report["max_abs_kappl_native"] = None
    candidate_npz = candidate_dir / "candidate.npz"
    raw_npz = raw_dir / "raw_capture.npz"
    if candidate_npz.is_file() and raw_npz.is_file():
        import numpy as np
        from .carsim_export_schema import EXPORT_NAMES

        with np.load(candidate_npz, allow_pickle=False) as archive:
            state = np.asarray(archive["state_si"], dtype=np.float64)
        v_long = state[:, 0]
        mask = v_long >= 5.0  # 低车速时 beta 可能失去稳定意义
        if np.any(mask):
            beta = np.rad2deg(np.arctan2(state[mask, 1], v_long[mask]))
            report["peak_abs_beta_deg"] = float(np.max(np.abs(beta)))

        names = ("KappL_L1", "KappL_R1", "KappL_L2", "KappL_R2")
        schema = tuple(EXPORT_NAMES)
        if all(schema.count(n) == 1 for n in names):
            cols = [schema.index(n) for n in names]
            with np.load(raw_npz, allow_pickle=False) as archive:
                raw_exports = np.asarray(
                    archive["raw_exports"], dtype=np.float64
                )
            report["max_abs_kappl_native"] = float(
                np.max(np.abs(raw_exports[:, cols]))
            )

    return candidate_dir, report


def summarize_coverage(reports):
    """全数据集实测覆盖率检查；若激励不足明确标注不通过。"""
    train = [r for r in reports if r["split"] == "train"]
    validation = [r for r in reports if r["split"] == "validation"]
    test = [r for r in reports if r["split"] == "test"]
    near = 0.75 * NOMINAL_MU
    strong = 0.90 * NOMINAL_MU

    # 这些阈值是项目定义的“数据集覆盖率门槛”，不是 ISO/NHTSA 强制判据。
    def near_reached(r):
        # >=0.1 s 高横向加速度，排除单采样点的尖峰。
        return (r.get("duration_ay_ge_0p75_mu_s", 0.0) >= 0.10
                and r["peak_abs_ay_g"] >= near)

    def strong_reached(r):
        # >=0.05 s 接近名义附着水平的强激励。
        return (r.get("duration_ay_ge_0p90_mu_s", 0.0) >= 0.05
                and r["peak_abs_ay_g"] >= strong)

    checks = {
        "train_near_limit_count_ge_4":
            sum(near_reached(r) for r in train) >= 4,
        "train_strong_limit_count_ge_2":
            sum(strong_reached(r) for r in train) >= 2,
        "train_positive_direction_near_limit":
            any(r["sign"] > 0 and near_reached(r) for r in train),
        "train_negative_direction_near_limit":
            any(r["sign"] < 0 and near_reached(r) for r in train),
        "validation_at_least_one_near_limit":
            any(near_reached(r) for r in validation),
        "test_at_least_one_near_limit":
            any(near_reached(r) for r in test),
    }
    return {
        "format": COVERAGE_FORMAT,
        "nominal_mu": NOMINAL_MU,
        "threshold_ay_near_limit_g": near,
        "threshold_ay_strong_limit_g": strong,
        "minimum_near_limit_duration_s": 0.10,
        "minimum_strong_limit_duration_s": 0.05,
        "criteria_origin": "project_defined_coverage_gate_not_iso_or_fmvss",
        "passed": all(checks.values()),
        "checks": checks,
        "scenarios": reports,
        "note": (
            "横向加速度阈值仅是近极限激励覆盖率代理指标，"
            "不能保证所有轮胎都进入非线性区，也没有验证纵向滑移利用率。"
        ),
    }


def build_scene_workspace(candidate_paths, workspace,
                          scenarios=SCENARIOS):
    """沿用已验证的 candidate 指纹契约，创建通用数量的 WM 分区清单。

    暂时复用 carsim_experimental._candidate_facts 和原有 Episode 加载结构，
    不能删除该模块。
    """
    from .carsim_experimental import _candidate_facts, MANIFEST_FORMAT
    from .offline import NOMINAL_MU as MODEL_NOMINAL_MU

    scenes = validate_scenarios(scenarios)
    if set(candidate_paths) != {s.name for s in scenes}:
        raise ValueError("候选轨迹和场景列表不一致")
    if not math.isclose(MODEL_NOMINAL_MU, NOMINAL_MU, abs_tol=1e-12):
        raise ValueError("原始 7DOF 名义 mu 与场景配置 mu 不一致")

    workspace = Path(workspace).expanduser().resolve()
    if workspace.exists():
        raise FileExistsError(f"workspace 已存在：{workspace}")

    verified = {}
    raw_hashes = set()
    candidate_hashes = set()

    # 全部候选通过校验后，才建立 workspace。
    for scene in scenes:
        path = Path(candidate_paths[scene.name]).expanduser().resolve(
            strict=True
        )
        facts = _candidate_facts(path)
        raw_sha = facts["raw_source_sha256"]
        candidate_sha = facts["candidate_sha256"]
        if raw_sha in raw_hashes or candidate_sha in candidate_hashes:
            raise ValueError(
                f"场景 {scene.name} 复用了其他训练/测试的原始轨迹"
            )
        raw_hashes.add(raw_sha)
        candidate_hashes.add(candidate_sha)
        verified[scene.name] = (path, facts)

    workspace.mkdir(parents=True, exist_ok=False)
    rows = []
    for scene in scenes:
        candidate_path, facts = verified[scene.name]
        episode_path = workspace / "episodes" / scene.name
        episode_path.mkdir(parents=True, exist_ok=False)
        manifest = {
            "format": MANIFEST_FORMAT,
            "trajectory_id": f"carsim_experimental_{scene.name}",
            "condition_id": scene.condition_id,
            "split": scene.split,
            "actual_mu": MODEL_NOMINAL_MU,
            "candidate_path": str(candidate_path),
            "candidate_sha256": facts["candidate_sha256"],
            "raw_source_sha256": facts["raw_source_sha256"],
            "assumption_acknowledged": True,
            "source_kind": "carsim_experimental",
            "ready_for_wm_training": False,
            "note": (
                "实验性数据，仅用户显式允许后用于训练；"
                "未确认四轮净转矩/制动接口；"
                "同角命令不等于 CarSim 实测左右前轮角严格相等"
            ),
        }
        _atomic_json(
            episode_path / "experimental_episode.json", manifest
        )
        rows.append({
            "name": scene.name,
            "split": scene.split,
            "condition_id": scene.condition_id,
            "episode_path": str(episode_path),
            "raw_source_sha256": facts["raw_source_sha256"],
            "candidate_sha256": facts["candidate_sha256"],
        })

    registry = {
        "format": "wm_recovery_experimental_registry_v1",
        "status": "experimental_not_physically_validated",
        "explicit_opt_in_env": EXPERIMENT_ENV,
        "no_cross_split_raw_duplicates": True,
        "nominal_mu": MODEL_NOMINAL_MU,
        "episode_count": len(rows),
        "episodes": rows,
        "ready_for_wm_training": False,
    }
    _atomic_json(workspace / "registry.json", registry)
    return registry


def capture_all(sim_path, output_dir, bridge_type=None,
                candidate_creator=None):
    """标定 A03 → 18 条轨迹采集 → 评价覆盖率 → 建立分区清单。"""
    scenes = validate_scenarios()
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"输出目录已存在：{root}，拒绝覆盖")
    root.mkdir(parents=True, exist_ok=False)

    calibration = calibrate_steer(sim_path, root, bridge_type)
    candidates = {}
    reports = []
    for scene in scenes:
        candidate, report = capture_one(
            scene, sim_path, root, calibration,
            bridge_type=bridge_type,
            candidate_creator=candidate_creator,
        )
        candidates[scene.name] = candidate
        reports.append(report)

    coverage = summarize_coverage(reports)
    _atomic_json(root / "coverage_report.json", coverage)
    build_scene_workspace(candidates, root / "workspace")

    print("\n采集和数据集分区完成：", root)
    print("高横向响应覆盖率门槛：", coverage["passed"])
    if not coverage["passed"]:
        print(
            "警告：CarSim 实际横向加速度尚未达到设定的数据集覆盖要求；"
            "请查看 coverage_report.json，不要将大转角当成真实极限。"
        )
    return root / "workspace"


def require_opt_in():
    if os.getenv(EXPERIMENT_ENV) != "1":
        raise ValueError(
            f"训练实验性 CarSim 数据前，需要设置 {EXPERIMENT_ENV}=1"
        )


def _run_wm_cli(*arguments):
    """使用当前 PyCharm/Conda Python 调用现有 WM 命令入口。"""
    command = [sys.executable, "-m", "wm_recovery", *map(str, arguments)]
    print("\n执行:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train_from_workspace(workspace, output_dir, epochs,
                         allow_underexcited=False):
    """训练 + 标定 + 评价；不重新采集 CarSim。"""
    require_opt_in()
    workspace = Path(workspace).expanduser().resolve(strict=True)
    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}")
    if epochs < 1:
        raise ValueError("epochs 必须大于 0")

    coverage_path = workspace.parent / "coverage_report.json"
    if coverage_path.is_file():
        coverage = json.loads(
            coverage_path.read_text(encoding="utf-8")
        )
        if coverage.get("format") != COVERAGE_FORMAT:
            raise ValueError("未知覆盖率报告格式")
        if not coverage.get("passed") and not allow_underexcited:
            raise RuntimeError(
                "极限工况数据未通过项目定义的覆盖率门槛；"
                "请检查 coverage_report.json。"
                "如仅做流程调试，可显式加 --allow-underexcited"
            )

    # 兼容旧的四条轨迹 workspace 和本版 18 条轨迹 workspace。
    registry_path = workspace / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("format") != "wm_recovery_experimental_registry_v1":
        raise ValueError("不支持的实验 registry 格式")
    rows = registry["episodes"]
    splits = {"train": [], "validation": [], "test": []}
    for entry in rows:
        split = entry["split"]
        if split not in splits:
            raise ValueError(f"未知 split：{split}")
        path = Path(entry["episode_path"]).expanduser().resolve()
        if not (path / "experimental_episode.json").is_file():
            raise FileNotFoundError(path)
        splits[split].append(path)

    if (len(splits["train"]) < 2
        or not splits["validation"]
        or not splits["test"]):
        raise ValueError("数据集必须至少有 2 train、1 validation、1 test")

    out.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = out / "model"
    calibration_path = out / "calibration.json"
    evaluate_dir = out / "evaluation"

    all_paths = [
        *splits["train"], *splits["validation"], *splits["test"]
    ]
    _run_wm_cli("check-data", *all_paths)
    _run_wm_cli(
        "train", "--episodes", *splits["train"],
        "--output", checkpoint_dir, "--epochs", epochs
    )
    checkpoint = checkpoint_dir / "world_model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"训练输出 Checkpoint 缺失：{checkpoint}")

    _run_wm_cli(
        "calibrate", "--checkpoint", checkpoint,
        "--episodes", *splits["validation"],
        "--output", calibration_path,
    )

    # 现有 WM evaluate 接口只支持一个 --episode，依次独立评价各 test 轨迹。
    for i, episode in enumerate(splits["test"], start=1):
        _run_wm_cli(
            "evaluate", "--checkpoint", checkpoint,
            "--calibration", calibration_path,
            "--episode", episode,
            "--output", evaluate_dir / f"test_{i:02d}",
        )

    print("\n模型、标定和测试已完成：", out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="标准启发的 CarSim 操纵极限数据集 + WM 实验训练"
    )
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("plan", help="只预览，不启动 CarSim")

    cap = subs.add_parser(
        "capture", help="双向 0.3g 标定 + 18 条激励轨迹采集"
    )
    cap.add_argument("--sim", default=DEFAULT_SIM_PATH)
    cap.add_argument("--output", type=Path, required=True)

    tr = subs.add_parser(
        "train-existing", help="对已有训练集训练/标定/评价"
    )
    tr.add_argument("--workspace", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument(
        "--allow-underexcited", action="store_true",
        help="仅用于流程调试：覆盖率不足时仍允许实验训练"
    )

    full = subs.add_parser(
        "full", help="采集并训练；建议新手先分步运行"
    )
    full.add_argument("--sim", default=DEFAULT_SIM_PATH)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args(argv)
    if args.action == "plan":
        print_plan()
    elif args.action == "capture":
        capture_all(args.sim, args.output)
    elif args.action == "train-existing":
        train_from_workspace(
            args.workspace, args.output, args.epochs,
            allow_underexcited=args.allow_underexcited,
        )
    elif args.action == "full":
        require_opt_in()
        root = Path(args.output).expanduser().resolve()
        workspace = capture_all(args.sim, root)
        train_from_workspace(
            workspace, root / "model_run_01", args.epochs
        )


if __name__ == "__main__":
    main()
