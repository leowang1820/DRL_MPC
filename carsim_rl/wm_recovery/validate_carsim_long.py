# -*- coding: utf-8 -*-
"""真实 CarSim 长轨迹：采集、SI 转换、运动学、转矩、转角和 LoadMemory 代码审计。

在 carsim_rl 根目录执行：python -m wm_recovery.validate_carsim_long
不修改 CarSim 数据库、7DOF 方程或既有训练数据。
"""
import csv
import inspect
import json
import math
from pathlib import Path

import numpy as np

from wm_recovery.carsim_bridge import CarSimBridge
from wm_recovery.carsim_si import load_si_view

SIM_PATH = (r"E:\modelcodeE\carsim2022"
            r"\CarSim2022.1_Data\simfile.sim")
OUT = Path("artifacts/wm_recovery/carsim_long_validation_01")
DT = 0.01
DURATION = 6.0                 # 6 秒；先前每组实验只有 0.05 秒
WINDOW = 21                   # 21 点多项式数值求导


def command_at(t):
    """三段平缓激励：四轮纵向扭矩、转向、两者组合。"""
    segment = DURATION / 3
    q = t % segment
    smooth = math.sin(math.pi * q / segment) ** 2
    if t < segment:
        torque = 40 * smooth * math.sin(2 * math.pi * 0.55 * q)
        return [torque, torque, torque, torque, 0.0]
    if t < 2 * segment:
        steer_deg = 0.35 * smooth * math.sin(2 * math.pi * 0.45 * q)
        return [0.0, 0.0, 0.0, 0.0, math.radians(steer_deg)]
    torque = 25 * smooth * math.sin(2 * math.pi * 0.60 * q)
    steer_deg = 0.25 * smooth * math.sin(2 * math.pi * 0.50 * q)
    return [torque, torque, torque, torque, math.radians(steer_deg)]


def derivative(signal, dt, window=WINDOW):
    """局部三阶多项式求导；边缘点标记 NaN，不拿去计算误差。"""
    h = window // 2
    if len(signal) < window or window % 2 == 0:
        raise ValueError("轨迹太短，无法计算局部导数")
    x = np.arange(-h, h + 1) * dt
    weight = np.linalg.pinv(np.vander(x, 4, increasing=True))[1]
    result = np.full(len(signal), np.nan)
    result[h:-h] = np.convolve(signal, weight[::-1], mode="valid")
    return result


def compare(measured, predicted, mask):
    """输出零时延误差；额外检查反号和 ±3 采样点的潜在时延。"""
    x = measured[mask]
    y = predicted[mask]
    if len(x) < 30:
        raise ValueError("可用于比较的采样点不足")
    error = x - y
    std = float(np.std(x))
    corr = (float(np.corrcoef(x, y)[0, 1])
            if std > 1e-8 and float(np.std(y)) > 1e-8 else None)
    locations = np.where(mask)[0]
    lags = {}
    for lag in range(-3, 4):
        j = locations + lag
        valid = (j >= 0) & (j < len(measured))
        i = locations[valid]
        j = j[valid]
        inside = mask[j]
        i, j = i[inside], j[inside]
        if len(i) >= 30:
            lags[str(lag)] = float(np.sqrt(np.mean(
                (predicted[i] - measured[j]) ** 2)))
    best_lag = min(lags, key=lags.get) if lags else None

    # 这些是保守的工程初筛条件，不构成物理接口验证证明
    rmse = float(np.sqrt(np.mean(error ** 2)))
    screen_limit = max(
        0.15, 0.30 * float(np.percentile(np.abs(x), 95))
    )
    if float(np.sqrt(np.mean((x + y) ** 2))) < 0.7 * rmse:
        screening = "CHECK_AXIS_SIGN"
    elif std < 0.02:
        screening = "WEAK_ACCELERATION_EXCITATION"
    elif (best_lag not in (None, "0")
          and lags[best_lag] < 0.80 * rmse):
        screening = "CHECK_TIME_ALIGNMENT"
    elif rmse <= screen_limit and corr is not None and corr >= 0.7:
        screening = "NUMERICAL_SCREEN_GOOD"
    else:
        screening = "REVIEW_KINEMATICS"

    return {
        "screening": screening,
        "screen_rmse_limit_mps2": screen_limit,
        "samples": len(x),
        "rmse_mps2": float(np.sqrt(np.mean(error ** 2))),
        "mae_mps2": float(np.mean(np.abs(error))),
        "bias_mps2": float(np.mean(error)),
        "p95_abs_error_mps2": float(np.percentile(np.abs(error), 95)),
        "measured_std_mps2": std,
        "measured_range_mps2": [float(x.min()), float(x.max())],
        "correlation": corr,
        "rmse_if_sign_reversed_mps2": float(np.sqrt(np.mean((x + y) ** 2))),
        "best_lag_samples_diagnostic_only": best_lag,
    }


def inspect_loadmemory():
    """记录 LoadMemory 定义与调用位置，不据此宣称运行时物理已验证。"""
    report = [
        "静态源码检查：供人工复核初始化、微步更新与是否使用未来数据。",
        "",
    ]
    try:
        from dynamics7dof.model import LoadMemory
        report += ["=== LoadMemory 定义 ===", inspect.getsource(LoadMemory)]
    except (ImportError, AttributeError, OSError) as err:
        report.append(f"无法直接读取 LoadMemory 定义：{err}")
    for folder in (Path("dynamics7dof"), Path("wm_recovery")):
        for file in sorted(folder.rglob("*.py")):
            if file.name == "validate_carsim_long.py":
                continue
            try:
                lines = file.read_text(encoding="utf-8-sig").splitlines()
            except (UnicodeError, OSError):
                continue
            hits = [
                i for i, line in enumerate(lines)
                if "LoadMemory" in line or "load_memory" in line
            ]
            if hits:
                report.append(f"\n=== {file} ===")
            for i in hits[:40]:
                start, end = max(0, i - 2), min(len(lines), i + 3)
                report.extend(f"{k+1}: {lines[k]}" for k in range(start, end))
                report.append("---")
    (OUT / "loadmemory_references.txt").write_text(
        "\n".join(report), encoding="utf-8")


def run():
    """执行真正的 Solver 采集并完成所有可自动化的验证。"""
    if OUT.exists():
        raise FileExistsError(
            f"输出目录已存在：{OUT}。请把 OUT 改成一个新的 _02 目录。"
        )
    steps = round(DURATION / DT)
    if not math.isclose(steps * DT, DURATION, abs_tol=1e-10):
        raise ValueError("采样时长必须为 10 ms 的整数倍")

    print("开始真实 CarSim 采集：", DURATION, "s，", steps, "个控制区间")
    with CarSimBridge(SIM_PATH, prediction_dt_s=DT) as bridge:
        t_start = bridge.latest.time_s

        # 使用 Solver 实际停止时间，避免错误地越过 CarSim TSTOP
        solver = getattr(bridge, "_env", None)
        config = getattr(solver, "config", {})
        if isinstance(config, dict) and config.get("t_stop") is not None:
            t_stop = float(config["t_stop"])
            print("Solver TSTOP:", t_stop)
            if t_start + DURATION > t_stop + 1e-8:
                raise RuntimeError(
                    f"当前 Run 只能运行到 {t_stop}s；请先在 CarSim "
                    f"延长 TSTOP 至至少 {t_start + DURATION:.2f}s，"
                    "重新生成 simfile。"
                )

        for step in range(steps):
            state = bridge.step(command_at(step * DT))
            if (step + 1) % 100 == 0:
                print(f"已运行 {(step + 1) * DT:.2f}/{DURATION:.2f} s")
            if getattr(bridge, "_ended", False) and step + 1 < steps:
                raise RuntimeError("CarSim Solver 提前结束，请核实 TSTOP")
        bridge.save_raw(OUT)

    # SI 转换模块已通过原有测试；同时复核时间、Import、外加转矩回显
    record = load_si_view(OUT, expected_dt_s=DT)
    time = record["time_s"]
    state = record["state_si"]
    accel = record["reported_accel_mps2"]
    steer = record["reported_steer_rad"]
    command = record["command_si"]

    # 对比来自 CarSim 的质心加速度与平面车辆运动学加速度
    vx, vy, yaw = state[:, 0], state[:, 1], state[:, 2]
    ax_kin = derivative(vx, DT) - yaw * vy
    ay_kin = derivative(vy, DT) + yaw * vx

    valid = np.isfinite(ax_kin) & np.isfinite(ay_kin)
    elapsed = time - time[0]
    phase_length = DURATION / 3
    for boundary in (phase_length, 2 * phase_length):
        valid &= np.abs(elapsed - boundary) > 0.12

    ax_result = compare(accel[:, 0], ax_kin, valid)
    ay_result = compare(accel[:, 1], ay_kin, valid)

    # 三段工况分别给出误差，不依靠单个全程均值掩盖问题
    phase_results = {}
    for label, begin, end in (
        ("longitudinal", 0, phase_length),
        ("steering", phase_length, 2 * phase_length),
        ("mixed", 2 * phase_length, DURATION),
    ):
        phase_mask = valid & (elapsed > begin + 0.15) & (
            elapsed < end - 0.15
        )
        phase_results[label] = {
            "ax": compare(accel[:, 0], ax_kin, phase_mask),
            "ay": compare(accel[:, 1], ay_kin, phase_mask),
        }

    # 保留实际左右转角，另存“共同指令”构造的同角名义 7DOF 输入
    nominal_input = np.column_stack((
        command[:, 4], command[:, 4], command[:, :4]
    ))
    steer_diff_deg = np.rad2deg(steer[:, 0] - steer[:, 1])
    wheel_speed_mps = state[:, 3:7] * 0.353

    np.savez_compressed(
        OUT / "physics_check_si.npz",
        time_s=time, state_si=state, measured_ax_ay_mps2=accel,
        estimated_ax_ay_mps2=np.column_stack((ax_kin, ay_kin)),
        measured_front_steer_rad=steer,
        nominal_7dof_input_si=nominal_input,
        commands_si=command,
    )

    # 生成可用 Excel 直接打开的逐点运动学对照表
    with (OUT / "kinematics_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_s", "vx_mps", "vy_mps", "yaw_rads",
            "Ax_measured_mps2", "Ax_kinematic_mps2",
            "Ay_measured_mps2", "Ay_kinematic_mps2",
            "steer_L_deg", "steer_R_deg", "include_in_error",
        ])
        for i in range(len(time)):
            writer.writerow([
                time[i], vx[i], vy[i], yaw[i],
                accel[i, 0], ax_kin[i], accel[i, 1], ay_kin[i],
                np.rad2deg(steer[i, 0]),
                np.rad2deg(steer[i, 1]), bool(valid[i]),
            ])

    # 如果环境有 matplotlib，额外保存两张对比图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key, actual, estimated in (
            ("ax", accel[:, 0], ax_kin),
            ("ay", accel[:, 1], ay_kin),
        ):
            fig, axis = plt.subplots(figsize=(10, 4))
            axis.plot(time, actual, label="CarSim")
            axis.plot(time, estimated, label="Kinematic estimate")
            axis.set(xlabel="Time (s)", ylabel="Acceleration (m/s^2)",
                     title=f"{key.upper()} consistency")
            axis.legend()
            axis.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / f"{key}_comparison.png", dpi=150)
            plt.close(fig)
    except ImportError:
        print("未安装 matplotlib，跳过绘图；CSV 和 JSON 不受影响。")

    inspect_loadmemory()

    report = {
        "status": "long_capture_kinematic_screen_completed",
        "duration_s": float(time[-1] - time[0]),
        "samples": len(time),
        "import_and_time_checked_by": "wm_recovery.carsim_si.load_si_view",
        "reported_torque_matches_import":
            record["report"]["reported_torque_matches_import"],
        "max_torque_echo_error_nm":
            record["report"]["max_torque_echo_error_nm"],
        "ax_comparison": ax_result,
        "ay_comparison": ay_result,
        "phase_comparisons": phase_results,
        "max_left_right_steer_difference_deg":
            float(np.max(np.abs(steer_diff_deg))),
        "wheel_linear_speed_minus_vx_median_mps":
            np.median(wheel_speed_mps - vx[:, None],
                      axis=0).astype(float).tolist(),
        "ax_ay_physical_semantics_fully_verified": False,
        "loadmemory_runtime_causality_verified": False,
        "nominal_input_is_equal_angle_command_approximation": True,
        "car_vehicle_parameters_and_mu": "user_confirmed",
        "ready_for_wm_training": False,
        "interpretation": [
            "运动学使用平面质心公式，完整三维滚转、俯仰等项未建模",
            "导数使用离线邻域采样，仅作验证，不能输入在线预测器",
            "LoadMemory 来源与微步更新仍需要核对静态调用报告",
            "名义同角转向为建模近似，真实两个转角仍分别保存",
        ],
    }
    (OUT / "physics_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n======== 完成：自动检查报告 ========")
    print("目录：", OUT.resolve())
    print("转矩回显：", report["reported_torque_matches_import"])
    print("最大实际转角差(deg)：",
          report["max_left_right_steer_difference_deg"])
    for key, result in (("Ax", ax_result), ("Ay", ay_result)):
        print(key, "RMSE(m/s2)=", result["rmse_mps2"],
              "Corr=", result["correlation"],
              "Best lag(samples)=", result["best_lag_samples_diagnostic_only"],
              "Screen=", result["screening"])
    print("报告：physics_validation.json")
    print("明细：kinematics_comparison.csv")
    print("LoadMemory：loadmemory_references.txt")
    print("可直接用于正式 WM 训练：False（尚未自动验证物理定义与微步因果性）")


if __name__ == "__main__":
    run()
