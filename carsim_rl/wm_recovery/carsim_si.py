"""CarSim 原始采集记录 -> SI 单位视图及审计报告（不生成训练轨迹）。

在 carsim_rl 目录执行：
    python -m wm_recovery.carsim_si artifacts/wm_recovery/carsim_steer_01
也可以加：
    --output artifacts/wm_recovery/carsim_steer_01_si_audit

本模块不启动 CarSim、不修改原始记录、不强制左右实际转角相等。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .carsim_export_schema import EXPORT_NAMES
from .controls import DIRECT_IMPORT_NAMES


GRAVITY_MPS2 = 9.80665
RAW_FORMAT = "wm_recovery_carsim_raw_capture_v1"

# 七维状态对应原始 Export 的名称
STATE_NAMES = (
    "Vx", "Vy", "AVz",
    "AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2",
)
STEER_NAMES = ("Steer_L1", "Steer_R1")
TORQUE_NAMES = ("My_US_L1", "My_US_R1", "My_US_L2", "My_US_R2")
ACCEL_NAMES = ("Ax", "Ay")
STATE_FACTORS = np.array([
    1 / 3.6, 1 / 3.6, math.pi / 180,
    2 * math.pi / 60, 2 * math.pi / 60,
    2 * math.pi / 60, 2 * math.pi / 60,
], dtype=np.float64)


def _columns(names):
    """按照固定的 61 通道定义查询位置，避免重复 Steer_SW 覆盖其他变量。"""
    schema = tuple(EXPORT_NAMES)
    result = []
    for name in names:
        if schema.count(name) != 1:
            raise ValueError(f"需要唯一 Export：{name}，实际出现 {schema.count(name)} 次")
        result.append(schema.index(name))
    return result


def load_si_view(source, expected_dt_s=0.01):
    """读入原始记录，返回 SI 数组和只读审计信息。

    注意：实际转向角是 CarSim 输出值，不等于 Python 命令角度。
    Ax/Ay 的坐标系、参考点和重力分量尚待物理确认。
    """
    source = Path(source).expanduser().resolve()
    raw_file = source / "raw_capture.npz" if source.is_dir() else source
    meta_file = raw_file.parent / "metadata.json"

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("format") != RAW_FORMAT:
        raise ValueError("不是 carsim_bridge.py 生成的原始记录格式")
    if tuple(metadata.get("export_names", ())) != tuple(EXPORT_NAMES):
        raise ValueError("Export 名称或 61 通道顺序与当前定义不一致")
    if tuple(metadata.get("import_names", ())) != tuple(DIRECT_IMPORT_NAMES):
        raise ValueError("Import 名称或六通道顺序与当前定义不一致")

    with np.load(raw_file, allow_pickle=False) as source_data:
        keys = ("time_s", "raw_exports", "command", "import_values")
        missing = set(keys) - set(source_data.files)
        if missing:
            raise ValueError(f"原始记录缺少字段：{sorted(missing)}")
        time_s, raw, commands, imports = [
            np.asarray(source_data[key], dtype=np.float64).copy()
            for key in keys
        ]

    if time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("至少需要初始状态和一个控制区间")
    n = len(time_s) - 1
    if raw.shape != (n + 1, len(EXPORT_NAMES)):
        raise ValueError(f"原始 Export 形状错误：{raw.shape}")
    if commands.shape != (n, 5) or imports.shape != (n, 6):
        raise ValueError("控制指令应为 (N,5)，CarSim Import 应为 (N,6)")
    if not all(np.isfinite(a).all() for a in (time_s, raw, commands, imports)):
        raise ValueError("存在 NaN 或无穷大")

    # 每一组 command[i] 对应 [time_s[i], time_s[i+1]) 区间
    dt = np.diff(time_s)
    if not np.all(dt > 0) or not np.allclose(
        dt, expected_dt_s, rtol=0, atol=1e-9
    ):
        raise ValueError("采样时间未按指定的 10 ms 周期严格对齐")

    # Python 控制量：四轮 N-m、前轮 rad；CarSim 接受四轮 N-m、两个 deg
    steer_deg = np.rad2deg(commands[:, 4])
    expected_imports = np.column_stack((
        commands[:, :4], steer_deg, steer_deg
    ))
    if not np.allclose(imports, expected_imports, rtol=0, atol=1e-8):
        raise ValueError("保存的六维 Import 与五维 Python 指令不匹配")

    # 按已确认的单位转换状态、真实转角和加速度
    state_si = raw[:, _columns(STATE_NAMES)] * STATE_FACTORS
    reported_steer_rad = raw[:, _columns(STEER_NAMES)] * (math.pi / 180)
    reported_torque_nm = raw[:, _columns(TORQUE_NAMES)].copy()
    reported_accel_mps2 = raw[:, _columns(ACCEL_NAMES)] * GRAVITY_MPS2

    # 只审计 reported My_US 与输入是否一致；不把它称为车轮总转矩
    torque_error = float(np.max(np.abs(
        reported_torque_nm[1:] - imports[:, :4]
    )))
    torque_echo_ok = bool(np.allclose(
        reported_torque_nm[1:], imports[:, :4], rtol=0, atol=1e-7
    ))

    # 原始 7DOF 当前要求左右实际角度几乎严格相等
    delta = reported_steer_rad[:, 0] - reported_steer_rad[:, 1]
    strict_equal = bool(np.allclose(
        reported_steer_rad[:, 0], reported_steer_rad[:, 1],
        rtol=0, atol=1e-10
    ))

    report = {
        "status": "si_converted_physics_not_validated",
        "source_path": str(raw_file),
        "samples": n + 1,
        "control_intervals": n,
        "sample_dt_s": float(expected_dt_s),
        "unit_mapping": {
            "Vx_Vy": "km/h -> m/s",
            "AVz": "deg/s -> rad/s",
            "AVy_wheels": "rpm -> rad/s",
            "Steer_L1_R1": "deg -> rad",
            "Ax_Ay": "g -> m/s^2",
            "My_US_wheels": "N-m -> N-m",
        },
        "reported_torque_matches_import": torque_echo_ok,
        "max_torque_echo_error_nm": torque_error,
        "steering_strictly_equal_for_7dof": strict_equal,
        "max_steering_difference_deg": float(
            np.rad2deg(np.max(np.abs(delta)))
        ),
        "max_steering_difference_rad": float(np.max(np.abs(delta))),
        "ax_ay_physical_definition_verified": False,
        "vehicle_parameter_alignment_verified": False,
        "actual_steering_physics_verified": False,
        "ready_for_wm_training": False,
        "warnings": [
            "仅完成数值单位换算，不能直接作为正式 WM Episode",
            "Ax/Ay 的坐标系、参考点和重力分量定义仍需核对",
            "左右前轮实际转角不能强制平均或自动视为相等",
            "My_US 不是已经验证的车轮净转矩",
        ],
    }

    return {
        "time_s": time_s,
        "state_si": state_si,
        "reported_steer_rad": reported_steer_rad,
        "reported_external_torque_nm": reported_torque_nm,
        "reported_accel_mps2": reported_accel_mps2,
        "command_si": commands,
        "import_native": imports,
        "report": report,
    }


def save_si_audit(source, output_dir, expected_dt_s=0.01):
    """另存审计文件；拒绝覆盖目录，不改原始数据。"""
    result = load_si_view(source, expected_dt_s)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    arrays = {k: v for k, v in result.items() if k != "report"}
    np.savez_compressed(output_dir / "si_audit.npz", **arrays)

    report = dict(result["report"])
    report["source_sha256"] = hashlib.sha256(
        Path(report["source_path"]).read_bytes()
    ).hexdigest()
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main():
    """提供单独的 -m 入口，不需要修改项目原有 __main__.py。"""
    parser = argparse.ArgumentParser(
        description="CarSim 61 通道原始数据 -> SI 单位审计"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    if args.output:
        report = save_si_audit(args.source, args.output, args.dt)
    else:
        report = load_si_view(args.source, args.dt)["report"]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
