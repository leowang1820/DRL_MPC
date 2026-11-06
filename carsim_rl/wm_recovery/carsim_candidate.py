# -*- coding: utf-8 -*-
"""CarSim 原始轨迹 -> 只读、可追溯的 World Model 数据候选。

这不是正式的 wm_recovery_trajectory_v1，也不是已经验证的训练 Episode。
区分三种量：
1. command_si：Python 发送的五维控制指令（N,5）。
2. nominal_7dof_input_si：六维名义模型输入（N,6），左右转角使用同一指令。
3. measured_steer_rad：CarSim 测得的两个真实转角（N+1,2），不强制相等。

运行方法：
python -m wm_recovery.carsim_candidate \
    artifacts/wm_recovery/carsim_long_validation_01 \
    --output artifacts/wm_recovery/carsim_candidate_01
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .carsim_si import load_si_view
from .carsim_export_schema import EXPORT_NAMES


FORMAT = "wm_recovery_carsim_candidate_v1"

# 四个非方向盘的车轮实际外加转矩原始输出名
MY_DR_NAMES = ("My_Dr_L1", "My_Dr_R1", "My_Dr_L2", "My_Dr_R2")


def _sha256_file(path):
    """对原始采集文件计算 SHA256，保证结果可追溯。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(arrays):
    """对物理数组计算内容指纹，防止简单重命名伪造新轨迹。"""
    digest = hashlib.sha256()
    for name, array in arrays.items():
        values = np.asarray(array, dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _my_dr_native(source):
    """保留四轮 My_Dr 原生输出，未确认其单位前不擅自转换。"""
    source = Path(source)
    raw_file = source / "raw_capture.npz" if source.is_dir() else source
    with np.load(raw_file, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw_exports"], dtype=np.float64)
    positions = []
    for name in MY_DR_NAMES:
        if tuple(EXPORT_NAMES).count(name) != 1:
            raise ValueError(f"Export {name} 不唯一，不能安全提取")
        positions.append(tuple(EXPORT_NAMES).index(name))
    return raw[:, positions].copy()


def create_candidate(source, output_dir):
    """生成候选数据和审计报告；拒绝覆盖目录和跳过单位检查。"""
    source = Path(source).expanduser().resolve()
    raw_file = source / "raw_capture.npz" if source.is_dir() else source
    if not raw_file.is_file():
        raise FileNotFoundError(f"找不到原始记录：{raw_file}")

    # 复用已验证的 SI 单位与时序检查模块
    view = load_si_view(source, expected_dt_s=0.01)
    time = view["time_s"]
    state = view["state_si"]
    command = view["command_si"]
    reported_steer = view["reported_steer_rad"]
    reported_torque = view["reported_external_torque_nm"]
    measured_accel = view["reported_accel_mps2"]
    original_imports = view["import_native"]
    my_dr_native = _my_dr_native(source)

    # 命令顺序：[T_FL,T_FR,T_RL,T_RR,delta_f_rad]
    # 名义 7DOF 输入：[delta_FL,delta_FR,T_FL,T_FR,T_RL,T_RR]
    # 两个 delta 都是共同指令；不是实际 CarSim 测得的前轮角度。
    delta = command[:, 4]
    nominal = np.column_stack((delta, delta, command[:, :4]))

    n = len(time) - 1
    if nominal.shape != (n, 6):
        raise ValueError("名义 7DOF 输入维度错误")
    if reported_steer.shape != (n + 1, 2):
        raise ValueError("实际前轮转角测量维度错误")
    if reported_torque.shape != (n + 1, 4):
        raise ValueError("四轮外加转矩测量维度错误")

    # 检查两个名义角度相等，同时不能覆盖原始实测值
    if not np.array_equal(nominal[:, 0], nominal[:, 1]):
        raise ValueError("名义输入应由共同转向指令生成")

    # 用“区间结束时刻的 My_US”核对该区间已发送的外加转矩
    torque_error = np.max(
        np.abs(reported_torque[1:] - original_imports[:, :4])
    )
    torque_echo_match = bool(torque_error <= 1e-7)

    arrays = {
        "time_s": time,
        "state_si": state,
        "command_si": command,
        "nominal_7dof_input_si": nominal,
        "measured_steer_rad": reported_steer,
        "reported_external_torque_nm": reported_torque,
        "reported_accel_mps2": measured_accel,
        "import_native": original_imports,
        "my_dr_native": my_dr_native,
        # 该数值是名义模型初始化约定，不是 CarSim 测得的 LoadMemory
        "nominal_initial_load_memory_mps2": np.zeros(2, dtype=np.float64),
    }
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError("候选数据存在非有限数值")

    # 从采集元数据继承事实，不推断其物理验证为真
    capture_meta = json.loads(
        (raw_file.parent / "metadata.json").read_text(encoding="utf-8")
    )

    measured_difference = (
        reported_steer[:, 0] - reported_steer[:, 1]
    )
    report = {
        "format": FORMAT,
        "status": "candidate_created_not_training_ready",
        "source_kind": capture_meta.get("source_kind", "unknown"),
        "raw_source": str(raw_file),
        "raw_source_sha256": _sha256_file(raw_file),
        "canonical_physical_arrays_sha256": _fingerprint(arrays),
        "capture_sim_sha256": capture_meta.get("sim_sha256"),
        "capture_parameter_sha256": capture_meta.get("parameter_sha256"),
        "samples": int(n + 1),
        "intervals": int(n),
        "dt_s": 0.01,
        "actual_mu": "user_confirmed_nominal_0.8_not_rechecked",
        "vehicle_parameter_alignment": "user_confirmed_not_rechecked",
        "split": "unassigned",
        "model_input_definition": (
            "equal_front_commanded_steer_plus_four_commanded_external_torques"
        ),
        "interval_alignment": "command_i_holds_from_t_i_to_t_i_plus_1",
        "measured_steer_definition": "two_independent_carSim_outputs_at_state_times",
        "torque_echo_matches": torque_echo_match,
        "max_torque_echo_error_nm": float(torque_error),
        "max_measured_steer_difference_rad": float(
            np.max(np.abs(measured_difference))
        ),
        "max_measured_steer_difference_deg": float(
            np.rad2deg(np.max(np.abs(measured_difference)))
        ),
        "load_memory_initialization": (
            "reference_7dof_zero_by_model_convention_not_carsim_ax_ay"
        ),
        "ax_ay_only_for_diagnostics": True,
        "other_wheel_actuation_verified_zero": False,
        "friction_brake_actuation_verified_zero": False,
        "load_memory_runtime_causality_verified": False,
        "ready_for_wm_training": False,
        "reasons_not_training_ready": [
            "现有 offline.py 的 V1 applied_input 表示实际输入，不能将名义指令混装",
            "需明确新数据契约，保留名义输入与真实左右前轮测量的区别",
            "尚无完整轮端内部传动与摩擦制动均为零的独立核验",
            "尚未分配独立 train/validation/test 工况及轨迹标识",
            "仍需独立验证零初始化的 7DOF 载荷记忆用于真实轨迹的适用范围",
        ],
        "note": (
            "本候选不能直接用于 Episode.load、train 或论文物理验证结论；"
            "禁止通过重命名文件或将验证标志强制改为 True 来绕过检查。"
        ),
    }

    # 所有校验完成后才建立目标目录，禁止意外覆盖既有实验
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(output_dir / "candidate.npz", **arrays)
    (output_dir / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main():
    """在项目根目录通过 python -m wm_recovery.carsim_candidate 调用。"""
    parser = argparse.ArgumentParser(
        description="将 6 Import / 61 Export 原始 CarSim 轨迹转换为可审计候选数据"
    )
    parser.add_argument("source", type=Path, help="含 raw_capture.npz 的目录")
    parser.add_argument(
        "--output", type=Path, required=True, help="新的、尚不存在的输出目录"
    )
    args = parser.parse_args()
    report = create_candidate(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
