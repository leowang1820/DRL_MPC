# -*- coding: utf-8 -*-
"""CarSim 候选轨迹 -> 名义 7DOF 离线接口审计，始终不生成正式训练集。

运行：
python -m wm_recovery.carsim_candidate_audit \
    artifacts/wm_recovery/carsim_candidate_01 \
    --output artifacts/wm_recovery/carsim_candidate_nominal_audit_01

不会启动 CarSim、修改已有文件或开始神经网络训练。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .offline import CandidateEpisode
from .dataset import audit_carsim_candidate


def run(source, output_dir, max_substep_s=0.0005):
    """在通过候选格式检查后进行名义模型预测；另存审计结果。"""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"输出目录已存在：{output_dir}，请改成新的 _02 路径"
        )

    candidate = CandidateEpisode.load(source)

    # 非训练诊断：从已观测状态做单步预测，物理载荷记忆跨区间延续。
    audit = audit_carsim_candidate(
        candidate, max_substep_s=max_substep_s
    )
    report = dict(audit.report)

    # My_Dr 只是已选 61 Export 中的内部驱动观测量，不能证明制动为零。
    drive = np.asarray(candidate.my_dr_native, dtype=float)
    drive_abs = np.max(np.abs(drive), axis=0)
    report["my_dr_native_max_abs_by_wheel"] = drive_abs.tolist()
    report["my_dr_native_all_zero_with_1e_minus_8_tolerance"] = bool(
        np.all(np.abs(drive) <= 1e-8)
    )
    report["actual_steer_max_difference_deg"] = float(
        np.rad2deg(np.max(
            np.abs(candidate.measured_steer[:, 0] -
                   candidate.measured_steer[:, 1])
        ))
    )
    report["friction_brake_verified_zero"] = False
    report["net_shaft_torque_verified"] = False
    report["note"] = (
        "My_Dr 原生输出接近零仅能说明该观测变量接近零；"
        "仍需确认制动器配置和轮端净转矩定义。"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "candidate_nominal_diagnostics.npz",
        time_s=candidate.time,
        state_si=candidate.state,
        nominal_input_si=candidate.nominal_input,
        measured_front_steer_rad=candidate.measured_steer,
        measured_ax_ay_mps2=candidate.measured_ax_ay,
        model_memory_at_start=audit.memory_at_start,
        model_nominal_next=audit.nominal_next,
        valid_interval_mask=audit.valid,
    )
    (output_dir / "candidate_nominal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="运行真实 CarSim 候选轨迹的名义 7DOF 接口审计"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-substep-s", type=float, default=0.0005
    )
    args = parser.parse_args()
    report = run(
        args.source, args.output, max_substep_s=args.max_substep_s
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
