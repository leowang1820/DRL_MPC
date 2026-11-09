# -*- coding: utf-8 -*-
"""只运行一条 CarSim 开环工况，便于在 CarSim Video 中检查车辆响应。

用法（在项目根目录执行）：
    python -m wm_recovery.preview_openloop --scene train_step_left \
        --output artifacts/wm_recovery/preview_step_left_01

说明：
1. 不训练 World Model，不运行全部 18 条工况。
2. 调用已有 CarSimBridge.step()，每 0.01 s 发送一次五维控制指令。
3. 保存 Python 读取的原始数据，并打印最大横向加速度等基本响应。
4. CarSim Video 需要 Solver 自己生成的动画结果文件；本脚本不伪造该文件。
5. 如未提供 --a03-deg，使用初始车速推算一个近似 A03 以快速预览；
   这个数不是经 CarSim 0.3g 实测标定的参考值，不能用于标准符合性结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


# 从当前 wm_recovery 包中导入场景配置
from .scenario_config import (
    DEFAULT_SIM_PATH,
    DT_S,
    SCENARIOS,
    command_at,
    roadwheel_delta_for_ay,
)

SCENE_BY_NAME = {scene.name: scene for scene in SCENARIOS}


def run_preview(scene_name, output_dir, sim_path=DEFAULT_SIM_PATH,
                a03_deg=None, bridge_factory=None):
    """按一条已定义工况发送控制并记录车辆输出。"""
    if scene_name not in SCENE_BY_NAME:
        raise ValueError(f"未知场景：{scene_name}，请先运行 scenario_runner plan")

    scene = SCENE_BY_NAME[scene_name]
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"输出文件夹已经存在，请换新名称：{root}")

    if bridge_factory is None:
        from .carsim_bridge import CarSimBridge
        bridge_factory = CarSimBridge

    if a03_deg is not None:
        a03_deg = float(a03_deg)
        if not math.isfinite(a03_deg) or a03_deg <= 0:
            raise ValueError("--a03-deg 必须是大于零的有限数")

    # 输出目录不覆盖既有数据。
    root.mkdir(parents=True, exist_ok=False)

    rows = []
    with bridge_factory(str(sim_path), prediction_dt_s=DT_S) as bridge:
        init = bridge.latest
        speed_kmh = float(init.data.Vx)  # 已核对的 CarSim 原生单位：km/h
        if not math.isfinite(speed_kmh) or not 30 <= speed_kmh <= 110:
            raise ValueError(
                f"当前 CarSim 初始速度 {speed_kmh:.2f} km/h，"
                "请先在 CarSim Run 中设置约 40~80 km/h 的正向初始速度。"
            )

        if a03_deg is None:
            # 运动学近似：仅用于快速观察，不等于实测转向参考角。
            a03_deg = roadwheel_delta_for_ay(0.3, speed_kmh)
            a03_source = "kinematic_estimate_not_measured"
        else:
            a03_source = "user_supplied_not_verified_by_this_script"

        print(f"场景：{scene.name}，预计运行 {scene.duration_s:g} 秒")
        print(f"CarSim 初始纵向速度：{speed_kmh:.2f} km/h")
        print(f"本次采用的参考前轮角 A03：{a03_deg:.3f} 度")
        print("注意：本脚本不直接生成 CarSim Video 文件。")

        steps = round(scene.duration_s / DT_S)
        for i in range(steps):
            t = i * DT_S
            control = command_at(scene, t, a03_deg)
            # 真正把转矩、前轮转角发送给 CarSim 的位置。
            snapshot = bridge.step(control)

            d = snapshot.data
            rows.append({
                "time_s": float(snapshot.time_s),
                "command_steer_deg": math.degrees(control[4]),
                "actual_steer_l_deg": float(d.Steer_L1),
                "actual_steer_r_deg": float(d.Steer_R1),
                "vx_native_kmh": float(d.Vx),
                "ay_native_g": float(d.Ay),
                "yaw_native_deg_s": float(d.AVz),
            })
            if (i + 1) % 100 == 0 or i + 1 == steps:
                print(f"  CarSim 运行到 {snapshot.time_s:.2f} s")

        # 仅保存原始采集数据，不做候选转换和 World Model 训练。
        bridge.save_raw(root / "raw")

    # 离开 with 后已调用 CarSimBridge.close()，允许 Solver 完成收尾。
    csv_path = root / "response_preview.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scene": scene.name,
        "simfile": str(Path(sim_path).expanduser().resolve()),
        "duration_s": scene.duration_s,
        "control_interval_s": DT_S,
        "samples": len(rows),
        "initial_speed_kmh": speed_kmh,
        "reference_a03_deg": a03_deg,
        "reference_a03_source": a03_source,
        "peak_abs_steer_command_deg": max(
            abs(r["command_steer_deg"]) for r in rows
        ),
        "peak_abs_actual_steer_deg": max(
            max(abs(r["actual_steer_l_deg"]),
                abs(r["actual_steer_r_deg"])) for r in rows
        ),
        "peak_abs_lateral_accel_g": max(
            abs(r["ay_native_g"]) for r in rows
        ),
        "peak_abs_yaw_rate_deg_s": max(
            abs(r["yaw_native_deg_s"]) for r in rows
        ),
        "carsim_video_ready": "not_verified",
        "note": (
            "仅确认 CarSimBridge 采集完成；是否可在 CarSim Video 播放，"
            "取决于 CarSim Solver 是否写出本次 Run 对应的动画结果文件。"
        ),
    }
    (root / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n单场景运行完成！")
    print("响应数据：", csv_path)
    print("原始输出：", root / "raw" / "raw_capture.npz")
    print("横向加速度峰值：",
          f'{summary["peak_abs_lateral_accel_g"]:.3f} g')
    print("实际前轮转角峰值：",
          f'{summary["peak_abs_actual_steer_deg"]:.3f} 度')
    print(
        "下一步：返回 CarSim 原先生成此 simfile 的 Run Control，"
        "尝试 Video + Plot；确认播放的是刚才的结果，不是旧 Run。"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="只运行一条 CarSim 开环工况，不训练神经网络"
    )
    parser.add_argument(
        "--scene", default="train_step_left",
        choices=sorted(SCENE_BY_NAME),
        help="scene_config.py 中已有的场景名称",
    )
    parser.add_argument(
        "--sim", default=DEFAULT_SIM_PATH,
        help="CarSim GUI 生成的 simfile.sim",
    )
    parser.add_argument(
        "--output", required=True,
        help="新的、尚不存在的预览输出目录",
    )
    parser.add_argument(
        "--a03-deg", type=float, default=None,
        help="可选：给出已标定的 0.3g 参考前轮角；默认使用运动学近似",
    )
    args = parser.parse_args()
    run_preview(
        args.scene,
        args.output,
        sim_path=args.sim,
        a03_deg=args.a03_deg,
    )


if __name__ == "__main__":
    main()
