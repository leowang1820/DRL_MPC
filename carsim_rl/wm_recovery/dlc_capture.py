# -*- coding: utf-8 -*-
"""CarSim 内置预瞄驾驶员 DLC：逐次运行并采集全部已配置 Export。

运行方式：
    python -m wm_recovery.dlc_capture \
        --sim "E:\\...\\40kmh\\simfile.sim" \
        --sim "E:\\...\\60kmh\\simfile.sim" \
        --output artifacts/wm_recovery/dlc_preview_01 \
        --sample-dt 0.01

前提：
1. 在 CarSim GUI 已选中内置预瞄驾驶员追踪 DLC 并成功 Run Math Model。
2. 此 DLC 的 SIM/PAR 是独立生成的平铺配置；不包含外部 IMPORT。
3. 需要采集的变量全部在 Export 中声明，名称及顺序由 inspection.py 读取。
4. 通过 dlc_env.make_preview_env 对现有 CarSimEnv 做零 Import 适配。
5. 不会发送油门、制动或转向命令；不修改 SIM/PAR，也不启动 GUI。
6. 记录 CarSim 原生单位，不猜测任何变量的单位或符号。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .inspection import inspect_simfile


FORMAT = "wm_recovery_dlc_preview_native_exports_v2"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_label(path, index):
    """避免不同 Run Control 都叫 simfile.sim 时发生输出目录冲突。"""
    part = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(path).parent.name).strip("_")
    return f"run_{index:03d}_{part or 'dlc'}"


def _validated_snapshot(values, n_export):
    """每个采样时刻显式复制缓冲区，防止 DLL 下一步修改旧数据。"""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (n_export,):
        raise ValueError(
            f"Export 维度变化：预期 {n_export}，实际 {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("CarSim Export 存在 NaN/Inf")
    return vector.copy()


def check_sampling_times(
    times, expected_dt_s, *, solver_done, tolerance_s=1e-7,
    terminal_partial_policy="split",
):
    """严格诊断 CarSim 时间戳；只可容纳最后一段短于采样间隔的终止步。

    参数：
    - solver_done: 求解器明确报告正常结束，才允许末尾短步。
    - split: 主数据保持等间隔；最后短步另存独立文件，不丢弃。
    - reject: 任何不规则间隔都拒绝。

    时间戳不由 Python 重新生成；绝不删除中间异常或插值补齐。
    返回中包含要写到主 NPZ 的帧数与完整采样统计。
    """
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
        raise ValueError("CarSim 时间戳必须为至少两个有限值的一维数组")
    if (not np.isfinite(expected_dt_s)) or expected_dt_s <= 0:
        raise ValueError("expected_dt_s 必须是正数")
    if (not np.isfinite(tolerance_s) or tolerance_s <= 0
        or tolerance_s > expected_dt_s * 1e-3):
        raise ValueError(
            "时间容差必须为正，且不超过 sample_dt 的 0.1%"
        )
    if terminal_partial_policy not in ("split", "reject"):
        raise ValueError("terminal_partial_policy 只能是 split 或 reject")

    dt = np.diff(t)
    if not np.all(dt > 0):
        bad = np.flatnonzero(dt <= 0)
        raise ValueError(
            "CarSim 时间未严格递增；前几个异常索引："
            + repr(bad[:10].tolist())
        )
    bad = np.flatnonzero(np.abs(dt - expected_dt_s) > tolerance_s)
    result = {
        "raw_sample_count": int(len(t)),
        "uniform_sample_count": int(len(t)),
        "expected_interval_s": float(expected_dt_s),
        "time_tolerance_s": float(tolerance_s),
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "dt_median_s": float(np.median(dt)),
        "nonuniform_interval_count": int(len(bad)),
        "terminal_partial_detected": False,
        "terminal_partial_dt_s": None,
        "terminal_partial_time_s": None,
        "all_recorded_timestamps_uniform": bool(len(bad) == 0),
    }
    if len(bad) == 0:
        return result

    # 仅放行最后一个时间区间的小幅截断，前面的每个区间必须严格正确。
    terminal_last = (
        len(bad) == 1 and bad[0] == len(dt)-1 and bool(solver_done)
        and 0 < dt[-1] < expected_dt_s - tolerance_s
    )
    if terminal_last and terminal_partial_policy == "split":
        result["uniform_sample_count"] -= 1
        result["terminal_partial_detected"] = True
        result["terminal_partial_dt_s"] = float(dt[-1])
        result["terminal_partial_time_s"] = float(t[-1])
        return result

    # 所有其他情况都拒绝，输出可供直接定位的时间戳与间隔。
    details = [
        {
            "i": int(i),
            "t0": round(float(t[i]), 12),
            "t1": round(float(t[i+1]), 12),
            "dt": round(float(dt[i]), 12),
        }
        for i in bad[:12]
    ]
    raise ValueError(
        "CarSim 采样时间异常（不会静默删除或重采样）。"
        f" 总帧数={len(t)}，期望 dt={expected_dt_s:.12g}s，"
        f" 实际 dt 范围=[{dt.min():.12g}, {dt.max():.12g}]s，"
        f" 中位数={np.median(dt):.12g}s，异常数量={len(bad)}，"
        f" Solver done={bool(solver_done)}，"
        f" 第一个异常时刻={details}，"
        f" 最后十个时间戳={t[-10:].tolist()}"
    )


def capture_dlc(sim_path, output_dir, *, sample_dt_s=0.01,
                max_time_s=120.0, max_samples=100_000,
                env_factory=None, inspect_fn=inspect_simfile,
                time_tolerance_s=1e-7, terminal_partial_policy="split"):
    """从 CarSim 内置驾驶员获取完整时序；仅接受零外部 Import 的 DLC。

    输出：
        dlc_exports.npz:
            time_s         (N,)    实际 CarSim 时间戳
            raw_exports    (N, M)  CarSim 原生通道
            export_names   (M,)    保留重复名称和准确顺序
        metadata.json:
            SIM/PAR 哈希、采样时间、输出名字、是否正常终止等

    不将原生 km/h、deg/s、g 等单位自动转换为 SI。
    """
    sim = Path(sim_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"输出目录已存在，不允许覆盖：{destination}")
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0:
        raise ValueError("sample_dt_s 必须为正")
    if not math.isfinite(max_time_s) or max_time_s <= 0:
        raise ValueError("max_time_s 必须为正")
    if isinstance(max_samples, bool) or max_samples <= 1:
        raise ValueError("max_samples 必须大于 1")

    # 在加载 CarSim DLL 之前，先进行只读接口检查。
    inventory = inspect_fn(sim)
    imports = inventory["imports"]
    names = [str(name) for name in inventory["exports"]]
    if imports:
        raise ValueError(
            "内置预瞄驾驶员 DLC 采集要求零个外部 Import。"
            "请复制独立 Run Control，移除六通道转矩/前轮角度外部控制，"
            "重新在 GUI 生成 SIM/PAR。不能沿用旧 CarSimBridge。"
        )
    if len(names) == 0:
        raise ValueError(
            "未声明任何 Export。请在 CarSim 中添加要采集的输出，"
            "重新运行 GUI 生成 flat SIM/PAR。"
        )

    if env_factory is None:
        # 此适配器专门支持 n_import=0；普通 CarSimEnv.reset()
        # 在常见版本中会拒绝零 Import。
        from .dlc_env import make_preview_env
        env_factory = make_preview_env

    env = None
    timestamps = []
    observations = []
    done = False
    end_info = {}
    config = {}
    try:
        env = env_factory(str(sim))
        initial = env.reset()
        config = dict(env.config)
        n_import = int(config["n_import"])
        n_export = int(config["n_export"])

        if n_import != 0:
            raise ValueError(
                f"Solver 实际报告 n_import={n_import}。"
                "为避免覆盖内置驾驶员，采集程序拒绝发送外部控制。"
            )
        if n_export != len(names):
            raise ValueError(
                f"Solver 实际 Export 数量 {n_export}，"
                f"与 flat 参数文件的 {len(names)} 个名称不匹配。"
            )

        solver_step = float(config["t_step"])
        if not math.isfinite(solver_step) or solver_step <= 0:
            raise ValueError("CarSim Solver t_step 无效")
        ratio = sample_dt_s / solver_step
        inner_steps = int(round(ratio))
        if inner_steps < 1 or not math.isclose(
            inner_steps * solver_step, sample_dt_s,
            rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                f"目标采集步长 {sample_dt_s} 不是 Solver "
                f"t_step={solver_step} 的整数倍"
            )

        t0 = float(env.t_current)
        if not math.isfinite(t0):
            raise ValueError("CarSim 初始时刻不是有限数")
        timestamps.append(t0)
        observations.append(_validated_snapshot(initial, n_export))
        done = bool(getattr(env, "done", False))

        # 关键：给 CarSim 发送的是长度 0 的 Import 数组！
        # 真实转向/驱动/制动全部由 CarSim 内置预瞄控制器生成。
        no_external_actions = np.empty((0,), dtype=np.float64)

        while not done:
            if len(timestamps) >= max_samples:
                raise RuntimeError("超过 max_samples，停止采集防止无限循环")
            if timestamps[-1] - t0 >= max_time_s - 1e-10:
                raise RuntimeError("超过 max_time_s，疑似仿真未正常结束")

            values, _, ended, info = env.control_step(
                no_external_actions,
                inner_steps=inner_steps,
            )
            info = info if isinstance(info, dict) else {}
            if info.get("error"):
                raise RuntimeError(f"CarSim Solver 报错：{info}")
            current_time = float(env.t_current)
            if (not math.isfinite(current_time)
                or current_time <= timestamps[-1]):
                raise ValueError(
                    "CarSim 时间没有向前推进；可能提前结束或 DLL 步进出错"
                )
            timestamps.append(current_time)
            observations.append(_validated_snapshot(values, n_export))
            done = bool(ended)
            end_info = info

        times = np.asarray(timestamps, dtype=np.float64)
        raw = np.stack(observations)
        dt_actual = np.diff(times)

        # 使用真实时间戳诊断，禁止将异常间隔强行标记成 10 ms。
        # 即使异常，也要保留这次昂贵的 CarSim 原始采集以便排障，
        # 但保存到 quarantine 文件名，绝不能混入训练数据。
        try:
            sampling = check_sampling_times(
                times, sample_dt_s, solver_done=done,
                tolerance_s=time_tolerance_s,
                terminal_partial_policy=terminal_partial_policy,
            )
        except ValueError as exc:
            destination.mkdir(parents=True, exist_ok=False)
            quarantine = destination / "INVALID_TIMING_do_not_train.npz"
            np.savez_compressed(
                quarantine,
                time_s=times.copy(),
                raw_exports=raw.copy(),
                export_names=np.asarray(names, dtype=str),
                interval_dt_s=dt_actual.copy(),
            )
            (destination / "INVALID_TIMING_details.txt").write_text(
                f"采样质量判定：不合格，请勿直接用于 10ms 训练。\n"
                f"配置要求的采样间隔：{sample_dt_s:.12g}s\n"
                f"CarSim Solver 内部步长：{solver_step:.12g}s\n"
                f"CarSim 是否正常结束：{done}\n"
                f"CarSim 返回信息：{end_info}\n"
                + str(exc) + "\n",
                encoding="utf-8",
            )
            raise ValueError(
                f"{exc}\n"
                f"原始时序已隔离保存：{quarantine}\n"
                "请不要将此隔离文件直接用于残差标签或训练。"
            ) from exc
        print(
            f"采样检查：{len(times)} 帧，"
            f"期望 dt={sample_dt_s:.9g}s，"
            f"实际中位 dt={sampling['dt_median_s']:.9g}s，"
            f"最小={sampling['dt_min_s']:.9g}s，"
            f"最大={sampling['dt_max_s']:.9g}s",
            flush=True,
        )
        if sampling["terminal_partial_detected"]:
            print(
                "检测到终止短步："
                f"前一时刻={times[-2]:.9f}s，"
                f"终止时刻={times[-1]:.9f}s，"
                f"末尾 dt={dt_actual[-1]:.9f}s；"
                "将短步单独保存，主 NPZ 只包含等间隔数据。",
                flush=True,
            )

    finally:
        if env is not None:
            env.close()

    destination.mkdir(parents=True, exist_ok=False)

    # 原始最后一帧即使不足一个采样间隔，也不会被丢弃。
    # 将其单独保存在 sidecar，防止 downstream 默认把短步当 10ms。
    if sampling["terminal_partial_detected"]:
        np.savez_compressed(
            destination / "terminal_partial_step.npz",
            time_s=np.asarray([times[-1]], dtype=np.float64),
            previous_time_s=np.asarray([times[-2]], dtype=np.float64),
            interval_dt_s=np.asarray([dt_actual[-1]], dtype=np.float64),
            raw_exports=raw[-1:].copy(),
            export_names=np.asarray(names, dtype=str),
        )

    n_uniform = sampling["uniform_sample_count"]
    times = times[:n_uniform]
    raw = raw[:n_uniform]
    np.savez_compressed(
        destination / "dlc_exports.npz",
        time_s=times,
        raw_exports=raw,
        export_names=np.asarray(names, dtype=str),
        interval_dt_s=np.diff(times),
    )
    metadata = {
        "format": FORMAT,
        "sim_path": str(sim),
        "sim_sha256": inventory.get("sim_sha256", _sha256(sim)),
        "parameter_path": inventory.get("parameter_path"),
        "parameter_sha256": inventory.get("parameter_sha256"),
        "solver_step_s": float(config["t_step"]),
        "recording_interval_s": float(sample_dt_s),
        "actual_initial_time_s": float(times[0]),
        "actual_final_time_s": float(times[-1]),
        "solver_final_time_s": float(
            sampling["terminal_partial_time_s"]
            if sampling["terminal_partial_detected"] else times[-1]
        ),
        "n_samples": int(len(times)),
        "raw_solver_samples": int(sampling["raw_sample_count"]),
        "sampling_diagnostics": sampling,
        "time_grid_uniform_0p01s": bool(
            np.allclose(np.diff(times), sample_dt_s,
                        atol=time_tolerance_s, rtol=0)
        ),
        "n_import": 0,
        "n_export": int(raw.shape[1]),
        "export_names": names,
        "duplicate_export_names_allowed": True,
        "internal_driver_intended": True,
        "internal_driver_mode_gui_verification_required": True,
        "termination_reported_done": bool(done),
        "termination_info": {
            str(key): str(value) for key, value in end_info.items()
        },
        "native_units_not_converted": True,
        "si_units_verified": False,
        "full_internal_state_coverage_claimed": False,
        "notes": [
            "保存的是所有已配置 Export，并非 CarSim 数学模型的每一个内部状态。",
            "采集时没有任何外部 Import 控制，DLC 转向由 CarSim 内置驾驶员负责。",
            "采集只验证时序与数值，预瞄控制功能仍需核对 GUI/轨迹输出。",
            "下游 NMPC 残差数据集仍需单位、转向角及等效力输入的物理对齐。",
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="逐次运行 CarSim 内置预瞄驾驶员 DLC，采集全部配置好的 Export"
    )
    parser.add_argument(
        "--sim", action="append", required=True,
        help="每个已生成的 DLC SIM 路径；可重复指定 --sim",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="全新的批量输出根目录，不允许覆盖",
    )
    parser.add_argument("--sample-dt", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument(
        "--time-tolerance", type=float, default=1e-7,
        help="时间戳误差容差，默认 1e-7 秒；不能超过 sample-dt 的 0.1%",
    )
    parser.add_argument(
        "--terminal-partial-policy", choices=("split", "reject"),
        default="split",
        help="split：终止短步单独保存；reject：所有短步均报错",
    )
    args = parser.parse_args(argv)

    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"输出根目录已存在，拒绝覆盖：{root}")

    # 对多条 CarSim 仿真配置预检查，禁止把同一个 SIM 当作不同实验。
    sim_paths = [Path(p).expanduser().resolve(strict=True) for p in args.sim]
    if len(set(sim_paths)) != len(sim_paths):
        raise ValueError("--sim 包含重复路径，请检查是否真的生成了独立 Run")
    root.mkdir(parents=True, exist_ok=False)

    reports = []
    for i, path in enumerate(sim_paths, 1):
        name = _safe_label(path, i)
        print(f"\n[{i}/{len(sim_paths)}] CarSim DLC: {path}", flush=True)
        item = capture_dlc(
            path, root / name,
            sample_dt_s=args.sample_dt, max_time_s=args.max_time,
            time_tolerance_s=args.time_tolerance,
            terminal_partial_policy=args.terminal_partial_policy,
        )
        reports.append({
            "directory": str(root / name),
            "sim_path": str(path),
            "samples": item["n_samples"],
            "exports": item["n_export"],
            "duration_s": item["actual_final_time_s"]
                - item["actual_initial_time_s"],
        })
        print(
            f"完成: {item['n_samples']} 帧、{item['n_export']} 个通道、"
            f"{reports[-1]['duration_s']:.3f}s，"
            f"Solver 终点 {item['solver_final_time_s']:.6f}s",
            flush=True,
        )
    (root / "batch_manifest.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
