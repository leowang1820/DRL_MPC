# -*- coding: utf-8 -*-
"""独立 CarSim 实验训练入口：保留严格 V1，另开显式隔离的候选轨迹通道。

工作流程：
1. 通过 carsim_candidate.py 获得 candidate.npz。
2. 本文件建立相互独立的 train/validation/test 清单。
3. WM_RECOVERY_EXPERIMENTAL_CARSIM=1 时，旧 Episode.load 才能读取清单。
4. Episode.applied 在此试验路径中是“名义 7DOF 同角命令”，不是
   CarSim 左右实测转角。该字段仅用作兼容旧神经网络张量维度。

运行一次完整试验性采集：
python -m wm_recovery.carsim_experimental pilot \
  --existing-candidate artifacts/wm_recovery/carsim_candidate_01 \
  --output artifacts/wm_recovery/carsim_experiment_01

此过程调用 CarSim Solver 采集三条新轨迹，不改变数据库或车辆参数。
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


EXPERIMENT_ENV = "WM_RECOVERY_EXPERIMENTAL_CARSIM"
MANIFEST_FORMAT = "wm_recovery_experimental_episode_manifest_v1"
DEFAULT_SIM_PATH = (
    r"E:\modelcodeE\carsim2022"
    r"\CarSim2022.1_Data\simfile.sim"
)
SPLITS = ("train", "validation", "test")
PILOT_NAMES = (
    ("train1", "train", "pilot_existing_train"),
    ("train2", "train", "pilot_new_train"),
    ("validation", "validation", "pilot_validation_inputs"),
    ("test", "test", "pilot_test_inputs"),
)


def _check_nonempty(value, name):
    """检查可追溯的轨迹/工况名称。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _candidate_facts(path):
    """从候选数据原始报告核对来源、完整性及固定的实验语义。"""
    from .offline import CandidateEpisode

    ep = CandidateEpisode.load(path)
    meta = ep.metadata

    if meta.get("source_kind") != "carsim_env":
        raise ValueError(
            f"{path} 不是由真实 CarSimEnv 记录的候选轨迹，"
            "模拟数据不能充当真实训练轨迹"
        )
    if meta.get("ready_for_wm_training") is not False:
        raise ValueError("候选报告不得自称已通过正式训练验证")
    if meta.get("dt_s") != 0.01:
        raise ValueError("当前实验采用固定 10 ms 时间栅格")
    if meta.get("model_input_definition") != (
        "equal_front_commanded_steer_plus_four_commanded_external_torques"
    ):
        raise ValueError("候选没有声明正确的名义同角模型输入")
    if meta.get("torque_echo_matches") is not True:
        raise ValueError("外加转矩回显不匹配，不能加入实验训练分区")

    return {
        "candidate": ep,
        "candidate_sha256": ep.source_sha256,
        "raw_source_sha256": _check_nonempty(
            meta.get("raw_source_sha256"), "raw_source_sha256"
        ),
    }


def load_experimental_episode(directory):
    """显式实验路径加载函数：只在用户设置环境开关时允许调用。"""
    from .offline import Episode, NOMINAL_MU

    if os.getenv(EXPERIMENT_ENV) != "1":
        raise ValueError(
            f"候选数据为实验性训练数据。需要先执行 "
            f"$env:{EXPERIMENT_ENV}='1'，"
            "未开启时保留原来的严格验证规则"
        )

    directory = Path(directory).expanduser().resolve()
    meta_path = directory / "experimental_episode.json"
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))

    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("不支持的实验清单版本")
    if manifest.get("split") not in SPLITS:
        raise ValueError("实验清单只支持 train、validation、test")
    if manifest.get("actual_mu") != NOMINAL_MU:
        raise ValueError("当前实验仅处理名义 mu=0.8 的真实轨迹")
    if manifest.get("assumption_acknowledged") is not True:
        raise ValueError("实验训练必须明确承认同角名义近似和净转矩未验证")
    _check_nonempty(manifest.get("trajectory_id"), "trajectory_id")
    _check_nonempty(manifest.get("condition_id"), "condition_id")

    candidate_path = Path(manifest["candidate_path"]).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = directory / candidate_path
    candidate_path = candidate_path.resolve(strict=True)

    facts = _candidate_facts(candidate_path)
    candidate = facts["candidate"]
    if (
        facts["candidate_sha256"] != manifest.get("candidate_sha256")
        or facts["raw_source_sha256"] != manifest.get("raw_source_sha256")
    ):
        raise ValueError("实验清单与候选物理数据来源指纹不一致")

    if len(candidate.time) - 1 < 100:
        raise ValueError("实验轨迹太短，无法构造足够的因果历史窗口")

    # 一律标记为实验性。即使用户已经人工确认参数，也不冒充核验标志。
    meta = {
        "format": "wm_recovery_experimental_episode_v1",
        "trajectory_id": manifest["trajectory_id"],
        "condition_id": manifest["condition_id"],
        "split": manifest["split"],
        "source_kind": "carsim_experimental",
        "nominal_mu": NOMINAL_MU,
        "actual_mu": NOMINAL_MU,
        "parameter_alignment": "user_confirmed_not_rechecked",
        "actuation_definition": (
            "external_command_as_nominal_model_torque_not_verified_net"
        ),
        "input_alignment": "held_over_interval_t_i_to_t_i_plus_1",
        "input_semantics": "nominal_equal_steer_command",
        "initial_memory_source": "reference_model_zero_then_force_carried",
        "experimental_training_only": True,
        "ready_for_wm_training": False,
        "nominal_steer_not_measured": True,
        "actual_measured_steer_preserved_in_candidate": True,
        "other_wheel_actuation_verified_zero": False,
        "friction_brake_actuation_verified_zero": False,
        "candidate_sha256": candidate.source_sha256,
        "raw_source_sha256": facts["raw_source_sha256"],
    }

    # 为兼容已有 World Model 张量约定，将名义 7DOF 指令放入
    # Episode.applied，字段命名仅是历史兼容，绝不代表实测 applied。
    # CarSim 两个真实转角仍完整保存在独立 candidate.npz。
    return Episode(
        time=candidate.time,
        state=candidate.state,
        applied=candidate.nominal_input,
        command=candidate.command,
        initial_memory=candidate.initial_memory,
        metadata=meta,
        source_sha256=candidate.source_sha256,
    )


def build_workspace(candidates, output_dir):
    """建立四个独立轨迹清单；禁止复制同一记录穿越数据集分区。

    参数 candidates：字典，键依次为 train1/train2/validation/test。
    返回路径结构，可直接传给原 wm_recovery train/calibrate/evaluate。
    """
    required = {item[0] for item in PILOT_NAMES}
    if set(candidates) != required:
        raise ValueError(f"必须提供 {sorted(required)} 四条候选轨迹")

    facts_by_name = {}
    seen_candidate = {}
    seen_raw = {}

    for name, split, condition_id in PILOT_NAMES:
        path = Path(candidates[name]).expanduser().resolve(strict=True)
        facts = _candidate_facts(path)

        for label, identifier, seen in (
            ("候选物理数组", facts["candidate_sha256"], seen_candidate),
            ("原始 CarSim 记录", facts["raw_source_sha256"], seen_raw),
        ):
            if identifier in seen:
                raise ValueError(
                    f"{label} 出现在多条轨迹中：{seen[identifier]} 和 {name}；"
                    "不能把一段记录复制成训练/验证/测试集"
                )
            seen[identifier] = name
        facts_by_name[name] = (path, split, condition_id, facts)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    episodes_dir = output_dir / "episodes"
    manifest_rows = []
    for name, (path, split, condition_id, facts) in facts_by_name.items():
        episode_dir = episodes_dir / name
        episode_dir.mkdir(parents=True, exist_ok=False)

        manifest = {
            "format": MANIFEST_FORMAT,
            "trajectory_id": f"carsim_experimental_{name}",
            "condition_id": condition_id,
            "split": split,
            "actual_mu": 0.8,
            "candidate_path": str(path),
            "candidate_sha256": facts["candidate_sha256"],
            "raw_source_sha256": facts["raw_source_sha256"],
            "assumption_acknowledged": True,
            "source_kind": "carsim_experimental",
            "ready_for_wm_training": False,
            "note": (
                "该清单仅在环境开关显式开启时可用于程序开发实验；"
                "名义输入不是实际前轮角度/轮端净转矩的物理证明"
            ),
        }
        (episode_dir / "experimental_episode.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_rows.append({
            "name": name, "split": split,
            "episode_path": str(episode_dir.resolve()),
            "raw_source_sha256": facts["raw_source_sha256"],
            "candidate_sha256": facts["candidate_sha256"],
        })

    registry = {
        "format": "wm_recovery_experimental_registry_v1",
        "status": "experimental_not_physically_validated",
        "explicit_opt_in_env": EXPERIMENT_ENV,
        "no_cross_split_raw_duplicates": True,
        "nominal_mu": 0.8,
        "episode_count": len(manifest_rows),
        "episodes": manifest_rows,
        "ready_for_wm_training": False,
    }
    (output_dir / "registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return registry


def _smooth_envelope(t, duration):
    """让每条激励在仿真起点/终点渐进，避免不必要的突然跳变。"""
    return math.sin(math.pi * t / duration) ** 2


def _pilot_command(t, duration, kind):
    """三个互不相同的小幅激励，分别用于第二训练、验证和测试轨迹。

    [四轮 N-m, 一个前轮 rad]。车辆运动的安全适用范围仍需独立评估。
    """
    s = _smooth_envelope(t, duration)

    if kind == "train2":
        base = 30 * math.sin(2*math.pi*0.49*t) * s
        diff = 9 * math.sin(2*math.pi*0.23*t) * s
        angle = 0.26 * math.sin(2*math.pi*0.38*t) * s
    elif kind == "validation":
        base = 25 * math.sin(2*math.pi*0.71*t) * s
        diff = 7 * math.sin(2*math.pi*0.41*t) * s
        angle = 0.31 * math.sin(2*math.pi*0.52*t) * s
    elif kind == "test":
        base = 32 * math.sin(2*math.pi*0.34*t) * s
        diff = 11 * math.sin(2*math.pi*0.57*t) * s
        angle = 0.40 * math.sin(2*math.pi*0.63*t) * s
    else:
        raise ValueError(f"未知激励类型：{kind}")

    # 同向基准转矩 + 左右车轮小幅差动，便于覆盖横摆工况
    return [
        base - diff, base + diff,
        0.9*base - 0.7*diff, 0.9*base + 0.7*diff,
        math.radians(angle),
    ]


def _capture_one(sim_path, duration, dt, kind, root):
    """每一条独立重置 CarSim，保存 61 通道原始记录并转换成候选。"""
    from .carsim_bridge import CarSimBridge
    from .carsim_candidate import create_candidate

    steps = int(round(duration / dt))
    if not math.isclose(steps*dt, duration, rel_tol=0, abs_tol=1e-10):
        raise ValueError("采样时长必须是 10ms 的整数倍")

    raw_dir = root / f"raw_{kind}"
    candidate_dir = root / f"candidate_{kind}"
    print(f"\n采集 {kind}：时长 {duration:g}s，{steps} 个控制区间")

    with CarSimBridge(sim_path, prediction_dt_s=dt) as bridge:
        stop = bridge._env.config.get("t_stop")
        if stop is not None:
            start = float(bridge.latest.time_s)
            if start + duration > float(stop) + 1e-8:
                raise RuntimeError(
                    f"CarSim TSTOP={stop}s 不足以完成 {duration}s 采集，"
                    "请在当前 Run 中延长结束时间并重新生成 simfile"
                )
        for i in range(steps):
            bridge.step(_pilot_command(i*dt, duration, kind))
            if (i+1) % 100 == 0:
                print(f"  {kind}: {(i+1)*dt:.1f}/{duration:g} s")
        bridge.save_raw(raw_dir)

    report = create_candidate(raw_dir, candidate_dir)
    if report.get("torque_echo_matches") is not True:
        raise RuntimeError(
            f"{kind} 的实际转矩回显不一致，拒绝建立实验性训练清单"
        )
    return candidate_dir


def pilot(existing_candidate, output_dir, sim_path=DEFAULT_SIM_PATH,
          duration_s=6.0):
    """复用已采集的真实候选 train1，再采三条并创建独立分区。

    生成的是开发实验用数据集；不能冒充严谨的跨工况论文验证。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"目录已存在：{output_dir}，请换成 _02 等新目录"
        )

    _candidate_facts(existing_candidate)
    if duration_s < 3.0 or not math.isclose(
        duration_s / 0.01, round(duration_s/0.01), abs_tol=1e-10
    ):
        raise ValueError("每条实验轨迹至少 3 秒，长度必须为 10 ms 整数倍")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = {
        "train1": Path(existing_candidate).expanduser().resolve(strict=True)
    }
    for kind in ("train2", "validation", "test"):
        candidates[kind] = _capture_one(
            sim_path, duration_s, 0.01, kind, output_dir
        )

    registry = build_workspace(candidates, output_dir / "workspace")
    print("\n实验清单创建完成：", output_dir / "workspace" / "registry.json")
    print("实验训练数据由两条训练轨迹 + 一条验证 + 一条测试组成。")
    print("请明确区分：数据集分区独立 ≠ 不同车辆/不同附着条件的独立外推验证。")
    return registry


def main():
    """命令行入口：pilot 自动采集；build 用已有四条候选直接分区。"""
    parser = argparse.ArgumentParser(
        description="CarSim 实验性训练数据集：真实独立轨迹 + 显式近似"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    cmd_pilot = sub.add_parser("pilot", help="采集三条新轨迹并创建分区")
    cmd_pilot.add_argument("--existing-candidate", type=Path, required=True)
    cmd_pilot.add_argument("--output", type=Path, required=True)
    cmd_pilot.add_argument("--sim", default=DEFAULT_SIM_PATH)
    cmd_pilot.add_argument("--duration", type=float, default=6.0)

    cmd_build = sub.add_parser("build", help="直接对四条已有候选分区")
    for key in ("train1", "train2", "validation", "test"):
        cmd_build.add_argument(f"--{key}", type=Path, required=True)
    cmd_build.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.action == "pilot":
        result = pilot(
            args.existing_candidate, args.output,
            sim_path=args.sim, duration_s=args.duration,
        )
    else:
        paths = {
            name: getattr(args, name)
            for name in ("train1", "train2", "validation", "test")
        }
        result = build_workspace(paths, args.output)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
