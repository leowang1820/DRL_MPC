# -*- coding: utf-8 -*-
"""统一 CarSim 场景采集和 World Model 实验入口。

首次采集四条独立轨迹：
    python -m wm_recovery.scenario_runner plan
    python -m wm_recovery.scenario_runner capture \
        --output artifacts/wm_recovery/scenario_batch_01

已有 workspace 进行实验性 WM 训练、标定和评价：
    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"
    python -m wm_recovery.scenario_runner train-existing \
        --workspace artifacts/wm_recovery/scenario_batch_01/workspace \
        --output artifacts/wm_recovery/scenario_batch_01/model_run_01 \
        --epochs 1

或者在一个新的目录中一次性采集+训练：
    python -m wm_recovery.scenario_runner full \
        --output artifacts/wm_recovery/scenario_batch_02 --epochs 1

说明：
1. 仿真采集仍使用已经验证过的 CarSimBridge，不直接调用 DLL。
2. 输出目录必须不存在，绝不覆盖以前的实验。
3. 实验训练必须由用户显式设置环境开关。
4. 四条轨迹互不复用，独立建立 train/validation/test。
5. 不声称完成了四轮净转矩、制动或其他物理接口验证。
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
    SCENARIOS,
    Scenario,
    command_at,
    validate_scenarios,
)


EXPERIMENT_ENV = "WM_RECOVERY_EXPERIMENTAL_CARSIM"


def print_plan():
    """只打印场景计划，不启动 CarSim，也不写入任何文件。"""
    scenes = validate_scenarios()
    for scene in scenes:
        print(json.dumps(
            asdict(scene),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))
    print("采集阶段不需要实验训练开关，训练阶段必须显式启用。")


def capture_one(scene: Scenario, sim_path, root,
                bridge_type=None, candidate_creator=None):
    """用 CarSimBridge 采集一条轨迹，并生成 candidate.npz。

    bridge_type/candidate_creator 用于无 DLL 测试；通常不必传入。
    """
    if bridge_type is None:
        from .carsim_bridge import CarSimBridge
        bridge_type = CarSimBridge
    if candidate_creator is None:
        from .carsim_candidate import create_candidate
        candidate_creator = create_candidate

    root = Path(root)
    raw_dir = root / f"raw_{scene.name}"
    candidate_dir = root / f"candidate_{scene.name}"
    count = round(scene.duration_s / DT_S)
    if not math.isclose(count * DT_S, scene.duration_s,
                        rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("轨迹时长不是 10 ms 的整数倍")

    print(f"\n采集 {scene.name}：{count} 个控制区间")
    with bridge_type(sim_path, prediction_dt_s=DT_S) as bridge:
        # CarSim Run 的停止时间必须覆盖整个场景。
        env = getattr(bridge, "_env", None)
        config = getattr(env, "config", {})
        if isinstance(config, dict) and config.get("t_stop") is not None:
            end = float(config["t_stop"])
            required_end = float(bridge.latest.time_s) + scene.duration_s
            if required_end > end + 1e-8:
                raise RuntimeError(
                    f"CarSim TSTOP={end}s 不足以运行到 {required_end}s；"
                    "请调整 CarSim Run 并重新生成 simfile"
                )

        # 每 10 ms 计算控制，发送一次六维 Import。
        for index in range(count):
            t = index * DT_S
            command = command_at(scene, t)
            bridge.step(command)

            if (index + 1) % 100 == 0 or index + 1 == count:
                print(
                    f"  {scene.name}: {(index + 1) * DT_S:.2f}/"
                    f"{scene.duration_s:.2f} s"
                )

            if getattr(bridge, "_ended", False) and index + 1 < count:
                raise RuntimeError("CarSim Solver 在场景完成前提前结束")

        # N+1 个状态，N 个控制区间。
        bridge.save_raw(raw_dir)

    # 使用原有候选数据模块进行单位转换和可追溯记录。
    report = candidate_creator(raw_dir, candidate_dir)
    if report.get("torque_echo_matches") is not True:
        raise RuntimeError(
            f"{scene.name} 外加转矩回显未通过检查；"
            "原始记录已保存，但拒绝将其加入实验训练数据"
        )
    return candidate_dir


def build_scene_workspace(candidate_paths, workspace):
    """由四条独立候选轨迹生成实验性训练/验证/测试清单。

    复用既有 _candidate_facts 对候选数组进行校验；
    单独生成清单是为了使用 scenario_config 的真实 condition_id，
    而不是旧 pilot 固定写死的工况名称。
    """
    from .carsim_experimental import _candidate_facts, MANIFEST_FORMAT
    from .offline import NOMINAL_MU

    scenes = validate_scenarios()
    if set(candidate_paths) != {scene.name for scene in scenes}:
        raise ValueError("必须提供 train1、train2、validation、test 四条候选")

    workspace = Path(workspace).expanduser().resolve()
    if workspace.exists():
        raise FileExistsError(f"workspace 已经存在：{workspace}")

    # 先完成所有校验，再开始创建目录。
    facts = {}
    raw_ids = {}
    candidate_ids = {}
    for scene in scenes:
        path = Path(candidate_paths[scene.name]).expanduser().resolve(
            strict=True
        )
        check = _candidate_facts(path)
        raw_sha = check["raw_source_sha256"]
        candidate_sha = check["candidate_sha256"]

        if raw_sha in raw_ids or candidate_sha in candidate_ids:
            raise ValueError(
                f"{scene.name} 与其他轨迹使用相同的 CarSim 数据；"
                "训练、验证、测试不得复用同一条原始记录"
            )
        raw_ids[raw_sha] = scene.name
        candidate_ids[candidate_sha] = scene.name
        facts[scene.name] = (path, check)

    workspace.mkdir(parents=True, exist_ok=False)
    rows = []
    for scene in scenes:
        path, check = facts[scene.name]
        episode_dir = workspace / "episodes" / scene.name
        episode_dir.mkdir(parents=True, exist_ok=False)

        manifest = {
            "format": MANIFEST_FORMAT,
            "trajectory_id": f"carsim_experimental_{scene.name}",
            "condition_id": scene.condition_id,
            "split": scene.split,
            "actual_mu": NOMINAL_MU,
            "candidate_path": str(path),
            "candidate_sha256": check["candidate_sha256"],
            "raw_source_sha256": check["raw_source_sha256"],
            "assumption_acknowledged": True,
            "source_kind": "carsim_experimental",
            "ready_for_wm_training": False,
            "note": (
                "仅用于用户明确开启的开发性实验；"
                "真实左右前轮转角分别保留在 candidate.npz；"
                "不宣称轮端净转矩、制动等物理语义已经验证。"
            ),
        }
        (episode_dir / "experimental_episode.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append({
            "name": scene.name,
            "split": scene.split,
            "condition_id": scene.condition_id,
            "episode_path": str(episode_dir),
            "raw_source_sha256": check["raw_source_sha256"],
            "candidate_sha256": check["candidate_sha256"],
        })

    registry = {
        "format": "wm_recovery_experimental_registry_v1",
        "status": "experimental_not_physically_validated",
        "explicit_opt_in_env": EXPERIMENT_ENV,
        "no_cross_split_raw_duplicates": True,
        "nominal_mu": NOMINAL_MU,
        "episode_count": len(rows),
        "episodes": rows,
        "ready_for_wm_training": False,
    }
    (workspace / "registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return registry


def capture_all(sim_path, output_dir):
    """一个入口连续采集四条轨迹，并创建 WM 可读取的工作区。"""
    scenes = validate_scenarios()
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"实验输出目录已存在：{root}；"
            "请换用新目录，不能覆盖或清空旧数据"
        )
    root.mkdir(parents=True, exist_ok=False)

    candidate_paths = {}
    for scene in scenes:
        candidate_paths[scene.name] = capture_one(
            scene, sim_path=sim_path, root=root
        )

    build_scene_workspace(candidate_paths, root / "workspace")
    print("\n四条 CarSim 轨迹采集与分区完成：")
    print(root / "workspace" / "registry.json")
    return root / "workspace"


def require_experimental_opt_in():
    """强制保留之前的实验训练显式开关。"""
    if os.getenv(EXPERIMENT_ENV) != "1":
        raise ValueError(
            f"训练实验性 CarSim 数据前必须设置 {EXPERIMENT_ENV}=1"
        )


def _run_wm_cli(*arguments):
    """用当前 PyCharm/Conda 解释器调用原来的 WM CLI。"""
    command = [sys.executable, "-m", "wm_recovery", *map(str, arguments)]
    print("\n执行：", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train_from_workspace(workspace, output_dir, epochs):
    """不重跑 CarSim：对已有 workspace 执行训练→标定→评价。"""
    require_experimental_opt_in()
    workspace = Path(workspace).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs 必须是正整数")
    if output_dir.exists():
        raise FileExistsError(
            f"训练输出目录已存在：{output_dir}；请使用新目录"
        )

    episodes = workspace / "episodes"
    paths = {
        name: episodes / name
        for name in ("train1", "train2", "validation", "test")
    }
    for name, path in paths.items():
        if not (path / "experimental_episode.json").is_file():
            raise FileNotFoundError(
                f"找不到 {name} 的实验清单：{path}"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "model"
    calibration_path = output_dir / "calibration.json"
    evaluation_dir = output_dir / "evaluation"

    # 1. 独立的数据集检查
    _run_wm_cli(
        "check-data",
        paths["train1"], paths["train2"],
        paths["validation"], paths["test"],
    )

    # 2. 只用两条训练轨迹更新五网络的权重
    _run_wm_cli(
        "train",
        "--episodes", paths["train1"], paths["train2"],
        "--output", model_dir,
        "--epochs", epochs,
    )
    checkpoint = model_dir / "world_model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f"训练命令执行结束但 checkpoint 不存在或为空：{checkpoint}"
        )

    # 3. 单独使用验证轨迹标定
    _run_wm_cli(
        "calibrate",
        "--checkpoint", checkpoint,
        "--episodes", paths["validation"],
        "--output", calibration_path,
    )

    # 4. 单独使用测试轨迹评价；不在这里自动挑选训练轮数
    _run_wm_cli(
        "evaluate",
        "--checkpoint", checkpoint,
        "--calibration", calibration_path,
        "--episode", paths["test"],
        "--output", evaluation_dir,
    )

    print("\n实验完成。")
    print("模型文件：", checkpoint)
    print("验证标定：", calibration_path)
    print("测试评价：", evaluation_dir)
    return output_dir


def main(argv=None):
    """统一命令行入口，供 PyCharm 的 Module name 方式调用。"""
    parser = argparse.ArgumentParser(
        description="统一 CarSim 工况采集与实验性 World Model 训练"
    )
    actions = parser.add_subparsers(dest="action", required=True)

    actions.add_parser("plan", help="预览四条场景，不启动 CarSim")

    capture = actions.add_parser("capture", help="只采集四条轨迹")
    capture.add_argument("--sim", default=DEFAULT_SIM_PATH)
    capture.add_argument("--output", type=Path, required=True)

    training = actions.add_parser(
        "train-existing", help="使用已有轨迹训练、标定、评价"
    )
    training.add_argument("--workspace", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=1)

    full = actions.add_parser(
        "full", help="采集四条轨迹，再训练、标定、评价"
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
            args.workspace, args.output, args.epochs
        )
    elif args.action == "full":
        # 在新一轮采集之前确认用户已显式同意实验性训练。
        require_experimental_opt_in()
        output_dir = Path(args.output).expanduser().resolve()
        workspace = capture_all(args.sim, output_dir)
        train_from_workspace(
            workspace, output_dir / "model_run_01", args.epochs
        )


if __name__ == "__main__":
    main()
