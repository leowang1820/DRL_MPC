# -*- coding: utf-8 -*-
"""NMPC 物理残差网络：单时域 / 多时域可微自由滚动微调（v3）。

兼容旧用法：
    --horizon 50
新用法：
    --horizons "50,100,200" --horizon-weights "0.4,0.35,0.25"
    --terminal-weight 0.3 --scene-balance family

网络与输入输出顺序保持不变：8 -> 16 -> 16 -> 3。
参考模型调用已有的 nmpc_reference_torch.py；执行器量采用同一时刻
CarSim 已知未来等效值，预测的 [vy, r, vx] 永远不被未来真值覆盖。

训练和验证均严格按连续段取窗口，不跨数据断点。只读取 Train 和
Validation，不使用 Test 做归一化、抽样、早停或调参。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch

from .nmpc_reference import NMPCReferenceParams
from .nmpc_reference_torch import hybrid_rk4_step_torch
from .nmpc_residual_model import (
    INPUT_NAMES, TARGET_NAMES, MODEL_FORMAT, load_model_checkpoint,
)
from .nmpc_residual_train import (
    _check_manifest, load_split_npz, sha256,
)
from .nmpc_rollout_evaluate import _segment_slices

REPORT_FORMAT = "wm_recovery_nmpc_multihorizon_finetune_v2"
CONDITION_MODE = "future_CarSim_equivalent_actuation_is_exogenous"


def contiguous_window_starts(data, horizon, *, stride=50):
    """取所有合法连续段中可容纳 H 个区间的窗口起点。

    区间索引 / episode / segment_step / 时间戳的连续性由
    _segment_slices() 核验，不能跨越 lambda cancellation 丢弃的区间。
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 500:
        raise ValueError("horizon 必须是 1~500 的整数")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride 必须是正整数")
    starts = []
    for start, end in _segment_slices(data):
        if end - start >= horizon:
            starts.extend(range(start, end - horizon + 1, stride))
    return np.asarray(starts, dtype=np.int64)


def window_batch_torch(
    data, starts, horizon, *, device="cpu", dtype=torch.float64,
    use_train_weights=True,
):
    """将不跨段的连续 H 个真实区间整理成可微自由滚动的监督批次。

    只有 q0 用作预测初始状态，后续 CarSim 真实 q 存入 true_next，
    只能计算 Loss，不能参与下一步状态积分。
    """
    idx = np.asarray(starts, dtype=np.int64)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError("窗口起点必须是非空一维数组")
    loc = idx[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    if np.min(loc) < 0 or np.max(loc) >= len(data["z_si"]):
        raise ValueError("窗口越过数据边界")

    segments = data["segment_id"][loc]
    steps = data["segment_step"][loc]
    episodes = data["episode_index"][loc]
    original_indices = data["interval_index"][loc]
    expected_step = np.arange(horizon, dtype=np.int64)[None, :]
    if (not np.all(segments == segments[:, :1])
        or not np.all(episodes == episodes[:, :1])
        or not np.array_equal(steps, steps[:, :1] + expected_step)
        or not np.array_equal(
            original_indices, original_indices[:, :1] + expected_step
        )):
        raise ValueError("窗口跨越 episode/segment 或原始时间索引不连续")

    # 对齐时间戳也必须连续；不能只靠数组中恰好相邻的位置。
    time = data["time_k_s"][loc]
    dt = data["dt_s"][loc]
    if horizon > 1 and not np.allclose(
        time[:, 1:], time[:, :-1] + dt[:, :-1], rtol=0, atol=1e-8
    ):
        raise ValueError("窗口时间戳不连续")
    if horizon > 1 and not np.allclose(
        data["q_true_next_si"][loc[:, :-1]],
        data["q_k_si"][loc[:, 1:]], rtol=0, atol=1e-7,
    ):
        raise ValueError("窗口真实状态前后不连续")

    def as_tensor(array):
        return torch.as_tensor(
            np.asarray(array), dtype=dtype, device=device
        )

    return {
        "q0": as_tensor(data["q_k_si"][idx]),
        "external": as_tensor(data["z_si"][loc, 3:]),
        "dt": as_tensor(dt),
        "true_next": as_tensor(data["q_true_next_si"][loc]),
        "weights": as_tensor(
            data["sample_weight"][loc]
            if use_train_weights
            else np.ones(loc.shape, dtype=np.float64)
        ),
        "starts": idx.copy(),
    }


def parse_horizon_configuration(horizon=50, horizons=None, horizon_weights=None):
    """解析训练长度与各时域权重，自动归一化；支持旧 horizon 参数。"""
    if horizons is None:
        values = [horizon]
    elif isinstance(horizons, str):
        values = horizons.split(",")
    else:
        values = list(horizons)
    if any(
        isinstance(x, (bool, np.bool_))
        or (isinstance(x, (float, np.floating)) and not float(x).is_integer())
        for x in values
    ):
        raise ValueError("horizon 只能使用整数")
    try:
        steps = tuple(int(x) for x in values)
    except (ValueError, TypeError) as exc:
        raise ValueError("horizons 应为逗号分隔的整数，例如 50,100,200") from exc
    if (not steps or len(set(steps)) != len(steps)
        or any(v < 1 or v > 500 for v in steps)
        or any(isinstance(v, bool) for v in values)):
        raise ValueError("各 horizon 必须互异且在 1~500")
    if isinstance(horizons, str) and any(
        str(x).strip() != str(int(x)) for x in values
    ):
        raise ValueError("horizons 必须为整数，如 50,100,200")

    if horizon_weights is None:
        weights = np.ones(len(steps), dtype=np.float64)
    else:
        raw = (horizon_weights.split(",")
               if isinstance(horizon_weights, str)
               else list(horizon_weights))
        try:
            weights = np.asarray([float(v) for v in raw], dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("horizon_weights 必须是数字列表") from exc
        if weights.shape != (len(steps),):
            raise ValueError("horizon_weights 必须与 horizons 数量一致")

    if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
        raise ValueError("所有 horizon 权重必须为正的有限数")
    weights = weights / np.sum(weights)
    return steps, {h: float(w) for h, w in zip(steps, weights)}


def _scenario_family(name):
    """将不同速度和左右方向的轨迹归纳为相应工况。

    如 v60_train_brake_turn_left -> v60:brake_turn。
    同时覆盖 40/60 km/h，避免短轨迹被全体区间比例淹没。
    """
    match = re.fullmatch(r"(v\d+)_(?:train|validation|test)_(.+)", str(name))
    if match:
        speed, suffix = match.groups()
    else:
        speed, suffix = "unknown", str(name)
    suffix = re.sub(r"_(left|right)$", "", suffix)
    return f"{speed}:{suffix}"


def _episode_name(index, manifest, split):
    episodes = manifest.get("episodes", [])
    i = int(index)
    if i < 0 or i >= len(episodes):
        raise ValueError(f"Episode {i} 不在 dataset_manifest.json 中")
    episode = episodes[i]
    if episode.get("split") != split:
        raise ValueError(f"Episode {i} 不属于当前 {split} 分区")
    name = episode.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Episode {i} 缺少合法 name")
    return name


class BalancedIndexSampler:
    """按 Train 内部的 trajectory 或 speed×family 等概率轮转采样。

    mode=none: 对候选区间均匀采样（兼容旧实验）
    mode=scene: 各轨迹名称等权
    mode=family: 各 speed×工况类型等权，SWD、Brake-turn 等不再仅由
                 其可用区间长度决定出现次数。

    本类仅接收 Train 的 data、窗口起点和 Train Episode 元信息。
    """

    def __init__(self, candidates, data, manifest, *, mode="none", seed=2026):
        if mode not in ("none", "scene", "family"):
            raise ValueError("scene_balance 只能为 none、scene 或 family")
        self.candidates = np.asarray(candidates, dtype=np.int64)
        if self.candidates.ndim != 1 or len(self.candidates) == 0:
            raise ValueError("Train 候选窗口不能为空")
        if np.any(self.candidates < 0) or np.any(
            self.candidates >= len(data["episode_index"])
        ):
            raise ValueError("采样候选索引超过 Train 范围")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        groups = defaultdict(list)
        if mode != "none":
            for index in self.candidates:
                name = _episode_name(
                    data["episode_index"][index], manifest, "train"
                )
                key = name if mode == "scene" else _scenario_family(name)
                groups[key].append(int(index))
        else:
            groups["all_train"] = self.candidates.tolist()
        self.groups = {
            name: np.asarray(group, dtype=np.int64)
            for name, group in sorted(groups.items())
        }
        self.group_names = tuple(self.groups)
        self._cycle = np.empty(0, dtype=np.int64)
        self._cursor = 0
        self.observed_group_counts = defaultdict(int)

    def draw(self, size):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("batch size 必须是正整数")
        if self.mode == "none":
            sample = self.rng.choice(
                self.candidates, size=size,
                replace=(size > len(self.candidates))
            ).astype(np.int64)
            self.observed_group_counts["all_train"] += size
            return sample
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            if self._cursor >= len(self._cycle):
                # 一轮遍历所有 group 后才重新洗牌，减少稀有工况漏采。
                self._cycle = self.rng.permutation(len(self.group_names))
                self._cursor = 0
            key = self.group_names[int(self._cycle[self._cursor])]
            self._cursor += 1
            out[i] = self.rng.choice(self.groups[key])
            self.observed_group_counts[key] += 1
        return out

    def summary(self):
        return {
            "mode": self.mode,
            "group_count": len(self.group_names),
            "candidate_counts": {
                key: int(len(values)) for key, values in self.groups.items()
            },
            "observed_sampled_counts": {
                key: int(self.observed_group_counts.get(key, 0))
                for key in self.group_names
            },
        }


def differentiable_rollout_components(
    model, batch, q_scale, *, params,
    integration="continuous", rk4_substeps=1,
):
    """一次计算完整 H 步自由滚动的路径损失和终点损失。

    各状态先除以 Train-only q_scale，防止 vy/r/vx 单位不同。
    终点损失不除以 H，与平均路径损失单独加权组合。
    """
    q = batch["q0"]
    external = batch["external"]
    true_next = batch["true_next"]
    dt = batch["dt"]
    weights = batch["weights"]
    if q.ndim != 2 or q.shape[1] != 3 or external.shape[-1] != 5:
        raise ValueError("滚动 batch 的状态/外部输入维数不正确")
    if true_next.shape != (len(q), external.shape[1], 3):
        raise ValueError("真实下一状态张量形状不正确")
    if dt.shape != external.shape[:2] or weights.shape != dt.shape:
        raise ValueError("滚动 dt/权重张量形状不正确")
    if q_scale.shape != (3,) or not bool(torch.isfinite(q_scale).all()) or not bool(
        (q_scale > 0).all()
    ):
        raise ValueError("q_scale 必须是 Train-only 的三个正向状态尺度")

    path_num = torch.zeros(len(q), dtype=q.dtype, device=q.device)
    path_den = torch.zeros(len(q), dtype=q.dtype, device=q.device)
    last = None
    for step in range(external.shape[1]):
        q = hybrid_rk4_step_torch(
            q, external[:, step, :], dt[:, step],
            residual_model=model, params=params,
            integration=integration, substeps=rk4_substeps,
        )
        # 真实状态只计算监督误差，不回灌 q。
        error = (q - true_next[:, step, :]) / q_scale[None, :]
        error_sq = error.square().mean(dim=1)
        path_num = path_num + weights[:, step] * error_sq
        path_den = path_den + weights[:, step]
        last = error_sq

    if not bool((path_den > 0).all()):
        raise ValueError("至少有一个窗口的样本权重总和不为正")
    path_per_window = path_num / path_den
    return {
        # 每个窗口等权，更适合按轨迹均衡采样后优化。
        "path": path_per_window.mean(),
        "terminal": last.mean(),
        "path_per_window": path_per_window,
        "terminal_per_window": last,
    }


def differentiable_rollout_loss(
    model, batch, q_scale, *, params,
    integration="continuous", rk4_substeps=1,
):
    """兼容 v2 两返回值接口：(path_mse, endpoint_mse)。"""
    result = differentiable_rollout_components(
        model, batch, q_scale, params=params,
        integration=integration, rk4_substeps=rk4_substeps,
    )
    return result["path"], result["terminal"]


def one_step_residual_loss(model, z, y, weights=None):
    """原有 10ms 等效加速度残差 MSE，用作多时域联合训练的辅助约束。"""
    diff = ((model(z) - y) / model.y_scale).square().mean(dim=1)
    if weights is None:
        return diff.mean()
    return (diff * weights).sum() / weights.sum()


@torch.no_grad()
def validation_joint_metrics(
    model, validation_data, fixed_starts, horizons, horizon_weights, *,
    manifest, params, q_scale, device, batch_size,
    integration, rk4_substeps, terminal_weight=0.0,
    validation_scene_weight=0.0,
):
    """固定各时域 Validation 起点，联合计算路径/终点/逐场景 MSE。

    validation_scene_weight=0 时各窗口等权（兼容旧口径）。
    >0 时将 scene-macro MSE 纳入选模，避免长轨迹过度主导。
    任何统计都不使用 Test。
    """
    model.eval()
    per_horizon = {}
    joint_score = 0.0

    for horizon in horizons:
        starts = fixed_starts[horizon]
        per_window_path = []
        per_window_terminal = []
        for begin in range(0, len(starts), batch_size):
            chunk = starts[begin:begin + batch_size]
            batch = window_batch_torch(
                validation_data, chunk, horizon, device=device,
                use_train_weights=False,
            )
            components = differentiable_rollout_components(
                model, batch, q_scale, params=params,
                integration=integration, rk4_substeps=rk4_substeps,
            )
            per_window_path.extend(
                components["path_per_window"].cpu().numpy().tolist()
            )
            per_window_terminal.extend(
                components["terminal_per_window"].cpu().numpy().tolist()
            )
        path_arr = np.asarray(per_window_path, dtype=np.float64)
        end_arr = np.asarray(per_window_terminal, dtype=np.float64)
        if not len(path_arr) or not np.isfinite(path_arr).all() or not np.isfinite(
            end_arr
        ).all():
            raise FloatingPointError(f"Validation horizon={horizon} 产生无效指标")
        scene_names = [
            _episode_name(
                validation_data["episode_index"][int(s)], manifest,
                "validation",
            )
            for s in starts
        ]
        per_scene = {}
        for name in sorted(set(scene_names)):
            mask = np.asarray([scene == name for scene in scene_names])
            per_scene[name] = {
                "windows": int(mask.sum()),
                "path_mse": float(path_arr[mask].mean()),
                "terminal_mse": float(end_arr[mask].mean()),
            }
        path_mean = float(path_arr.mean())
        terminal_mean = float(end_arr.mean())
        scene_macro_path = float(np.mean([
            info["path_mse"] for info in per_scene.values()
        ]))
        scene_macro_terminal = float(np.mean([
            info["terminal_mse"] for info in per_scene.values()
        ]))
        blend = validation_scene_weight
        selected_path = (1.0 - blend) * path_mean + blend * scene_macro_path
        selected_terminal = (
            (1.0 - blend) * terminal_mean + blend * scene_macro_terminal
        )
        selected = selected_path + terminal_weight * selected_terminal
        joint_score += horizon_weights[horizon] * selected
        worst_scene = max(
            per_scene,
            key=lambda n: per_scene[n]["path_mse"] +
            terminal_weight * per_scene[n]["terminal_mse"]
        )
        per_horizon[str(horizon)] = {
            "horizon_s": horizon * 0.01,
            "windows": len(starts),
            "scene_count": len(per_scene),
            "path_mse": path_mean,
            "terminal_mse": terminal_mean,
            "scene_macro_path_mse": scene_macro_path,
            "scene_macro_terminal_mse": scene_macro_terminal,
            "selected_joint_mse": float(selected),
            "worst_scene": worst_scene,
            "per_scene": per_scene,
        }

    z = torch.as_tensor(
        validation_data["z_si"], dtype=torch.float64, device=device
    )
    y = torch.as_tensor(
        validation_data["residual_y_si"], dtype=torch.float64, device=device
    )
    single = float(one_step_residual_loss(model, z, y).item())
    if not math.isfinite(joint_score):
        raise FloatingPointError("Validation 联合损失非有限")
    return {
        "validation_joint_normalized_mse": float(joint_score),
        "validation_single_normalized_mse": single,
        "per_horizon": per_horizon,
        "horizon_weights": {str(k): v for k, v in horizon_weights.items()},
        "terminal_weight": float(terminal_weight),
        "validation_scene_weight": float(validation_scene_weight),
    }


@torch.no_grad()
def validation_metrics(
    model, validation_data, fixed_starts, horizon, *,
    params, q_scale, device, batch_size, integration, rk4_substeps,
):
    """保留 v2 外部调用的单时域 Validation 接口及字段。"""
    model.eval()
    all_path, all_terminal = [], []
    for begin in range(0, len(fixed_starts), batch_size):
        selection = fixed_starts[begin:begin + batch_size]
        batch = window_batch_torch(
            validation_data, selection, horizon,
            device=device, use_train_weights=False,
        )
        comp = differentiable_rollout_components(
            model, batch, q_scale, params=params,
            integration=integration, rk4_substeps=rk4_substeps,
        )
        all_path.extend(comp["path_per_window"].cpu().tolist())
        all_terminal.extend(comp["terminal_per_window"].cpu().tolist())
    z = torch.as_tensor(
        validation_data["z_si"], dtype=torch.float64, device=device
    )
    y = torch.as_tensor(
        validation_data["residual_y_si"], dtype=torch.float64, device=device
    )
    return {
        "validation_rollout_normalized_mse": float(np.mean(all_path)),
        "validation_endpoint_normalized_mse": float(np.mean(all_terminal)),
        "validation_single_normalized_mse": float(
            one_step_residual_loss(model, z, y).item()
        ),
        "window_count": int(len(fixed_starts)),
    }


def fine_tune_multistep(
    dataset_dir, pretrained_checkpoint, output_dir, *,
    horizon=50, epochs=5, batch_size=8, steps_per_epoch=10,
    validation_stride=50, train_stride=10,
    learning_rate=1e-4, single_weight=0.25,
    rollout_weight=1.0, regularization_weight=1e-4,
    weight_decay=1e-6, patience=6,
    seed=2026, device="cpu", integration="continuous", rk4_substeps=1,
    horizons=None, horizon_weights=None, horizon_strategy="joint",
    terminal_weight=0.0, scene_balance="none",
    validation_scene_weight=0.0,
):
    """保留原模型架构，增加多时域自由滚动损失和场景均衡采样。

    horizon_strategy:
       joint: 每个优化步骤计算全部 H 的加权平均（严格联合目标）。
       mixed: 每个步骤按 horizon_weights 随机抽一个 H，期望目标
              与 joint 相同，适合 CPU 低成本初筛。

    validation 固定每个 H 的起点且完整计算所有 H，以联合目标选模。
    """
    if integration not in ("continuous", "discrete"):
        raise ValueError("integration 必须为 continuous 或 discrete")
    if horizon_strategy not in ("joint", "mixed"):
        raise ValueError("horizon_strategy 必须为 joint 或 mixed")
    if scene_balance not in ("none", "scene", "family"):
        raise ValueError("scene_balance 必须为 none、scene 或 family")
    for name, value in (
        ("epochs", epochs), ("batch_size", batch_size),
        ("steps_per_epoch", steps_per_epoch), ("patience", patience),
        ("validation_stride", validation_stride),
        ("train_stride", train_stride),
        ("rk4_substeps", rk4_substeps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")

    hs, hweights = parse_horizon_configuration(
        horizon=horizon, horizons=horizons, horizon_weights=horizon_weights,
    )
    for name, value in (
        ("learning_rate", learning_rate),
        ("single_weight", single_weight),
        ("rollout_weight", rollout_weight),
        ("regularization_weight", regularization_weight),
        ("weight_decay", weight_decay),
        ("terminal_weight", terminal_weight),
    ):
        if not math.isfinite(value) or (
            value <= 0 if name == "learning_rate" else value < 0
        ):
            raise ValueError(f"{name} 数值必须合法且有限")
    if rollout_weight <= 0:
        raise ValueError("rollout_weight 必须大于 0")
    if (not math.isfinite(validation_scene_weight)
        or not 0 <= validation_scene_weight <= 1):
        raise ValueError("validation_scene_weight 必须在 0~1")

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{output_dir}")
    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    pretrained_checkpoint = Path(pretrained_checkpoint).expanduser().resolve(
        strict=True
    )
    manifest = _check_manifest(dataset_dir)
    if manifest["dataset_settings"]["rk4_substeps"] != rk4_substeps:
        raise ValueError("RK4 子步数必须与数据集名义标签构造配置一致")
    model, payload = load_model_checkpoint(pretrained_checkpoint, device="cpu")
    manifest_hash = sha256(dataset_dir / "dataset_manifest.json")
    train_hash = sha256(dataset_dir / "train.npz")
    validation_hash = sha256(dataset_dir / "validation.npz")
    if payload.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("源 checkpoint 与当前数据集 manifest SHA256 不一致")
    if payload.get("train_npz_sha256") != train_hash:
        raise ValueError("源 checkpoint 与 train.npz 不一致")
    if payload.get("validation_npz_sha256") != validation_hash:
        raise ValueError("源 checkpoint 与 validation.npz 不一致")

    # 只加载 Train/Validation。Test 不参与任何训练、验证和统计拟合。
    train = load_split_npz(dataset_dir, "train")
    valid = load_split_npz(dataset_dir, "validation")
    for split, data in (("train", train), ("validation", valid)):
        if len(data["z_si"]) != manifest["split_counts"][split]["intervals"]:
            raise ValueError(f"{split} 数据样本数与 manifest 不一致")

    train_starts = {
        h: contiguous_window_starts(train, h, stride=train_stride)
        for h in hs
    }
    fixed_validation_starts = {
        h: contiguous_window_starts(valid, h, stride=validation_stride)
        for h in hs
    }
    for h in hs:
        if len(train_starts[h]) == 0 or len(fixed_validation_starts[h]) == 0:
            raise ValueError(
                f"Train 或 Validation 没有长度 {h} 的合法连续窗口"
            )
        print(
            f"固定时域 {h * .01:g}s：Train 候选窗口 {len(train_starts[h])}，"
            f"Validation 固定窗口 {len(fixed_validation_starts[h])}",
            flush=True,
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(1)
    model = model.to(dev).double()
    params = NMPCReferenceParams(**manifest["reference_parameters"])
    params.validate()
    q_scale = model.x_scale[:3].detach().clone()
    if not bool(torch.isfinite(q_scale).all()) or not bool((q_scale > 0).all()):
        raise ValueError("原模型的 Train-only 状态标准差非法")

    sampler_by_h = {
        h: BalancedIndexSampler(
            train_starts[h], train, manifest, mode=scene_balance,
            seed=seed + 101 * i,
        )
        for i, h in enumerate(hs)
    }
    # 单步辅助损失也按同样的场景模式抽 Train 样本。
    single_sampler = BalancedIndexSampler(
        np.arange(len(train["z_si"]), dtype=np.int64),
        train, manifest, mode=scene_balance, seed=seed + 997,
    )

    optimizer = torch.optim.Adam(
        model.net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_z = torch.as_tensor(
        train["z_si"], dtype=torch.float64, device=dev
    )
    train_y = torch.as_tensor(
        train["residual_y_si"], dtype=torch.float64, device=dev
    )
    train_weights = torch.as_tensor(
        train["sample_weight"], dtype=torch.float64, device=dev
    )

    initial_metrics = validation_joint_metrics(
        model, valid, fixed_validation_starts, hs, hweights,
        manifest=manifest, params=params, q_scale=q_scale, device=dev,
        batch_size=batch_size, integration=integration,
        rk4_substeps=rk4_substeps,
        terminal_weight=terminal_weight,
        validation_scene_weight=validation_scene_weight,
    )
    best_score = initial_metrics["validation_joint_normalized_mse"]
    print(
        "微调前 Validation 多时域联合 MSE:",
        f"{best_score:.8g}",
        "| horizons:", list(hs),
        "| 终点权重:", terminal_weight,
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    history = []
    best_epoch = 0
    no_improve = 0
    metadata = {
        "format": MODEL_FORMAT,
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "hidden_sizes": [16, 16],
        "dataset_manifest_sha256": manifest_hash,
        "train_npz_sha256": train_hash,
        "validation_npz_sha256": validation_hash,
        "training_mode": "multistep_rollout_finetune",
        "training_version": "v3_joint_multihorizon",
        "source_checkpoint_sha256": sha256(pretrained_checkpoint),
        "condition": CONDITION_MODE,
        "horizon_intervals": int(max(hs)),
        "horizons_intervals": list(hs),
        "horizon_weights": {str(h): hweights[h] for h in hs},
        "terminal_weight": terminal_weight,
        "scene_balance": scene_balance,
        "horizon_strategy": horizon_strategy,
        "validation_scene_weight": validation_scene_weight,
        "integration": integration,
        "seed": int(seed),
    }

    def save_best(epoch, score):
        state = {
            key: value.detach().cpu().clone().to(torch.float32)
            for key, value in model.state_dict().items()
        }
        torch.save({
            **metadata,
            "best_epoch": int(epoch),
            "best_validation_normalized_mse": float(score),
            "model_state_dict": state,
        }, output_dir / "best.pt")

    # Epoch 0 基线；后面微调再差也不会强行覆盖预训练权重。
    save_best(0, best_score)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "train_rollout_loss": 0.0,
            "train_terminal_loss": 0.0,
            "train_single_loss": 0.0,
            "train_regularization": 0.0,
            "train_total": 0.0,
        }
        sampled_h = defaultdict(int)

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss_path = torch.zeros((), dtype=torch.float64, device=dev)
            loss_terminal = torch.zeros((), dtype=torch.float64, device=dev)

            if horizon_strategy == "joint":
                active = [(h, hweights[h]) for h in hs]
            else:
                # 按权重随机选 H，期望目标等于联合加权损失；每步只算一个 H。
                chosen = hs[int(rng.choice(
                    len(hs), p=np.asarray([hweights[h] for h in hs])
                ))]
                active = [(chosen, 1.0)]

            for h, coeff in active:
                sampled_h[h] += 1
                starts = sampler_by_h[h].draw(batch_size)
                batch = window_batch_torch(
                    train, starts, h, device=dev, use_train_weights=True,
                )
                comp = differentiable_rollout_components(
                    model, batch, q_scale, params=params,
                    integration=integration, rk4_substeps=rk4_substeps,
                )
                loss_path = loss_path + coeff * comp["path"]
                loss_terminal = loss_terminal + coeff * comp["terminal"]

            # 保留独立的单步监督约束，防止滚动修正完全偏离残差定义。
            single_indices = single_sampler.draw(
                min(len(train_z), batch_size * 8)
            )
            ids = torch.as_tensor(
                single_indices, dtype=torch.long, device=dev
            )
            z = train_z.index_select(0, ids)
            target = train_y.index_select(0, ids)
            w = train_weights.index_select(0, ids)
            single = one_step_residual_loss(model, z, target, w)

            if regularization_weight > 0:
                regularization = (model(z) / model.y_scale).square().mean()
            else:
                regularization = torch.zeros(
                    (), dtype=torch.float64, device=dev
                )
            total = (
                rollout_weight * (loss_path + terminal_weight * loss_terminal)
                + single_weight * single
                + regularization_weight * regularization
            )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(
                    "多时域 Loss 出现 NaN/Inf；需检查外部量和学习率"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.net.parameters(), max_norm=1.0
            )
            optimizer.step()
            for key, term in (
                ("train_rollout_loss", loss_path),
                ("train_terminal_loss", loss_terminal),
                ("train_single_loss", single),
                ("train_regularization", regularization),
                ("train_total", total),
            ):
                totals[key] += float(term.detach().item())

        metrics = validation_joint_metrics(
            model, valid, fixed_validation_starts, hs, hweights,
            manifest=manifest, params=params, q_scale=q_scale, device=dev,
            batch_size=batch_size, integration=integration,
            rk4_substeps=rk4_substeps,
            terminal_weight=terminal_weight,
            validation_scene_weight=validation_scene_weight,
        )
        score = metrics["validation_joint_normalized_mse"]
        entry = {
            "epoch": epoch,
            **{name: value / steps_per_epoch for name, value in totals.items()},
            **metrics,
            "sampled_horizon_updates": {
                str(h): int(sampled_h.get(h, 0)) for h in hs
            },
        }
        history.append(entry)
        print(
            f"epoch {epoch:03d}"
            f" | train_path={entry['train_rollout_loss']:.7f}"
            f" | train_terminal={entry['train_terminal_loss']:.7f}"
            f" | val_joint={score:.7f}"
            f" | val_single={entry['validation_single_normalized_mse']:.7f}",
            flush=True,
        )

        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            save_best(epoch, best_score)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("根据固定 Validation 多时域联合指标早停。", flush=True)
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "normalization.json").write_text(
        json.dumps(model.normalization_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "format": REPORT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "output": str(output_dir),
        "model_architecture": "8-16-16-3-Tanh",
        "horizon_intervals": int(max(hs)),
        "horizons_intervals": list(hs),
        "horizons_s": [h * 0.01 for h in hs],
        "horizon_weights": {str(h): hweights[h] for h in hs},
        "horizon_strategy": horizon_strategy,
        "integration": integration,
        "rk4_substeps": rk4_substeps,
        "training_hyperparameters": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": learning_rate,
            "rollout_weight": rollout_weight,
            "terminal_weight": terminal_weight,
            "single_weight": single_weight,
            "regularization_weight": regularization_weight,
            "weight_decay": weight_decay,
            "patience": patience,
            "scene_balance": scene_balance,
            "validation_scene_weight": validation_scene_weight,
            "train_stride_intervals": train_stride,
            "validation_stride_intervals": validation_stride,
            "seed": seed,
            "device": device,
        },
        "train_window_candidates_by_horizon": {
            str(h): len(train_starts[h]) for h in hs
        },
        "validation_fixed_window_count_by_horizon": {
            str(h): len(fixed_validation_starts[h]) for h in hs
        },
        "training_sampling_by_horizon": {
            str(h): sampler_by_h[h].summary() for h in hs
        },
        "training_single_sampling": single_sampler.summary(),
        "initial_validation": initial_metrics,
        "best_epoch": int(best_epoch),
        "best_validation_joint_normalized_mse": float(best_score),
        # 兼容旧报告字段；多时域时应优先查看上面的 joint 字段。
        "best_validation_rollout_normalized_mse": float(best_score),
        "best_checkpoint_sha256": sha256(output_dir / "best.pt"),
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "dataset_manifest_sha256": manifest_hash,
        "train_npz_sha256": train_hash,
        "validation_npz_sha256": validation_hash,
        "q_scale_train_only_si": q_scale.detach().cpu().numpy().tolist(),
        "test_not_used": True,
        "closed_loop_nmpc_validated": False,
        "physics_verified": False,
        "notes": [
            "多时域联合目标按 horizon_weights 加权，各 H 路径 MSE 均已归一化。",
            "终点误差损失独立加权，不除以 horizon 步数。",
            "仅使用 Train 场景均衡采样；Validation 固定窗口选模，Test 不读取。",
            "微调未改善时 best.pt 保留源模型第 0 轮权重。",
            "未来 CarSim 等效执行器参数已知，车辆状态完全自由滚动。",
            "该微调指标不等于完整 5s 验证，仍需运行固定 3s/5s 消融。",
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "多时域微调最佳 Epoch:", best_epoch,
        "| 最佳联合 Validation MSE:", f"{best_score:.8g}",
        "| 模型:", output_dir / "best.pt",
        flush=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NMPC 可微多时域自由滚动残差微调（v3）"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50,
                        help="单时域旧参数；新实验建议使用 --horizons")
    parser.add_argument("--horizons", default=None,
                        help='多时域，例如 "50,100,200"，覆盖 --horizon')
    parser.add_argument("--horizon-weights", default=None,
                        help='例如 "0.4,0.35,0.25"，自动归一化')
    parser.add_argument("--horizon-strategy", choices=("joint", "mixed"),
                        default="joint",
                        help="joint 每步算所有 H；mixed 每步按权重随机算一个 H")
    parser.add_argument("--terminal-weight", type=float, default=0.0,
                        help="额外终点误差损失系数，建议从 0.2～0.5 试验")
    parser.add_argument("--scene-balance",
                        choices=("none", "scene", "family"), default="none",
                        help="Train 场景均衡：none/按轨迹/按速度和工况类别")
    parser.add_argument("--validation-scene-weight", type=float, default=0.0,
                        help="Validation scene-macro 指标占比，范围 0~1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--single-weight", type=float, default=0.25)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--validation-stride", type=int, default=50)
    parser.add_argument("--integration", choices=("continuous", "discrete"),
                        default="continuous")
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    fine_tune_multistep(
        args.dataset, args.pretrained, args.output,
        horizon=args.horizon, horizons=args.horizons,
        horizon_weights=args.horizon_weights,
        horizon_strategy=args.horizon_strategy,
        terminal_weight=args.terminal_weight,
        scene_balance=args.scene_balance,
        validation_scene_weight=args.validation_scene_weight,
        epochs=args.epochs, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        validation_stride=args.validation_stride,
        train_stride=args.train_stride,
        learning_rate=args.lr, single_weight=args.single_weight,
        rollout_weight=args.rollout_weight,
        regularization_weight=args.reg_weight,
        patience=args.patience, seed=args.seed,
        device=args.device, integration=args.integration,
        rk4_substeps=args.rk4_substeps,
    )


if __name__ == "__main__":
    main()
