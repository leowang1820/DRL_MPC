# -*- coding: utf-8 -*-
"""将已清洗 CarSim Workspace 构造为 NMPC 8 输入 / 3 输出残差数据集。

使用现有模块：
    nmpc_actuation.load_from_candidate: 原始61通道 -> [T,8]、lambda三状态
    offline.CandidateEpisode.load: 严格候选格式、来源指纹、时序检查
    nmpc_reference.rk4_nominal_step: 名义单轨动力学 ZOH-RK4

网络输入 z_k 的顺序：
    [vy, r, vx, delta_f, Mz_tv_estimated, Fx_all, lamda, mu_hat]
训练标签（区间平均加速度残差近似）：
    y_k = (q_carSim[k+1] - RK4_nominal(q_carSim[k], z_k, dt_k)) / dt_k
    q = [vy, r, vx]

关键边界：
1. 只使用 t_k 的轮胎力/转角/状态作为输入；t_{k+1} 只作为监督真值。
2. lambda 的 ratio / neutral / cancellation 三状态完整保留；neutral 是建模
   约定值，不得当作测量真值；cancellation 区间跳过。
3. 时间上有空洞时为两条独立连续段，不能跨无效点构造滚动预测窗口。
4. 不用测试集计算归一化统计量；此模块只保存 SI 原始数据。
5. CarSim 的 Mz_tv 是依据同一时刻测得轮胎力估计的“等效输入”，
   尚未与 NMPC 的预测控制状态进行执行器层面的物理核验。
6. 当前参考模型 RK4 在单个10ms区间对 delta/Mz/Fx/lambda/mu 使用 ZOH，
   不积分完整 NMPC 的执行器速率状态，标签属于区间等效残差。

命令：
    python -m wm_recovery.nmpc_residual_dataset audit \
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02

    python -m wm_recovery.nmpc_residual_dataset build \
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02 \
      --output artifacts/wm_recovery/nmpc_residual_dataset_40_60_01

输出：train.npz、validation.npz、test.npz、dataset_manifest.json。
每个分区 npz 中附带 episode_index / segment_id / interval_index，
长时域训练必须通过 iter_contiguous_windows() 保证窗口不跨连续段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .nmpc_actuation import (
    NMPCActuationConfig,
    NMPC_INPUT_NAMES,
    contiguous_true_runs,
    load_from_candidate,
)
from .nmpc_reference import (
    NMPCReferenceParams,
    NMPC_Q_NAMES,
    NMPC_Y_NAMES,
    NMPC_Z_NAMES,
    REFERENCE_VERSION,
    nmpc_accelerations,
    rk4_nominal_step,
)

DATASET_VERSION = "wm_recovery_nmpc_residual_dataset_v1"
SPLITS = ("train", "validation", "test")
LAMBDA_RATIO = 0
LAMBDA_NEUTRAL = 1

# 统一在独立分区文件中使用这些稳定键，不借用原 7DOF Tensor 约定。
_FLOAT_FIELDS = (
    "z_si", "residual_y_si", "q_k_si", "q_true_next_si",
    "q_nominal_next_si", "nominal_accel_si", "time_k_s", "dt_s",
    "sample_weight",
)
_INT_FIELDS = (
    "episode_index", "segment_id", "segment_step", "interval_index",
    "lamda_regime_code",
)


@dataclass(frozen=True)
class ResidualDatasetConfig:
    """数据集生成选项（不根据测试误差调节）。"""

    min_segment_intervals: int = 1
    neutral_train_weight: float = 1.0
    rk4_substeps: int = 1
    expected_dt_s: float = 0.01

    def validate(self):
        if isinstance(self.min_segment_intervals, bool) or not (
            isinstance(self.min_segment_intervals, int) and
            self.min_segment_intervals >= 1
        ):
            raise ValueError("min_segment_intervals 必须是正整数")
        if isinstance(self.rk4_substeps, bool) or not (
            isinstance(self.rk4_substeps, int) and self.rk4_substeps >= 1
        ):
            raise ValueError("rk4_substeps 必须是正整数")
        if not (math.isfinite(self.neutral_train_weight) and
                0 < self.neutral_train_weight <= 1):
            raise ValueError("neutral_train_weight 必须在 (0,1] 内")
        if not (math.isfinite(self.expected_dt_s) and self.expected_dt_s > 0):
            raise ValueError("expected_dt_s 必须为正且有限")


def _sha256_file(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(root, entry):
    """核对 registry 及原 episode manifest，严格保留 split/来源事实。"""
    name = entry.get("name")
    split = entry.get("split")
    if not isinstance(name, str) or not name or split not in SPLITS:
        raise ValueError("registry 存在无效场景名称或数据分区")
    path = Path(entry["episode_path"]).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    manifest_path = path / "experimental_episode.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "wm_recovery_experimental_episode_manifest_v1":
        raise ValueError(f"{name} 不是受支持的 experimental_episode")
    if manifest.get("split") != split:
        raise ValueError(f"{name} 的 registry 与 episode split 不一致")
    for field in ("candidate_sha256", "raw_source_sha256", "condition_id"):
        if manifest.get(field) != entry.get(field):
            raise ValueError(f"{name} 的 {field} 与 registry 不一致")
    if manifest.get("assumption_acknowledged") is not True:
        raise ValueError(f"{name} 没有明确声明实验性名义输入假设")
    if manifest.get("ready_for_wm_training") is not False:
        raise ValueError(f"{name} 不能冒充已物理验证的正式训练数据")
    candidate_path = Path(manifest["candidate_path"]).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = path / candidate_path
    candidate_path = candidate_path.resolve(strict=True)
    return name, split, path, manifest, candidate_path


def prepare_episode(name, split, candidate_dir, actuation=None,
                    reference=None, settings=None):
    """构造单条轨迹的有效样本及连续段，保持状态与标签同一时间索引。"""
    from .offline import CandidateEpisode

    if actuation is None:
        actuation = NMPCActuationConfig()
    if reference is None:
        reference = NMPCReferenceParams()
    if settings is None:
        settings = ResidualDatasetConfig()
    actuation.validate()
    reference.validate()
    settings.validate()
    if split not in SPLITS:
        raise ValueError(f"不支持的 split: {split}")
    if tuple(NMPC_INPUT_NAMES) != tuple(NMPC_Z_NAMES):
        raise ValueError("nmpc_actuation 与 nmpc_reference 输入顺序不一致")

    candidate_dir = Path(candidate_dir).expanduser().resolve(strict=True)
    ep = CandidateEpisode.load(candidate_dir)
    t, inputs = load_from_candidate(candidate_dir, actuation)
    if not np.array_equal(t, ep.time):
        raise ValueError(f"{name} 的候选与 NMPC 输入状态时间戳不一致")
    if len(t) != len(inputs.z_si) or len(t) != len(ep.state):
        raise ValueError(f"{name} 的状态、61通道、时间长度不一致")

    # 原严格候选状态：[vx,vy,r,omega_FL,omega_FR,omega_RL,omega_RR]。
    q = ep.state[:, [1, 2, 0]].copy()  # q=[vy,r,vx]
    n = len(t) - 1
    dt = np.diff(t)
    if not np.allclose(dt, settings.expected_dt_s, atol=1e-8, rtol=0):
        raise ValueError(f"{name} 的采样周期与 {settings.expected_dt_s}s 不匹配")
    if not np.isfinite(q).all():
        raise ValueError(f"{name} 的车辆状态含非有限值")

    # 区间 k 仅使用 t_k 的8维输入：绝不读取 z[k+1] 作为网络输入。
    valid = inputs.valid_mask[:-1].copy()
    # 已清洗数据应有正纵向速度；其余异常拒绝通过而不是静默剪裁。
    if np.any(q[:-1, 2] <= 0):
        raise ValueError(f"{name} 存在非正 Vx：应先使用 data_cleaning 清洗")
    if not np.array_equal(
        valid,
        inputs.lamda_observable_mask[:-1] | inputs.lamda_neutral_mask[:-1],
    ):
        raise ValueError(f"{name} 的 lambda 三状态分类与有效掩码不一致")

    # 仅用有效输入积分；cancellation 的 NaN 无法送入名义模型。
    interval_ids = np.flatnonzero(valid)
    z_k = inputs.z_si[interval_ids]
    q_k = q[interval_ids]
    q_true_next = q[interval_ids + 1]
    dt_k = dt[interval_ids]
    if len(interval_ids):
        q_nominal_next = rk4_nominal_step(
            z_k, dt_k, reference, substeps=settings.rk4_substeps
        )
        nominal_accel = nmpc_accelerations(z_k, reference)
        # 残差是 区间差 / 采样周期，不宣称为 CarSim 原生瞬时加速度。
        target_y = (q_true_next - q_nominal_next) / dt_k[:, None]
        if not np.isfinite(target_y).all():
            raise FloatingPointError(f"{name} 的三通道标签非有限")
    else:
        q_nominal_next = np.empty((0, 3), dtype=np.float64)
        nominal_accel = np.empty((0, 3), dtype=np.float64)
        target_y = np.empty((0, 3), dtype=np.float64)

    # 有效区间可能被 1 个或多个 cancellation 分开，不允许错误拼接。
    runs = contiguous_true_runs(valid, t)
    approved_runs = [r for r in runs
                     if r["intervals"] >= settings.min_segment_intervals]
    # 不满足最短连续长度的片段，整段不进入数据集，但仍计入审计报告。
    approved = np.zeros(n, dtype=bool)
    for run in approved_runs:
        approved[run["start_index"]:run["end_index_exclusive"]] = True
    take = approved[interval_ids]
    included_interval_ids = interval_ids[take]

    # 按全局样本索引对齐；第二个掩码只改变训练可用范围，不改变时间轴。
    ratio = inputs.lamda_observable_mask[included_interval_ids]
    neutral = inputs.lamda_neutral_mask[included_interval_ids]
    regime_code = np.where(ratio, LAMBDA_RATIO, LAMBDA_NEUTRAL)
    weights = np.ones(len(included_interval_ids), dtype=np.float64)
    if split == "train":
        weights[neutral] = settings.neutral_train_weight

    data = {
        "z_si": z_k[take],
        "residual_y_si": target_y[take],
        "q_k_si": q_k[take],
        "q_true_next_si": q_true_next[take],
        "q_nominal_next_si": q_nominal_next[take],
        "nominal_accel_si": nominal_accel[take],
        "time_k_s": t[included_interval_ids],
        "dt_s": dt[included_interval_ids],
        "sample_weight": weights,
        "interval_index": included_interval_ids.astype(np.int64),
        "lamda_regime_code": regime_code.astype(np.int64),
    }
    if not all(np.isfinite(data[k]).all() for k in _FLOAT_FIELDS):
        raise ValueError(f"{name} 生成了非有限的训练张量")

    return {
        "name": name,
        "split": split,
        "candidate_sha256": ep.source_sha256,
        "raw_source_sha256": ep.metadata["raw_source_sha256"],
        "original_intervals": n,
        "valid_interval_inputs": int(valid.sum()),
        "cancellation_intervals": int((~valid).sum()),
        "discarded_short_intervals": int((valid & ~approved).sum()),
        "included_intervals": len(included_interval_ids),
        "ratio_intervals": int(ratio.sum()),
        "neutral_intervals": int(neutral.sum()),
        "runs": approved_runs,
        "all_runs": runs,
        "data": data,
    }


def iter_contiguous_windows(segment_id, horizon_intervals):
    """返回多步训练可用的 [start,end_exclusive) 切片。

    只要 segment_id 变化，就必须开启新窗口；禁止跨场景或无效点。
    输入 segment_id 必须按每个连续段分块，且每个编号最多出现一个块。
    """
    ids = np.asarray(segment_id)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("segment_id 必须是一维整数数组")
    if isinstance(horizon_intervals, bool) or not (
        isinstance(horizon_intervals, int) and horizon_intervals >= 1
    ):
        raise ValueError("horizon_intervals 必须是正整数")
    if len(ids) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    if len(set(map(int, ids[starts]))) != len(starts):
        raise ValueError("同一个 segment_id 出现在不相邻的区间块中")
    windows = []
    for start, end in zip(starts, ends):
        for pos in range(int(start), int(end)-horizon_intervals+1):
            windows.append((pos, pos + horizon_intervals))
    return windows


def _empty_split():
    shape_map = {
        "z_si": (0, 8),
        "residual_y_si": (0, 3),
        "q_k_si": (0, 3),
        "q_true_next_si": (0, 3),
        "q_nominal_next_si": (0, 3),
        "nominal_accel_si": (0, 3),
    }
    values = {}
    for name in _FLOAT_FIELDS:
        values[name] = np.empty(shape_map.get(name, (0,)), dtype=np.float64)
    for name in _INT_FIELDS:
        values[name] = np.empty((0,), dtype=np.int64)
    return values


def _collate_episodes(prepared):
    """同 split 可平铺为矩阵；segment_id 防止模型跨断点建窗口。"""
    result = {split: {key: [] for key in (*_FLOAT_FIELDS, *_INT_FIELDS)}
              for split in SPLITS}
    segment_report = []
    per_episode_report = []
    next_segment_id = 0

    for episode_index, ep in enumerate(prepared):
        split = ep["split"]
        data = ep["data"]
        selected_ids = data["interval_index"]
        if len(selected_ids):
            for run in ep["runs"]:
                chosen = (
                    (selected_ids >= run["start_index"])
                    & (selected_ids < run["end_index_exclusive"])
                )
                local_positions = np.flatnonzero(chosen)
                length = len(local_positions)
                if length != run["intervals"]:
                    raise ValueError(f"{ep['name']} 连续段样本数不匹配")
                start, end = int(local_positions[0]), int(local_positions[-1]+1)
                # 一个连续段保留了真实原始 interval_index，不以拼接序号充当时间。
                expected_ids = np.arange(run["start_index"],run["end_index_exclusive"])
                if not np.array_equal(selected_ids[start:end],expected_ids):
                    raise ValueError("连续区间被错误拼接或重排")
                segment_report.append({
                    "segment_id": next_segment_id,
                    "episode_index": episode_index,
                    "scene": ep["name"],
                    "split": split,
                    **run,
                    "ratio_intervals": int(np.sum(
                        data["lamda_regime_code"][start:end] == LAMBDA_RATIO
                    )),
                    "neutral_intervals": int(np.sum(
                        data["lamda_regime_code"][start:end] == LAMBDA_NEUTRAL
                    )),
                })
                next_segment_id += 1

        # 每条有效样本的 segment_id、segment_step，防止多步时误跨片段。
        assigned_segment = np.full(len(selected_ids), -1, dtype=np.int64)
        assigned_step = np.full(len(selected_ids), -1, dtype=np.int64)
        for segment in segment_report:
            if segment["episode_index"] != episode_index:
                continue
            in_run = (
                (selected_ids >= segment["start_index"])
                & (selected_ids < segment["end_index_exclusive"])
            )
            assigned_segment[in_run] = segment["segment_id"]
            assigned_step[in_run] = (
                selected_ids[in_run] - segment["start_index"]
            )
        if np.any(assigned_segment < 0) or np.any(assigned_step < 0):
            raise ValueError("某些样本没有连续区间编号")

        for key in _FLOAT_FIELDS:
            result[split][key].append(data[key])
        for key in ("interval_index", "lamda_regime_code"):
            result[split][key].append(data[key])
        result[split]["episode_index"].append(
            np.full(len(selected_ids), episode_index, dtype=np.int64)
        )
        result[split]["segment_id"].append(assigned_segment)
        result[split]["segment_step"].append(assigned_step)

        per_episode_report.append({key: ep[key] for key in (
            "name", "split", "candidate_sha256", "raw_source_sha256",
            "original_intervals", "valid_interval_inputs", "cancellation_intervals",
            "discarded_short_intervals", "included_intervals", "ratio_intervals",
            "neutral_intervals",
        )})

    split_arrays = {}
    for split in SPLITS:
        bundle = {}
        for key in (*_FLOAT_FIELDS, *_INT_FIELDS):
            parts = result[split][key]
            if parts:
                bundle[key] = np.concatenate(parts, axis=0)
            else:
                bundle[key] = _empty_split()[key]
        split_arrays[split] = bundle
        if len(bundle["z_si"]) != len(bundle["residual_y_si"]):
            raise ValueError("训练输入和三维标签行数不一致")
        if len(bundle["z_si"]):
            if not np.all(bundle["episode_index"][:-1] <= bundle["episode_index"][1:]):
                raise ValueError("Episode 编号异常交错")
    return split_arrays, segment_report, per_episode_report


def _summarize_dataset(split_arrays, segments, episodes):
    summary = {"split_counts": {}, "window_counts": {}}
    for split, arrays in split_arrays.items():
        count = len(arrays["z_si"])
        ratio = int(np.sum(arrays["lamda_regime_code"] == LAMBDA_RATIO))
        neutral = count - ratio
        split_segments = [s for s in segments if s["split"] == split]
        summary["split_counts"][split] = {
            "intervals": count,
            "ratio_intervals": ratio,
            "neutral_intervals": neutral,
            "segments": len(split_segments),
            "episodes": int(sum(e["split"] == split for e in episodes)),
            "max_contiguous_intervals": max(
                (s["intervals"] for s in split_segments), default=0
            ),
        }
        summary["window_counts"][split] = {
            f"h{h:03d}_intervals": int(sum(max(0, s["intervals"] - h + 1)
                                           for s in split_segments))
            for h in (1, 10, 50, 100)
        }

        # 这些是“尚未训练网络时”的名义单轨误差与标签幅值诊断。
        # 对测试集只做描述性报告，不计算训练归一化、也不用于调参。
        if count:
            diff = arrays["q_true_next_si"] - arrays["q_nominal_next_si"]
            residual = arrays["residual_y_si"]
            summary.setdefault("split_diagnostics", {})[split] = {
                "nominal_one_step_rmse_q_si": [
                    float(x) for x in np.sqrt(np.mean(diff**2, axis=0))
                ],
                "residual_mean_y_si": [
                    float(x) for x in np.mean(residual, axis=0)
                ],
                "residual_abs_p99_y_si": [
                    float(x) for x in np.percentile(np.abs(residual), 99, axis=0)
                ],
                "residual_rmse_y_si": [
                    float(x) for x in np.sqrt(np.mean(residual**2, axis=0))
                ],
            }
        else:
            summary.setdefault("split_diagnostics", {})[split] = None
    summary["total_intervals"] = int(sum(
        s["intervals"] for s in summary["split_counts"].values()
    ))
    return summary


def make_dataset(workspace, output=None, actuation=None,
                 reference=None, settings=None):
    """对已清洗 Workspace 做完整检查；output=None 时只审计不写文件。"""
    if actuation is None:
        actuation = NMPCActuationConfig()
    if reference is None:
        reference = NMPCReferenceParams()
    if settings is None:
        settings = ResidualDatasetConfig()
    actuation.validate()
    reference.validate()
    settings.validate()

    root = Path(workspace).expanduser().resolve(strict=True)
    registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    if registry.get("format") != "wm_recovery_experimental_registry_v1":
        raise ValueError("workspace 不是原项目支持的 experimental registry")
    # 禁止误用 60km/h Brake-turn 尚未裁去严重侧滑的原始 Workspace。
    # 当前 data_cleaning v2 的干净工作区已明确声明 quality_cleaned=True。
    if registry.get("quality_cleaned") is not True:
        raise ValueError(
            "需要输入 data_cleaning clean 生成的 quality_cleaned=True Workspace，"
            "不要使用尚未清洗的原始 merge Workspace"
        )
    if not registry.get("episodes"):
        raise ValueError("workspace 中没有训练轨迹")

    dest = None
    if output is not None:
        dest = Path(output).expanduser().resolve()
        if dest.exists():
            raise FileExistsError(f"目标已存在：{dest}（拒绝覆盖）")

    prepared = []
    used_raw = {}
    used_candidate = {}
    scene_names = set()
    condition_ids = set()
    for entry in registry["episodes"]:
        name, split, ep_dir, manifest, cand_dir = _load_manifest(root, entry)
        if name in scene_names:
            raise ValueError(f"同名轨迹重复：{name}")
        scene_names.add(name)
        condition = manifest["condition_id"]
        if condition in condition_ids:
            raise ValueError(f"同一 condition_id 被重复使用：{condition}")
        condition_ids.add(condition)
        episode = prepare_episode(
            name, split, cand_dir, actuation=actuation,
            reference=reference, settings=settings,
        )
        for label, seen in (
            ("raw_source_sha256", used_raw),
            ("candidate_sha256", used_candidate),
        ):
            actual = episode[label]
            expected = manifest[label]
            if actual != expected:
                raise ValueError(f"{name} 的 {label} 与 manifest 来源指纹不匹配")
            if actual in seen:
                raise ValueError(
                    f"{name} 与 {seen[actual]} 共用了同一 {label}，禁止跨分区泄漏"
                )
            seen[actual] = name
        prepared.append(episode)
        print(
            f"{name:36} "
            f"有效 {episode['included_intervals']:4}/{episode['original_intervals']:4}，"
            f"ratio {episode['ratio_intervals']:4}，"
            f"neutral {episode['neutral_intervals']:4}，"
            f"连续段 {len(episode['runs'])}"
        )

    split_arrays, segments, episodes = _collate_episodes(prepared)
    report = _summarize_dataset(split_arrays, segments, episodes)
    for split in SPLITS:
        if report["split_counts"][split]["intervals"] < 1:
            raise ValueError(f"{split} 无有效训练标签，必须重新检查数据")
    report.update({
        "format": DATASET_VERSION,
        "source_workspace": str(root),
        "source_registry_sha256": _sha256_file(root / "registry.json"),
        "input_names": list(NMPC_Z_NAMES),
        "q_names": list(NMPC_Q_NAMES),
        "target_names": list(NMPC_Y_NAMES),
        "reference_version": REFERENCE_VERSION,
        "reference_parameters": asdict(reference),
        "actuation_assumptions": asdict(actuation),
        "dataset_settings": asdict(settings),
        "regime_code_definition": {
            str(LAMBDA_RATIO): "ratio_from_axle_forces",
            str(LAMBDA_NEUTRAL): "neutral_assumed_not_measured",
        },
        "episode_count": len(episodes),
        "episodes": episodes,
        "segments": segments,
        "training_ready_for_closed_loop_nmpc": False,
        "notes": [
            "仅为三通道残差的实验数据，不是原 7DOF World Model 数据格式。",
            "用 t_k 的等效轮胎力构造 z_k，t_{k+1} 只用于监督目标。",
            "neutral=0.5 是约定值；不得作为前后力分配的实测标签。",
            "输出为区间等效加速度残差，不等于真实瞬时加速度传感器数据。",
            "Mz_tv 与 NMPC 可执行控制力矩的对应仍未经完整验证。",
            "训练归一化只能用 train.npz 拟合；验证/测试不可参与拟合。",
        ],
    })

    print("\n分区样本:", {
        split: data["intervals"]
        for split, data in report["split_counts"].items()
    })
    print("有效残差样本总数:", report["total_intervals"])
    print("可用 1 秒（100×10ms）连续窗口:", {
        split: windows["h100_intervals"]
        for split, windows in report["window_counts"].items()
    })

    # 先把全部数据审计通过，再创建输出，避免不完整目录误当成功。
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=False)
        for split, values in split_arrays.items():
            np.savez_compressed(dest / f"{split}.npz", **values)
        with (dest / "dataset_manifest.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("输出数据集:", dest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CarSim 清洗轨迹 -> NMPC 单轨名义 RK4 -> 八输入三残差数据集"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("audit", "build"):
        p = sub.add_parser(action)
        p.add_argument("--workspace", required=True, type=Path)
        if action == "build":
            p.add_argument("--output", required=True, type=Path)
        p.add_argument("--min-segment-intervals", type=int, default=1)
        p.add_argument("--neutral-train-weight", type=float, default=1.0)
        p.add_argument("--rk4-substeps", type=int, default=1)
        p.add_argument("--front-track-m", type=float, default=1.605)
        p.add_argument("--rear-track-m", type=float, default=1.605)
        p.add_argument("--mu-hat", type=float, default=0.8)
        p.add_argument("--min-abs-fx-all-n", type=float, default=50.0)
        p.add_argument("--max-abs-neutral-axle-force-n", type=float, default=50.0)
        p.add_argument("--neutral-lamda", type=float, default=0.5)
    args = parser.parse_args(argv)
    acts = NMPCActuationConfig(
        front_track_m=args.front_track_m,
        rear_track_m=args.rear_track_m,
        mu_hat=args.mu_hat,
        min_abs_fx_all_n=args.min_abs_fx_all_n,
        max_abs_neutral_axle_force_n=args.max_abs_neutral_axle_force_n,
        neutral_lamda=args.neutral_lamda,
    )
    settings = ResidualDatasetConfig(
        min_segment_intervals=args.min_segment_intervals,
        neutral_train_weight=args.neutral_train_weight,
        rk4_substeps=args.rk4_substeps,
    )
    make_dataset(
        args.workspace,
        output=args.output if args.command == "build" else None,
        actuation=acts,
        reference=NMPCReferenceParams(),
        settings=settings,
    )


if __name__ == "__main__":
    main()
