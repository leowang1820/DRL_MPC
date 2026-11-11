# -*- coding: utf-8 -*-
"""从 CarSim 61 个原生 Export 构造 NMPC 八维物理输入（只读）。

本模块仅负责构造等效物理量，不训练神经网络，不修改 CarSim。

网络输入顺序，全部采用 SI 单位：
    z = [vy, r, vx, delta_f, Mz_tv, Fx_all, lamda, mu_hat]
        vy:       m/s
        r:        rad/s
        vx:       m/s
        delta_f:  rad
        Mz_tv:    N*m（由左右纵向力差估计的附加横摆力矩）
        Fx_all:   N（四轮胎局部纵向力之和）
        lamda:    无量纲（前轴纵向力 / 全车纵向力）
        mu_hat:   无量纲（默认 0.8 的已假定工况系数）

当前明确采用以下建模假设（实际物理等效性尚未独立验证）：
1. Vx/Vy 的 CarSim 原生单位为 km/h；AVz 为 deg/s。
2. Steer_L1/R1 为前轮实际转向角，原生单位 deg。
3. 四轮 Fx/Fy 为各轮胎局部坐标系下的力，单位 N，x 向前、y 向左。
4. 后轮无转角；车辆车身 y 向左，正横摆为逆时针。
5. 前/后轮距暂时都使用 1.605m（来自用户的 NMPC 代码）。
6. Mz_tv 仅考虑轮距与各轮车身坐标纵向力的差，
   不等于 CarSim 实测的总横摆力矩，也不等于控制器指令已经被执行器跟踪。
7. mu_hat 取实验确认的名义路面参数 0.8；不使用 MuX_* 代替在线附着估计。
8. lamda 不进行 0.4～0.6 限幅，分成三类：
   * ratio：总纵向力足够大，直接用 Fx_front/Fx_all；
   * neutral：总纵向力和两轴力都很小，约定 lamda=0.5（不是实测比例）；
   * cancellation：总力接近零但两轴力明显抵消，lamda 保持 NaN。
   保留每个采样时刻的分类与连续有效区间信息。

时间语义：
  raw_exports[k]、z[k] 为 t_k 时刻状态；
  若用于监督 [t_k,t_{k+1}) 区间，应使用 z[:-1]，
  不允许把 t_{k+1} 的轮胎力输入到预测 t_{k+1} 的网络中。

现有项目接口：
  from wm_recovery.nmpc_actuation import (
      NMPCActuationConfig, derive_nmpc_actuation, load_from_candidate
  )

  result = derive_nmpc_actuation(raw_exports)   # (N+1,61) -> (N+1,8)
  z_k = result.z_si[:-1]                         # 对应 N 个控制区间

  # 检查已清洗的候选轨迹及它的原始 61 通道来源：
  time_s, result = load_from_candidate(candidate_dir)

命令行只读审计：
  python -m wm_recovery.nmpc_actuation audit \\
    --workspace artifacts/wm_recovery/workspace_40_60_clean_02
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .carsim_export_schema import EXPORT_COUNT, get_export_index


# ---------------------------------------------------------------------
# 网络八个输入的顺序（不能随意调整，否则训练和部署权重会错位）
# ---------------------------------------------------------------------
NMPC_INPUT_NAMES = (
    "vy_mps",
    "yaw_rate_radps",
    "vx_mps",
    "delta_front_rad",
    "mz_tv_estimated_nm",
    "fx_all_n",
    "lamda",
    "mu_hat",
)

# 对应 CarSim 导出的关键原生信号。
_REQUIRED_EXPORTS = (
    "Vx", "Vy", "AVz", "Steer_L1", "Steer_R1",
    "Fx_L1", "Fx_R1", "Fx_L2", "Fx_R2",
    "Fy_L1", "Fy_R1",
)

_KMH_TO_MPS = 1.0 / 3.6
_DEG_TO_RAD = math.pi / 180.0


@dataclass(frozen=True)
class NMPCActuationConfig:
    """可追溯的 CarSim -> NMPC 等效映射假设。"""

    front_track_m: float = 1.605
    rear_track_m: float = 1.605
    mu_hat: float = 0.8

    # 这是分母的数值可辨识门槛，不是轮胎力/执行器的物理上限。
    # 若 0 < |Fx_all| <= 此阈值，强行除法会放大数值噪声。
    min_abs_fx_all_n: float = 50.0

    # 前后轴都接近零纵向力时，允许采用中性分配约定的各轴阈值。
    max_abs_neutral_axle_force_n: float = 50.0
    neutral_lamda: float = 0.5

    # 仅用于统计是否偏离用户预期的 0.4~0.6，不做实际裁剪。
    expected_lamda_low: float = 0.4
    expected_lamda_high: float = 0.6

    def validate(self):
        for key in (
            "front_track_m", "rear_track_m", "mu_hat",
            "min_abs_fx_all_n", "max_abs_neutral_axle_force_n",
            "neutral_lamda", "expected_lamda_low", "expected_lamda_high",
        ):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} 必须是有限数字")
        if self.front_track_m <= 0 or self.rear_track_m <= 0:
            raise ValueError("前、后轮距必须大于 0")
        if self.mu_hat <= 0:
            raise ValueError("mu_hat 必须为正")
        if self.min_abs_fx_all_n < 0:
            raise ValueError("分母最小幅值不得为负")
        if self.max_abs_neutral_axle_force_n < 0:
            raise ValueError("中性分配的轴向力阈值不得为负")
        if self.expected_lamda_low >= self.expected_lamda_high:
            raise ValueError("预期 lamda 区间的上下限不正确")


@dataclass
class NMPCActuationResult:
    """状态时刻一一对齐的 NMPC 输入与可审计诊断数组。

    z_si: shape=(T,8)；顺序固定为 NMPC_INPUT_NAMES。
    valid_mask: shape=(T,)；标识八维输入均为有限值的样本。
    lamda_observable_mask: shape=(T,)；标识 Fx_all 足够远离零的样本。

    提示：valid_mask 不意味着物理模型已验证、轮胎极限有效，
    也不意味着该时刻符合其他独立数据清洗条件。
    """

    z_si: np.ndarray
    valid_mask: np.ndarray
    lamda_observable_mask: np.ndarray  # 仅表示实测轴力比值可计算
    lamda_neutral_mask: np.ndarray     # 采用约定值 0.5 的样本
    lamda_cancellation_mask: np.ndarray  # 轴力相互抵消、无法定义比例的样本
    lamda_regime: np.ndarray          # 各状态时刻的三分类标签
    lamda_in_expected_band_mask: np.ndarray
    delta_left_right_diff_deg: np.ndarray
    fx_front_n: np.ndarray
    fx_rear_n: np.ndarray
    fx_all_n: np.ndarray
    mz_tv_estimated_nm: np.ndarray
    assumptions: dict

    def as_summary(self):
        """统计状态时刻的三类比例，明确区分实际力比值和中性约定值。"""
        ratio = self.lamda_observable_mask
        neutral = self.lamda_neutral_mask
        cancelled = self.lamda_cancellation_mask
        defined = ratio | neutral
        lmd = self.z_si[:, 6]
        n = self.z_si.shape[0]
        if int(ratio.sum() + neutral.sum() + cancelled.sum()) != n:
            raise ValueError("三个 lamda 状态未完整覆盖所有样本")
        return {
            "format": "wm_recovery_nmpc_actuation_audit_v2",
            "state_samples": int(n),
            "valid_z_samples": int(self.valid_mask.sum()),
            "lamda_observable_samples": int(ratio.sum()),
            "lamda_unobservable_samples": int((~ratio).sum()),
            "lamda_ratio_samples": int(ratio.sum()),
            "lamda_neutral_assumed_samples": int(neutral.sum()),
            "lamda_cancellation_samples": int(cancelled.sum()),
            "lamda_defined_samples": int(defined.sum()),
            "lamda_outside_expected_band_samples": int(
                (defined & ~self.lamda_in_expected_band_mask).sum()
            ),
            "lamda_ratio_outside_expected_band_samples": int(
                (ratio & ~self.lamda_in_expected_band_mask).sum()
            ),
            "lamda_min": float(np.min(lmd[defined])) if defined.any() else None,
            "lamda_max": float(np.max(lmd[defined])) if defined.any() else None,
            "lamda_ratio_min": float(np.min(lmd[ratio])) if ratio.any() else None,
            "lamda_ratio_max": float(np.max(lmd[ratio])) if ratio.any() else None,
            "min_fx_all_n": float(np.min(self.fx_all_n)),
            "max_fx_all_n": float(np.max(self.fx_all_n)),
            "min_mz_tv_nm": float(np.min(self.mz_tv_estimated_nm)),
            "max_mz_tv_nm": float(np.max(self.mz_tv_estimated_nm)),
            "max_delta_lr_difference_deg": float(
                np.max(np.abs(self.delta_left_right_diff_deg))
            ),
            "assumptions": dict(self.assumptions),
            "note": (
                "lamda 不作数值限幅；中性比例属于建模约定而非真实测量，"
                "抵消状态保持 NaN；Mz_tv 仍是未经独立物理验证的附加横摆力矩估计。"
            ),
        }


def _as_2d_exports(raw_exports):
    """严格检查固定 61 通道原生输出，允许单个状态或完整时间序列。"""
    values = np.asarray(raw_exports, dtype=np.float64)
    if values.shape == (EXPORT_COUNT,):
        values = values.reshape(1, EXPORT_COUNT)
    if values.ndim != 2 or values.shape[1] != EXPORT_COUNT:
        raise ValueError(
            f"CarSim Export 需要 (61,) 或 (T,{EXPORT_COUNT})，"
            f"实际得到 {values.shape}"
        )
    if values.shape[0] < 1:
        raise ValueError("Export 不允许为空")
    if not np.isfinite(values).all():
        raise ValueError("CarSim Export 含 NaN 或 Inf，禁止静默清洗")
    return values


def derive_nmpc_actuation(raw_exports, config=None):
    """将 CarSim 61 通道原生输出转换为 NMPC 八输入。

    核心公式（在约定的 tire-local Fx/Fy 坐标系下）：

      delta_f = 0.5 * (delta_FL + delta_FR)

      Fx_front = Fx_FL + Fx_FR
      Fx_rear  = Fx_RL + Fx_RR
      Fx_all   = Fx_front + Fx_rear
      lamda    = Fx_front / Fx_all        # 不裁剪

      Fx_b_FL = Fx_FL*cos(delta_FL) - Fy_FL*sin(delta_FL)
      Fx_b_FR = Fx_FR*cos(delta_FR) - Fy_FR*sin(delta_FR)
      Fx_b_RL = Fx_RL                     # 假定后轮不转向
      Fx_b_RR = Fx_RR

      Mz_tv = (Bf/2)*(Fx_b_FR-Fx_b_FL)
            + (Br/2)*(Fx_b_RR-Fx_b_RL)

    注意：这是 NMPC 附加横摆力矩的等效估计，而非总偏航力矩。
    """
    if config is None:
        config = NMPCActuationConfig()
    config.validate()
    raw = _as_2d_exports(raw_exports)

    # get_export_index 会拒绝重复变量（例如 Steer_SW），避免错误取列。
    columns = {name: get_export_index(name) for name in _REQUIRED_EXPORTS}

    def col(name):
        return raw[:, columns[name]]

    # 速度、横摆角速度：沿用现有 carsim_si.py 的单位约定。
    vx = col("Vx") * _KMH_TO_MPS
    vy = col("Vy") * _KMH_TO_MPS
    yaw_rate = col("AVz") * _DEG_TO_RAD

    # 用左右实际转向角平均，建立单轨等效前轮角。
    delta_fl = col("Steer_L1") * _DEG_TO_RAD
    delta_fr = col("Steer_R1") * _DEG_TO_RAD
    delta_front = 0.5 * (delta_fl + delta_fr)
    delta_lr_diff_deg = np.rad2deg(delta_fl - delta_fr)

    # 前/后轴纵向力仍保留 tire-local 语义；供 NMPC 摩擦圆公式使用。
    fx_fl = col("Fx_L1")
    fx_fr = col("Fx_R1")
    fx_rl = col("Fx_L2")
    fx_rr = col("Fx_R2")
    fy_fl = col("Fy_L1")
    fy_fr = col("Fy_R1")

    fx_front = fx_fl + fx_fr
    fx_rear = fx_rl + fx_rr
    fx_all = fx_front + fx_rear

    # 不进行 0.4～0.6 的数值限幅：分类并保留来源标记。
    # 对每一个状态时刻进行互斥且完备的三状态划分。
    # 状态一：总纵向力足够大，按实际前轴/总力求比例。
    lamda_observable = np.abs(fx_all) > config.min_abs_fx_all_n

    # 状态二：总纵向力较小，前后轴纵向力也都较小。
    # 这里赋予中性约定值，不能把它视为实测力分配比例。
    lamda_neutral = (
        ~lamda_observable
        & (np.abs(fx_front) <= config.max_abs_neutral_axle_force_n)
        & (np.abs(fx_rear) <= config.max_abs_neutral_axle_force_n)
    )

    # 状态三：总纵向力虽小，但至少一个轴存在较大纵向力；
    # 此时前后轴力可能明显抵消，单一 lambda 无法可靠表达。
    lamda_cancellation = ~(lamda_observable | lamda_neutral)

    # 不对实测比值限幅，允许 lambda 大于 1 或小于 0。
    lamda = np.full(fx_all.shape, np.nan, dtype=np.float64)
    np.divide(fx_front, fx_all, out=lamda, where=lamda_observable)
    lamda[lamda_neutral] = config.neutral_lamda

    lamda_regime = np.full(fx_all.shape, "cancellation", dtype="<U12")
    lamda_regime[lamda_observable] = "ratio"
    lamda_regime[lamda_neutral] = "neutral"

    lamda_defined = lamda_observable | lamda_neutral
    lamda_expected = (
        lamda_defined
        & (lamda >= config.expected_lamda_low)
        & (lamda <= config.expected_lamda_high)
    )

    # 将轮胎坐标系纵向力投影到车身纵向，用于估计附加横摆力矩。
    fx_fl_body = fx_fl * np.cos(delta_fl) - fy_fl * np.sin(delta_fl)
    fx_fr_body = fx_fr * np.cos(delta_fr) - fy_fr * np.sin(delta_fr)
    fx_rl_body = fx_rl
    fx_rr_body = fx_rr

    # 正方向：右轮向前的纵向力更大 -> 正的逆时针附加横摆力矩。
    mz_tv = (
        0.5 * config.front_track_m * (fx_fr_body - fx_fl_body)
        + 0.5 * config.rear_track_m * (fx_rr_body - fx_rl_body)
    )

    mu_hat = np.full_like(vx, config.mu_hat, dtype=np.float64)

    # 保持训练与未来 NMPC 中完全相同的 8 维物理单位顺序。
    z = np.column_stack((
        vy, yaw_rate, vx, delta_front, mz_tv, fx_all, lamda, mu_hat
    ))

    valid_mask = np.isfinite(z).all(axis=1)
    # 非数值问题与“只因 lamda 未能定义”相区分。
    if np.any(~valid_mask & lamda_defined):
        raise ValueError("已定义的 lamda 仍产生了非有限网络输入")
    if not np.array_equal(valid_mask, lamda_defined):
        raise ValueError("八维输入有效掩码与 lamda 三状态分类不匹配")

    return NMPCActuationResult(
        z_si=z,
        valid_mask=valid_mask,
        lamda_observable_mask=lamda_observable,
        lamda_neutral_mask=lamda_neutral,
        lamda_cancellation_mask=lamda_cancellation,
        lamda_regime=lamda_regime,
        lamda_in_expected_band_mask=lamda_expected,
        delta_left_right_diff_deg=delta_lr_diff_deg,
        fx_front_n=fx_front.copy(),
        fx_rear_n=fx_rear.copy(),
        fx_all_n=fx_all.copy(),
        mz_tv_estimated_nm=mz_tv.copy(),
        assumptions=asdict(config),
    )


def _file_sha256(path):
    """记录原始轨迹文件的 SHA256；清洗不改变来源文件。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_from_candidate(candidate_dir, config=None):
    """读取已清洗/未清洗的 CandidateEpisode，并核对原始 61 通道来源。

    返回：
      time_s: 原始状态采样时间数组 (N+1,)
      result: NMPCActuationResult，其 z_si 为 (N+1,8)

    下游训练应该使用：
      z_k = result.z_si[:-1][result.valid_mask[:-1]]
    但必须以同一掩码筛选 [t_k,t_{k+1}] 对应的监督标签，
    且不得跨越已删除的时间区间。
    """
    from .offline import CandidateEpisode

    candidate_path = Path(candidate_dir).expanduser().resolve(strict=True)
    ep = CandidateEpisode.load(candidate_path)

    raw_source = ep.metadata.get("raw_source")
    expected_sha = ep.metadata.get("raw_source_sha256")
    if not isinstance(raw_source, str) or not raw_source:
        raise ValueError("candidate_report.json 缺少 raw_source")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError("candidate_report.json 缺少 raw_source_sha256")

    raw_path = Path(raw_source).expanduser()
    if not raw_path.is_absolute():
        raw_path = candidate_path / raw_path
    raw_path = raw_path.resolve(strict=True)
    if _file_sha256(raw_path) != expected_sha:
        raise ValueError("原始 raw_capture.npz SHA256 已变化，拒绝训练前映射")

    # 保留的候选可能只是原始记录的前缀；必须按真实时刻一一核对。
    with np.load(raw_path, allow_pickle=False) as saved:
        if "time_s" not in saved or "raw_exports" not in saved:
            raise ValueError("raw_capture.npz 缺少时间或原生输出")
        raw_time = np.asarray(saved["time_s"], dtype=np.float64)
        raw_export = np.asarray(saved["raw_exports"], dtype=np.float64)

    keep_states = len(ep.time)
    if raw_export.shape != (len(raw_time), EXPORT_COUNT):
        raise ValueError("原始 61 通道 Export 与时间轴长度不一致")
    if keep_states > len(raw_time):
        raise ValueError("候选轨迹比原始记录更长，无法对齐")
    if not np.allclose(raw_time[:keep_states], ep.time, atol=1e-8, rtol=0):
        raise ValueError("清洗后候选时间轴与原始 CarSim 记录前缀不一致")

    result = derive_nmpc_actuation(raw_export[:keep_states], config)

    # 与当前 7DOF Candidate 状态读取器比较三个共同物理状态。
    # Candidate.state 列顺序：[vx,vy,r,wFL,wFR,wRL,wRR]。
    if not np.allclose(
        result.z_si[:, [0, 1, 2]],
        ep.state[:, [1, 2, 0]],
        rtol=0, atol=1e-7,
    ):
        raise ValueError("NMPC 的 vx/vy/r 与已有候选的 SI 状态不一致")
    if not np.allclose(
        result.z_si[:, 3],
        np.mean(ep.measured_steer, axis=1),
        rtol=0, atol=1e-9,
    ):
        raise ValueError("平均实际前轮转角与候选角度不一致")

    return ep.time.copy(), result


def contiguous_true_runs(mask, time_s):
    """寻找最长的连续有效控制区间，不跨越任何无效采样。

    mask[k] 对应 [time_s[k], time_s[k+1]) 时间区间。
    start_index 包含在区间内，end_index_exclusive 不包含。
    """
    mask = np.asarray(mask)
    t = np.asarray(time_s, dtype=np.float64)
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("有效区间掩码必须是一维布尔数组")
    if t.ndim != 1 or len(t) != len(mask) + 1:
        raise ValueError("时间戳长度必须比控制区间掩码多 1")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("时间戳必须有限且严格递增")

    changes = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        {
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "intervals": int(end - start),
            "start_time_s": float(t[start]),
            "end_time_s": float(t[end]),
            "duration_s": float(t[end] - t[start]),
        }
        for start, end in zip(starts, ends)
    ]


def _longest_run(runs):
    """返回实际最长连续区间；若无有效区间则返回 None。"""
    return max(runs, key=lambda r: r["intervals"], default=None)


def _interval_counts(res):
    """最后一个状态点没有对应的下一个控制区间，因此只统计前 N 个。"""
    measured = res.lamda_observable_mask[:-1]
    neutral = res.lamda_neutral_mask[:-1]
    cancelled = res.lamda_cancellation_mask[:-1]
    valid = res.valid_mask[:-1]
    n = len(valid)
    if int(measured.sum() + neutral.sum() + cancelled.sum()) != n:
        raise ValueError("三种 lamda 状态不能完整划分所有控制区间")
    if not np.array_equal(valid, measured | neutral):
        raise ValueError("有效控制区间只能属于 ratio 或 neutral")
    return {
        "control_intervals": n,
        "ratio_interval_inputs": int(measured.sum()),
        "neutral_interval_inputs": int(neutral.sum()),
        "cancellation_interval_inputs": int(cancelled.sum()),
        "valid_interval_inputs": int(valid.sum()),
        "unobservable_interval_inputs": int((~measured).sum()),
        "invalid_interval_inputs": int((~valid).sum()),
    }


def audit_workspace(workspace, config=None):
    """只读审计：逐轨迹、分车速、分数据分区统计三状态及连续段。"""
    if config is None:
        config = NMPCActuationConfig()
    config.validate()

    root = Path(workspace).expanduser().resolve(strict=True)
    registry_path = root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("format") != "wm_recovery_experimental_registry_v1":
        raise ValueError("输入不是受支持的多车速实验 Workspace")

    episodes = []
    speed_groups = {}
    split_groups = {}
    all_longest = None

    for item in registry["episodes"]:
        ep_dir = Path(item["episode_path"])
        manifest = json.loads(
            (ep_dir / "experimental_episode.json").read_text(encoding="utf-8")
        )
        cand_dir = Path(manifest["candidate_path"])
        if not cand_dir.is_absolute():
            cand_dir = ep_dir / cand_dir

        times, res = load_from_candidate(cand_dir, config)
        summary = res.as_summary()
        counts = _interval_counts(res)
        valid_runs = contiguous_true_runs(res.valid_mask[:-1], times)
        measured_runs = contiguous_true_runs(
            res.lamda_observable_mask[:-1], times
        )
        longest_valid = _longest_run(valid_runs)
        longest_measured = _longest_run(measured_runs)
        speed_key = str(item["name"]).split("_", 1)[0]
        if not speed_key.startswith("v") or not speed_key[1:].isdigit():
            speed_key = "unspecified_speed"
        split_key = item["split"]

        summary.update(counts)
        summary.update({
            "scene": item["name"],
            "split": split_key,
            "speed": speed_key,
            "valid_runs_count": len(valid_runs),
            "ratio_runs_count": len(measured_runs),
            "longest_contiguous_valid_run": longest_valid,
            "longest_contiguous_ratio_run": longest_measured,
            "longest_contiguous_valid_intervals": (
                longest_valid["intervals"] if longest_valid else 0
            ),
            "longest_contiguous_valid_duration_s": (
                longest_valid["duration_s"] if longest_valid else 0.0
            ),
            "longest_contiguous_ratio_duration_s": (
                longest_measured["duration_s"] if longest_measured else 0.0
            ),
            "valid_interval_fraction": (
                counts["valid_interval_inputs"] / counts["control_intervals"]
                if counts["control_intervals"] else 0.0
            ),
        })
        episodes.append(summary)
        speed_groups.setdefault(speed_key, []).append(summary)
        split_groups.setdefault(split_key, []).append(summary)

        if longest_valid is not None and (
            all_longest is None
            or longest_valid["intervals"] > all_longest["intervals"]
        ):
            all_longest = {"scene": item["name"], **longest_valid}

        print(
            f"{item['name']:36} "
            f"直接比例 {counts['ratio_interval_inputs']:4}  "
            f"中性约定 {counts['neutral_interval_inputs']:4}  "
            f"抵消不可用 {counts['cancellation_interval_inputs']:4}  "
            f"有效 {counts['valid_interval_inputs']:4}/{counts['control_intervals']:4}  "
            f"最长连续 {summary['longest_contiguous_valid_duration_s']:.2f}s"
        )

    count_fields = (
        "control_intervals", "ratio_interval_inputs", "neutral_interval_inputs",
        "cancellation_interval_inputs", "valid_interval_inputs",
        "invalid_interval_inputs", "unobservable_interval_inputs",
    )

    def aggregate(group):
        counts = {key: int(sum(ep[key] for ep in group)) for key in count_fields}
        n = counts["control_intervals"]
        counts.update({
            "episode_count": len(group),
            "valid_fraction": counts["valid_interval_inputs"] / n if n else 0.0,
            "ratio_fraction": counts["ratio_interval_inputs"] / n if n else 0.0,
            "neutral_fraction": counts["neutral_interval_inputs"] / n if n else 0.0,
            "cancellation_fraction": (
                counts["cancellation_interval_inputs"] / n if n else 0.0
            ),
            "longest_contiguous_valid_run": (
                max(
                    (
                        {
                            "scene": ep["scene"],
                            **ep["longest_contiguous_valid_run"],
                        }
                        for ep in group
                        if ep["longest_contiguous_valid_run"] is not None
                    ),
                    key=lambda r: r["intervals"], default=None,
                )
            ),
        })
        return counts

    totals = aggregate(episodes)
    return {
        "format": "wm_recovery_nmpc_actuation_workspace_audit_v2",
        "workspace": str(root),
        "episode_count": len(episodes),
        "total_interval_inputs": totals["control_intervals"],
        "valid_interval_inputs": totals["valid_interval_inputs"],
        "regime_totals": totals,
        "by_speed": {k: aggregate(v) for k, v in sorted(speed_groups.items())},
        "by_split": {k: aggregate(v) for k, v in sorted(split_groups.items())},
        "longest_contiguous_valid_run": all_longest,
        "assumptions": asdict(config),
        "episodes": episodes,
        "warning": (
            "实际力比值绝不限幅。中性 lambda 是明确标记的模型约定值，"
            "并非实测。抵消区间不可直接训练，连续段不能跨轨迹或跨无效间隔。"
            "本报告不证明执行器物理映射已完成验证。"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CarSim 61 Export -> NMPC 8D 等效输入，支持只读审计"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("audit", help="分析已合并/已清洗的 Workspace")
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument(
        "--front-track-m", type=float, default=1.605,
        help="前轮距，默认沿用 NMPC 暂定参数 1.605 m",
    )
    p.add_argument(
        "--rear-track-m", type=float, default=1.605,
        help="后轮距，默认沿用 NMPC 暂定参数 1.605 m",
    )
    p.add_argument("--mu-hat", type=float, default=0.8)
    p.add_argument(
        "--min-abs-fx-all-n", type=float, default=50.0,
        help="计算 lamda 时的分母可辨识门槛，非幅值限幅",
    )
    p.add_argument(
        "--max-abs-neutral-axle-force-n", type=float, default=50.0,
        help="中性约定要求的每轴纵向力绝对值上限（N）",
    )
    p.add_argument(
        "--neutral-lamda", type=float, default=0.5,
        help="前后轴纵向力都很小时的模型约定比例，非测量值",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="可选：另存只读审计结果为 JSON；拒绝覆盖已有文件",
    )
    args = parser.parse_args(argv)

    config = NMPCActuationConfig(
        front_track_m=args.front_track_m,
        rear_track_m=args.rear_track_m,
        mu_hat=args.mu_hat,
        min_abs_fx_all_n=args.min_abs_fx_all_n,
        max_abs_neutral_axle_force_n=args.max_abs_neutral_axle_force_n,
        neutral_lamda=args.neutral_lamda,
    )
    report = audit_workspace(args.workspace, config)
    print(
        f"\n总计：{report['valid_interval_inputs']}/"
        f"{report['total_interval_inputs']} 个区间具有有限的 NMPC 八维输入。"
    )
    total = report["regime_totals"]
    print(
        "分布：直接比例={ratio}，中性约定={neutral}，抵消不可用={cancel}；"
        "有效率={pct:.2f}%".format(
            ratio=total["ratio_interval_inputs"],
            neutral=total["neutral_interval_inputs"],
            cancel=total["cancellation_interval_inputs"],
            pct=100.0 * total["valid_fraction"],
        )
    )
    if report["longest_contiguous_valid_run"]:
        run = report["longest_contiguous_valid_run"]
        print(
            f"全局最长连续有效区间：{run['scene']}，"
            f"{run['duration_s']:.2f}s（不跨轨迹）"
        )
    if args.output is not None:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("审计报告已保存：", target)


if __name__ == "__main__":
    main()
