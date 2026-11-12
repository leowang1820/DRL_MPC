# -*- coding: utf-8 -*-
"""NMPC 三通道名义模型与残差模型的长时域自由滚动器。

本模块采取用户确认的“已知未来等效执行器响应”条件：
- 每个 10ms 区间的 delta_f, Mz_tv, Fx_all, lamda, mu_hat 是已知外部序列。
- 不使用这些时刻的 CarSim vy/r/vx 纠正预测状态。
- 名义模型和残差模型都以同一 q0 独立积分，
  每个 RK4 子阶段都用自身预测 q 重新计算轮胎力与网络残差。
- 数据采样率 100 Hz；最多 500 个区间对应 NMPC 的 5 秒预测时域。

这不是“同轮端转矩指令，执行器也由模型完全独立仿真”的实验；
它是已知未来实际等效执行器轨迹条件下的自由车辆状态预测。
"""

from __future__ import annotations

import math
import numpy as np

from .nmpc_reference import (
    NMPCReferenceParams, nmpc_accelerations,
)


def _state_and_exogenous(q_si, external_si):
    q = np.asarray(q_si, dtype=np.float64)
    external = np.asarray(external_si, dtype=np.float64)
    one = q.ndim == 1
    if one:
        q = q.reshape(1, 3)
    if external.ndim == 1:
        external = external.reshape(1, 5)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0:
        raise ValueError("车辆状态应为 (3,) 或 (B,3)")
    if external.shape != (len(q), 5):
        raise ValueError("外部执行器量应为 (5,) 或与车辆数一致的 (B,5)")
    if not np.isfinite(q).all() or not np.isfinite(external).all():
        raise ValueError("滚动状态或外部已知量包含 NaN/Inf")
    if np.any(external[:, 4] <= 0):
        raise ValueError("mu_hat 必须为正")
    return q.copy(), external.copy(), one


def rk4_free_step(
    q_si, external_si, dt_s=0.01, *,
    residual_model=None, reference=None, substeps=1,
):
    """单次自由 RK4 积分，调用的每一级都使用当前预测的 q。

    q_si: [vy, r, vx]，SI
    external_si: [delta_f, Mz_tv, Fx_all, lamda, mu_hat]，SI
    residual_model: None=纯名义单轨，传入 MLP=名义动力学+三维残差。
    """
    if reference is None:
        reference = NMPCReferenceParams()
    reference.validate()
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps 必须为正整数")
    q, ext, one = _state_and_exogenous(q_si, external_si)
    dt = np.asarray(dt_s, dtype=np.float64)
    if dt.ndim == 0:
        dt = np.full((len(q),), float(dt))
    if dt.shape != (len(q),) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("dt_s 需要是正标量或与 batch 一致的 (B,) 数组")
    h = dt[:, None] / substeps

    def derivative(q_state):
        z = np.concatenate((q_state, ext), axis=1)
        nominal = nmpc_accelerations(z, reference)
        if residual_model is None:
            return nominal
        residual = np.asarray(
            residual_model.predict_numpy(z), dtype=np.float64
        )
        if residual.shape != nominal.shape or not np.isfinite(residual).all():
            raise ValueError("残差网络输出必须是有限的 (B,3)")
        return nominal + residual

    for _ in range(substeps):
        k1 = derivative(q)
        k2 = derivative(q + 0.5 * h * k1)
        k3 = derivative(q + 0.5 * h * k2)
        k4 = derivative(q + h * k3)
        q = q + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(q).all():
            raise FloatingPointError("自由滚动出现 NaN/Inf")
    return q[0] if one else q


def rollout_known_future(
    initial_q_si, observed_z_si, *,
    residual_model=None, dt_s=0.01,
    reference=None, substeps=1,
):
    """单条开环轨迹，允许未来外部执行器量已知，但禁止未来真值车辆状态回灌。

    observed_z_si: (H,8)；其中第0～2列除了检查形状外完全不使用！
    每个时间区间只提取第3～7列作为已知的外部执行器参数。

    返回预测状态轨迹 (H+1,3)，第0行等于 initial_q_si。
    """
    q0 = np.asarray(initial_q_si, dtype=np.float64)
    schedule = np.asarray(observed_z_si, dtype=np.float64)
    if q0.shape != (3,) or not np.isfinite(q0).all():
        raise ValueError("initial_q_si 必须是有限的 (3,) 状态")
    if schedule.ndim != 2 or schedule.shape[1] != 8:
        raise ValueError("observed_z_si 应为 (H,8)")
    if not np.isfinite(schedule).all():
        raise ValueError("已知未来输入时间序列不可包含 NaN/Inf")
    h = len(schedule)
    dt = np.asarray(dt_s, dtype=np.float64)
    if dt.ndim == 0:
        dt = np.full(h, float(dt))
    if dt.shape != (h,) or np.any(dt <= 0) or not np.isfinite(dt).all():
        raise ValueError("dt_s 必须为正标量或长度 H 的正向量")
    trajectory = np.empty((h+1, 3), dtype=np.float64)
    trajectory[0] = q0
    for k in range(h):
        trajectory[k+1] = rk4_free_step(
            trajectory[k], schedule[k, 3:], dt[k],
            residual_model=residual_model, reference=reference,
            substeps=substeps,
        )
    return trajectory


def rollout_pair_known_future(
    initial_q_si, observed_z_si, *, model, dt_s=0.01,
    reference=None, substeps=1,
):
    """公平对比：同一初值、同一执行器序列、同一 RK4 设置。"""
    nominal = rollout_known_future(
        initial_q_si, observed_z_si, dt_s=dt_s,
        reference=reference, substeps=substeps,
    )
    corrected = rollout_known_future(
        initial_q_si, observed_z_si, residual_model=model,
        dt_s=dt_s, reference=reference, substeps=substeps,
    )
    return nominal, corrected
