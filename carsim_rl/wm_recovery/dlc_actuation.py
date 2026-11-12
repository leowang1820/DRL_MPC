# -*- coding: utf-8 -*-
"""DLC CarSim 61 Export -> NMPC 八维输入，明确区分物理事实与假设。

z = [vy, r, vx, delta_f, Mz_tv, Fx_all, lamda, mu_hat]

默认是“单场景方法调通”：
  delta_f = (Steer_L1 + Steer_R1) / 2
  Mz_tv = 0  （假设内置驾驶员无独立差动横摆执行器）
  Fx_all = 四轮实测轮胎局部 Fx 之和
  lamda = 0.5（固定模型假设，不代表实测前后轴力分配）
  mu_hat = 用户明确提供的路面名义附着系数

重要：当前和未来时刻的 Fx 是 CarSim 实际轮胎力，不是已知转矩命令。
因此 3s/5s 对比属于“未来实际等效力已知的离线重放诊断”，
不属于可部署的、无未来真值泄漏的闭环 NMPC 预测。
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .dlc_si import DLCStates


@dataclass
class DLCActuation:
    z_si: np.ndarray
    diagnostics: dict
    assumptions: dict


def build_dlc_inputs(
    dlc: DLCStates, *,
    mu_hat=0.8,
    mz_mode="zero",
    lambda_mode="fixed-half",
    front_track_m=1.605,
    rear_track_m=1.605,
    min_abs_fx_n=50.0,
    neutral_axle_limit_n=50.0,
):
    """把状态与等效输入对齐为 (T,8)，全部保持 SI 单位。"""
    if not (math.isfinite(mu_hat) and mu_hat > 0):
        raise ValueError("mu_hat 必须是已确认的正附着系数")
    if not all(math.isfinite(x) and x > 0 for x in
               (front_track_m, rear_track_m)):
        raise ValueError("轮距必须为正")
    if min_abs_fx_n < 0 or neutral_axle_limit_n < 0:
        raise ValueError("纵向力可辨识阈值必须非负")

    dfl, dfr = dlc.delta_fl_rad, dlc.delta_fr_rad
    delta = 0.5*(dfl+dfr)
    fx = dlc.fx_tire_n
    fx_front = fx[:, 0]+fx[:, 1]
    fx_rear = fx[:, 2]+fx[:, 3]
    fx_all = fx_front+fx_rear

    if mz_mode == "zero":
        mz = np.zeros_like(fx_all)
    elif mz_mode == "force-diff-diagnostic":
        fyfl = dlc.fy_front_n[:, 0]
        fyfr = dlc.fy_front_n[:, 1]
        fxfl_body = fx[:, 0]*np.cos(dfl) - fyfl*np.sin(dfl)
        fxfr_body = fx[:, 1]*np.cos(dfr) - fyfr*np.sin(dfr)
        mz = (0.5*front_track_m*(fxfr_body-fxfl_body)
              + 0.5*rear_track_m*(fx[:, 3]-fx[:, 2]))
    else:
        raise ValueError("mz_mode 只能是 zero 或 force-diff-diagnostic")

    ratio_defined = np.abs(fx_all) > min_abs_fx_n
    neutral = (~ratio_defined
               & (np.abs(fx_front) <= neutral_axle_limit_n)
               & (np.abs(fx_rear) <= neutral_axle_limit_n))
    cancellation = ~(ratio_defined | neutral)
    if lambda_mode == "fixed-half":
        lamda = np.full_like(fx_all, 0.5)
    elif lambda_mode == "ratio-strict":
        if cancellation.any():
            indices = np.flatnonzero(cancellation)[:12].tolist()
            raise ValueError(
                f"实测 lamda 存在轴力抵消/不可辨识区间 {int(cancellation.sum())}"
                f" 帧，前几个索引={indices}。请改用 --lambda-mode fixed-half "
                "作显式简化假设，或单独处理这些区间。"
            )
        lamda = np.full_like(fx_all, np.nan)
        np.divide(fx_front, fx_all, out=lamda, where=ratio_defined)
        lamda[neutral] = 0.5
    else:
        raise ValueError("lambda_mode 只能是 fixed-half 或 ratio-strict")

    mu = np.full_like(fx_all, mu_hat)
    z = np.column_stack((dlc.q_si, delta, mz, fx_all, lamda, mu))
    if z.shape != (len(dlc.time_s), 8) or not np.isfinite(z).all():
        raise ValueError("DLC 八维输入不完整或存在 NaN/Inf")
    assumptions = {
        "delta_front": "arithmetic mean of measured front road-wheel steering angles",
        "mz_mode": mz_mode,
        "mz_zero_is_assumption_not_measured": bool(mz_mode == "zero"),
        "fx_all": "sum of CarSim measured tire-local longitudinal forces",
        "lambda_mode": lambda_mode,
        "lambda_fixed_half_is_assumption": bool(lambda_mode == "fixed-half"),
        "mu_hat": float(mu_hat),
        "front_track_m": float(front_track_m),
        "rear_track_m": float(rear_track_m),
        "tire_forces_are_observed_future_exogenous_not_true_command": True,
        "physical_interface_verified": False,
    }
    diagnostics = {
        "lambda_ratio_available_count": int(ratio_defined.sum()),
        "lambda_neutral_count": int(neutral.sum()),
        "lambda_cancellation_count": int(cancellation.sum()),
        "max_abs_front_steer_difference_deg": float(
            np.rad2deg(np.max(np.abs(dfl-dfr)))
        ),
        "min_fx_all_n": float(fx_all.min()),
        "max_fx_all_n": float(fx_all.max()),
        "min_mz_tv_nm": float(mz.min()),
        "max_mz_tv_nm": float(mz.max()),
        "lambda_min": float(lamda.min()),
        "lambda_max": float(lamda.max()),
    }
    return DLCActuation(z_si=z, diagnostics=diagnostics,
                        assumptions=assumptions)
