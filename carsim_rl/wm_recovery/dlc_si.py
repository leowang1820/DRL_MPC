# -*- coding: utf-8 -*-
"""读取当前 CarSim DLC 的原生 Export，并形成经过运动学验证的 SI 状态。

当前约定由用户的运动学交叉验证支持：
    Vx, Vy: km/h -> m/s
    AVz, Yaw, Steer_L1/R1: deg or deg/s -> rad or rad/s
    Ax, Ay: g -> m/s^2
    Fx_*, Fy_*: N（轮胎局部坐标；仍属于待核实的物理约定）

保留所有 61 列原始值；重复的 Steer_SW 不会被偷偷合并。
本模块只读取数据，不修改 CarSim 数据库或调用 DLL。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


REQUIRED = (
    "Xo", "Yo", "Vx", "Vy", "Yaw", "AVz", "Ax", "Ay",
    "Steer_L1", "Steer_R1",
    "Fx_L1", "Fx_R1", "Fx_L2", "Fx_R2",
    "Fy_L1", "Fy_R1",
)


@dataclass
class DLCStates:
    path: Path
    file_sha256: str
    time_s: np.ndarray
    raw: np.ndarray
    names: tuple
    q_si: np.ndarray          # [vy, r, vx], (T,3)
    yaw_rad: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    delta_fl_rad: np.ndarray
    delta_fr_rad: np.ndarray
    fx_tire_n: np.ndarray     # (T,4): FL FR RL RR
    fy_front_n: np.ndarray   # (T,2): FL FR
    diagnostics: dict

    def get(self, name):
        indices = [i for i, item in enumerate(self.names) if item == name]
        if len(indices) != 1:
            raise ValueError(f"Export {name} 应恰好出现一次，实际位置={indices}")
        return self.raw[:, indices[0]]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def load_dlc_si(npz_path, *, expected_dt_s=0.01, tolerance_s=1e-7):
    """严格读取 DLC 原始结果，拒绝时间异常或必需通道缺失。

    保存的原生单位不被修改；所有结果在单独数组中转换为 SI。
    """
    path = Path(npz_path).expanduser().resolve(strict=True)
    with np.load(path, allow_pickle=False) as store:
        needed = {"time_s", "raw_exports", "export_names"}
        if not needed.issubset(set(store.files)):
            raise ValueError(f"原始 NPZ 缺少字段：{sorted(needed-set(store.files))}")
        t = np.asarray(store["time_s"], dtype=np.float64).copy()
        raw = np.asarray(store["raw_exports"], dtype=np.float64).copy()
        names = tuple(str(v) for v in store["export_names"].tolist())

    if t.ndim != 1 or len(t) < 3 or raw.shape != (len(t), len(names)):
        raise ValueError("DLC time_s / raw_exports / export_names 形状不一致")
    if not np.isfinite(t).all() or not np.isfinite(raw).all():
        raise ValueError("DLC 中有 NaN/Inf，禁止进入训练")
    if expected_dt_s <= 0 or tolerance_s <= 0:
        raise ValueError("时间步长和容差必须为正")
    dt = np.diff(t)
    if (not (dt > 0).all()
        or not np.allclose(dt, expected_dt_s, atol=tolerance_s, rtol=0)):
        raise ValueError(
            f"时间轴异常：dt_min={dt.min():.9g}, "
            f"dt_max={dt.max():.9g}, expected={expected_dt_s:.9g}；"
            "不得使用 INVALID_TIMING_do_not_train.npz"
        )
    for name in REQUIRED:
        indices = [i for i, item in enumerate(names) if item == name]
        if len(indices) != 1:
            raise ValueError(f"必需 Export {name} 应恰好出现一次：{indices}")

    def c(name):
        return raw[:, names.index(name)]

    vx = c("Vx") / 3.6
    vy = c("Vy") / 3.6
    r = np.deg2rad(c("AVz"))
    yaw = np.unwrap(np.deg2rad(c("Yaw")))
    q = np.column_stack((vy, r, vx))
    ax = c("Ax") * 9.81
    ay = c("Ay") * 9.81
    dfl = np.deg2rad(c("Steer_L1"))
    dfr = np.deg2rad(c("Steer_R1"))
    fx = np.column_stack([c(name) for name in
                          ("Fx_L1", "Fx_R1", "Fx_L2", "Fx_R2")])
    fy_front = np.column_stack((c("Fy_L1"), c("Fy_R1")))

    # CarSim 之前的运动学交叉验证给出很小的误差；继续记录检查结果。
    # 这些检查只能支持单位/坐标自洽，不能证明每个轮胎力已物理对齐。
    dyn_ax = np.gradient(vx, t) - r*vy
    dyn_ay = np.gradient(vy, t) + r*vx
    r_from_yaw = np.gradient(yaw, t)
    sl = slice(5, -5) if len(t) > 12 else slice(None)
    diagnostics = {
        "n_samples": int(len(t)),
        "n_exports": int(len(names)),
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "time_axis_uniform": True,
        "initial_speed_kmh": float(c("Vx")[0]),
        "max_abs_beta_deg": float(
            np.rad2deg(np.max(np.abs(np.arctan2(vy, vx))))
        ),
        "max_abs_ay_g": float(np.max(np.abs(c("Ay")))),
        "yaw_rate_consistency_rmse_radps": _rms((r_from_yaw-r)[sl]),
        "ax_consistency_rmse_g": _rms(((dyn_ax-ax)/9.81)[sl]),
        "ay_consistency_rmse_g": _rms(((dyn_ay-ay)/9.81)[sl]),
        "duplicates": {
            name: [i for i, x in enumerate(names) if x == name]
            for name in dict.fromkeys(names)
            if names.count(name) > 1
        },
        "unit_assumptions": {
            "Vx,Vy": "km/h",
            "AVz": "deg/s",
            "Yaw,Steer_L1,Steer_R1": "deg",
            "Ax,Ay": "g",
            "Fx_*,Fy_*": "N; wheel-local assumed",
        },
    }
    return DLCStates(
        path=path, file_sha256=_sha256(path),
        time_s=t, raw=raw, names=names, q_si=q,
        yaw_rad=yaw, x_m=c("Xo").copy(), y_m=c("Yo").copy(),
        ax_mps2=ax, ay_mps2=ay,
        delta_fl_rad=dfl, delta_fr_rad=dfr,
        fx_tire_n=fx, fy_front_n=fy_front,
        diagnostics=diagnostics,
    )
