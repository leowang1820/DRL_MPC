# -*- coding: utf-8 -*-
"""与用户 MATLAB continuousDynamics() 三个车辆动力学方程对齐的名义模型。

注意：本模块不使用原 7DOF 动力学，也不调用 CarSim/FORCESPRO。
目标是得到 NMPC 的 [vy_dot, r_dot, vx_dot]，并使用 RK4 生成名义一步预测。

输入 z 的固定顺序和单位：
    [vy(m/s), r(rad/s), vx(m/s), delta(rad), Mz(N*m),
     Fx_all(N), lamda(-), mu_hat(-)]

物理公式（对应用户上传的 lineartirewcytest49.m continuousDynamics）：
    Fx_f = Fx_all * lamda
    Fx_r = Fx_all * (1 - lamda)
    Fz_f = (m*g*lr - h*Fx_all)/L
    Fz_r = (m*g*lf + h*Fx_all)/L
    ...（轴荷平滑、横向摩擦圆、刚度、tanh 饱和）...
    vy_dot = -vx*r + sum_Fy/m
    r_dot  = sum_Mz/Iz
    vx_dot = sum_Fx/m + vy*r

保持原 MATLAB 的数值常数、正负号、平滑形式及轮胎模型，不擅自修正。
不含 Frenet e_phi/e_y/s、执行器变化率 u；RK4 期间对 delta、Mz、Fx、
lamda、mu 采用区间起点的零阶保持。完整 NMPC 中执行器状态通过 u 演化，
两者可能存在离散化差异，因此由此生成的标签称为“区间等效残差”。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


NMPC_Z_NAMES = (
    "vy_mps", "yaw_rate_radps", "vx_mps", "delta_front_rad",
    "mz_tv_estimated_nm", "fx_all_n", "lamda", "mu_hat",
)
NMPC_Q_NAMES = ("vy_mps", "yaw_rate_radps", "vx_mps")
NMPC_Y_NAMES = ("delta_ay_mps2", "delta_yaw_accel_radps2", "delta_ax_mps2")
REFERENCE_VERSION = "nmpc_singletrack_matlab_continuousDynamics_v1"


@dataclass(frozen=True)
class NMPCReferenceParams:
    """逐项抄录当前 MATLAB NMPC 的数值参数，单位使用 SI。"""

    lf_m: float = 1.265
    wheelbase_m: float = 3.16
    mass_kg: float = 1820.0 + 116.9 * 1.0+ 89.5
    inertia_z_kgm2: float = 4095.0
    cg_height_m: float = 0.59
    gravity_mps2: float = 9.81
    front_rear_load_floor_n: float = 1000.0
    axle_load_smooth_const_n2: float = 10000.0
    margin_smooth_const_n4: float = 1e6
    lateral_force_floor_n: float = 0.001
    cornering_stiffness_c1: float = 8735.0825
    cornering_stiffness_c2: float = 59.6128
    vx_safe_square_m2ps2: float = 0.01

    def validate(self):
        for field, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"NMPC 参数 {field} 必须为有限数字")
        if not (0 < self.lf_m < self.wheelbase_m):
            raise ValueError("须满足 0 < lf < L")
        for key in (
            "wheelbase_m", "mass_kg", "inertia_z_kgm2", "gravity_mps2",
            "front_rear_load_floor_n", "axle_load_smooth_const_n2",
            "margin_smooth_const_n4", "lateral_force_floor_n",
            "vx_safe_square_m2ps2",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"NMPC 参数 {key} 必须大于零")

    @property
    def lr_m(self):
        return self.wheelbase_m - self.lf_m


def _validate_z(z):
    """允许单时刻 (8,) 或多时刻 (N,8)，拒绝 NaN/Inf。"""
    a = np.asarray(z, dtype=np.float64)
    single = a.ndim == 1
    if single:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 8 or len(a) == 0:
        raise ValueError(f"z 应为 (8,) 或 (N,8)，实际 {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("z 中有 NaN/Inf，需先用 nmpc_actuation.valid_mask 筛选")
    if np.any(a[:, 7] <= 0):
        raise ValueError("路面附着输入 mu_hat 必须大于零")
    return a, single


def nmpc_force_balance(z, params=None):
    """计算 MATLAB 模型中的轮胎合力和附加横摆力矩（用于审计）。

    输入/返回均允许批量化；返回字段含 sum_fx_n、sum_fy_n、
    sum_mz_nm、fy_front_n、fy_rear_n、fz_front_n、fz_rear_n。

    这些是单轨**模型预测**的力，不等于 CarSim 逐轮测得的真值。
    """
    if params is None:
        params = NMPCReferenceParams()
    params.validate()
    a, single = _validate_z(z)
    vy, r, vx, delta, mz, fx_all, lamda, mu = (
        a[:, i] for i in range(8)
    )

    lf, lr, L = params.lf_m, params.lr_m, params.wheelbase_m
    m, g, h = params.mass_kg, params.gravity_mps2, params.cg_height_m
    fx_f = fx_all * lamda
    fx_r = fx_all * (1.0 - lamda)

    # 纵向载荷转移、前/后轴轴荷；与 continuousDynamics 完全相同。
    fz_f = (m * g * lr - h * fx_all) / L
    fz_r = (m * g * lf + h * fx_all) / L
    floor = params.front_rear_load_floor_n
    eps_fz = params.axle_load_smooth_const_n2
    fz_f = floor + 0.5 * ((fz_f - floor) + np.sqrt((fz_f - floor)**2 + eps_fz))
    fz_r = floor + 0.5 * ((fz_r - floor) + np.sqrt((fz_r - floor)**2 + eps_fz))

    # 摩擦圆剩余横向容量：平方单位为 N^2；内层平滑常数为 N^4。
    fy_limit_f = mu * fz_f
    fy_limit_r = mu * fz_r
    margin_f = fy_limit_f**2 - fx_f**2
    margin_r = fy_limit_r**2 - fx_r**2
    eps_margin = params.margin_smooth_const_n4
    margin_f_smooth = 0.5 * (margin_f + np.sqrt(margin_f**2 + eps_margin))
    margin_r_smooth = 0.5 * (margin_r + np.sqrt(margin_r**2 + eps_margin))
    fy_max_f = np.sqrt(margin_f_smooth) + params.lateral_force_floor_n
    fy_max_r = np.sqrt(margin_r_smooth) + params.lateral_force_floor_n

    # 按轴分配两条轮胎的名义侧偏刚度；与 MATLAB 公式一致。
    fz_single_f_kn = 0.5 * fz_f / 1000.0
    fz_single_r_kn = 0.5 * fz_r / 1000.0
    c1, c2 = params.cornering_stiffness_c1, params.cornering_stiffness_c2
    ky_f = c1 * fz_single_f_kn - c2 * fz_single_f_kn**2
    ky_r = c1 * fz_single_r_kn - c2 * fz_single_r_kn**2

    vx_safe = np.sqrt(vx**2 + params.vx_safe_square_m2ps2)
    alpha_f = delta - (vy + lf * r) / vx_safe
    alpha_r = (lr * r - vy) / vx_safe
    fy_lin_f = (2.0 * ky_f) * alpha_f
    fy_lin_r = (2.0 * ky_r) * alpha_r

    # 保留原 tanh 非线性饱和，不以 CarSim 的 Fy 真值替换名义 Fy。
    fy_f = fy_max_f * np.tanh(fy_lin_f / fy_max_f)
    fy_r = fy_max_r * np.tanh(fy_lin_r / fy_max_r)

    sin_d, cos_d = np.sin(delta), np.cos(delta)
    sum_fx = fx_f * cos_d - fy_f * sin_d + fx_r
    sum_fy = fy_f * cos_d + fx_f * sin_d + fy_r
    sum_mz = lf * (fy_f * cos_d + fx_f * sin_d) - lr * fy_r + mz

    vals = {
        "sum_fx_n": sum_fx,
        "sum_fy_n": sum_fy,
        "sum_mz_nm": sum_mz,
        "fy_front_n": fy_f,
        "fy_rear_n": fy_r,
        "fz_front_n": fz_f,
        "fz_rear_n": fz_r,
    }
    if any(not np.isfinite(x).all() for x in vals.values()):
        raise FloatingPointError("NMPC 力学公式产生非有限值；请检查输入量级")
    if single:
        return {key: float(value[0]) for key, value in vals.items()}
    return vals


def nmpc_accelerations(z, params=None):
    """NMPC 名义 [vy_dot, r_dot, vx_dot]，形状 (3,) 或 (N,3)。"""
    if params is None:
        params = NMPCReferenceParams()
    a, single = _validate_z(z)
    force = nmpc_force_balance(a, params)
    vy, r, vx = a[:, 0], a[:, 1], a[:, 2]
    vy_dot = -vx * r + force["sum_fy_n"] / params.mass_kg
    r_dot = force["sum_mz_nm"] / params.inertia_z_kgm2
    vx_dot = force["sum_fx_n"] / params.mass_kg + vy * r
    out = np.column_stack((vy_dot, r_dot, vx_dot))
    if not np.isfinite(out).all():
        raise FloatingPointError("NMPC 名义动力学出现 NaN/Inf")
    return out[0] if single else out


def rk4_nominal_step(z_k, dt_s=0.01, params=None, substeps=1):
    """RK4 计算 [vy,r,vx] 的下一状态；物理控制量保持 t_k 值。

    注意：仅修改 q=[vy,r,vx]，其余五项控制状态按区间起点 ZOH。
    与完整 NMPC 的执行器速率积分不同，因此它只适合第一版
    “区间等效加速度残差”标签构造和基准预测。
    """
    if params is None:
        params = NMPCReferenceParams()
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps 必须为正整数")
    a, single = _validate_z(z_k)
    dt = np.asarray(dt_s, dtype=np.float64)
    if dt.ndim == 0:
        dt = np.full(len(a), float(dt))
    if dt.ndim != 1 or len(dt) != len(a):
        raise ValueError("dt_s 必须为标量或与样本数对应的一维数组")
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("dt_s 必须为正的有限数")
    q = a[:, :3].copy()
    controls = a[:, 3:].copy()
    h = (dt / substeps)[:, None]

    def f(q_state):
        stage = np.concatenate((q_state, controls), axis=1)
        return nmpc_accelerations(stage, params)

    for _ in range(substeps):
        k1 = f(q)
        k2 = f(q + 0.5 * h * k1)
        k3 = f(q + 0.5 * h * k2)
        k4 = f(q + h * k3)
        q += (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    if not np.isfinite(q).all():
        raise FloatingPointError("RK4 名义积分出现 NaN/Inf")
    return q[0] if single else q
