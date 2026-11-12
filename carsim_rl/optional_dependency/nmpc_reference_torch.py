# -*- coding: utf-8 -*-
"""可微 PyTorch NMPC 单轨动力学（与 nmpc_reference.py 的 NumPy 公式一致）。

z = [vy, yaw_rate, vx, delta_f, Mz_tv, Fx_all, lamda, mu_hat]
输出导数 = [d(vy)/dt, d(yaw_rate)/dt, d(vx)/dt]

重要约定：
1. 参数来自 dataset_manifest.json，不擅自更改 MATLAB 的平滑/饱和公式。
2. 用 float64 对齐 NumPy 的名义模型及 10 ms 区间积分。
3. 已知的未来执行器等效量在一个 10 ms 区间内采用 ZOH。
4. hybrid_rk4_step_torch() 的 continuous/discrete 模式分别实现
   RK4(f_nom + g_theta) 和 RK4(f_nom) + dt*g_theta(q_k, a_k)。
5. 全部计算可反向传播，不对中间状态 detach，也不回灌未来 CarSim 真值。
"""
from __future__ import annotations

import torch

from .nmpc_reference import NMPCReferenceParams


def _as_batch(z):
    if not isinstance(z, torch.Tensor) or not z.is_floating_point():
        raise TypeError("输入必须是浮点 PyTorch Tensor")
    single = z.ndim == 1
    if single:
        z = z.unsqueeze(0)
    if z.ndim != 2 or z.shape[1] != 8 or z.shape[0] < 1:
        raise ValueError("z 应为 (8,) 或 (B,8)")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("z 不能包含 NaN 或 Inf")
    if not bool((z[:, 7] > 0).all()):
        raise ValueError("mu_hat 必须为正")
    return z, single


def nmpc_force_balance_torch(z, params=None):
    """逐项复现 MATLAB/NumPy 名义轴荷、摩擦圆、刚度和轮胎力公式。"""
    if params is None:
        params = NMPCReferenceParams()
    params.validate()
    a, single = _as_batch(z)
    vy, r, vx, delta, mz, fx_all, lamda, mu = a.unbind(dim=1)
    m, g, lf = params.mass_kg, params.gravity_mps2, params.lf_m
    lr, L, h = params.lr_m, params.wheelbase_m, params.cg_height_m

    fx_f = fx_all * lamda
    fx_r = fx_all * (1.0 - lamda)
    fz_f = (m * g * lr - h * fx_all) / L
    fz_r = (m * g * lf + h * fx_all) / L

    # 连续可导轴荷下限；不直接 clamp。
    floor = params.front_rear_load_floor_n
    c_load = params.axle_load_smooth_const_n2
    fz_f = floor + 0.5 * ((fz_f - floor) + torch.sqrt((fz_f-floor)**2 + c_load))
    fz_r = floor + 0.5 * ((fz_r - floor) + torch.sqrt((fz_r-floor)**2 + c_load))

    margin_f = (mu * fz_f)**2 - fx_f**2
    margin_r = (mu * fz_r)**2 - fx_r**2
    c_margin = params.margin_smooth_const_n4
    sm_f = 0.5 * (margin_f + torch.sqrt(margin_f**2 + c_margin))
    sm_r = 0.5 * (margin_r + torch.sqrt(margin_r**2 + c_margin))
    fy_max_f = torch.sqrt(sm_f) + params.lateral_force_floor_n
    fy_max_r = torch.sqrt(sm_r) + params.lateral_force_floor_n

    fzf_kn = 0.5 * fz_f / 1000.0
    fzr_kn = 0.5 * fz_r / 1000.0
    c1, c2 = params.cornering_stiffness_c1, params.cornering_stiffness_c2
    ky_f = c1 * fzf_kn - c2 * fzf_kn**2
    ky_r = c1 * fzr_kn - c2 * fzr_kn**2

    vx_safe = torch.sqrt(vx**2 + params.vx_safe_square_m2ps2)
    alpha_f = delta - (vy + lf*r) / vx_safe
    alpha_r = (lr*r-vy) / vx_safe
    fy_f = fy_max_f * torch.tanh((2.0 * ky_f * alpha_f) / fy_max_f)
    fy_r = fy_max_r * torch.tanh((2.0 * ky_r * alpha_r) / fy_max_r)

    sin_d, cos_d = torch.sin(delta), torch.cos(delta)
    sum_fx = fx_f*cos_d - fy_f*sin_d + fx_r
    sum_fy = fy_f*cos_d + fx_f*sin_d + fy_r
    sum_mz = lf*(fy_f*cos_d + fx_f*sin_d) - lr*fy_r + mz

    ret = {
        "sum_fx_n": sum_fx, "sum_fy_n": sum_fy, "sum_mz_nm": sum_mz,
        "fy_front_n": fy_f, "fy_rear_n": fy_r,
        "fz_front_n": fz_f, "fz_rear_n": fz_r,
    }
    if single:
        return {k: v.squeeze(0) for k, v in ret.items()}
    return ret


def nmpc_accelerations_torch(z, params=None):
    """返回 (3,) 或 (B,3)；不使用任何真实 CarSim 的未来状态。"""
    if params is None:
        params = NMPCReferenceParams()
    a, single = _as_batch(z)
    f = nmpc_force_balance_torch(a, params)
    vy, r, vx = a[:, 0], a[:, 1], a[:, 2]
    out = torch.stack((
        -vx*r + f["sum_fy_n"]/params.mass_kg,
        f["sum_mz_nm"]/params.inertia_z_kgm2,
        f["sum_fx_n"]/params.mass_kg + vy*r,
    ), dim=1)
    return out[0] if single else out


def _batch_q_external(q, external):
    if not isinstance(q, torch.Tensor) or not isinstance(external, torch.Tensor):
        raise TypeError("q 和 external 必须是 torch.Tensor")
    single = q.ndim == 1
    if single:
        q = q.unsqueeze(0)
    if external.ndim == 1:
        external = external.unsqueeze(0)
    if q.ndim != 2 or q.shape[1] != 3 or external.shape != (len(q), 5):
        raise ValueError("q 必须是 (B,3)，external 必须是 (B,5)")
    if q.device != external.device or q.dtype != external.dtype:
        raise ValueError("q 与 external 的 dtype/device 必须相同")
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(external).all()):
        raise ValueError("q/external 不得含 NaN/Inf")
    return q, external, single


def hybrid_rk4_step_torch(
    q, external, dt_s, *,
    residual_model=None, params=None, integration="continuous", substeps=1,
):
    """连续或离散残差单步积分，全部保持可微。

    continuous: RK4(f_nom + residual_model)。
    discrete:   RK4(f_nom) + dt * residual_model([q_k, external_k])。
    当 residual_model=None 时，两种方法完全退化为名义 RK4。
    """
    if integration not in ("continuous", "discrete"):
        raise ValueError("integration 只能是 continuous 或 discrete")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps 必须为正整数")
    if params is None:
        params = NMPCReferenceParams()
    params.validate()
    state, ext, single = _batch_q_external(q, external)

    if isinstance(dt_s, torch.Tensor):
        dt = dt_s.to(device=state.device, dtype=state.dtype)
    else:
        dt = torch.as_tensor(dt_s, dtype=state.dtype, device=state.device)
    if dt.ndim == 0:
        dt = dt.expand(len(state))
    if dt.shape != (len(state),) or not bool((dt > 0).all()) or not bool(torch.isfinite(dt).all()):
        raise ValueError("dt_s 必须是正标量或批量正时间间隔 (B,)")
    dt_col = dt[:, None]
    step_col = dt_col / substeps

    def rhs(current, with_residual):
        combined = torch.cat((current, ext), dim=1)
        nominal = nmpc_accelerations_torch(combined, params)
        return nominal + residual_model(combined) if with_residual else nominal

    # 离散区间残差只在区间起点求一次；名义积分内部仍使用 RK4。
    use_cont_residual = integration == "continuous" and residual_model is not None
    for _ in range(substeps):
        k1 = rhs(state, use_cont_residual)
        k2 = rhs(state + 0.5*step_col*k1, use_cont_residual)
        k3 = rhs(state + 0.5*step_col*k2, use_cont_residual)
        k4 = rhs(state + step_col*k3, use_cont_residual)
        state = state + (step_col/6.0) * (k1+2.0*k2+2.0*k3+k4)

    if integration == "discrete" and residual_model is not None:
        original = q[None, :] if single else q
        correction_z = torch.cat((original, ext), dim=1)
        state = state + dt_col * residual_model(correction_z)

    return state[0] if single else state


def rk4_nominal_step_torch(z, dt_s=0.01, params=None, substeps=1):
    """名义 RK4，与 nmpc_reference.rk4_nominal_step 数值校验。"""
    a, single = _as_batch(z)
    q_new = hybrid_rk4_step_torch(
        a[:, :3], a[:, 3:], dt_s,
        residual_model=None, params=params, integration="continuous", substeps=substeps,
    )
    return q_new[0] if single else q_new
