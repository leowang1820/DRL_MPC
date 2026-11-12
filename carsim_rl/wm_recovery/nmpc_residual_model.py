# -*- coding: utf-8 -*-
"""NMPC 三通道物理残差 MLP：8 -> 16 -> 16 -> 3。

输入均为 SI 物理单位：
    [vy, r, vx, delta_f, Mz_tv, Fx_all, lamda, mu_hat]
输出均为 SI 等效加速度残差：
    [delta_ay, delta_yaw_accel, delta_ax]

训练归一化统计量仅能从 train.npz 得到，保存到模型的 buffer 中；
外部调用 forward() / predict_numpy() 一律输入/输出 SI 物理量。

特别注意：
- 当前 Mz_tv 是 CarSim 轮胎力的等效估计，不代表已验证的执行器力矩。
- 预测时可以使用已知未来的等效执行器序列，但不得重置预测的车辆状态。
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn


MODEL_FORMAT = "wm_recovery_nmpc_residual_mlp_v1"
INPUT_NAMES = (
    "vy_mps", "yaw_rate_radps", "vx_mps", "delta_front_rad",
    "mz_tv_estimated_nm", "fx_all_n", "lamda", "mu_hat",
)
TARGET_NAMES = (
    "delta_ay_mps2", "delta_yaw_accel_radps2", "delta_ax_mps2",
)


def fit_normalization(train_z, train_y, *, min_std=1e-10):
    """仅在训练集上拟合均值、标准差；对恒定 mu 等维度安全设 scale=1。"""
    z = np.asarray(train_z, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 8 or len(z) == 0:
        raise ValueError("train_z 必须是非空 (N,8)")
    if y.shape != (len(z), 3):
        raise ValueError("train_y 必须是 (N,3)，与 train_z 行数相同")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("训练数据含 NaN/Inf，禁止参与归一化")
    if not np.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std 必须为正的有限数")

    def stats(a):
        mean = np.mean(a, axis=0)
        original_std = np.std(a, axis=0)
        constant = original_std < min_std
        scale = np.where(constant, 1.0, original_std)
        return mean, scale, constant

    zm, zs, z_constant = stats(z)
    ym, ys, y_constant = stats(y)
    return {
        "x_mean": zm,
        "x_scale": zs,
        "y_mean": ym,
        "y_scale": ys,
        "x_constant_mask": z_constant,
        "y_constant_mask": y_constant,
    }


class NMPCResidualMLP(nn.Module):
    """物理单位输入输出的可部署 MLP；内部执行训练集统计量归一化。"""

    def __init__(self, stats, hidden_sizes=(16, 16)):
        super().__init__()
        if tuple(hidden_sizes) != (16, 16):
            raise ValueError("本版固定使用 8->16->16->3，便于后续 MATLAB 导出")
        for key, width in (
            ("x_mean", 8), ("x_scale", 8),
            ("y_mean", 3), ("y_scale", 3),
        ):
            value = np.asarray(stats[key], dtype=np.float64)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"{key} 必须是有限的 {width} 维向量")
            if "scale" in key and not np.all(value > 0):
                raise ValueError(f"{key} 不能包含零或负值")
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )

    def forward(self, z_si):
        """返回 SI 单位残差；训练和滚动评价使用同一前向公式。"""
        if z_si.ndim == 1:
            if z_si.shape[0] != 8:
                raise ValueError("单行输入必须恰好 8 维")
        elif z_si.ndim != 2 or z_si.shape[-1] != 8:
            raise ValueError("输入应为 (8,) 或 (N,8)")
        x = (z_si - self.x_mean) / self.x_scale
        y_normalized = self.net(x)
        return y_normalized * self.y_scale + self.y_mean

    @torch.no_grad()
    def predict_numpy(self, z_si):
        """NumPy 数组 -> NumPy SI 残差；用于离线 RK4 自由滚动。"""
        arr = np.asarray(z_si, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape != (8,):
                raise ValueError("单时刻应为 (8,)")
        elif arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError("批量输入应为 (N,8)")
        if not np.isfinite(arr).all():
            raise ValueError("预测输入包含 NaN/Inf")
        self.eval()
        device = self.x_mean.device
        x = torch.as_tensor(arr, dtype=torch.float32, device=device)
        out = self(x).detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(out).all():
            raise FloatingPointError("残差 MLP 前向计算产生 NaN/Inf")
        return out

    def normalization_json(self):
        """导出训练集归一化参数，便于审核或后续 MATLAB 移植。"""
        out = {}
        for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            out[key] = self.get_buffer(key).detach().cpu().numpy().tolist()
        out["input_names"] = list(INPUT_NAMES)
        out["target_names"] = list(TARGET_NAMES)
        return out


def load_model_checkpoint(path, *, device="cpu"):
    """严格加载自定义 MLP 权重，不能加载任意 pickle 对象。"""
    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
        raise ValueError("不是 NMPC 三通道 MLP v1 模型检查点")
    if tuple(payload.get("input_names", ())) != INPUT_NAMES:
        raise ValueError("模型八维输入顺序与当前代码不一致")
    if tuple(payload.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("模型三维残差顺序与当前代码不一致")
    if tuple(payload.get("hidden_sizes", ())) != (16, 16):
        raise ValueError("模型隐藏层结构与固定接口不一致")

    state = payload["model_state_dict"]
    stats = {key: state[key].detach().cpu().numpy()
             for key in ("x_mean", "x_scale", "y_mean", "y_scale")}
    model = NMPCResidualMLP(stats=stats)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload
