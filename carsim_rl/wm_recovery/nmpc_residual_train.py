# -*- coding: utf-8 -*-
"""NMPC 8->16->16->3 三通道加速度残差网络训练（仅 Python）。

训练目标：
  y = [Delta a_y, Delta yaw_accel, Delta a_x]
  由清洗后 CarSim 状态与 NMPC 名义 RK4 的 10ms 差值计算。

约束：
  * 只使用 train 拟合输入与标签归一化。
  * train 使用 dataset.npz 中已经保存的 sample_weight；
    validation 不使用 train 的采样权重，也不参与归一化拟合。
  * test 数据完全不参与训练、早停和选模；由独立 evaluate 使用。
  * 保存的模型 forward 始终采用 SI 输入与 SI 输出。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .nmpc_residual_model import (
    INPUT_NAMES, TARGET_NAMES, MODEL_FORMAT,
    NMPCResidualMLP, fit_normalization,
)


DATASET_FORMAT = "wm_recovery_nmpc_residual_dataset_v1"


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _check_manifest(dataset_dir):
    """验证数据版本与标签定义，防止错用原始 7DOF World Model 数据。"""
    path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("输入不是 NMPC 三通道残差数据集 v1")
    if tuple(manifest.get("input_names", ())) != INPUT_NAMES:
        raise ValueError("dataset_manifest 八维输入顺序不一致")
    if tuple(manifest.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("dataset_manifest 三维标签顺序不一致")
    if manifest.get("dataset_settings", {}).get("expected_dt_s") != 0.01:
        raise ValueError("目前仅验证过 CarSim 10ms 采样数据")
    return manifest


def load_split_npz(dataset_dir, split):
    """严格读取分区，不允许文件名与内含的数据接口不一致。"""
    if split not in ("train", "validation", "test"):
        raise ValueError("非法 split")
    source = Path(dataset_dir) / f"{split}.npz"
    with np.load(source, allow_pickle=False) as file:
        expected = (
            "z_si", "residual_y_si", "q_k_si", "q_true_next_si",
            "q_nominal_next_si", "dt_s", "time_k_s", "sample_weight",
            "lamda_regime_code", "episode_index", "segment_id",
            "segment_step", "interval_index",
        )
        missing = set(expected) - set(file.files)
        if missing:
            raise ValueError(f"{split}.npz 缺少字段：{sorted(missing)}")
        data = {key: np.asarray(file[key]).copy() for key in expected}

    n = len(data["z_si"])
    for key, width in (
        ("z_si", 8), ("residual_y_si", 3), ("q_k_si", 3),
        ("q_true_next_si", 3), ("q_nominal_next_si", 3),
    ):
        if data[key].shape != (n, width):
            raise ValueError(f"{split}:{key} 形状不正确")
    for key in (
        "dt_s", "time_k_s", "sample_weight", "lamda_regime_code",
        "episode_index", "segment_id", "segment_step", "interval_index",
    ):
        if data[key].shape != (n,):
            raise ValueError(f"{split}:{key} 形状不正确")

    floats = ("z_si", "residual_y_si", "q_k_si", "q_true_next_si",
              "q_nominal_next_si", "dt_s", "time_k_s", "sample_weight")
    if not all(np.isfinite(data[k]).all() for k in floats):
        raise ValueError(f"{split} 存在非有限数据")
    if n == 0:
        raise ValueError(f"{split} 没有可用样本")
    if not np.all(data["sample_weight"] > 0):
        raise ValueError(f"{split} 样本权重必须为正")
    if not np.all(data["dt_s"] > 0) or not np.allclose(
        data["dt_s"], 0.01, rtol=0, atol=1e-8
    ):
        raise ValueError(f"{split} 采样时间不符合 10ms 契约")
    if not np.allclose(data["z_si"][:, :3], data["q_k_si"], rtol=0, atol=1e-7):
        raise ValueError(f"{split} 输入状态与 q_k 不一致")
    if not np.allclose(
        data["residual_y_si"] * data["dt_s"][:, None]
        + data["q_nominal_next_si"],
        data["q_true_next_si"],
        rtol=0, atol=2e-7
    ):
        raise ValueError(f"{split} 残差标签与 NMPC 名义一步预测不一致")
    for key in ("lamda_regime_code", "episode_index", "segment_id",
                "segment_step", "interval_index"):
        if data[key].dtype.kind not in "iu":
            raise ValueError(f"{split}:{key} 必须为整数数组")
    if not np.all(np.isin(data["lamda_regime_code"], (0, 1))):
        raise ValueError(f"{split}:lamda_regime_code 只能是 ratio=0 或 neutral=1")
    if not np.all(data["segment_step"] >= 0):
        raise ValueError(f"{split}:segment_step 不能为负")
    return data


def normalized_weighted_mse(model, z_tensor, y_tensor, weights=None):
    """物理残差先按仅 Train 拟合的 y_scale 标准化，再计算三个通道 MSE。"""
    y_pred = model(z_tensor)
    normalized_sq = ((y_pred - y_tensor) / model.y_scale).square()
    row_loss = normalized_sq.mean(dim=1)
    if weights is None:
        return row_loss.mean()
    return (row_loss * weights).sum() / weights.sum()


def run_training(
    dataset_dir,
    output_dir,
    *,
    epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-6,
    patience=15,
    min_delta=1e-6,
    seed=2026,
    device="cpu",
):
    """训练并保存最佳模型；不允许覆盖已存在的训练目录。"""
    if isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs 必须是正整数")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    if isinstance(patience, bool) or patience < 1:
        raise ValueError("patience 必须是正整数")
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError("学习率必须为正")
    if not (math.isfinite(weight_decay) and weight_decay >= 0):
        raise ValueError("权重衰减不能为负")
    if not (math.isfinite(min_delta) and min_delta >= 0):
        raise ValueError("min_delta 不能为负")

    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不允许覆盖：{output_dir}")

    manifest = _check_manifest(dataset_dir)
    train = load_split_npz(dataset_dir, "train")
    valid = load_split_npz(dataset_dir, "validation")
    expected_counts = manifest.get("split_counts", {})
    for split, data in (("train", train), ("validation", valid)):
        expected = expected_counts.get(split, {}).get("intervals")
        if expected != len(data["z_si"]):
            raise ValueError(f"{split} 样本数与 manifest 不一致")

    # 不读取 test.npz；测试集数据完全隔离在训练和选模之外。
    stats = fit_normalization(train["z_si"], train["residual_y_si"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    # 对两层 16 单元的小网络，CPU 单线程通常能避免批量矩阵运算的调度开销。
    if dev.type == "cpu":
        torch.set_num_threads(1)

    model = NMPCResidualMLP(stats).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    xt = torch.as_tensor(train["z_si"], dtype=torch.float32, device=dev)
    yt = torch.as_tensor(train["residual_y_si"], dtype=torch.float32, device=dev)
    wt = torch.as_tensor(train["sample_weight"], dtype=torch.float32, device=dev)
    xv = torch.as_tensor(valid["z_si"], dtype=torch.float32, device=dev)
    yv = torch.as_tensor(valid["residual_y_si"], dtype=torch.float32, device=dev)
    order = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=False)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_metadata = {
        "format": MODEL_FORMAT,
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "hidden_sizes": [16, 16],
        "dataset_manifest_sha256": sha256(dataset_dir / "dataset_manifest.json"),
        "train_npz_sha256": sha256(dataset_dir / "train.npz"),
        "validation_npz_sha256": sha256(dataset_dir / "validation.npz"),
        "training_mode": "one_step_interval_residual_supervision",
        "condition": "future_CarSim_equivalent_actuation_is_exogenous",
        "seed": int(seed),
    }

    for epoch in range(1, epochs + 1):
        model.train()
        permutations = order.permutation(len(xt))
        total_weighted = 0.0
        total_weights = 0.0
        for start in range(0, len(permutations), batch_size):
            indices = torch.as_tensor(
                permutations[start:start + batch_size],
                device=dev, dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            w = wt.index_select(0, indices)
            loss = normalized_weighted_mse(
                model, xt.index_select(0, indices),
                yt.index_select(0, indices), w
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("训练 Loss 变成 NaN/Inf，请检查训练数据")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_weight = float(w.sum().item())
            total_weighted += float(loss.item()) * batch_weight
            total_weights += batch_weight

        model.eval()
        with torch.inference_mode():
            val_loss = float(normalized_weighted_mse(model, xv, yv).item())
            val_yhat = model(xv)
            val_rmse = torch.sqrt(
                ((val_yhat - yv) ** 2).mean(dim=0)
            ).detach().cpu().numpy().astype(float).tolist()
        if not math.isfinite(val_loss):
            raise FloatingPointError("验证集 Loss 变成 NaN/Inf")

        record = {
            "epoch": epoch,
            "train_weighted_normalized_mse": total_weighted / total_weights,
            "validation_normalized_mse": val_loss,
            "validation_rmse_y_si": val_rmse,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} | train={record['train_weighted_normalized_mse']:.7f}"
            f" | val={val_loss:.7f}"
            f" | val_rmse=[{val_rmse[0]:.5g},"
            f" {val_rmse[1]:.5g}, {val_rmse[2]:.5g}]",
            flush=True
        )
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(
                {**checkpoint_metadata, "best_epoch": epoch,
                 "best_validation_normalized_mse": val_loss,
                 "model_state_dict": state},
                output_dir / "best.pt"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"早停：连续 {patience} 轮验证集未改善。")
            break

    (output_dir / "normalization.json").write_text(
        json.dumps(model.normalization_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "format": "wm_recovery_nmpc_residual_training_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "output": str(output_dir),
        "train_samples": len(train["z_si"]),
        "validation_samples": len(valid["z_si"]),
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "best_validation_normalized_mse": best_loss,
        "hyperparameters": {
            "max_epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "patience": patience, "min_delta": min_delta, "seed": seed,
            "device": device,
        },
        "checkpoint_sha256": sha256(output_dir / "best.pt"),
        "dataset_manifest_sha256": checkpoint_metadata[
            "dataset_manifest_sha256"
        ],
        "train_npz_sha256": checkpoint_metadata["train_npz_sha256"],
        "validation_npz_sha256": checkpoint_metadata["validation_npz_sha256"],
        "input_constant_channels": [
            INPUT_NAMES[i] for i, flag in enumerate(stats["x_constant_mask"])
            if flag
        ],
        "physics_verified": False,
        "evaluation_ready": True,
        "closed_loop_nmpc_validated": False,
        "notes": [
            "只使用 Train 数据拟合归一化，Validation 用于早停；不读取 Test。",
            "单步残差监督不保证 5 秒滚动预测改善，需要独立滚动评价。",
            "已知未来执行器响应是离线条件预测，不声称在线未知执行器的预测。",
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"最佳 epoch={best_epoch}, val_loss={best_loss:.7g}")
    print("最佳模型:", output_dir / "best.pt")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="NMPC 三通道 Residual MLP 训练")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_training(
        args.dataset, args.output, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
