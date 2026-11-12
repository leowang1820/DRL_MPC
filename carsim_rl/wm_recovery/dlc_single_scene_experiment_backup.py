# -*- coding: utf-8 -*-
"""CarSim DLC 单轨迹 NMPC 残差实验：训练 + 3/5 秒自由滚动对照。

运行：
python -m wm_recovery.dlc_single_scene_experiment ^
  --data ".../dlc_exports.npz" --output ".../dlc_single_scene_01" ^
  --mu-hat 0.8 --train-end 5.0 --epochs 250

本实验用同一条 DLC 轨迹的前 5s 训练、后 5s 验证：
- 验证区间的真实 q 不参与拟合归一化、梯度和选模型；
- 模型 checkpoint 固定使用最后一个 epoch；
- 3s/5s 的 RK4 不回灌未来车辆真实状态；
- 但使用 **CarSim 未来实测转角与轮胎 Fx** 作为已知外生序列；
  因此是 offline known-response replay (oracle-actuation diagnostic)，
  不是仅凭将来实际可知驾驶员指令的闭环 NMPC 预测。
- 单场景验证不代表跨场景泛化，5s 验证通常只有 1 个起点。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .dlc_actuation import build_dlc_inputs
from .dlc_si import load_dlc_si
from .nmpc_reference import (
    NMPCReferenceParams, NMPC_Q_NAMES, NMPC_Y_NAMES, rk4_nominal_step,
)
from .nmpc_residual_model import (
    MODEL_FORMAT, INPUT_NAMES, TARGET_NAMES,
    NMPCResidualMLP, fit_normalization,
)
from .nmpc_rollout import rollout_pair_known_future


FORMAT = "wm_recovery_dlc_single_scene_residual_experiment_v1"


def _json_dump(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8",
    )


def _metric_rms(v):
    return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64)**2)))


def _per_state_rms(matrix):
    a = np.asarray(matrix, dtype=np.float64).reshape(-1, 3)
    return np.sqrt(np.mean(a*a, axis=0)).astype(float).tolist()


def _improvement(before, after):
    return None if before <= 1e-12 else float(100*(before-after)/before)


def _split_index(times, train_end_s, horizons):
    """以时间划分 interval；训练 [0,split)、验证 [split,N)。"""
    t = np.asarray(times, dtype=np.float64)
    index = int(np.argmin(np.abs(t-train_end_s)))
    if abs(float(t[index])-train_end_s) > 1e-6:
        raise ValueError(
            f"训练截止时间 {train_end_s}s 不在 10ms 采样网格上；"
            f"最近时间戳是 {t[index]}"
        )
    n_intervals = len(t)-1
    minimum = max(horizons)
    if index < minimum or n_intervals-index < minimum:
        raise ValueError(
            f"前段 {index}、后段 {n_intervals-index} 个区间，"
            f"均须不少于最长预测窗口 {minimum} 个区间。"
            "请用10s 轨迹配合 --train-end 5.0 --horizons 300,500。"
        )
    return index


def make_interval_targets(dlc, z, *, reference=None):
    """1001 帧 -> 1000 个监督样本，严格使用区间左端的控制。"""
    if reference is None:
        reference = NMPCReferenceParams()
    dt = np.diff(dlc.time_s)
    if z.shape != (len(dlc.time_s), 8):
        raise ValueError("状态与输入时间对齐失败")
    nominal_next = rk4_nominal_step(z[:-1], dt_s=dt, params=reference)
    q_next = dlc.q_si[1:]
    residual = (q_next-nominal_next)/dt[:, None]
    if not np.isfinite(residual).all():
        raise FloatingPointError("残差训练标签含 NaN/Inf")
    return nominal_next, residual


def train_single_scene(
    train_z, train_y, *,
    epochs=250, batch_size=128, lr=1e-3,
    weight_decay=1e-5, seed=42, verbose=True,
):
    """只用 Train 时段训练固定 MLP，不使用 Validation early stopping。"""
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs、batch_size、lr 必须为正")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.set_num_threads(1)
    stats = fit_normalization(train_z, train_y)
    model = NMPCResidualMLP(stats)
    model.train()
    xt = torch.as_tensor(train_z, dtype=torch.float32)
    yt = torch.as_tensor(train_y, dtype=torch.float32)
    scale = model.y_scale.detach().clone()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    history = []
    log_every = max(1, epochs // 10)

    for epoch in range(1, epochs+1):
        permutation = rng.permutation(len(train_z))
        for begin in range(0, len(permutation), batch_size):
            index = torch.as_tensor(
                permutation[begin:begin+batch_size],
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xt[index])
            loss = (((prediction-yt[index])/scale)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("训练 Loss 出现 NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            objective = float(
                torch.mean(((model(xt)-yt)/scale)**2).item()
            )
        history.append({
            "epoch": epoch,
            "train_normalized_mse": objective,
        })
        if verbose and (epoch == 1 or epoch % log_every == 0
                        or epoch == epochs):
            print(
                f"epoch {epoch:03d}/{epochs} | "
                f"Train residual normalized MSE={objective:.7f}",
                flush=True,
            )
        model.train()

    model.eval()
    return model, stats, history


def _one_step_assessment(model, z, y, y_scale):
    estimated = model.predict_numpy(z)
    error = estimated-y
    return {
        "normalized_mse": float(np.mean((error/y_scale)**2)),
        "residual_rmse_si": _per_state_rms(error),
        "n_intervals": int(len(z)),
    }


def evaluate_windows(
    dlc, z, model, *, start_interval, stop_interval,
    horizon, stride, q_scale, reference=None,
):
    """每个窗口纯开环积分；计算逐点与终点误差，不回灌状态。"""
    if stride < 1 or horizon < 1:
        raise ValueError("stride、horizon 必须为正整数")
    if reference is None:
        reference = NMPCReferenceParams()
    anchors = list(range(
        start_interval, stop_interval-horizon+1, stride
    ))
    if not anchors:
        raise ValueError("没有足够长度的完整预测窗口")
    nominal_error = []
    residual_error = []
    nominal_end = []
    residual_end = []
    examples = []
    for start in anchors:
        stop = start+horizon
        true = dlc.q_si[start:stop+1]
        nominal, corrected = rollout_pair_known_future(
            initial_q_si=dlc.q_si[start],
            observed_z_si=z[start:stop],
            model=model,
            dt_s=np.diff(dlc.time_s[start:stop+1]),
            reference=reference,
        )
        if nominal.shape != true.shape or corrected.shape != true.shape:
            raise AssertionError("自由滚动状态形状与 CarSim 不一致")
        nominal_error.append(nominal-true)
        residual_error.append(corrected-true)
        nominal_end.append(nominal[-1]-true[-1])
        residual_end.append(corrected[-1]-true[-1])
        if len(examples) == 0:
            examples.append({
                "time_s": dlc.time_s[start:stop+1].copy(),
                "true": true.copy(),
                "nominal": nominal.copy(),
                "residual": corrected.copy(),
                "known_external": z[start:stop, 3:].copy(),
            })

    en = np.concatenate(nominal_error, axis=0)
    er = np.concatenate(residual_error, axis=0)
    zn = _metric_rms(en/q_scale[None, :])
    zr = _metric_rms(er/q_scale[None, :])
    kn = np.asarray(nominal_end, dtype=np.float64)
    kr = np.asarray(residual_end, dtype=np.float64)

    result = {
        "horizon_s": float(
            dlc.time_s[anchors[0]+horizon]-dlc.time_s[anchors[0]]
        ),
        "horizon_steps": int(horizon),
        "n_windows": int(len(anchors)),
        "anchor_times_s": [
            float(dlc.time_s[start]) for start in anchors
        ],
        "nominal_normalized_path_rmse": zn,
        "residual_normalized_path_rmse": zr,
        "normalized_path_improvement_pct": _improvement(zn, zr),
        "nominal_path_rmse_si": _per_state_rms(en),
        "residual_path_rmse_si": _per_state_rms(er),
        "nominal_terminal_rmse_si": _per_state_rms(kn),
        "residual_terminal_rmse_si": _per_state_rms(kr),
        "state_names": list(NMPC_Q_NAMES),
        "state_units": ["m/s", "rad/s", "m/s"],
        "normalized_by_train_q_scale": q_scale.tolist(),
    }
    return result, examples[0]


def _save_example(output, example, label):
    """保存可审计的完整滚动数据和独立的3张物理量对照图。"""
    np.savez_compressed(
        output/f"{label}_rollout.npz",
        **example,
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("缺少 matplotlib：已保存 NPZ，跳过曲线绘制。")
        return

    labels = (
        ("vy", "Lateral velocity (m/s)"),
        ("yaw_rate", "Yaw rate (rad/s)"),
        ("vx", "Longitudinal velocity (m/s)"),
    )
    t = example["time_s"]-example["time_s"][0]
    for channel, (filename, ylabel) in enumerate(labels):
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        ax.plot(t, example["true"][:, channel],
                label="CarSim truth", linewidth=2)
        ax.plot(t, example["nominal"][:, channel],
                label="Pure NMPC", linestyle="--", linewidth=1.5)
        ax.plot(t, example["residual"][:, channel],
                label="NMPC + residual MLP", linestyle=":", linewidth=1.8)
        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label}: {ylabel}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/f"{label}_{filename}.png", dpi=160)
        plt.close(fig)


def run_experiment(
    data_path, output_dir, *, mu_hat=0.8,
    train_end_s=5.0, epochs=250, batch_size=128,
    lr=1e-3, weight_decay=1e-5, seed=42,
    horizons=(300, 500), anchor_stride=50,
    mz_mode="zero", lambda_mode="fixed-half",
    make_plots=True, verbose=True,
):
    """单入口：载入、转换、训练、滚动比较、保存报告。"""
    horizons = tuple(sorted(set(int(x) for x in horizons)))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons 必须为正采样步数")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不允许覆盖：{output}")

    dlc = load_dlc_si(data_path)
    config_path = dlc.path.parent/"metadata.json"
    if config_path.is_file():
        capture_metadata = json.loads(config_path.read_text(encoding="utf-8"))
        if capture_metadata.get("n_import") != 0:
            raise ValueError("不是零外部 Import 的 DLC 采集记录")
        if capture_metadata.get("n_export") != len(dlc.names):
            raise ValueError("元数据与实际 Export 通道数不匹配")
        if (tuple(capture_metadata.get("export_names", ()))
            != dlc.names):
            raise ValueError("元数据的 Export 顺序与 NPZ 不一致")

    actuation = build_dlc_inputs(
        dlc, mu_hat=mu_hat,
        mz_mode=mz_mode, lambda_mode=lambda_mode,
    )
    z = actuation.z_si
    q = dlc.q_si
    split = _split_index(dlc.time_s, train_end_s, horizons)
    ref = NMPCReferenceParams()
    nominal_next, labels = make_interval_targets(
        dlc, z, reference=ref,
    )
    train_z, valid_z = z[:split], z[split:-1]
    train_y, valid_y = labels[:split], labels[split:]
    if len(train_z) != split or len(valid_z) != len(z)-1-split:
        raise AssertionError("训练/验证时间索引不连续")

    if verbose:
        print(
            f"数据：{len(dlc.time_s)} 帧、{len(dlc.names)} 个 Export，"
            f"Train {len(train_z)} 个区间 "
            f"[{dlc.time_s[0]:.2f},{dlc.time_s[split]:.2f})s，"
            f"Validation {len(valid_z)} 个区间 "
            f"[{dlc.time_s[split]:.2f},{dlc.time_s[-1]:.2f}]s",
            flush=True,
        )
        print(
            "物理输入假设：Mz=" + mz_mode
            + ", lamda=" + lambda_mode
            + f", mu_hat={mu_hat}",
            flush=True,
        )
        print(
            "注意：未来实测 CarSim 轮胎 Fx/转角是离线重放的已知外生序列；"
            "这不是闭环 NMPC 的独立预测。",
            flush=True,
        )

    model, stats, history = train_single_scene(
        train_z, train_y, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, seed=seed,
        verbose=verbose,
    )
    q_scale = np.asarray(stats["x_scale"][:3], dtype=np.float64)
    output.mkdir(parents=True, exist_ok=False)

    checkpoint = {
        "format": MODEL_FORMAT,
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "hidden_sizes": [16, 16],
        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "training_mode": "one_step_fixed_epochs_train_time_segment_only",
        "evaluation_mode": "oracle_future_carsim_tire_force_replay",
        "dataset_sha256": dlc.file_sha256,
        "train_end_s": float(dlc.time_s[split]),
        "epochs": int(epochs),
        "seed": int(seed),
        "actuation_assumptions": actuation.assumptions,
    }
    torch.save(checkpoint, output/"final.pt")

    # 完整保存监督标签，后续可用同一数据重复核对。split 是 interval 索引。
    np.savez_compressed(
        output/"dlc_residual_dataset.npz",
        time_s=dlc.time_s,
        q_si=q,
        z_si=z,
        nominal_next_si=nominal_next,
        residual_target_si=labels,
        train_end_interval=np.asarray(split, dtype=np.int64),
        dt_s=np.diff(dlc.time_s),
    )
    single_step = {
        "train": _one_step_assessment(
            model, train_z, train_y, stats["y_scale"]
        ),
        "validation": _one_step_assessment(
            model, valid_z, valid_y, stats["y_scale"]
        ),
    }
    rollouts = {"train": {}, "validation": {}}
    for split_name, begin, end in (
        ("train", 0, split),
        ("validation", split, len(dlc.time_s)-1),
    ):
        for h in horizons:
            scores, example = evaluate_windows(
                dlc, z, model,
                start_interval=begin, stop_interval=end,
                horizon=h, stride=anchor_stride,
                q_scale=q_scale, reference=ref,
            )
            rollouts[split_name][f"{h}_steps"] = scores
            if verbose:
                print(
                    f"[{split_name:10s} H={h:3d} ({h/100:.1f}s)] "
                    f"windows={scores['n_windows']:2d} | "
                    f"pure={scores['nominal_normalized_path_rmse']:.7f} "
                    f"| residual={scores['residual_normalized_path_rmse']:.7f} "
                    f"| improvement="
                    f"{scores['normalized_path_improvement_pct']:+.2f}%",
                    flush=True,
                )
            if split_name == "validation" and make_plots:
                _save_example(output, example,
                              f"validation_{h/100:g}s")
            elif split_name == "validation":
                # 即使未装 matplotlib，仍保留最长时域的数值轨迹。
                if h == max(horizons):
                    np.savez_compressed(
                        output/f"validation_{h/100:g}s_rollout.npz",
                        **example,
                    )

    report = {
        "format": FORMAT,
        "original_npz": str(dlc.path),
        "original_npz_sha256": dlc.file_sha256,
        "n_samples": len(dlc.time_s),
        "n_exports": len(dlc.names),
        "train_intervals": len(train_z),
        "validation_intervals": len(valid_z),
        "train_interval_time_range_s": [
            float(dlc.time_s[0]), float(dlc.time_s[split])
        ],
        "validation_interval_time_range_s": [
            float(dlc.time_s[split]), float(dlc.time_s[-1])
        ],
        "train_validation_no_interval_overlap": True,
        "model_hidden_sizes": [16, 16],
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "normalization_train_only": model.normalization_json(),
        "nmpc_reference_params": asdict(ref),
        "nmpc_reference_is_existing_project_matlab_aligned_model": True,
        "dataset_assumptions": actuation.assumptions,
        "dataset_diagnostics": {
            "si": dlc.diagnostics,
            "actuation": actuation.diagnostics,
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "used_validation_labels_for_training_or_checkpoint_selection": False,
            "history": history,
        },
        "one_step_assessment": single_step,
        "rollouts": rollouts,
        "evaluation_scope": (
            "one DLC trajectory with non-overlapping time train/validation;"
            " future recorded CarSim tire forces and front road-wheel steer"
            " used as oracle exogenous schedule; this is not independent"
            " multi-scene generalization or deployable closed-loop NMPC"
        ),
        "validation_5s_n_windows_warning": (
            "At 10 s total and 5 s train/5 s validation, only one"
            " disjoint validation 5 s rollout exists."
        ),
        "genuine_closed_loop_nmpc_verified": False,
        "physics_actuator_interface_verified": False,
    }
    _json_dump(output/"report.json", report)
    _json_dump(output/"history.json", history)
    _json_dump(output/"normalization.json",
               model.normalization_json())
    print(f"完成：{output/'report.json'}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description="单个内置 DLC 场景训练 NMPC 残差并对比 3s/5s 预测"
    )
    p.add_argument("--data", required=True, type=Path,
                   help="dlc_exports.npz 的实际完整路径")
    p.add_argument("--output", required=True, type=Path,
                   help="新建的结果目录，禁止覆盖")
    p.add_argument("--mu-hat", type=float, default=0.8,
                   help="CarSim Procedure 已确认的路面附着系数")
    p.add_argument("--train-end", type=float, default=5.0,
                   help="Train 与 Validation 的时间边界，默认 5s")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizons", default="300,500",
                   help="逗号分隔的预测步数：默认 300,500")
    p.add_argument("--anchor-stride", type=int, default=50)
    p.add_argument("--mz-mode", choices=[
        "zero", "force-diff-diagnostic"
    ], default="zero")
    p.add_argument("--lambda-mode", choices=[
        "fixed-half", "ratio-strict"
    ], default="fixed-half")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    try:
        horizons = tuple(int(x.strip()) for x in args.horizons.split(","))
    except ValueError as exc:
        p.error(f"--horizons 格式错误：{exc}")
    run_experiment(
        args.data, args.output,
        mu_hat=args.mu_hat, train_end_s=args.train_end,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, horizons=horizons,
        anchor_stride=args.anchor_stride,
        mz_mode=args.mz_mode, lambda_mode=args.lambda_mode,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
