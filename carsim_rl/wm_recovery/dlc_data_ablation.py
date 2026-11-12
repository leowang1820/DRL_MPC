# -*- coding: utf-8 -*-
"""CarSim DLC 单场景 A/B/C/D 样本量与动力学时间覆盖消融实验。

A: 只用前 500 个区间；B: 全 0~T 均匀抽样 500 个区间；
C: B + 全 0~T 余下区间均匀补充到 750；D: 全部区间（当前为 1000）。
四组采用相同的 MLP 结构、随机初始化种子和优化更新次数。
所有组都在相同的完整 DLC 轨迹起点进行 3s/5s 自由滚动评价。
该评价为 in-sample / partially-seen-trajectory diagnostic, 不是独立验证。

更重要的物理限制：未来 CarSim 实测 Steer_L1/R1 和 Fx_* 作为
oracle 外生时间表使用，属于已知响应的离线重放，而非真实可部署 MPC。

不修改任何 CarSim 文件、DLL、现有 NMPC 模型或旧训练检查点。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .dlc_si import load_dlc_si
from .dlc_actuation import build_dlc_inputs
from .nmpc_reference import (
    NMPCReferenceParams, rk4_nominal_step, NMPC_Q_NAMES,
)
from .nmpc_residual_model import (
    MODEL_FORMAT, INPUT_NAMES, TARGET_NAMES,
    NMPCResidualMLP, fit_normalization,
)
from .nmpc_rollout import rk4_free_step


FORMAT = "wm_recovery_dlc_abcd_time_coverage_v1"
ARMS = ("A", "B", "C", "D")


def _json_write(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2, axis=axis))


def make_abcd_indices(n_intervals, a_size=500, b_size=500, c_size=750):
    """B/C 是嵌套的、确定性均匀覆盖全程的唯一索引。

    固定采样间隔时，索引上的均匀分布等价于时间上的均匀分布。
    C 包含 B，D 包含 C，便于隔离额外 250/250 样本的作用。
    """
    n = int(n_intervals)
    if not (1 <= a_size == b_size < c_size <= n):
        raise ValueError(
            f"需要 1 <= A样本数=B样本数 < C样本数 <= 总区间数；"
            f"当前 A={a_size},B={b_size},C={c_size},n={n}"
        )
    a = np.arange(a_size, dtype=np.int64)
    b = np.unique(
        np.rint(np.linspace(0, n-1, b_size)).astype(np.int64)
    )
    if len(b) != b_size:
        raise AssertionError("B 均匀取样产生了重复位置")
    remaining = np.setdiff1d(
        np.arange(n, dtype=np.int64), b, assume_unique=True
    )
    extra_n = c_size-b_size
    extra = remaining[
        np.rint(
            np.linspace(0, len(remaining)-1, extra_n)
        ).astype(np.int64)
    ]
    c = np.sort(np.concatenate((b, extra)))
    d = np.arange(n, dtype=np.int64)
    result = {"A": a, "B": b, "C": c, "D": d}
    for key, idx in result.items():
        if len(idx) != len(set(idx.tolist())):
            raise AssertionError(f"{key} 有重复时间区间")
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            raise AssertionError(f"{key} 存在非法时间索引")
        assert np.all(np.diff(idx) > 0)
    if not set(b).issubset(set(c)):
        raise AssertionError("C 必须包含 B 的全部样本")
    return result


def make_fixed_anchor_indices(n_intervals, horizon, stride=50):
    """所有模型/随机种子使用同一批滚动起点，包括最后完整窗口。"""
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon 与 stride 必须为正数")
    if horizon > n_intervals:
        raise ValueError(
            f"完整轨迹只有 {n_intervals} 个区间，"
            f"无法做 {horizon} 步滚动"
        )
    anchors = np.arange(
        0, n_intervals-horizon+1, stride, dtype=np.int64
    )
    # 若最后一个窗口不能按 stride 对齐，只用原定锚点不偷偷增加窗口。
    return anchors


def _shared_state_scale(q):
    """公用状态尺度取全部区间左端状态，匹配之前全量训练 D 的指标。

    只用于 RMSE 的统一量纲，不进入任何 MLP 训练或参数拟合。
    """
    all_q=np.asarray(q, dtype=np.float64)
    if all_q.ndim != 2 or all_q.shape[1] != 3 or len(all_q) < 3:
        raise ValueError("q 应为至少三个状态时刻的 (T,3)")
    std=np.std(all_q[:-1],axis=0)
    return np.where(std < 1e-10, 1.0, std)


def _train_one_arm(
    z, residual_y, selected, *, seed, updates, batch_size,
    lr, weight_decay, log_every,
):
    """固定 Adam optimizer steps，实现公平计算预算比较。

    每个 Arm 的归一化仅由该 Arm 选中的训练样本计算；
    对应 seed 的网络初始化保持一致（Tanh MLP 的初始权重相同）。
    训练采用有放回小批量抽样，避免小数据组的 Epoch 含糊。
    """
    selected = np.asarray(selected, dtype=np.int64)
    if updates < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("updates、batch_size、lr 必须为正")
    if not np.isfinite(z[selected]).all() or not np.isfinite(
        residual_y[selected]
    ).all():
        raise ValueError("训练样本存在 NaN/Inf")

    stats = fit_normalization(z[selected], residual_y[selected])
    torch.manual_seed(int(seed))
    model = NMPCResidualMLP(stats)
    model.train()
    # 预先转成 tensor，避免 2000 次反复拷贝 NumPy 数组。
    xt = torch.as_tensor(z[selected], dtype=torch.float32)
    yt = torch.as_tensor(residual_y[selected], dtype=torch.float32)
    yscale = model.y_scale.detach().clone()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    rng = np.random.default_rng(int(seed)+20261111)
    log = []
    for step in range(1, updates+1):
        idx = torch.as_tensor(
            rng.integers(0, len(selected), size=batch_size),
            dtype=torch.long,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xt[idx])
        loss = torch.mean(((prediction-yt[idx])/yscale)**2)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"step={step} 出现非有限 loss，"
                "不要继续训练或自动将其截断。"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step in (1, updates):
            model.eval()
            with torch.inference_mode():
                full_selected_loss = float(
                    torch.mean(((model(xt)-yt)/yscale)**2).item()
                )
            log.append({
                "optimizer_step": step,
                "selected_train_normalized_mse": full_selected_loss,
            })
            model.train()

    model.eval()
    return model, log


def _batched_rollout(q, z, dt, anchors, horizon, *, model, reference):
    """将多个起点一起用 RK4 推进，绝不回灌任何未来 CarSim 真值。

    对每个起点，仅第零步使用 CarSim 真状态。未来仅从
    z[k, 3:] 读取 CarSim 实测等效控制/轮胎力。
    """
    a = np.asarray(anchors, dtype=np.int64)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("必须有至少一个预测窗口起点")
    if a.min() < 0 or a.max()+horizon > len(dt):
        raise ValueError("预测窗口越界")
    trajectory = np.empty((horizon+1, len(a), 3), dtype=np.float64)
    trajectory[0] = q[a]
    for offset in range(horizon):
        previous = trajectory[offset]
        exo = z[a+offset, 3:]
        interval_dt = dt[a+offset]
        try:
            trajectory[offset+1] = rk4_free_step(
                previous, exo, dt_s=interval_dt,
                reference=reference, residual_model=model,
            )
        except Exception as exc:
            raise RuntimeError(
                f"第 {offset} 个自由滚动积分步失败；"
                f"窗口起点={a.tolist()}。"
                "保留原始预测失败，不能静默裁剪车辆状态。"
            ) from exc
    if not np.isfinite(trajectory).all():
        raise FloatingPointError("自由滚动结果有 NaN/Inf")
    return trajectory


def _truth_for_windows(q, anchors, horizon):
    idx = anchors[None, :]+np.arange(horizon+1)[:, None]
    return q[idx].copy()


def _per_window_metric(pred, truth, state_scale):
    diff = np.asarray(pred, dtype=np.float64)-truth
    norm = diff/state_scale[None, None, :]
    # 时间 x 起点 x 状态 -> 每个起点的归一化整段 RMSE
    return _rms(norm, axis=(0, 2))


def _rollout_metrics(pred, truth, state_scale):
    error = np.asarray(pred, dtype=np.float64)-truth
    norm = error/state_scale[None, None, :]
    windows = _rms(norm, axis=(0, 2))
    return {
        "path_normalized_rmse": float(_rms(norm)),
        "path_rmse_si_per_state": _rms(error, axis=(0, 1)).tolist(),
        "terminal_rmse_si_per_state": _rms(error[-1], axis=0).tolist(),
        "terminal_normalized_rmse": float(_rms(norm[-1])),
        "per_window_path_normalized_rmse": windows.tolist(),
        "per_timestep_normalized_rmse": _rms(norm, axis=(1, 2)).tolist(),
    }


def _share_of_validation_outside_training(z, indices):
    """按单个变量 min/max 的粗略覆盖诊断，非高维支持集测度。"""
    result = {}
    train = z[indices]
    evaluate = z[:-1]
    for i, name in enumerate(INPUT_NAMES):
        lo, hi = float(np.min(train[:, i])), float(np.max(train[:, i]))
        outside = (evaluate[:, i] < lo) | (evaluate[:, i] > hi)
        result[str(name)] = {
            "train_min": lo,
            "train_max": hi,
            "evaluation_min": float(np.min(evaluate[:, i])),
            "evaluation_max": float(np.max(evaluate[:, i])),
            "evaluation_outside_train_range_pct":
                float(np.mean(outside)*100.0),
        }
    return result


def _make_rollout_plot(output, *, horizon, anchors_s,
                       nominal_metrics, arm_metrics):
    """横坐标为预测时域，纵坐标是所有相同起点上的状态归一化 RMSE。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    timeline = np.linspace(0.0, horizon*0.01, horizon+1)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(
        timeline,
        nominal_metrics["per_timestep_normalized_rmse"],
        linewidth=2.2, label="Pure NMPC",
    )
    for key in ARMS:
        ax.plot(
            timeline, arm_metrics[key]["per_timestep_normalized_rmse"],
            label=f"{key}: residual MLP", linewidth=1.6,
        )
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("Normalized state RMSE")
    ax.set_title(
        f"DLC in-sample ablation: {horizon*0.01:.0f}s"
        f" | {len(anchors_s)} overlapping windows"
    )
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/f"mean_error_curve_{horizon}steps.png",dpi=170)
    plt.close(fig)


def _make_summary_bar_plot(output, summary, horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.arange(len(horizons),dtype=float)
    width=0.16
    fig,ax=plt.subplots(figsize=(9.5,5.0))
    series = ["Pure NMPC"]+list(ARMS)
    for i, series_name in enumerate(series):
        if series_name=="Pure NMPC":
            values=[
                summary["baseline"][str(h)]["path_normalized_rmse"]
                for h in horizons
            ]
        else:
            values=[
                summary["groups"][series_name][str(h)]
                ["mean_residual_path_normalized_rmse"]
                for h in horizons
            ]
        ax.bar(
            x+(i-2)*width,values,width,
            label=series_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h/100:g}s" for h in horizons])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Normalized path RMSE (common full-data scale)")
    ax.set_title("DLC A/B/C/D same-trajectory ablation")
    ax.legend()
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"ablation_rmse_summary.png",dpi=180)
    plt.close(fig)


def run_ablation(
    data_path, output_dir, *,
    mu_hat=0.8, a_size=500, b_size=500, c_size=750,
    updates=2000, batch_size=128, lr=1e-3, weight_decay=1e-5,
    seeds=(42,), horizons=(300,500), stride=50,
    mz_mode="zero", lambda_mode="fixed-half",
    make_plots=True, verbose=True,
):
    """完整跑四组消融并保存每个 seed/窗口的可追溯结果。"""
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{root}")
    seed_list = tuple(int(x) for x in seeds)
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not seed_list or len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds 不能为空或重复")
    if any(s < 0 for s in seed_list):
        raise ValueError("随机种子必须非负")
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons 必须包含正整数")
    if stride <= 0 or updates < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("stride、updates、batch-size、lr 必须为正")
    torch.set_num_threads(1)

    dlc = load_dlc_si(data_path)
    cfg_file = dlc.path.parent/"metadata.json"
    if cfg_file.is_file():
        metadata = json.loads(cfg_file.read_text(encoding="utf-8"))
        if int(metadata.get("n_export",-1)) != len(dlc.names):
            raise ValueError("metadata Export 数量与 NPZ 不一致")
        if tuple(metadata.get("export_names",())) != dlc.names:
            raise ValueError("metadata Export 顺序与 NPZ 不一致")
        if int(metadata.get("n_import",-1)) != 0:
            raise ValueError("当前文件并非零外部 Import 的 DLC")

    actuation = build_dlc_inputs(
        dlc,mu_hat=mu_hat,mz_mode=mz_mode,lambda_mode=lambda_mode
    )
    z = actuation.z_si
    q = dlc.q_si
    dt = np.diff(dlc.time_s)
    n=len(dt)
    selection = make_abcd_indices(
        n, a_size=a_size, b_size=b_size, c_size=c_size
    )
    scale = _shared_state_scale(q)
    reference=NMPCReferenceParams()

    # 计算区间单步残差：评估标签覆盖所有区间，但每个模型只使用自己的 index 训练。
    nominal_next=rk4_nominal_step(z[:-1],dt_s=dt,params=reference)
    target=(q[1:]-nominal_next)/dt[:,None]
    if not np.isfinite(target).all():
        raise FloatingPointError("单步 NMPC 残差标签包含 NaN/Inf")
    # 此尺度只用于所有模型的共同一步预测评价，不参与任何模型训练，
    # 避免 A/B/C/D 各自 target std 不同造成一步指标不可比。
    full_target_std=np.std(target,axis=0)
    common_target_scale=np.where(
        full_target_std<1e-10,1.0,full_target_std
    )

    anchor_map = {
        h: make_fixed_anchor_indices(n,h,stride) for h in horizons
    }
    # 单独预计算一次真实路径/名义模型滚动：各组共用，避免重复计算。
    baseline = {}
    for h in horizons:
        anchors=anchor_map[h]
        truth=_truth_for_windows(q,anchors,h)
        pure=_batched_rollout(
            q,z,dt,anchors,h,model=None,reference=reference
        )
        metric=_rollout_metrics(pure,truth,scale)
        baseline[h] = {
            "anchors": anchors,
            "truth": truth,
            "pure": pure,
            "metrics": metric,
        }
        if verbose:
            print(
                f"共同基线 H={h} ({h/100:g}s) | {len(anchors)} 个窗口 "
                f"| Pure NMPC={metric['path_normalized_rmse']:.7f}",
                flush=True,
            )

    root.mkdir(parents=True,exist_ok=False)
    (root/"models").mkdir()
    (root/"rollouts").mkdir()
    np.savez_compressed(
        root/"training_indices.npz",
        **{f"{key}_interval_indices":idx for key,idx in selection.items()},
        **{f"anchors_{h}steps":anchor_map[h] for h in horizons},
        state_scale_common_eval=scale,
        target_scale_common_eval=common_target_scale,
        train_target_si=target,
        source_time_s=dlc.time_s,
    )
    run_config={
        "format":FORMAT,
        "data":str(dlc.path),
        "data_sha256":dlc.file_sha256,
        "n_intervals":n,
        "n_exports":len(dlc.names),
        "seeds":list(seed_list),
        "horizons_steps":list(horizons),
        "anchor_stride_steps":int(stride),
        "equal_optimizer_updates_per_arm":int(updates),
        "train_batch_size":int(batch_size),
        "learning_rate":float(lr),
        "weight_decay":float(weight_decay),
        "common_state_eval_scale":scale.tolist(),
        "common_target_eval_scale":common_target_scale.tolist(),
        "normalization":(
            "Each arm fits its own normalization on selected training rows;"
            " evaluation RMSE uses the same full-trajectory state scale."
        ),
        "selection":{
            key:{
                "n_train_intervals":int(len(idx)),
                "first_index":int(idx[0]),
                "last_index":int(idx[-1]),
                "selected_index_sha256":hashlib.sha256(
                    idx.tobytes()
                ).hexdigest(),
                "index_picking":{
                    "A":"first contiguous training intervals",
                    "B":"uniformly dispersed across entire trajectory",
                    "C":"all B plus uniform supplement from remaining indices",
                    "D":"all trajectory intervals",
                }[key],
                "coverage":_share_of_validation_outside_training(z,idx),
            } for key,idx in selection.items()
        },
        "dataset_assumptions":actuation.assumptions,
        "dataset_diagnostics":{
            "si":dlc.diagnostics,
            "actuation":actuation.diagnostics,
        },
        "reference_params":asdict(reference),
        "evaluation_scope":(
            "All arms evaluated on exactly same 3s/5s overlapping windows "
            "from the one DLC trajectory. B/C/D have trained on portions "
            "or all of those windows. NOT unseen-scene validation."
        ),
        "known_future_inputs_include_measured_tire_forces":True,
        "physical_interface_verified":False,
        "generalization_test":False,
    }
    _json_write(root/"experiment_config.json",run_config)

    csv_rows=[]
    window_rows=[]
    group_results={group:{} for group in ARMS}
    for key in ARMS:
        idx=selection[key]
        for seed in seed_list:
            start=time.perf_counter()
            model,log=_train_one_arm(
                z[:-1],target,idx,
                seed=seed,updates=updates,
                batch_size=batch_size,lr=lr,
                weight_decay=weight_decay,
                log_every=max(1,updates//20),
            )
            train_norm=fit_normalization(z[idx],target[idx])
            checksum=hashlib.sha256(
                idx.tobytes()
            ).hexdigest()
            checkpoint={
                "format":MODEL_FORMAT,
                "input_names":list(INPUT_NAMES),
                "target_names":list(TARGET_NAMES),
                "hidden_sizes":[16,16],
                "model_state_dict":{
                    k:v.detach().cpu().clone()
                    for k,v in model.state_dict().items()
                },
                "dataset_sha256":dlc.file_sha256,
                "training_selection":key,
                "training_index_sha256":checksum,
                "training_optimizer_updates":int(updates),
                "training_seed":int(seed),
                "training_is_from_single_dlc_trajectory":True,
                "actuation_assumptions":actuation.assumptions,
            }
            torch.save(
                checkpoint,root/"models"/f"{key}_seed_{seed}.pt"
            )

            model_y=model.predict_numpy(z[:-1])
            all_step_norm=float(
                _rms((model_y-target)/
                     common_target_scale[None,:])
            )
            single_selected=float(
                _rms((model_y[idx]-target[idx])/
                     train_norm["y_scale"][None,:])
            )

            for h in horizons:
                common=baseline[h]
                trajectory=_batched_rollout(
                    q,z,dt,common["anchors"],h,
                    model=model,reference=reference
                )
                metrics=_rollout_metrics(
                    trajectory,common["truth"],scale
                )
                pure_metrics=common["metrics"]
                pure_rmse=pure_metrics["path_normalized_rmse"]
                residual_rmse=metrics["path_normalized_rmse"]
                gain=float(100*(pure_rmse-residual_rmse)/pure_rmse)
                window_b=np.asarray(
                    pure_metrics["per_window_path_normalized_rmse"]
                )
                window_m=np.asarray(
                    metrics["per_window_path_normalized_rmse"]
                )
                winner_fraction=float(
                    np.mean(window_m<window_b)
                )
                group_results[key].setdefault(str(h),[]).append({
                    "seed":int(seed),
                    "train_interval_count":len(idx),
                    "selected_training_normalized_rmse":single_selected,
                    "all_interval_one_step_normalized_rmse":all_step_norm,
                    "final_selected_training_normalized_mse":
                        float(log[-1]["selected_train_normalized_mse"]),
                    "nominal_path_normalized_rmse":pure_rmse,
                    "residual_path_normalized_rmse":residual_rmse,
                    "path_improvement_pct":gain,
                    "nominal_terminal_rmse_per_state_si":
                        pure_metrics["terminal_rmse_si_per_state"],
                    "residual_terminal_rmse_per_state_si":
                        metrics["terminal_rmse_si_per_state"],
                    "fraction_of_windows_improved":winner_fraction,
                    "residual_worst_window_normalized_rmse":
                        float(np.max(window_m)),
                    "nominal_worst_window_normalized_rmse":
                        float(np.max(window_b)),
                    "rollout_state_rmse_si":metrics["path_rmse_si_per_state"],
                    "elapsed_training_and_evaluation_seconds":
                        float(time.perf_counter()-start),
                })
                csv_rows.append({
                    "arm":key,"seed":seed,
                    "n_train_intervals":len(idx),
                    "horizon_s":h*.01,"n_windows":len(common["anchors"]),
                    "nominal_rmse":pure_rmse,
                    "residual_rmse":residual_rmse,
                    "improvement_pct":gain,
                    "fraction_windows_improved":winner_fraction,
                    "worst_residual_window_rmse":float(np.max(window_m)),
                    "selected_train_one_step_normalized_rmse":single_selected,
                    "full_one_step_normalized_rmse":all_step_norm,
                })
                for k, anchor in enumerate(common["anchors"]):
                    window_rows.append({
                        "arm":key,"seed":seed,
                        "horizon_steps":h,
                        "anchor_interval":int(anchor),
                        "anchor_time_s":float(dlc.time_s[anchor]),
                        "nominal_normalized_path_rmse":float(window_b[k]),
                        "residual_normalized_path_rmse":float(window_m[k]),
                        "improvement_pct": (
                            float(100*(window_b[k]-window_m[k])/window_b[k])
                            if window_b[k] > 1e-12 else None
                        ),
                    })
                # 保留每个窗口的真实状态、预测状态及起点；用于分析最坏误差。
                np.savez_compressed(
                    root/"rollouts"/f"{key}_seed_{seed}_{h}steps.npz",
                    time_s=dlc.time_s[common["anchors"][None,:]
                                      + np.arange(h+1)[:,None]],
                    anchors=common["anchors"],
                    truth_si=common["truth"],
                    pure_nmpc_si=common["pure"],
                    residual_nmpc_si=trajectory,
                )
                if verbose:
                    print(
                        f"[{key} seed={seed} H={h/100:g}s] "
                        f"n={len(idx)} windows={len(common['anchors'])} "
                        f"pure={pure_rmse:.6f} "
                        f"residual={residual_rmse:.6f} "
                        f"gain={gain:+.2f}% "
                        f"win_better={winner_fraction*100:.1f}%",
                        flush=True,
                    )

            _json_write(
                root/"models"/f"{key}_seed_{seed}_history.json",
                log,
            )

    # 对多 seed 的结果进行宏观汇总，避免只挑最佳 seed。
    summary={
        "format":FORMAT,
        "baseline":{},
        "groups":{},
        "seeds":list(seed_list),
        "horizons_steps":list(horizons),
        "n_intervals":n,
        "train_and_evaluation_same_dlc_trajectory":True,
        "genuine_independent_validation":False,
        "common_evaluation_scale":scale.tolist(),
    }
    for h in horizons:
        m=baseline[h]["metrics"]
        summary["baseline"][str(h)]={
            "n_windows":int(len(anchor_map[h])),
            "anchor_times_s":dlc.time_s[anchor_map[h]].tolist(),
            "path_normalized_rmse":m["path_normalized_rmse"],
            "terminal_rmse_per_state_si":m["terminal_rmse_si_per_state"],
            "per_timestep_normalized_rmse":
                m["per_timestep_normalized_rmse"],
        }
    for group in ARMS:
        summary["groups"][group]={}
        for h in horizons:
            samples=group_results[group][str(h)]
            values=np.asarray(
                [r["residual_path_normalized_rmse"] for r in samples],
                dtype=np.float64
            )
            nominal=baseline[h]["metrics"]["path_normalized_rmse"]
            summary["groups"][group][str(h)]={
                "n_train_intervals":int(len(selection[group])),
                "per_seed_results":samples,
                "mean_residual_path_normalized_rmse":float(np.mean(values)),
                "std_residual_path_normalized_rmse":float(np.std(values)),
                "pure_nominal_path_normalized_rmse":nominal,
                "improvement_of_mean_rmse_pct":
                    float(100*(nominal-np.mean(values))/nominal),
            }

    _json_write(root/"summary.json",summary)

    # 按 Arm / 时域聚合种子结果，便于 Excel 直接比较样本覆盖与样本数量。
    summary_table=[]
    for h in horizons:
        nominal=baseline[h]["metrics"]["path_normalized_rmse"]
        summary_table.append({
            "arm":"Pure NMPC","horizon_s":h*.01,
            "n_train_intervals":0,"n_windows":len(anchor_map[h]),
            "mean_residual_path_rmse":nominal,
            "std_residual_path_rmse":0.,
            "improvement_pct":0.,
            "n_seeds":0,
        })
        for group in ARMS:
            r=summary["groups"][group][str(h)]
            summary_table.append({
                "arm":group,"horizon_s":h*.01,
                "n_train_intervals":r["n_train_intervals"],
                "n_windows":len(anchor_map[h]),
                "mean_residual_path_rmse":
                    r["mean_residual_path_normalized_rmse"],
                "std_residual_path_rmse":
                    r["std_residual_path_normalized_rmse"],
                "improvement_pct":r["improvement_of_mean_rmse_pct"],
                "n_seeds":len(seed_list),
            })
    with (root/"summary_table.csv").open(
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(summary_table[0].keys()))
        writer.writeheader()
        writer.writerows(summary_table)

    for path,rows in (
        (root/"metrics_by_seed.csv",csv_rows),
        (root/"per_window_metrics.csv",window_rows),
    ):
        if rows:
            with path.open("w",newline="",encoding="utf-8-sig") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    if make_plots:
        try:
            _make_summary_bar_plot(root,summary,horizons)
            # 将多个 seed 的逐时间误差先均值化，再画共同横坐标的诊断曲线。
            for h in horizons:
                arm_paths={}
                for group in ARMS:
                    records=group_results[group][str(h)]
                    time_rmse=[]
                    for rec in records:
                        seed=rec["seed"]
                        with np.load(
                            root/"rollouts"/f"{group}_seed_{seed}_{h}steps.npz",
                            allow_pickle=False
                        ) as d:
                            traj=d["residual_nmpc_si"]
                        m=_rollout_metrics(
                            traj,baseline[h]["truth"],scale
                        )
                        time_rmse.append(
                            m["per_timestep_normalized_rmse"]
                        )
                    arm_paths[group]={
                        "per_timestep_normalized_rmse":
                            np.mean(time_rmse,axis=0).tolist()
                    }
                _make_rollout_plot(
                    root,horizon=h,
                    anchors_s=dlc.time_s[anchor_map[h]],
                    nominal_metrics=baseline[h]["metrics"],
                    arm_metrics=arm_paths,
                )
        except ImportError:
            print(
                "matplotlib 缺失，仅保存 JSON/CSV/NPZ，不影响消融指标。",
                flush=True,
            )

    print("\n=== A/B/C/D 同轨迹样本覆盖消融：所有组同一评价窗口 ===")
    print("注意：这不是独立验证；使用了已记录 CarSim 未来轮胎力和转向。")
    for h in horizons:
        pure=baseline[h]["metrics"]["path_normalized_rmse"]
        print(
            f"H={h/100:g}s | N_windows={len(anchor_map[h])} "
            f"| pure NMPC={pure:.7f}"
        )
        for key in ARMS:
            x=summary["groups"][key][str(h)]
            print(
                f"  {key} n={x['n_train_intervals']:4d}: "
                f"residual={x['mean_residual_path_normalized_rmse']:.7f} "
                f"±{x['std_residual_path_normalized_rmse']:.7f} "
                f"| gain={x['improvement_of_mean_rmse_pct']:+.2f}%"
            )
    print(f"\n结果目录：{root}",flush=True)
    return summary


def _parse_int_list(text, name):
    try:
        numbers=tuple(int(x.strip()) for x in text.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} 必须用逗号分隔整数") from exc
    if not numbers:
        raise ValueError(f"{name} 不能为空")
    return numbers


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="DLC 同轨迹 A/B/C/D 数据量与动态覆盖消融"
    )
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--mu-hat",type=float,default=0.8)
    parser.add_argument("--a-size",type=int,default=500)
    parser.add_argument("--b-size",type=int,default=500)
    parser.add_argument("--c-size",type=int,default=750)
    parser.add_argument("--updates",type=int,default=2000,
                        help="每组每随机种子的固定优化更新次数")
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--weight-decay",type=float,default=1e-5)
    parser.add_argument("--seeds",default="42",
                        help="随机种子列表，如 42,43,44")
    parser.add_argument("--horizons",default="300,500",
                        help="滚动预测步数，如 300,500")
    parser.add_argument("--anchor-stride",type=int,default=50)
    parser.add_argument(
        "--mz-mode",choices=("zero","force-diff-diagnostic"),
        default="zero"
    )
    parser.add_argument(
        "--lambda-mode",choices=("fixed-half","ratio-strict"),
        default="fixed-half"
    )
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args(argv)
    try:
        seeds=_parse_int_list(args.seeds,"seeds")
        horizons=_parse_int_list(args.horizons,"horizons")
    except ValueError as exc:
        parser.error(str(exc))
    run_ablation(
        args.data,args.output,mu_hat=args.mu_hat,
        a_size=args.a_size,b_size=args.b_size,c_size=args.c_size,
        updates=args.updates,batch_size=args.batch_size,
        lr=args.lr,weight_decay=args.weight_decay,
        seeds=seeds,horizons=horizons,stride=args.anchor_stride,
        mz_mode=args.mz_mode,lambda_mode=args.lambda_mode,
        make_plots=not args.no_plots,
    )


if __name__=="__main__":
    main()
