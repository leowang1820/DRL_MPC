# -*- coding: utf-8 -*-
"""固定窗口 NMPC 自由滚动残差消融实验（validation 优先）。

所有预测时域、残差强度 gamma、连续/离散 RK4 均使用同一批预测起点。
车身状态只从 t0 读取 CarSim 真值；未来 CarSim 的执行器等效量按时刻已知。

支持：
- 固定 5s (500*10ms)；另可取 3s (300*10ms) 分析截短的 Brake-turn。
- gamma=0,0.25,0.5,0.75,1.0 默认强度扫描，均可自定义。
- continuous: RK4(f_nom + gamma*g_theta)
- discrete:   RK4(f_nom) + dt*gamma*g_theta(q_k,a_k)
- 相同窗口/相同时间点的 endpoint/path RMSE 和最坏窗口误差。
- 仅使用检查点训练集的 q 标准差作不同单位之间的归一化比较。
- 不会修改 train/validation/test.npz 或旧模型。

为保证比较公平，所有变体共用同一批 10ms 时间戳和相同初始状态。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .nmpc_reference import NMPCReferenceParams, nmpc_accelerations
from .nmpc_residual_model import load_model_checkpoint
from .nmpc_residual_train import _check_manifest, load_split_npz, sha256
from .nmpc_rollout_evaluate import _make_anchors, HORIZONS


FORMAT = "wm_recovery_nmpc_fixed_cohort_ablation_v1"


def parse_gamma_grid(gamma_grid):
    """将 0,0.25,0.5,0.75,1.0 解析为互异的非负缩放系数。"""
    if isinstance(gamma_grid, str):
        gamma_grid = gamma_grid.split(",")
    values = [float(x) for x in gamma_grid]
    if not values or not all(np.isfinite(v) and 0 <= v <= 2 for v in values):
        raise ValueError("gamma-grid 必须是非空、有限、范围 0~2 的数字列表")
    if len(set(values)) != len(values):
        raise ValueError("gamma-grid 不允许重复系数")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError("为了严格对照，gamma-grid 必须包含 0 和 1")
    return tuple(values)


def make_variants(gamma_grid, channel_ablation=False):
    """构造完整、不重复的实验条件并保留可追踪的唯一名称。"""
    variants = []
    for integration in ("continuous", "discrete"):
        for g in gamma_grid:
            variants.append({
                "name": f"{integration}_g{g:g}",
                "integration": integration,
                "gamma": [g, g, g],
            })
    if channel_ablation:
        for gvec in (
            (0.0,1.0,1.0), (1.0,0.0,1.0), (1.0,1.0,0.0),
            (0.5,1.0,1.0), (1.0,0.5,1.0), (1.0,1.0,0.5),
        ):
            variants.append({
                "name": "continuous_gamma_"+"_".join(f"{v:g}" for v in gvec),
                "integration": "continuous",
                "gamma": list(gvec),
            })
    return variants


def select_fixed_cohort(data, fixed_horizon, anchor_stride):
    """只选具有完整 H 个连续真实区间的起点，所有变体和时长共享。"""
    if (isinstance(fixed_horizon, bool) or not isinstance(fixed_horizon, int)
            or not 10 <= fixed_horizon <= 500):
        raise ValueError("fixed_horizon 必须是 10~500 的整数区间数")
    anchors = _make_anchors(data, anchor_stride)
    eligible = anchors["available"] >= fixed_horizon
    starts = anchors["start"][eligible]
    if not len(starts):
        raise ValueError(
            f"没有连续 {fixed_horizon} 个 10ms 区间的窗口，请缩短 fixed_horizon"
        )
    return {
        key: value[eligible] for key, value in anchors.items()
    }


def _variant_step(q_variants, ext, dt, model, params, variants, substeps):
    """跨变体/窗口并行一步更新；每个变体完全使用自己的状态。

    q_variants: (V,B,3), ext: (B,5), dt:(B,)
    """
    v, b, _ = q_variants.shape
    ext_all = np.broadcast_to(ext[None, :, :], (v, b, 5)).reshape(v*b, 5)
    gamma = np.repeat(
        np.asarray([item["gamma"] for item in variants], dtype=np.float64),
        b, axis=0,
    )
    state = q_variants.reshape(v*b, 3).copy()
    h = np.tile(dt, v)[:, None] / substeps
    initial = state.copy()
    is_continuous = np.repeat(
        np.array([item["integration"] == "continuous" for item in variants],
                 dtype=bool),
        b,
    )
    cont_gamma = gamma * is_continuous[:, None]

    nonzero_cont = np.any(cont_gamma != 0.0, axis=1)

    def f(q_state):
        stage_z = np.concatenate((q_state, ext_all), axis=1)
        nominal = nmpc_accelerations(stage_z, params)
        # gamma=0 的纯物理基线绝不依赖 MLP 计算是否成功。
        if np.any(nonzero_cont):
            neural = model.predict_numpy(stage_z[nonzero_cont])
            nominal[nonzero_cont] += (
                cont_gamma[nonzero_cont] * neural
            )
        return nominal

    for _ in range(substeps):
        k1 = f(state)
        k2 = f(state+0.5*h*k1)
        k3 = f(state+0.5*h*k2)
        k4 = f(state+h*k3)
        state = state + (h/6.0)*(k1+2*k2+2*k3+k4)

    # 离散模式：修正只能使用区间起点 q_k，不在 RK4 子阶段重复计算。
    if np.any(~is_continuous):
        disc_ids = np.flatnonzero(
            (~is_continuous) & np.any(gamma != 0.0, axis=1)
        )
        z_start = np.concatenate((initial[disc_ids], ext_all[disc_ids]), axis=1)
        disc_correction = model.predict_numpy(z_start)
        state[disc_ids] += (
            np.tile(dt,v)[disc_ids, None]
            * gamma[disc_ids] * disc_correction
        )
    if not np.isfinite(state).all():
        raise FloatingPointError("某个固定窗口消融条件产生 NaN/Inf")
    return state.reshape(v,b,3)


def _stats(error_end, error_sq_path, horizon, qscale):
    """相同一组窗口中按三个物理通道分别算 RMSE 与最坏终点偏差。"""
    if not len(error_end):
        raise ValueError("缺少有效固定窗口")
    ep_rmse = np.sqrt(np.mean(error_end**2, axis=0))
    path_rmse = np.sqrt(np.mean(error_sq_path/horizon,axis=0))
    return {
        "windows": int(len(error_end)),
        "endpoint_rmse_q_si": ep_rmse.tolist(),
        "endpoint_mae_q_si": np.mean(np.abs(error_end),axis=0).tolist(),
        "endpoint_max_abs_q_si": np.max(np.abs(error_end),axis=0).tolist(),
        "endpoint_p90_abs_q_si": np.quantile(np.abs(error_end),0.9,axis=0).tolist(),
        "path_rmse_q_si": path_rmse.tolist(),
        # 三状态各有不同物理单位，综合指标只在 Train 的尺度归一化后计算。
        "endpoint_normalized_rmse": float(
            np.sqrt(np.mean(np.sum((error_end/qscale[None,:])**2,axis=1)/3.0))
        ),
        "path_normalized_rmse": float(
            np.sqrt(np.mean(np.sum(error_sq_path/qscale[None,:]**2,axis=1)
                            / (3.0*horizon)))
        ),
        "worst_window_endpoint_normalized_norm": float(
            np.max(np.linalg.norm(error_end/qscale[None,:],axis=1))
        ),
    }


def _snap_stats(snap, variants, select, qscale):
    """对固定子集（同一批窗口）统计全部变体的对照。"""
    h = snap["horizon"]
    err = snap["error"][:,select,:]
    sq = snap["sum_sq"][:,select,:]
    if not err.shape[1]:
        return None
    result = {
        v["name"]: _stats(err[i],sq[i],h,qscale)
        for i,v in enumerate(variants)
    }
    nominal = result["continuous_g0"]
    for name, item in result.items():
        e0 = np.asarray(nominal["endpoint_rmse_q_si"])
        e1 = np.asarray(item["endpoint_rmse_q_si"])
        p0 = np.asarray(nominal["path_rmse_q_si"])
        p1 = np.asarray(item["path_rmse_q_si"])
        item["endpoint_rmse_improvement_pct"] = [
            None if b<=1e-12 else float(100*(b-r)/b)
            for b,r in zip(e0,e1)
        ]
        item["path_rmse_improvement_pct"] = [
            None if b<=1e-12 else float(100*(b-r)/b)
            for b,r in zip(p0,p1)
        ]
        item["better_endpoint_window_fraction_by_channel"] = np.mean(
            np.abs(err[list(result).index(name)]) <
            np.abs(err[list(result).index("continuous_g0")]),
            axis=0,
        ).tolist()
    return result


def run_fixed_ablation_on_arrays(
    data, manifest, model, *, fixed_horizon=500, anchor_stride=50,
    gamma_grid=(0.0,0.25,0.5,0.75,1.0),
    channel_ablation=False, rk4_substeps=1,
):
    """不涉及 I/O 的核心固定窗口消融；合成数据可直接测试。"""
    if isinstance(rk4_substeps, bool) or not isinstance(rk4_substeps,int) or rk4_substeps<1:
        raise ValueError("rk4_substeps 必须为正整数")
    gamma_grid = parse_gamma_grid(gamma_grid)
    variants = make_variants(gamma_grid, channel_ablation)
    cohort = select_fixed_cohort(data, fixed_horizon, anchor_stride)
    starts = cohort["start"]
    b = len(starts)
    v = len(variants)
    q0 = data["q_k_si"][starts]
    states = np.broadcast_to(q0[None,:,:],(v,b,3)).copy()
    sum_sq = np.zeros((v,b,3),dtype=np.float64)

    qscale = model.x_scale.detach().cpu().numpy()[:3].astype(float)
    if not np.isfinite(qscale).all() or np.any(qscale <= 0):
        raise ValueError("检查点的训练集状态缩放系数非法")
    params = NMPCReferenceParams(**manifest["reference_parameters"])
    params.validate()
    observed = {}
    sample_idx = 0
    trace_true = [q0[sample_idx].copy()]
    trace_pred = [states[:,sample_idx,:].copy()]
    external_trace = []
    for k in range(fixed_horizon):
        loc = starts+k
        external = data["z_si"][loc,3:]
        dt = data["dt_s"][loc]
        states = _variant_step(
            states, external, dt, model, params, variants, rk4_substeps
        )
        # 只用于误差，绝不重置 states；模型不读取 z_si 的前三列。
        truth = data["q_true_next_si"][loc]
        diff = states-truth[None,:,:]
        sum_sq += diff**2
        trace_true.append(truth[sample_idx].copy())
        trace_pred.append(states[:,sample_idx,:].copy())
        external_trace.append(external[sample_idx].copy())
        h = k+1
        if h in set(HORIZONS) | {fixed_horizon}:
            observed[h] = {
                "horizon": h,
                "error": diff.copy(),
                "sum_sq": sum_sq.copy(),
            }

    zero_cont = [v["name"] for v in variants].index("continuous_g0")
    zero_disc = [v["name"] for v in variants].index("discrete_g0")
    if not np.allclose(
        trace_pred[-1][zero_cont],trace_pred[-1][zero_disc],rtol=0,atol=2e-9
    ):
        raise AssertionError("gamma=0 的连续和离散结果应退化为相同的名义积分")

    metrics = {}
    by_scene = {}
    by_speed = {}
    speed_labels = np.array([
        manifest["episodes"][int(idx)]["name"].split("_",1)[0]
        for idx in cohort["episode"]
    ])
    scene_labels = np.array([
        manifest["episodes"][int(idx)]["name"]
        for idx in cohort["episode"]
    ])
    for h, snapshot in sorted(observed.items()):
        label=f"{h*0.01:.2f}s"
        metrics[label]=_snap_stats(snapshot,variants,slice(None),qscale)
        for scene in sorted(set(scene_labels)):
            subset=scene_labels==scene
            by_scene.setdefault(scene,{})[label]=_snap_stats(snapshot,variants,subset,qscale)
        for speed in sorted(set(speed_labels)):
            subset=speed_labels==speed
            by_speed.setdefault(speed,{})[label]=_snap_stats(snapshot,variants,subset,qscale)

    final_key=f"{fixed_horizon*0.01:.2f}s"
    order=sorted(
        (item["name"] for item in variants),
        key=lambda name: metrics[final_key][name]["path_normalized_rmse"],
    )
    manifest_windows=[
        {
            "episode_index":int(cohort["episode"][i]),
            "scene":str(scene_labels[i]),
            "segment_id":int(cohort["segment"][i]),
            "interval_index":int(data["interval_index"][start]),
            "start_time_s":float(data["time_k_s"][start]),
            "horizon_intervals":int(fixed_horizon),
        }
        for i,start in enumerate(starts)
    ]
    window_fingerprint = hashlib.sha256(
        json.dumps(manifest_windows,ensure_ascii=False,
                   sort_keys=True,separators=(",",":")).encode("utf-8")
    ).hexdigest()
    report={
        "format":FORMAT,
        "fixed_cohort":True,
        "fixed_cohort_sha256":window_fingerprint,
        "fixed_horizon_intervals":int(fixed_horizon),
        "fixed_horizon_s":float(fixed_horizon*0.01),
        "window_count":int(b),
        "anchor_stride_intervals":int(anchor_stride),
        "window_scene_counts":{
            scene:int(np.count_nonzero(scene_labels==scene))
            for scene in sorted(set(scene_labels))
        },
        "train_only_state_scale_q_si":qscale.tolist(),
        "variant_definitions":variants,
        "horizons":metrics,
        "by_scene":by_scene,
        "by_speed":by_speed,
        "ranked_by_final_path_normalized_rmse":order,
        "windows":manifest_windows,
        "notes":[
            "所有时域/变体共享完全相同的固定窗口集合。",
            "仅将未来 CarSim 等效执行器参数视为已知；自由积分的车辆状态不会被真值重置。",
            "连续形式=RK4(f_nom+g*residual)，离散形式=RK4(f_nom)+dt*g*residual(q_k)。",
            "gamma=0 是纯物理模型，不是训练出的网络补偿。",
            "不同起点窗口可能重叠；最坏窗口以每个状态的绝对终点误差报告。",
            "归一化综合误差只使用 Train 检查点 q 均方差，无 Test 归一化泄漏。",
            "对照试验尚不等于闭环 NMPC 的物理验证。",
        ],
    }
    example={
        "scene":str(scene_labels[sample_idx]),
        "variant_names":np.asarray([item["name"] for item in variants]),
        "t_from_start_s":np.arange(fixed_horizon+1,dtype=float)*0.01,
        "q_true_si":np.stack(trace_true),
        "q_variants_si":np.stack(trace_pred,axis=1),
        "known_external_si":np.stack(external_trace),
    }
    return report,example


def evaluate_fixed_ablation(
    dataset_dir, checkpoint, *, split="validation", output_dir=None,
    fixed_horizon=500, anchor_stride=50,
    gamma_grid=(0.0,0.25,0.5,0.75,1.0),
    channel_ablation=False, rk4_substeps=1,
):
    """保持已有数据/模型只读，输出新的消融 JSON 和示例全时域轨迹。"""
    if split not in ("train","validation","test"):
        raise ValueError("split 必须是 train/validation/test")
    if output_dir is not None and Path(output_dir).expanduser().resolve().exists():
        raise FileExistsError("输出目录已存在，禁止覆盖历史实验")
    directory=Path(dataset_dir).expanduser().resolve(strict=True)
    checkpoint=Path(checkpoint).expanduser().resolve(strict=True)
    manifest=_check_manifest(directory)
    model,meta=load_model_checkpoint(checkpoint,device="cpu")
    if meta["dataset_manifest_sha256"] != sha256(directory/"dataset_manifest.json"):
        raise ValueError("检查点与数据集 manifest 的 SHA256 不一致")
    torch.set_num_threads(1)
    data=load_split_npz(directory,split)
    if len(data["z_si"]) != manifest["split_counts"][split]["intervals"]:
        raise ValueError("当前 split 的样本数与 manifest 不一致")
    report,example=run_fixed_ablation_on_arrays(
        data,manifest,model,
        fixed_horizon=fixed_horizon,anchor_stride=anchor_stride,
        gamma_grid=gamma_grid,channel_ablation=channel_ablation,
        rk4_substeps=rk4_substeps,
    )
    report.update({
        "split":split,
        "dataset":str(directory),
        "checkpoint":str(checkpoint),
        "dataset_manifest_sha256":meta["dataset_manifest_sha256"],
        "checkpoint_sha256":sha256(checkpoint),
    })
    if output_dir is not None:
        out=Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True,exist_ok=False)
        (out/"ablation_report.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),
            encoding="utf-8",
        )
        np.savez_compressed(out/"example_fixed.npz",**example)
        print("固定窗口报告：",out/"ablation_report.json")
        print("代表窗口轨迹：",out/"example_fixed.npz")
    return report
