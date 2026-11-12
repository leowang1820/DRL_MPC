# -*- coding: utf-8 -*-
"""DLC 已训练 MLP 的可微多步 RK4 微调：单步+滚动，不改模型结构。

仅使用已采集的一条 CarSim DLC 全轨迹（同数据重构诊断）：
- 加载现有 final.pt，继续训练原来的 8->16->16->3 MLP；
- 0.5s -> 1.0s -> 2.0s 的分阶段连续 RK4 自由滚动训练；
- 每个训练窗口只有初始 q 来自 CarSim，其后只能递推自己的 q；
- 每区间已知的未来前轮角/轮胎 Fx 来自 CarSim 实测（oracle exogenous）；
- 与微调前的 MLP 和纯 NMPC 在相同 3s/5s 起点进行比较；
- 不自动接受失败结果：负提升也会如实记录。

不是独立泛化验证；训练和评价共享同一个 CarSim 场景。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .dlc_si import load_dlc_si
from .dlc_actuation import build_dlc_inputs
from .nmpc_reference import NMPCReferenceParams, rk4_nominal_step
from .nmpc_reference_torch import hybrid_rk4_step_torch, rk4_nominal_step_torch
from .nmpc_residual_model import (
    INPUT_NAMES, TARGET_NAMES, MODEL_FORMAT, load_model_checkpoint,
)
from .nmpc_rollout import rollout_pair_known_future
from .dlc_all_windows_viewer import (
    compute_all_window_batches, window_error_arrays, show_all_windows,
)


REPORT_FORMAT = "wm_dlc_multistep_finetune_full_data_v1"


def _save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_stages(s):
    """解析 50:30,100:30,200:20（步数:优化更新次数），严格递增。"""
    stages=[]
    for segment in str(s).split(","):
        parts=segment.strip().split(":")
        if len(parts)!=2:
            raise ValueError(
                f"错误阶段定义 {segment!r}，格式示例：50:30,100:30,200:20"
            )
        try:
            horizon, updates=map(int,parts)
        except ValueError as exc:
            raise ValueError(f"阶段必须为整数 {segment!r}") from exc
        if horizon<=0 or updates<=0:
            raise ValueError("阶段预测步数与更新次数均必须 > 0")
        if stages and horizon<=stages[-1][0]:
            raise ValueError("阶段预测时域必须严格递增")
        stages.append((horizon,updates))
    if not stages:
        raise ValueError("至少设置一个训练阶段")
    return tuple(stages)


def _validate_pretraining_metadata(checkpoint, dlc, actuation):
    recorded_hash=checkpoint.get("dataset_sha256")
    if recorded_hash and recorded_hash != dlc.file_sha256:
        raise ValueError(
            "初始 final.pt 的训练数据 SHA256 与给定 dlc_exports.npz 不一致；"
            "为避免错误模型对比，本实验拒绝跨采集文件微调。"
        )
    assumptions=checkpoint.get("actuation_assumptions",{})
    for key in ("mz_mode","lambda_mode"):
        old=assumptions.get(key)
        current=actuation.assumptions.get(key)
        if old is not None and old != current:
            raise ValueError(
                f"模型训练时 {key}={old}, 当前 {key}={current}。"
                "保持原训练的物理输入假设一致。"
            )
    if "mu_hat" in assumptions:
        old=float(assumptions["mu_hat"])
        new=float(actuation.assumptions["mu_hat"])
        if abs(old-new)>1e-10:
            raise ValueError(
                f"模型训练时 mu_hat={old}，当前 mu_hat={new}"
            )


def verify_nominal_torch_numpy(z, dt, reference, *, tolerance=1e-3):
    """避免可微 Torch 动力学与 NumPy 物理模型在现实数据上不一致。

    对部分实际 CarSim 区间同时调用 NumPy 和 Torch 名义 RK4；
    实验开始前检查数值误差，若偏差过大直接报错。
    """
    n=len(dt)
    index=np.unique(np.rint(np.linspace(0,n-1,min(n,40))).astype(int))
    zn=z[index]
    dn=dt[index]
    nnext=rk4_nominal_step(zn,dt_s=dn,params=reference)
    with torch.no_grad():
        tnext=rk4_nominal_step_torch(
            torch.as_tensor(zn,dtype=torch.float32),
            dt_s=torch.as_tensor(dn,dtype=torch.float32),
            params=reference,
        ).cpu().numpy()
    absdiff=np.abs(nnext-tnext)
    value=float(np.max(absdiff))
    if not math.isfinite(value) or value>tolerance:
        raise RuntimeError(
            f"同一 NMPC 名义步的 NumPy 与可微 Torch 误差={value:.6g}，"
            f"超过容差 {tolerance}。请先排查物理参数、数据类型和计算公式。"
        )
    return {
        "n_tested":len(index),
        "nominal_rk4_numpy_torch_max_abs_si":value,
    }


def _single_step_target(dlc,z,ref):
    dt=np.diff(dlc.time_s)
    nominal=rk4_nominal_step(z[:-1],dt_s=dt,params=ref)
    labels=(dlc.q_si[1:]-nominal)/dt[:,None]
    if not np.isfinite(labels).all():
        raise FloatingPointError("单步残差目标含 NaN 或 Inf")
    return labels


def _state_scale(model):
    scale=model.x_scale.detach().cpu().numpy()[:3].astype(float)
    if np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError("原模型状态归一化尺度非法")
    return scale


def _per_state_rmse(a):
    x=np.asarray(a,dtype=float)
    return np.sqrt(np.mean(x*x,axis=0)).astype(float).tolist()


def _numeric_trajectory_metrics(pred,truth,scale):
    err=np.asarray(pred,dtype=float)-np.asarray(truth,dtype=float)
    norm=err/scale[None,:]
    return {
        "normalized_path_rmse":float(np.sqrt(np.mean(norm*norm))),
        "normalized_terminal_rmse":float(np.sqrt(np.mean(norm[-1]**2))),
        "path_rmse_si_by_state":_per_state_rmse(err),
        "terminal_abs_error_si_by_state":np.abs(err[-1]).tolist(),
        "min_vx_mps":float(np.min(pred[:,2])),
        "max_abs_vy_mps":float(np.max(np.abs(pred[:,0]))),
        "max_abs_r_radps":float(np.max(np.abs(pred[:,1]))),
    }


def _focus_diagnostic(
    dlc,z,model,scale,*,focus_start_s=5.5,
    horizon_steps=500,reference=None,
):
    """在失稳最明显的 5.5~10.5s 整个连续窗口中比较，不重置状态。"""
    if reference is None:
        reference=NMPCReferenceParams()
    t=dlc.time_s
    start=int(np.argmin(np.abs(t-focus_start_s)))
    if start+horizon_steps>=len(t):
        raise ValueError(
            f"focus_start={focus_start_s}s 对应采样帧 {start}，"
            f"没有完整的 {horizon_steps} 步预测窗口"
        )
    stop=start+horizon_steps
    qtrue=dlc.q_si[start:stop+1]
    no,mlp=rollout_pair_known_future(
        initial_q_si=qtrue[0],
        observed_z_si=z[start:stop],
        model=model,
        dt_s=np.diff(t[start:stop+1]),
        reference=reference,
    )
    # 不重置的同一条轨迹，在不同预测时刻累计到当前位置的 RMSE。
    # 和此前多次重置后拼接 RMSE 不同，这里可以定位首次明显失配时间。
    horizons_to_report=[
        h for h in (10,50,100,125,150,175,200,250,300,400,500)
        if h<=horizon_steps
    ]
    if horizon_steps not in horizons_to_report:
        horizons_to_report.append(horizon_steps)
    path_at_horizon={}
    for h in horizons_to_report:
        path_at_horizon[f"{h}_steps"]={
            "elapsed_s":float(t[start+h]-t[start]),
            "pure_normalized_path_rmse":
                _numeric_trajectory_metrics(no[:h+1],qtrue[:h+1],scale)
                ["normalized_path_rmse"],
            "mlp_normalized_path_rmse":
                _numeric_trajectory_metrics(mlp[:h+1],qtrue[:h+1],scale)
                ["normalized_path_rmse"],
        }
    instant_norm=np.sqrt(
        np.mean(((mlp-qtrue)/scale[None,:])**2,axis=1)
    )
    crossing={}
    for cutoff in (0.1,0.5,1.0,2.0):
        hits=np.flatnonzero(instant_norm>cutoff)
        crossing[str(cutoff)]=(
            float(t[start+hits[0]]-t[start]) if len(hits) else None
        )
    return {
        "start_s":float(t[start]),
        "end_s":float(t[stop]),
        "pure_nmpc":_numeric_trajectory_metrics(no,qtrue,scale),
        "residual_nmpc":_numeric_trajectory_metrics(mlp,qtrue,scale),
        "unreset_path_rmse_at_horizons":path_at_horizon,
        "first_instantaneous_normalized_rmse_above":crossing,
    }


def evaluate_same_windows(dlc,z,model,scale,*,horizons=(300,500),
                          stride=50,reference=None):
    """所有预测起点固定，从 0s 到总轨迹；与旧 full-data 评价一致。"""
    batches=compute_all_window_batches(
        dlc,z,model,horizons=horizons,anchor_stride=stride,
        reference=reference,
    )
    metrics={}
    for horizon,batch in batches.items():
        errors=window_error_arrays(batch,scale)
        pure=errors["nominal"]["overall_rmse"]
        residual=errors["residual"]["overall_rmse"]
        metrics[f"{horizon}_steps"]={
            "horizon_s":horizon*.01,
            "n_windows":len(batch.anchors),
            "anchor_times_s":batch.time_s[0].tolist(),
            "pure_normalized_path_rmse":pure,
            "mlp_normalized_path_rmse":residual,
            "improvement_over_pure_pct":
                float(100*(pure-residual)/pure) if pure>1e-12 else None,
            "per_window_mlp_rmse":errors["residual"]["per_window_rmse"].tolist(),
            "per_window_pure_rmse":errors["nominal"]["per_window_rmse"].tolist(),
        }
    return metrics,batches


def _update_stage(
    model, all_q, all_z, dt, target,
    optimizer, *, horizon, updates,
    batch_size, single_batch_size,
    q_scale, y_scale,
    single_weight, terminal_weight,
    seed, params, log_every=10,
):
    """在每个选取的训练起点进行完全可微的 H 步 RK4，自由滚动。

    未来只有等效转角/轮胎 Fx 等 exogenous 数据被读取；
    未来真实 q 仅进入损失函数，绝不进入模型下一步状态。
    """
    n=len(dt)
    if horizon>n or horizon<=0:
        raise ValueError("滚动训练步数超出采集轨迹范围")
    generator=np.random.default_rng(seed)
    device=model.x_mean.device
    dtype=model.x_mean.dtype
    q_t=torch.as_tensor(all_q,dtype=dtype,device=device)
    ext_t=torch.as_tensor(all_z[:,3:],dtype=dtype,device=device)
    z_t=torch.as_tensor(all_z[:-1],dtype=dtype,device=device)
    y_t=torch.as_tensor(target,dtype=dtype,device=device)
    dt_t=torch.as_tensor(dt,dtype=dtype,device=device)
    qs=torch.as_tensor(q_scale,dtype=dtype,device=device)
    ys=torch.as_tensor(y_scale,dtype=dtype,device=device)
    history=[]

    for update in range(1,updates+1):
        starts=generator.integers(
            0,n-horizon+1,size=batch_size
        ).astype(np.int64)
        starts=torch.as_tensor(starts,dtype=torch.long,device=device)
        # 独立的单步监督 mini-batch 保持原预训练能力。
        one_idx=generator.integers(0,n,size=single_batch_size)
        one_idx=torch.as_tensor(one_idx,dtype=torch.long,device=device)

        q=q_t[starts]
        total_path=0.0
        optimizer.zero_grad(set_to_none=True)
        for offset in range(horizon):
            # 对未来真实状态没有任何读取，只使用未来外生序列
            q=hybrid_rk4_step_torch(
                q,ext_t[starts+offset],dt_t[starts+offset],
                residual_model=model,params=params,
                integration="continuous",
            )
            # 未来 CarSim 真值只作为监督目标计算损失
            err=(q-q_t[starts+offset+1])/qs
            total_path=total_path+torch.mean(err*err)
        path_loss=total_path/horizon
        terminal_loss=torch.mean(err*err)
        single_loss=torch.mean(
            ((model(z_t[one_idx])-y_t[one_idx])/ys)**2
        )
        loss=(
            path_loss+terminal_weight*terminal_loss
            +single_weight*single_loss
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"horizon={horizon}, update={update} 出现 NaN/Inf loss；"
                "未进行状态裁剪，需检查模型预测轨迹。"
            )
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        if update==1 or update%log_every==0 or update==updates:
            entry={
                "horizon_steps":horizon,"update":update,
                "total_loss":float(loss.detach()),
                "path_loss":float(path_loss.detach()),
                "terminal_loss":float(terminal_loss.detach()),
                "single_step_loss":float(single_loss.detach()),
                "grad_norm_before_clip":float(grad_norm.detach()),
            }
            history.append(entry)
            print(
                f"[H={horizon:3d} update={update:4d}/{updates}] "
                f"path={entry['path_loss']:.5g} "
                f"terminal={entry['terminal_loss']:.5g} "
                f"single={entry['single_step_loss']:.5g} "
                f"grad={entry['grad_norm_before_clip']:.3g}",
                flush=True,
            )
    return history


def run_finetune(
    data_path,checkpoint_path,output_dir, *,
    mu_hat=0.8,mz_mode="zero",lambda_mode="fixed-half",
    stages=((50,30),(100,30),(150,20),(200,20)),
    batch_size=4,single_batch_size=128,
    lr=1e-4,weight_decay=1e-5,
    single_weight=0.2,terminal_weight=0.1,
    horizons=(300,500),stride=50,seed=42,
    focus_start_s=5.5,focus_horizon_steps=500,
    show_figures=False,backend="TkAgg",
):
    """从已有全时域 MLP 初始化，只修改 NN 权重，然后同窗口对照。"""
    out=Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(
            f"结果目录已存在，禁止覆盖：{out}；请改用新的输出目录"
        )
    stages=tuple((int(h),int(u)) for h,u in stages)
    if not stages or any(
        h<=0 or u<=0 or i>0 and h<=stages[i-1][0]
        for i,(h,u) in enumerate(stages)
    ):
        raise ValueError("训练阶段需按预测长度严格递增且更新次数为正")
    if any(int(x)<=0 for x in horizons) or not horizons:
        raise ValueError("评价时域必须是正采样步数")
    if batch_size<1 or single_batch_size<1 or lr<=0:
        raise ValueError("批量大小和学习率必须正")
    if single_weight<0 or terminal_weight<0:
        raise ValueError("损失权重必须非负")

    # 先验证数据与预训练模型匹配，任何输入错误都不生成目录。
    dlc=load_dlc_si(data_path)
    act=build_dlc_inputs(
        dlc,mu_hat=mu_hat,mz_mode=mz_mode,lambda_mode=lambda_mode,
    )
    z=act.z_si
    q=dlc.q_si
    dt=np.diff(dlc.time_s)
    if len(dt)<max(max(h for h,_ in stages),max(horizons)):
        raise ValueError("DLC 时长不足指定的多步训练/评价窗口")
    reference=NMPCReferenceParams()
    model,source_payload=load_model_checkpoint(checkpoint_path)
    _validate_pretraining_metadata(source_payload,dlc,act)
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    parity=verify_nominal_torch_numpy(z[:-1],dt,reference)
    target=_single_step_target(dlc,z,reference)
    qscale=_state_scale(model)
    yscale=model.y_scale.detach().cpu().numpy().astype(float)
    print("NumPy/Torch 名义 RK4 一致性检查通过:",parity,flush=True)
    print(
        f"数据 {len(q)} 帧、{len(dt)} 区间；"
        f"多步微调阶段={list(stages)}；"
        "仅当前窗口初值使用 CarSim 真实状态；"
        "未来 CarSim 车辆状态只用于训练损失。",
        flush=True,
    )

    before,before_batches=evaluate_same_windows(
        dlc,z,model,qscale,horizons=horizons,
        stride=stride,reference=reference,
    )
    focus_before=_focus_diagnostic(
        dlc,z,model,qscale,
        focus_start_s=focus_start_s,
        horizon_steps=focus_horizon_steps,
        reference=reference,
    )

    optimizer=torch.optim.Adam(
        model.parameters(),lr=lr,weight_decay=weight_decay
    )
    history=[]
    for index,(h,updates) in enumerate(stages):
        print(
            f"\n阶段 {index+1}/{len(stages)}："
            f"{h*.01:g}s 自由滚动微调，{updates} 次更新",
            flush=True,
        )
        logs=_update_stage(
            model,q,z,dt,target,optimizer,
            horizon=h,updates=updates,
            batch_size=batch_size,
            single_batch_size=single_batch_size,
            q_scale=qscale,y_scale=yscale,
            single_weight=single_weight,
            terminal_weight=terminal_weight,
            seed=int(seed)+1000*index,
            params=reference,
            log_every=max(1,updates//5),
        )
        history.extend(logs)

    after,after_batches=evaluate_same_windows(
        dlc,z,model,qscale,horizons=horizons,
        stride=stride,reference=reference,
    )
    focus_after=_focus_diagnostic(
        dlc,z,model,qscale,
        focus_start_s=focus_start_s,
        horizon_steps=focus_horizon_steps,
        reference=reference,
    )

    out.mkdir(parents=True,exist_ok=False)
    ckpt={
        "format":MODEL_FORMAT,
        "input_names":list(INPUT_NAMES),
        "target_names":list(TARGET_NAMES),
        "hidden_sizes":[16,16],
        "model_state_dict":{
            name:val.detach().cpu().clone()
            for name,val in model.state_dict().items()
        },
        "dataset_sha256":dlc.file_sha256,
        "actuation_assumptions":act.assumptions,
        "training_mode":"same_DLC_multistep_continuous_RK4_finetune",
        "pretrained_checkpoint":str(
            Path(checkpoint_path).resolve()
        ),
        "train_and_evaluation_same_data":True,
        "stages":list(stages),
        "seed":int(seed),
    }
    torch.save(ckpt,out/"fine_tuned.pt")

    result={
        "format":REPORT_FORMAT,
        "source_npz":str(dlc.path),
        "source_npz_sha256":dlc.file_sha256,
        "pretrained_checkpoint":str(
            Path(checkpoint_path).resolve()
        ),
        "finetuned_checkpoint":str(out/"fine_tuned.pt"),
        "train_and_evaluation_same_data":True,
        "independent_generalization_test":False,
        "pretrained_single_step_source":True,
        "network_architecture":"8-16-16-3 (unchanged)",
        "nmpc_reference_parameters":asdict(reference),
        "input_assumptions":act.assumptions,
        "oracle_future_measured_steer_and_Fx":True,
        "torch_numpy_consistency":parity,
        "state_scale_si":qscale.tolist(),
        "training":{
            "stages":[list(x) for x in stages],
            "batch_size":batch_size,
            "single_batch_size":single_batch_size,
            "lr":lr,"weight_decay":weight_decay,
            "single_weight":single_weight,
            "terminal_weight":terminal_weight,
            "seed":seed,"history":history,
            "fixed_epoch_and_updates_no_heldout_selection":True,
        },
        "all_windows_before":before,
        "all_windows_after":after,
        "focus_window_before":focus_before,
        "focus_window_after":focus_after,
    }
    _save_json(out/"report.json",result)
    _save_json(out/"history.json",history)

    rows=[]
    for horizon in sorted(set(horizons)):
        key=f"{horizon}_steps"
        a,b=before[key],after[key]
        initial=a["mlp_normalized_path_rmse"]
        final=b["mlp_normalized_path_rmse"]
        gain=float(100*(initial-final)/initial) if initial>1e-12 else None
        rows.append((horizon,a["pure_normalized_path_rmse"],initial,final,gain))
    print("\n=== 同一窗口微调前后（全程自由滚动，无状态重置） ===")
    for h,nom,old,new,gain in rows:
        print(
            f"{h*.01:g}s: pure={nom:.6f}; "
            f"MLP before={old:.6f}; "
            f"MLP after={new:.6f}; "
            f"fine-tune change={gain:+.2f}%",
            flush=True,
        )
    b0=focus_before["residual_nmpc"]["normalized_path_rmse"]
    b1=focus_after["residual_nmpc"]["normalized_path_rmse"]
    print(
        f"重点 {focus_before['start_s']:.2f}~"
        f"{focus_before['end_s']:.2f}s: "
        f"MLP before={b0:.6f}, after={b1:.6f}, "
        f"change={100*(b0-b1)/b0:+.2f}%",
        flush=True,
    )
    print(f"实验报告：{out/'report.json'}")
    if show_figures:
        show_all_windows(
            after_batches,
            dlc_time_s=dlc.time_s,
            dlc_y_m=dlc.y_m,
            scale=qscale,
            backend=backend,block=True,
        )
    return result


def _parse_csv_int(text,label):
    try:
        out=tuple(int(x.strip()) for x in str(text).split(","))
    except ValueError as exc:
        raise ValueError(f"{label} 格式应为逗号分隔整数") from exc
    if not out or any(x<=0 for x in out):
        raise ValueError(f"{label} 必须是正整数列表")
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="从已有 DLC MLP final.pt 做多步自由滚动微调，并比较纯 NMPC/原 MLP/新 MLP"
    )
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--mu-hat",type=float,default=.8)
    parser.add_argument("--mz-mode",default="zero",
                        choices=("zero","force-diff-diagnostic"))
    parser.add_argument("--lambda-mode",default="fixed-half",
                        choices=("fixed-half","ratio-strict"))
    parser.add_argument("--stages",default="50:30,100:30,150:20,200:20",
                        help="预测步数:优化更新次数,逗号分隔")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--single-batch-size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--weight-decay",type=float,default=1e-5)
    parser.add_argument("--single-weight",type=float,default=.2)
    parser.add_argument("--terminal-weight",type=float,default=.1)
    parser.add_argument("--horizons",default="300,500")
    parser.add_argument("--anchor-stride",type=int,default=50)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--focus-start",type=float,default=5.5)
    parser.add_argument("--focus-horizon",type=int,default=500)
    parser.add_argument("--show-figures",action="store_true")
    parser.add_argument("--backend",default="TkAgg",
                        choices=("TkAgg","QtAgg"))
    args=parser.parse_args(argv)
    try:
        stages=parse_stages(args.stages)
        horizons=_parse_csv_int(args.horizons,"--horizons")
    except ValueError as exc:
        parser.error(str(exc))
    run_finetune(
        args.data,args.checkpoint,args.output,
        mu_hat=args.mu_hat,
        mz_mode=args.mz_mode,lambda_mode=args.lambda_mode,
        stages=stages,batch_size=args.batch_size,
        single_batch_size=args.single_batch_size,
        lr=args.lr,weight_decay=args.weight_decay,
        single_weight=args.single_weight,
        terminal_weight=args.terminal_weight,
        horizons=horizons,stride=args.anchor_stride,
        seed=args.seed,focus_start_s=args.focus_start,
        focus_horizon_steps=args.focus_horizon,
        show_figures=args.show_figures,backend=args.backend,
    )


if __name__=="__main__":
    main()
