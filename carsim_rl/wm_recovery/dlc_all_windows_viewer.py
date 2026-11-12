# -*- coding: utf-8 -*-
"""CarSim DLC: 交互浏览所有 3s/5s 预测窗口，不生成 PNG 文件。

两种运行方式：

1) 训练结束时由 dlc_single_scene_experiment.py 自动调用：
   --show-all-windows

2) 不重新训练，直接用已有 final.pt 重新计算所有窗口并弹窗：
   python -m wm_recovery.dlc_all_windows_viewer --data ... --checkpoint ...

浏览方式：
- 3s、5s 各一个 Figure，每个窗口都有 vy、r、vx 三条状态的比较；
- 滑块 / 前一窗口 / 后一窗口 / 键盘左右箭头切换所有预测起点；
- 额外开一张误差热力图，横轴预测时长、纵轴窗口起点；
- 点击热力图中对应行可跳转至该预测窗口；
- 默认不保存 Figure PNG。只读打开 CarSim NPZ 与已训练的 checkpoint。

物理限制：使用 CarSim 未来实测的转向与轮胎力作为已知外生序列，
属于同轨迹 oracle-actuation 离线重放，非可部署闭环 NMPC 验证。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dlc_si import load_dlc_si
from .dlc_actuation import build_dlc_inputs
from .nmpc_reference import NMPCReferenceParams
from .nmpc_residual_model import load_model_checkpoint
from .nmpc_rollout import rk4_free_step


@dataclass
class HorizonWindows:
    horizon_steps: int
    anchors: np.ndarray           # (n_windows,), sample-index start
    time_s: np.ndarray            # (H+1, n_windows)
    true: np.ndarray              # (H+1, n_windows, 3)
    nominal: np.ndarray           # (H+1, n_windows, 3)
    residual: np.ndarray          # (H+1, n_windows, 3)


def _validate_window_batch(batch):
    """检查数据源、时间索引及三种预测的维度和数值。"""
    h = int(batch.horizon_steps)
    a = np.asarray(batch.anchors)
    if h <= 0 or a.ndim != 1 or len(a) < 1:
        raise ValueError("需要至少一个完整的正长度预测窗口")
    expected = (h + 1, len(a), 3)
    for name in ("true", "nominal", "residual"):
        val = np.asarray(getattr(batch, name))
        if val.shape != expected or not np.isfinite(val).all():
            raise ValueError(f"{name} 应为有限的 {expected} 数组")
    if np.asarray(batch.time_s).shape != expected[:2]:
        raise ValueError("time_s 与预测窗口维度不一致")
    if not np.isfinite(batch.time_s).all():
        raise ValueError("窗口时间戳含 NaN/Inf")
    if not np.all(np.diff(batch.time_s, axis=0) > 0):
        raise ValueError("窗口时间戳不是严格递增")
    if not np.all(np.diff(a) > 0) and len(a) > 1:
        raise ValueError("窗口起点应严格递增")


def batches_from_recorded_windows(records_by_horizon):
    """训练过程中已计算的所有窗口直接转换为可视化数组，无需重复积分。

    接受每个 h 的 list[{'anchor': sample_idx,
      'time_s': (H+1,), 'true': (H+1,3),
      'nominal': (H+1,3), 'residual': (H+1,3)}]。
    """
    result = {}
    for h, records in records_by_horizon.items():
        h = int(h)
        if not records:
            continue
        batch = HorizonWindows(
            horizon_steps=h,
            anchors=np.asarray(
                [entry["anchor"] for entry in records], dtype=np.int64
            ),
            time_s=np.stack(
                [np.asarray(entry["time_s"], dtype=float) for entry in records],
                axis=1,
            ),
            true=np.stack(
                [np.asarray(entry["true"], dtype=float) for entry in records],
                axis=1,
            ),
            nominal=np.stack(
                [np.asarray(entry["nominal"], dtype=float) for entry in records],
                axis=1,
            ),
            residual=np.stack(
                [np.asarray(entry["residual"], dtype=float) for entry in records],
                axis=1,
            ),
        )
        _validate_window_batch(batch)
        result[h] = batch
    if not result:
        raise ValueError("未采集到任何可展示的预测窗口")
    return result


def compute_all_window_batches(
    dlc, z, model, *, horizons=(300, 500),
    anchor_stride=50, start_interval=0, stop_interval=None,
    reference=None,
):
    """在同一个完整时段上批量自由滚动，全部 3s/5s 窗口均保留。

    仅起点 q0 取自 CarSim；后续积分只取未来 z[..., 3:] 外生序列，
    不回灌真值 q。对于大量窗口使用 NumPy batch 一起计算，避免
    逐窗口重复遍历 Python 和重复初始化绘图对象。
    """
    if reference is None:
        reference = NMPCReferenceParams()
    z = np.asarray(z, dtype=np.float64)
    q = np.asarray(dlc.q_si, dtype=np.float64)
    t = np.asarray(dlc.time_s, dtype=np.float64)
    dt = np.diff(t)

    if stop_interval is None:
        stop_interval = len(dt)
    start_interval = int(start_interval)
    stop_interval = int(stop_interval)
    anchor_stride = int(anchor_stride)
    if (
        z.shape != (len(t), 8) or q.shape != (len(t), 3)
        or not np.isfinite(z).all()
    ):
        raise ValueError("DLC 状态、时间和八维输入形状错误")
    if not (0 <= start_interval < stop_interval <= len(dt)):
        raise ValueError("所选预测区间必须在 CarSim 轨迹范围内")
    if anchor_stride <= 0:
        raise ValueError("--anchor-stride 必须大于零")

    batches = {}
    for h in sorted(set(int(n) for n in horizons)):
        if h <= 0:
            raise ValueError("预测步数必须是正整数")
        anchors = np.arange(
            start_interval, stop_interval-h+1, anchor_stride, dtype=np.int64,
        )
        if not len(anchors):
            raise ValueError(
                f"预测数据不足 {h} 个 10ms 区间；"
                "请缩短 --horizons 或增大 CarSim 采集时长。"
            )
        n = len(anchors)
        indexes = anchors[None, :] + np.arange(h+1)[:, None]
        truth = q[indexes]
        time_s = t[indexes]
        nom = np.empty((h+1, n, 3), dtype=np.float64)
        res = np.empty_like(nom)
        nom[0] = q[anchors]
        res[0] = q[anchors]

        for step in range(h):
            external = z[anchors+step, 3:]
            interval_dt = dt[anchors+step]
            nom[step+1] = rk4_free_step(
                nom[step], external, dt_s=interval_dt,
                residual_model=None, reference=reference,
            )
            res[step+1] = rk4_free_step(
                res[step], external, dt_s=interval_dt,
                residual_model=model, reference=reference,
            )

        batch = HorizonWindows(
            horizon_steps=h,
            anchors=anchors, time_s=time_s.copy(), true=truth.copy(),
            nominal=nom, residual=res,
        )
        _validate_window_batch(batch)
        batches[h] = batch
    return batches


def _validate_scale(scale):
    s=np.asarray(scale,dtype=np.float64)
    if s.shape != (3,) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("状态 RMSE 归一化尺度必须为 3 个正的有限数")
    return s


def window_error_arrays(batch, scale):
    """同时计算每个窗口和每个预测时刻的归一化状态 RMSE。"""
    _validate_window_batch(batch)
    s=_validate_scale(scale)
    out={}
    for label, prediction in (
        ("nominal",batch.nominal),("residual",batch.residual),
    ):
        err=(prediction-batch.true)/s[None,None,:]
        # (H+1,N,3) -> (N,H+1), 每窗口每步三个状态的 RMS
        matrix=np.sqrt(np.mean(err*err,axis=2)).T
        per_window=np.sqrt(np.mean(err*err,axis=(0,2)))
        overall=float(np.sqrt(np.mean(err*err)))
        out[label]={
            "time_window_matrix":matrix,
            "per_window_rmse":per_window,
            "overall_rmse":overall,
        }
    return out


def _selected_window_title(batch, index, errors):
    start=float(batch.time_s[0,index])
    stop=float(batch.time_s[-1,index])
    nom=float(errors["nominal"]["per_window_rmse"][index])
    corrected=float(errors["residual"]["per_window_rmse"][index])
    gain=(100*(nom-corrected)/nom) if nom > 1e-12 else float("nan")
    return (
        f"{batch.horizon_steps/100:g}s DLC prediction | "
        f"Window {index+1}/{len(batch.anchors)} | "
        f"{start:.2f}-{stop:.2f}s\n"
        f"Normalized RMSE  Pure NMPC: {nom:.4f}  | "
        f"NMPC + MLP: {corrected:.4f}  | "
        f"Improvement: {gain:+.1f}%"
    )


def build_window_figure(batch, dlc_t, dlc_y_m, scale):
    """构建交互式 Figure；在 Agg 测试中也可无 GUI 创建并驱动滑块。

    返回 (fig, controller)，controller 包含 slider、set_index、
    selected_index；需持有引用确保按钮和滑块回调不被回收。
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    _validate_window_batch(batch)
    errors = window_error_arrays(batch,scale)
    dlc_t=np.asarray(dlc_t,dtype=float)
    dlc_y_m=np.asarray(dlc_y_m,dtype=float)
    if (
        dlc_t.ndim != 1 or dlc_t.shape != dlc_y_m.shape
        or not np.isfinite(dlc_t).all() or not np.isfinite(dlc_y_m).all()
    ):
        raise ValueError("CarSim 全局时间及 Y 位置应为同长度有限一维数组")

    fig=plt.figure(figsize=(12.2,9.3))
    grid=fig.add_gridspec(
        4, 1, left=.095, right=.96, top=.885, bottom=.18,
        hspace=.42, height_ratios=(1.0,1.0,1.0,.7),
    )
    axes=[fig.add_subplot(grid[i,0]) for i in range(3)]
    overview=fig.add_subplot(grid[3,0])
    state_labels=("Lateral velocity vy (m/s)",
                  "Yaw rate r (rad/s)",
                  "Longitudinal velocity vx (m/s)")
    model_names=("CarSim truth","Pure NMPC","NMPC + residual MLP")
    styles=("-", "--", "-.")
    data=(batch.true,batch.nominal,batch.residual)
    lines=[]
    index=0
    first_x=batch.time_s[:,0]-batch.time_s[0,0]
    for channel,ax in enumerate(axes):
        state_lines=[]
        for k,arr in enumerate(data):
            (line,)=ax.plot(
                first_x,arr[:,0,channel],
                linestyle=styles[k], linewidth=2.1 if k==0 else 1.5,
                label=model_names[k],
            )
            state_lines.append(line)
        ax.set_ylabel(state_labels[channel])
        ax.set_xlim(0,float(first_x[-1]))
        ax.grid(alpha=.23)
        if channel==0:
            ax.legend(loc="best",ncol=3,fontsize=9)
        lines.append(state_lines)
    axes[-1].set_xlabel("Prediction horizon (s)")

    overview.plot(dlc_t,dlc_y_m,linewidth=1.3,label="CarSim Y")
    overview.set_xlim(float(dlc_t[0]),float(dlc_t[-1]))
    overview.set_xlabel("CarSim absolute time (s)")
    overview.set_ylabel("Lateral Y (m)")
    overview.grid(alpha=.2)
    span=[overview.axvspan(
        batch.time_s[0,0],batch.time_s[-1,0],alpha=.2
    )]

    # 下方的滑块和前后翻页控件；不是每个预测窗口创建一个独立窗口。
    ax_prev=fig.add_axes([.095,.065,.11,.044])
    ax_slider=fig.add_axes([.295,.071,.51,.024])
    ax_next=fig.add_axes([.825,.065,.13,.044])
    previous=Button(ax_prev,"< Previous")
    following=Button(ax_next,"Next >")
    n=len(batch.anchors)
    if n>1:
        slider=Slider(
            ax_slider,"Window",0,n-1,valinit=0,valstep=1,
            valfmt="%0.0f",
        )
    else:
        slider=None
        ax_slider.axis("off")
        ax_slider.text(.2,.3,"Only one complete window",
                       transform=ax_slider.transAxes)

    state={"index":0}
    def set_index(new_index):
        j=int(round(float(new_index)))
        j=max(0,min(n-1,j))
        state["index"]=j
        xx=batch.time_s[:,j]-batch.time_s[0,j]
        for channel,ax in enumerate(axes):
            for k,arr in enumerate(data):
                lines[channel][k].set_data(xx,arr[:,j,channel])
            ax.relim()
            ax.autoscale_view(scalex=False,scaley=True)
            ax.set_xlim(0,float(xx[-1]))
        span[0].remove()
        span[0]=overview.axvspan(
            batch.time_s[0,j],batch.time_s[-1,j],alpha=.2
        )
        fig.suptitle(
            _selected_window_title(batch,j,errors),
            fontsize=11,weight="semibold"
        )
        fig.canvas.draw_idle()

    def previous_click(event):
        if slider is not None:
            slider.set_val(max(0,state["index"]-1))
        else:
            set_index(0)

    def next_click(event):
        if slider is not None:
            slider.set_val(min(n-1,state["index"]+1))
        else:
            set_index(0)

    def key_press(event):
        if event.key in ("right","down"):
            next_click(event)
        elif event.key in ("left","up"):
            previous_click(event)

    def seek(j):
        """供热力图与测试调用，自动同步滑块和图形。"""
        if slider is not None:
            slider.set_val(max(0,min(n-1,int(j))))
        else:
            set_index(0)

    if slider is not None:
        slider.on_changed(set_index)
    previous.on_clicked(previous_click)
    following.on_clicked(next_click)
    fig.canvas.mpl_connect("key_press_event",key_press)
    set_index(0)
    controller={
        "slider":slider,
        "prev_button":previous,
        "next_button":following,
        "set_index":seek,
        "selected_index":lambda:state["index"],
        "state_lines":lines,
        "overview":overview,
        "error_arrays":errors,
        "batch":batch,
    }
    # Matplotlib 回调对象必须持续有引用。
    fig._dlc_controller=controller
    return fig,controller


def build_overview_heatmap(batches,scale,controllers=None):
    """把所有预测起点/预测时刻以热力图同时展示；可点击行跳转。"""
    import matplotlib.pyplot as plt

    horizons=sorted(batches)
    if len(horizons)==0:
        raise ValueError("至少要有一个预测时域")
    nrows=len(horizons)
    fig,axarr=plt.subplots(
        nrows,2,figsize=(12.3, max(4.3,3.25*nrows+1.0)),
        squeeze=False,layout="constrained",
    )
    matrices={}
    for h in horizons:
        matrices[h]=window_error_arrays(batches[h],scale)
    all_values=np.concatenate([
        matrices[h][name]["time_window_matrix"].ravel()
        for h in horizons
        for name in ("nominal","residual")
    ])
    vhigh=float(np.quantile(all_values,.98))
    if vhigh<=0:
        vhigh=1.0

    heatmap_axes={}
    for row,h in enumerate(horizons):
        batch=batches[h]
        xend=float(np.max(batch.time_s[-1]-batch.time_s[0]))
        for col,name in enumerate(("nominal","residual")):
            ax=axarr[row,col]
            matrix=matrices[h][name]["time_window_matrix"]
            artist=ax.imshow(
                matrix,aspect="auto",origin="lower",interpolation="nearest",
                extent=(0,xend,0,len(batch.anchors)),vmin=0,vmax=vhigh,
            )
            ax.set_title(
                f"{h/100:g}s | "
                +("Pure NMPC" if name=="nominal" else "NMPC + residual MLP")
            )
            ax.set_xlabel("Prediction horizon (s)")
            if col==0:
                ax.set_ylabel("Window start (CarSim time s)")
            count=len(batch.anchors)
            tickidx=np.unique(
                np.rint(np.linspace(0,count-1,min(8,count)))
                .astype(np.int64)
            )
            ax.set_yticks(
                tickidx+.5,
                [f"{batch.time_s[0,j]:.1f}" for j in tickidx],
            )
            heatmap_axes[ax]=(h,batch)
    fig.colorbar(
        artist,ax=axarr.ravel().tolist(),
        label="Normalized instantaneous state RMSE",
        shrink=.90,
    )
    fig.suptitle(
        "DLC all-window error overview | Click a row to open that window",
        fontsize=12,
    )

    def on_heatmap_click(event):
        found=heatmap_axes.get(event.inaxes)
        if found is None or event.ydata is None:
            return
        h,batch=found
        idx=int(np.floor(float(event.ydata)))
        idx=max(0,min(len(batch.anchors)-1,idx))
        if controllers is not None and h in controllers:
            controllers[h]["set_index"](idx)

    fig.canvas.mpl_connect("button_press_event",on_heatmap_click)
    return fig


def show_all_windows(
    batches, *, dlc_time_s, dlc_y_m, scale,
    backend="TkAgg", block=True,
):
    """在 Windows PowerShell 中启动本机可交互 Figure 窗口。

    警告：如果此前代码执行了 matplotlib.use('Agg')，
    请勿在同一进程内强行显示；集成训练的 --show-all-windows 会跳过 Agg。
    """
    try:
        import matplotlib
        matplotlib.use(backend,force=True)
        import matplotlib.pyplot as plt
    except (ImportError,RuntimeError) as exc:
        raise RuntimeError(
            f"无法加载 Matplotlib {backend} GUI 后端。"
            "请在本机 PowerShell 下运行，确认 matplotlib 与 tkinter 可导入；"
            "或使用 --backend QtAgg（需安装 PyQt/PySide）。"
        ) from exc

    all_controllers={}
    figures=[]
    try:
        for h in sorted(batches):
            fig,controller=build_window_figure(
                batches[h],dlc_time_s,dlc_y_m,scale
            )
            figures.append(fig)
            all_controllers[h]=controller
        summary_fig=build_overview_heatmap(
            batches,scale,controllers=all_controllers
        )
        figures.append(summary_fig)
        for h in sorted(batches):
            print(
                f"{h/100:g}s 预测：可浏览全部 "
                f"{len(batches[h].anchors)} 个窗口。"
            )
        print(
            "已弹出 Figure：滑块/Previous/Next/方向键切换窗口；"
            "热力图中单击行可跳转；关闭全部窗口后命令结束。",
            flush=True,
        )
        plt.show(block=block)
    except Exception as exc:
        for fig in figures:
            plt.close(fig)
        raise RuntimeError(
            "无法打开交互式窗口。请检查图形桌面与 TkAgg 后端，"
            "若在远程无桌面会话，请在 Windows 本机 PowerShell 运行。"
        ) from exc
    return figures


def _model_scale(model):
    return _validate_scale(
        model.x_scale.detach().cpu().numpy()[:3]
    )


def open_from_checkpoint(
    data_path, checkpoint_path, *, horizons=(300,500),
    anchor_stride=50, mu_hat=0.8,
    mz_mode="zero", lambda_mode="fixed-half",
    backend="TkAgg", display=True,
):
    """读取已有 final.pt，重算所有窗口，不再拟合/训练 MLP。

    对来源文件的 SHA256 和模型中记录的执行器假设做一致性核对。
    """
    dlc=load_dlc_si(data_path)
    model,ckpt=load_model_checkpoint(checkpoint_path)
    recorded_hash=ckpt.get("dataset_sha256")
    if recorded_hash and recorded_hash != dlc.file_sha256:
        raise ValueError(
            "final.pt 中记录的采集文件 SHA256 与当前 DLC NPZ 不一致；"
            "请传入当时用于训练的原始 dlc_exports.npz。"
        )
    assumptions=ckpt.get("actuation_assumptions",{})
    comparisons={
        "mz_mode":mz_mode,
        "lambda_mode":lambda_mode,
    }
    for key,wanted in comparisons.items():
        old=assumptions.get(key)
        if old is not None and old != wanted:
            raise ValueError(
                f"模型训练时 {key}={old}，本次给出 {wanted}，"
                "请保持物理输入假设一致"
            )
    if "mu_hat" in assumptions:
        if not np.isclose(float(assumptions["mu_hat"]),mu_hat,rtol=0,atol=1e-9):
            raise ValueError(
                f"模型训练时 mu_hat={assumptions['mu_hat']}，"
                f"本次 mu_hat={mu_hat}"
            )
    actuation=build_dlc_inputs(
        dlc,mu_hat=mu_hat,mz_mode=mz_mode,lambda_mode=lambda_mode
    )
    horizons=tuple(int(h) for h in horizons)
    print(
        "已加载已训练 final.pt，开始使用同一物理模型重算所有窗口，"
        "此过程不训练网络。",
        flush=True,
    )
    batches=compute_all_window_batches(
        dlc,actuation.z_si,model,
        horizons=horizons,anchor_stride=anchor_stride,
    )
    scale=_model_scale(model)
    for h,batch in batches.items():
        errors=window_error_arrays(batch,scale)
        nom=errors["nominal"]["overall_rmse"]
        corrected=errors["residual"]["overall_rmse"]
        gain=(100*(nom-corrected)/nom) if nom>1e-12 else float("nan")
        print(
            f"H={h/100:g}s，窗口数={len(batch.anchors)}，"
            f"Pure={nom:.7f}，MLP={corrected:.7f}，"
            f"改善率={gain:+.2f}%",
            flush=True,
        )
    if display:
        show_all_windows(
            batches,dlc_time_s=dlc.time_s,dlc_y_m=dlc.y_m,
            scale=scale,backend=backend
        )
    return batches


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="直接加载已有 final.pt，弹窗浏览全部 3s/5s 预测窗口，不重新训练"
    )
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--horizons",default="300,500")
    parser.add_argument("--anchor-stride",type=int,default=50)
    parser.add_argument("--mu-hat",type=float,default=0.8)
    parser.add_argument("--mz-mode",default="zero",choices=(
        "zero","force-diff-diagnostic",
    ))
    parser.add_argument("--lambda-mode",default="fixed-half",choices=(
        "fixed-half","ratio-strict",
    ))
    parser.add_argument("--backend",default="TkAgg",choices=(
        "TkAgg","QtAgg",
    ),help="Windows 上推荐 TkAgg")
    args=parser.parse_args(argv)
    try:
        horizons=tuple(int(x.strip()) for x in args.horizons.split(","))
    except ValueError as exc:
        parser.error(f"--horizons 格式应为 300,500：{exc}")
    open_from_checkpoint(
        args.data,args.checkpoint,horizons=horizons,
        anchor_stride=args.anchor_stride,mu_hat=args.mu_hat,
        mz_mode=args.mz_mode,lambda_mode=args.lambda_mode,
        backend=args.backend,display=True,
    )


if __name__=="__main__":
    main()
