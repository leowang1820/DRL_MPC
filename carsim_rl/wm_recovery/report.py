"""Self-contained local HTML/SVG evaluation report; no external scripts or fonts."""

from html import escape
from pathlib import Path
import re

import numpy as np

from dynamics7dof.comparison import LABELS, display_states, svg_panels
from dynamics7dof.model import STATE_NAMES

from .confidence import rolling_mean


def precise_panels(time, panels, title, subtitle):
    """Reuse the existing SVG layout, with enough tick precision for small changes."""
    svg = svg_panels(time, panels, title, subtitle)
    ticks = []
    for _, curves in panels:
        finite = np.concatenate([np.asarray(y)[np.isfinite(y)] for _, y in curves])
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        pad = max(abs(hi) * .05, .01) if hi - lo < 1e-8 else .09 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ticks.extend(f"{lo + tick / 4 * (hi - lo):.6g}" for tick in range(5))
    values = iter(ticks)
    return re.sub(r'(<text\b[^>]*text-anchor="end"[^>]*>)[^<]*(</text>)',
                  lambda match: match[1] + next(values) + match[2], svg)


def write_report(output, report, metrics, continuous, residual_scale, calibration):
    root = Path(output)
    synthetic = report["synthetic_demo_only"]
    label = "SYNTHETIC PIPELINE CHECK — NOT CARSIM VALIDATION" if synthetic else "Independent CarSim replay"
    time = continuous["continuous_time_s"]
    arrays = [display_states(continuous[key]) for key in ("continuous_truth", "continuous_nominal", "continuous_mean")]
    panels = [(name, [(legend, value[:, i]) for legend, value in zip(("Observed", "Nominal 7DOF", "World Model"), arrays)])
              for i, name in enumerate(LABELS)]
    (root / "continuous.svg").write_text(precise_panels(time - time[0], panels, "Continuous state prediction", label), encoding="utf-8")
    error = np.linalg.norm(metrics["error"] / residual_scale, axis=1)
    width = calibration["settings"]["window_steps"] if calibration else 1
    u, e = rolling_mean(metrics["uncertainty"], width), rolling_mean(error, width)
    u_curves, e_curves = [("Ensemble variance", u)], [("Observed residual", e)]
    if calibration:
        u_curves.append(("Validation threshold", np.full_like(u, calibration["uncertainty_threshold"])))
        e_curves.append(("Validation threshold", np.full_like(e, calibration["residual_threshold"])))
    (root / "confidence.svg").write_text(precise_panels(metrics["time_s"] - time[0],
        [("Windowed normalized ensemble variance", u_curves), ("Windowed normalized prediction residual", e_curves)],
        "Confidence diagnostics", label), encoding="utf-8")
    one = report["one_step"]
    rows = []
    for state in STATE_NAMES:
        vals = [one[key]["rmse"][state] if one[key]["rmse"] else None
                for key in ("nominal_common_mask", "world_model", "persistence_common_mask")]
        rows.append("<tr><td>" + escape(state) + "</td>" + "".join(f"<td>{value:.6g}</td>" if value is not None else "<td>—</td>" for value in vals) + "</tr>")
    horizon = report["rolling_horizon"]
    horizon_note = f"{horizon['windows']} 个 {horizon['horizon_s']:g} s 滚动窗口"
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Model 离线验证</title><style>body{{font:16px/1.6 'Segoe UI','Microsoft YaHei',sans-serif;max-width:1240px;margin:32px auto;padding:0 24px;color:#203247;background:#f4f7fb}}h1{{font-size:28px}}.notice{{background:#fff3cf;padding:16px;border-left:5px solid #ad7200}}img{{width:100%;height:auto}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}section{{margin:28px 0}}a{{color:#1767a6}}</style>
<h1>World Model 离线验证</h1><p class="notice">{escape(label)}<br>{'本报告使用合成轨迹，只验证代码流程；不能据此判断真实车辆精度或未知附着识别能力。' if synthetic else '使用独立测试轨迹；误差与门控结论限于本次数据。'}</p>
<p>轨迹：{escape(report['source']['trajectory_id'])} · 名义 μ=0.8 · {escape(horizon_note)}。完整指标见 <a href="evaluation.json">evaluation.json</a>。</p>
<section><h2>连续预测</h2><p>从历史窗口结束时的共同状态出发，此后不重置状态。使用记录的实际输入，超出模型适用域时曲线停止。</p><img src="continuous.svg" alt="七维状态与侧偏角的连续预测对比"></section>
<section><h2>模型分歧与真实预测残差</h2><p>残差在下一观测到达后计算；阈值来自独立名义验证集。</p><img src="confidence.svg" alt="模型分歧及预测残差与阈值对比"></section>
<section><h2>单步 RMSE（SI 单位、相同有效样本）</h2><p>有效覆盖率：{one['world_model']['coverage']:.1%}</p><table><tr><th>状态</th><th>原 7DOF</th><th>World Model</th><th>保持当前状态</th></tr>{''.join(rows)}</table></section>
<p>五成员残差 MLP；历史输入不包含真实路面附着。当前未生成 SE 恢复标签、训练 Risk Network 或接管 NMPC。</p></html>"""
    (root / "report.html").write_text(html, encoding="utf-8")
