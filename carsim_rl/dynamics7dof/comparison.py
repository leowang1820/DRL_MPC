"""Measured-input open-loop 7DOF replay and dependency-free SVG comparison.

The main rollout resets ONCE to the initial CarSim state. Future actual wheel
inputs are read from the recorded experiment; this is model validation, not a
claim that an online controller knows the future actuator response.
"""

import csv
from dataclasses import asdict
from html import escape
import json
from pathlib import Path

import numpy as np

from .audit import audit_capture
from .model import Controls, LoadMemory, Reference7DOF, STATE_NAMES, VehicleParameters
from .physics_check import load_capture, sha


WHEELS = ('fl', 'fr', 'rl', 'rr')
LABELS = ('Longitudinal velocity vx [m/s]', 'Lateral velocity vy [m/s]',
          'Yaw rate r [deg/s]', 'Sideslip beta [deg]',
          'Wheel FL [rad/s]', 'Wheel FR [rad/s]', 'Wheel RL [rad/s]', 'Wheel RR [rad/s]')
COLORS = ('#1767a6', '#e16c24', '#239178', '#9451a3')


def error_metrics(truth, predicted):
    a, b = np.asarray(truth), np.asarray(predicted)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return {'samples': 0, 'rmse': None, 'mae': None, 'max_abs': None,
                'bias': None, 'nrmse_by_truth_std': None}
    error = b[valid]-a[valid]
    scale = float(np.std(a[valid]))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {'samples': int(valid.sum()), 'rmse': rmse, 'mae': float(np.mean(np.abs(error))),
            'max_abs': float(np.max(np.abs(error))), 'bias': float(np.mean(error)),
            'nrmse_by_truth_std': rmse/scale if scale > 1e-6 else None}


def display_states(x):
    """Plot angles in degrees; computations and exported state columns stay SI."""
    return np.column_stack((x[:, 0], x[:, 1], np.degrees(x[:, 2]),
                            np.degrees(np.arctan2(x[:, 1], x[:, 0])), x[:, 3:]))


def replay_states(data, mu, vehicle=None, *, substep_s=.0005, allow_nominal=False):
    """Return a free rollout plus separate teacher-forced ONE-step diagnostics.

    Actual wheel angle and drive+signed-brake torque at snapshot i are held on
    [t_i, t_i+1). No estimated state or memory is reset within the free rollout.
    The first protected/clamped prediction ends that rollout; missing tail is
    explicit rather than silently scoring only an apparently successful trace.
    """
    actual = data.get('solver_time_s')
    if actual is not None and np.all(np.isfinite(actual)):
        t = actual.copy()
        basis = 'observed_solver_T'
    elif allow_nominal and (actual is None or not np.any(np.isfinite(actual))):
        t = data['wrapper_time_s'].copy()
        basis = 'NOMINAL_TIME_UNVERIFIED'
    else:
        raise ValueError('Complete solver_time_s required; legacy replay needs explicit --allow-nominal-time')
    if len(t) < 3 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('At least three increasing finite timestamps are required')
    if not np.isfinite(substep_s) or not 0 < substep_s <= .001:
        raise ValueError('Model substep must be in (0, 0.001] seconds')
    dt = np.diff(t)
    # Large gaps must not be treated as a held control interval.
    if np.max(dt) > 1.01*np.median(dt) or np.min(dt) < .49*np.median(dt):
        raise ValueError('Irregular/gapped capture: no interpolation over missing states')
    truth = np.column_stack([data[k] for k in STATE_NAMES])
    if not np.all(np.isfinite(truth)) or truth[0, 0] < 1:
        raise ValueError('Finite states and forward initial speed >= 1 m/s required')
    steer = np.column_stack([data[f'front_steer_{w}_rad'] for w in ('fl', 'fr')])
    torque = np.column_stack([data[f'drive_{w}_nm']+data[f'brake_{w}_nm'] for w in WHEELS])
    commands = [Controls(d, q) for d, q in zip(steer, torque)]
    model = Reference7DOF(vehicle=vehicle or VehicleParameters())
    predicted = np.full_like(truth, np.nan)
    one_step = np.full_like(truth, np.nan)
    predicted[0] = truth[0]
    memory = LoadMemory(0, 0)
    shadow_memory = LoadMemory(0, 0)
    stopped, shadow_stopped = None, None
    for i, step in enumerate(dt):
        if data['return_code'][i+1] != 0 or data['solver_done'][i+1]:
            stopped = stopped or {'index': i+1, 'reason': 'solver_terminal_or_error'}
            break
        substeps = max(1, int(np.ceil(step/substep_s-1e-8)))
        if stopped is None:
            try:
                p = model.predict(predicted[i], commands[i], mu=mu, load_memory=memory,
                                  horizon_s=float(step), substeps=substeps)
                if p.flags or p.state[0] < 1:
                    stopped = {'index': i+1, 'reason': 'model_outside_domain', 'flags': list(p.flags)}
                else:
                    predicted[i+1], memory = p.state, p.load_memory
            except (ValueError, FloatingPointError, OverflowError) as exc:
                stopped = {'index': i+1, 'reason': str(exc)}
        if shadow_stopped is None:
            try:
                p = model.predict(truth[i], commands[i], mu=mu, load_memory=shadow_memory,
                                  horizon_s=float(step), substeps=substeps)
                if p.flags or truth[i, 0] < 1:
                    shadow_stopped = {'index': i+1, 'reason': 'model_outside_domain', 'flags': list(p.flags)}
                else:
                    one_step[i+1], shadow_memory = p.state, p.load_memory
            except (ValueError, FloatingPointError, OverflowError) as exc:
                shadow_stopped = {'index': i+1, 'reason': str(exc)}
    return {'time': t, 'truth': truth, 'predicted': predicted, 'one_step': one_step,
            'time_basis': basis, 'steer': steer, 'torque': torque,
            'rollout_stop': stopped, 'one_step_stop': shadow_stopped, 'vehicle': asdict(model.vehicle)}


def _polyline_segments(t, y, limit=750):
    """Retain per-bucket extrema; do not bridge NaN gaps in partial rollouts."""
    valid = np.flatnonzero(np.isfinite(t) & np.isfinite(y))
    if not len(valid):
        return []
    groups = np.split(valid, np.flatnonzero(np.diff(valid) > 1)+1)
    result = []
    for group in groups:
        if len(group) > 2*limit:
            selected = [group[0], group[-1]]
            for bucket in np.array_split(group, limit):
                selected.extend((bucket[np.argmin(y[bucket])], bucket[np.argmax(y[bucket])]))
            group = np.unique(selected)
        result.append(group)
    return result


def svg_panels(t, panels, title, subtitle):
    """Small plotting backend: standalone SVG; no matplotlib installation needed.

    panels = [(label, [(legend, values), ...]), ...]. Exact metrics use full data.
    """
    width, height = 1220, 125+300*int(np.ceil(len(panels)/2))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f4f7fb"/>',
             '<g font-family="Segoe UI,Arial,sans-serif" fill="#1d2b3c">',
             f'<text x="28" y="36" font-size="23" font-weight="600">{escape(title)}</text>',
             f'<text x="28" y="63" font-size="13">{escape(subtitle)}</text>']
    t = np.asarray(t)
    xmin, xmax = float(t[0]), float(t[-1])
    for index, (label, curves) in enumerate(panels):
        left, top = 22+(index % 2)*600, 90+(index//2)*300
        px, py, pw, ph = left+80, top+59, 484, 189
        finite = np.concatenate([np.asarray(y)[np.isfinite(y)] for _, y in curves])
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if hi-lo < 1e-8:
            pad = max(abs(hi)*.05, .01)
        else:
            pad = .09*(hi-lo)
        lo, hi = lo-pad, hi+pad
        parts += [f'<rect x="{left}" y="{top}" width="582" height="284" rx="9" fill="white" stroke="#dce4ee"/>',
                  f'<text x="{left+15}" y="{top+25}" font-size="15" font-weight="600">{escape(label)}</text>']
        for n, (legend, _) in enumerate(curves):
            lx = left+15+n*141
            parts += [f'<line x1="{lx}" y1="{top+42}" x2="{lx+20}" y2="{top+42}" stroke="{COLORS[n%4]}" stroke-width="2"/>',
                      f'<text x="{lx+25}" y="{top+46}" font-size="11">{escape(legend)}</text>']
        for tick in range(5):
            fraction = tick/4
            xx, yy = px+fraction*pw, py+ph-fraction*ph
            parts += [f'<path d="M {xx:.2f} {py} V {py+ph} M {px} {yy:.2f} H {px+pw}" stroke="#e8edf3" fill="none"/>',
                      f'<text x="{xx:.2f}" y="{py+ph+17}" text-anchor="middle" font-size="11">{xmin+fraction*(xmax-xmin):.3g}</text>',
                      f'<text x="{px-7}" y="{yy+4:.2f}" text-anchor="end" font-size="11">{lo+fraction*(hi-lo):.3g}</text>']
        for n, (_, y) in enumerate(curves):
            y = np.asarray(y)
            for segment in _polyline_segments(t, y):
                points = ' '.join(f'{px+(t[i]-xmin)/(xmax-xmin)*pw:.2f},{py+ph-(y[i]-lo)/(hi-lo)*ph:.2f}' for i in segment)
                parts.append(f'<polyline points="{points}" stroke="{COLORS[n%4]}" fill="none" stroke-width="1.7"/>')
        parts.append(f'<text x="{px+pw/2}" y="{top+280}" text-anchor="middle" font-size="11">elapsed time [s]</text>')
    parts.append('</g></svg>')
    return '\n'.join(parts)


def write_comparison(capture, output, *, substep_s=.0005, warmup_s=.1,
                     allow_nominal=False, vehicle_json=None):
    capture, output = Path(capture).resolve(), Path(output).resolve()
    if output.exists() or output == capture or capture in output.parents:
        raise ValueError('Comparison output must be a NEW directory outside the capture')
    if not np.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError('Metric warmup must be nonnegative')
    audit = audit_capture(capture)
    if audit['integrity'] != 'PASS':
        raise ValueError(f'Capture audit failed: {audit["errors"]}')
    meta, data = load_capture(capture)
    vehicle = VehicleParameters()
    if vehicle_json:
        values = json.loads(Path(vehicle_json).read_text(encoding='utf-8-sig'))
        if set(values) != set(asdict(vehicle)):
            raise ValueError('Vehicle JSON must specify every VehicleParameters field, with no extras')
        vehicle = VehicleParameters(**values)
    result = replay_states(data, float(meta['scenario_mu_declared']), vehicle,
                           substep_s=substep_s, allow_nominal=allow_nominal)
    t = result['time']-result['time'][0]
    observed, estimated = display_states(result['truth']), display_states(result['predicted'])
    score = (t >= warmup_s) & (data['return_code'] == 0) & (data['solver_done'] == 0)
    if not np.any(score):
        raise ValueError('No samples after the metric warmup')
    pairs = [(label, [('CarSim', observed[:, i]), ('7DOF free rollout', estimated[:, i])])
             for i, label in enumerate(LABELS)]
    errors = [(label, [('7DOF - CarSim', estimated[:, i]-observed[:, i])]) for i, label in enumerate(LABELS)]
    inputs = [
        ('Steering-wheel command [deg]', [('command', data['steering_wheel_command_deg'])]),
        ('Actual front-wheel angles [deg]', [('front left', np.degrees(result['steer'][:, 0])),
                                            ('front right', np.degrees(result['steer'][:, 1]))]),
        ('Throttle command [0..1]', [('command', data['throttle_command'])]),
        ('Brake pressure command [MPa]', [('command', data['brake_pressure_command_mpa'])]),
        ('Front actuator torque: drive + brake [Nm]', [('FL', result['torque'][:, 0]), ('FR', result['torque'][:, 1])]),
        ('Rear actuator torque: drive + brake [Nm]', [('RL', result['torque'][:, 2]), ('RR', result['torque'][:, 3])]),
    ]
    schedule = meta.get('excitation')
    if schedule:
        from .open_loop import Excitation
        signal = Excitation(**schedule)
        inputs.append(('Requested nominal front angle [deg]', [('nominal request', np.array([signal.front_angle(v) for v in t]))]))
    headline = f'{capture.name} | {result["time_basis"]} | init once; measured actual input, left hold'
    metrics = {name: error_metrics(result['truth'][score, i], result['predicted'][score, i])
               for i, name in enumerate(STATE_NAMES)}
    one = {name: error_metrics(result['truth'][score, i], result['one_step'][score, i])
           for i, name in enumerate(STATE_NAMES)}
    baseline = np.vstack([np.full((1, 7), np.nan), result['truth'][:-1]])
    persistence = {name: error_metrics(result['truth'][score, i], baseline[score, i])
                   for i, name in enumerate(STATE_NAMES)}
    coverage = np.all(np.isfinite(result['predicted']), axis=1)
    report = {'format': 'open_loop_comparison_v1', 'capture': str(capture),
              'capture_status': meta['status'], 'final_info': meta.get('final_info'),
              'source_sha256': audit['source_sha256'], 'time_basis': result['time_basis'],
              'vehicle': result['vehicle'], 'mu': meta['scenario_mu_declared'],
              'substep_s': substep_s, 'metric_warmup_s': warmup_s,
              'initial_state_si': result['truth'][0].tolist(), 'initial_memory': [0., 0.],
              'rollout_stop': result['rollout_stop'], 'one_step_stop': result['one_step_stop'],
              'rollout_scored_samples': int(np.sum(score & coverage)), 'eligible_samples': int(score.sum()),
              'rollout_coverage_fraction': float(np.sum(score & coverage)/score.sum()),
              'last_valid_rollout_elapsed_s': float(t[coverage][-1]),
              'continuous_rollout_metrics_SI': metrics, 'teacher_forced_one_step_metrics_SI': one,
              'one_step_persistence_metrics_SI': persistence,
              'ready_for_residual_training': False,
              'notes': ['Main curves initialize from CarSim only once; load memory is carried, never reset.',
                        'One-step diagnostic DOES reset state to measurement; do not interpret it as a free rollout.',
                        'Input is actual wheel angle and drive+signed brake torque; steering ratio is not an actuator model.',
                        'Recorded u_i held over [t_i,t_i+1); actuator discontinuities retain sampling uncertainty.',
                        'Scores use SI; plotted yaw rate and beta use degrees. Near-zero variance => NRMSE null.',
                        'No learned correction; original reference model and fit unchanged.']}
    for name, digest in audit['source_sha256'].items():
        if sha(capture/name) != digest:
            raise RuntimeError('Capture changed during replay')
    output.mkdir(parents=True, exist_ok=False)
    for name, panels, title in [('states', pairs, 'CarSim vs 7DOF | Continuous state rollout'),
                                 ('errors', errors, 'Continuous rollout errors'),
                                 ('inputs', inputs, 'Commands and measured actuator inputs')]:
        (output/f'{name}.svg').write_text(svg_panels(t, panels, title, headline), encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'comparison.csv').open('x', newline='', encoding='utf-8-sig') as f:
        fields = ['elapsed_s', 'rollout_valid']+[f'{prefix}_{k}' for prefix in ('carsim', 'rollout', 'one_step') for k in STATE_NAMES]
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(len(t)):
            vals = [t[i], int(coverage[i]), *result['truth'][i], *result['predicted'][i], *result['one_step'][i]]
            writer.writerow([v if np.isfinite(v) else '' for v in vals])
    table = '\n'.join(f'<tr><td>{escape(k)}</td><td>{v["rmse"]:.6g}</td><td>{v["mae"]:.6g}</td><td>{v["max_abs"]:.6g}</td></tr>'
                      if v['samples'] else f'<tr><td>{escape(k)}</td><td colspan="3">no valid samples</td></tr>'
                      for k, v in metrics.items())
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CarSim / 7DOF 对比</title>
<style>body{{font:16px/1.65 "Segoe UI",sans-serif;background:#f4f7fb;color:#1d2b3c;max-width:1240px;margin:30px auto;padding:0 20px}}h1,h2{{line-height:1.3}}img{{width:100%;height:auto}}table{{border-collapse:collapse;background:white}}td,th{{padding:9px 18px;border:1px solid #dce4ee}}code{{overflow-wrap:anywhere}}.note{{padding:16px;background:#fff4dc;border-left:4px solid #e9ac39}}</style>
<h1>CarSim 与 7DOF：开环动力学对比</h1><p><code>{escape(str(capture))}</code></p>
<div class="note">时间依据：{escape(result['time_basis'])}。原模型参数不变，未训练网络。
主曲线只在初始时刻对齐状态，之后连续积分；使用记录的实际前轮角和轮端扭矩，不是油门/方向盘指令直接代入。
这是实测输入条件下的动力学验证，不验证执行器模型。<br>
采集状态：{escape(meta['status'])}；连续预测覆盖率：{report['rollout_coverage_fraction']:.1%}；停止信息：{escape(str(result['rollout_stop']))}。
比较不足完整时长时不能按局部误差宣称通过。旧 DLC 数据重放不是新正弦试验。</div>
<h2>1. 状态对比</h2><p>蓝色为 CarSim；橙色为从同一初始状态连续积分的 7DOF。角速度图用 deg/s，CSV 与指标保持 SI。</p><img src="states.svg" alt="状态对比">
<h2>2. 连续预测误差</h2><img src="errors.svg" alt="误差曲线">
<h2>3. 输入检查</h2><p>先确认实际前轮角存在期望激励、实际制动扭矩为负，再解释状态误差。</p><img src="inputs.svg" alt="指令及实际执行器输入">
<h2>4. 指标</h2><p>排除前 {warmup_s:g} s 和终止行。RMSE、MAE、最大绝对误差均使用各状态 SI 单位；零方差附近不计算百分比精度。</p>
<table><tr><th>状态</th><th>RMSE</th><th>MAE</th><th>最大绝对误差</th></tr>{table}</table>
<p>完整指标、独立的一步预测诊断及保持当前值基准见 <a href="metrics.json">metrics.json</a>；逐点数据见 <a href="comparison.csv">comparison.csv</a>。</p>
<p>判断顺序：实际输入是否正确 → 初值/单位/参数与时间 → 符号、幅值、相位 → 长时漂移。
当前没有全工况“通过阈值”，一次曲线吻合不能代替独立工况验证。</p></html>'''
    (output/'comparison.html').write_text(html, encoding='utf-8')
    print('Comparison:', output/'comparison.html', flush=True)
    print('Rollout coverage:', f'{report["rollout_coverage_fraction"]:.1%}', '| stop:', result['rollout_stop'], flush=True)
    return report
