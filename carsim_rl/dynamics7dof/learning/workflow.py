"""Independent episode evaluation and chronological shadow replay reports."""

from html import escape
import json
from pathlib import Path

import numpy as np

from ..comparison import LABELS, display_states, error_metrics, replay_states, svg_panels
from ..model import LoadMemory, STATE_NAMES
from ..physics_check import load_capture, sha
from .data import controls_at, prepare_capture
from .dynamics import Learned7DOF
from .ensemble import ResidualEnsemble, score_model
from .online import OnlineShadow


def source_role(model, info):
    for episode in model.metadata.get('episodes', []):
        if episode['source_sha256']['signals_si.csv'] == info['source_sha256']['signals_si.csv']:
            return 'TRAINING_EPISODE' if episode['split'] == 'train' else 'HELD_OUT_EPISODE'
    return 'NEW_EPISODE_UNREVIEWED'


def evaluate_checkpoint(checkpoint, capture, output):
    out, capture = Path(output).resolve(), Path(capture).resolve()
    if out.exists() or out == capture or capture in out.parents:
        raise ValueError('Evaluation output must be new and outside capture')
    samples, info = prepare_capture(capture)
    model, _ = ResidualEnsemble.load(checkpoint)
    derivative_scores = score_model(model, samples['features'], samples['targets'])
    meta, data = load_capture(capture)
    baseline = replay_states(data, meta['scenario_mu_declared'])
    t, truth = baseline['time'], baseline['truth']
    augmented = np.full_like(truth, np.nan)
    augmented[0] = truth[0]
    memory, dynamics = LoadMemory(0, 0), Learned7DOF(model)
    uncertainties, fallbacks = np.full(len(t), np.nan), np.full(len(t), np.nan)
    stop = None
    for i, dt in enumerate(np.diff(t)):
        if data['return_code'][i+1] != 0 or data['solver_done'][i+1]:
            stop = {'index': i+1, 'reason': 'solver_terminal_or_error'}
            break
        try:
            p = dynamics.predict(augmented[i], controls_at(data, i), mu=meta['scenario_mu_declared'],
                                 load_memory=memory, horizon_s=float(dt),
                                 substeps=max(1, int(np.ceil(dt/.0005-1e-8))))
            if p.flags or p.state[0] < 1:
                stop = {'index': i+1, 'reason': 'domain_or_protection', 'flags': list(p.flags)}
                break
            augmented[i+1], memory = p.state, p.load_memory
            uncertainties[i+1], fallbacks[i+1] = p.uncertainty_mean, p.fallback_fraction
        except (ValueError, FloatingPointError, OverflowError) as exc:
            stop = {'index': i+1, 'reason': str(exc)}
            break
    elapsed = t-t[0]
    eligible = (elapsed >= .1) & (data['return_code'] == 0) & (data['solver_done'] == 0)
    common = eligible & np.all(np.isfinite(augmented), axis=1) & np.all(np.isfinite(baseline['predicted']), axis=1)
    role = source_role(model, info)
    report = {'capture': str(capture), 'checkpoint': str(Path(checkpoint).resolve()),
              'checkpoint_sha256': sha(checkpoint), 'source': info, 'role': role,
              'derivative_diagnostic': derivative_scores, 'augmented_stop': stop,
              'baseline_stop': baseline['rollout_stop'],
              'common_coverage_fraction': float(common.sum()/max(1, eligible.sum())),
              'continuous_rollout_common_samples': int(common.sum()),
              'baseline_rmse': {}, 'augmented_rmse': {}, 'automatic_deployment': False,
              'notes': ['Both continuous curves initialize only once; recorded actual future inputs are replayed.',
                        'Derivative diagnostics and long rollout are different tests.',
                        'Ensemble disagreement/OOD gate is heuristic, not a stability or safety proof.']}
    for i, k in enumerate(STATE_NAMES):
        report['baseline_rmse'][k] = error_metrics(truth[common, i], baseline['predicted'][common, i])
        report['augmented_rmse'][k] = error_metrics(truth[common, i], augmented[common, i])
    a, b, c = map(display_states, (truth, baseline['predicted'], augmented))
    panels = [(label, [('CarSim', a[:, i]), ('7DOF', b[:, i]), ('7DOF + residual', c[:, i])]) for i, label in enumerate(LABELS)]
    gate_panels = [('Normalized ensemble disagreement (NOT Risk)', [('U_f', uncertainties)]),
                   ('Fallback fraction per integration interval', [('physics fallback', fallbacks)])]
    for name, digest in info['source_sha256'].items():
        if sha(capture/name) != digest:
            raise RuntimeError('Capture changed during evaluation')
    out.mkdir(parents=True, exist_ok=False)
    (out/'states.svg').write_text(svg_panels(elapsed, panels, 'CarSim / nominal / learned residual', role), encoding='utf-8')
    (out/'uncertainty.svg').write_text(svg_panels(elapsed, gate_panels, 'Uncertainty proxy and fallback', role), encoding='utf-8')
    (out/'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(out/'rollout.npz', time=t, truth=truth, baseline=baseline['predicted'],
                        augmented=augmented, uncertainty=uncertainties, fallback=fallbacks)
    rows = []
    for k in STATE_NAMES:
        n, r = report['baseline_rmse'][k]['rmse'], report['augmented_rmse'][k]['rmse']
        rows.append(f'<tr><td>{k}</td><td>{n if n is not None else "NA"}</td><td>{r if r is not None else "NA"}</td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>7DOF 学习补偿</title>
<style>body{{font:16px/1.6 sans-serif;max-width:1240px;margin:30px auto;padding:20px;background:#f4f7fb;color:#1d2b3c}}img{{width:100%}}td,th{{padding:8px;border:1px solid #bbb}}table{{border-collapse:collapse}}.note{{background:#fff4dc;padding:15px}}</style>
<h1>7DOF 学习补偿对比</h1><p>{escape(str(capture))}</p><p class="note">数据角色：{role}。
蓝色为 CarSim，橙色为原模型，绿色为学习补偿。两种模型都只初始化一次，不重置到真值。
共同有效覆盖率：{report['common_coverage_fraction']:.1%}。训练回合改善不等于泛化；只有完整留出回合才用于独立验证。
未接管控制、未生成 Risk 标签；高分歧/超范围回退只是工程防护，不是安全证明。</p>
<img src="states.svg" alt="三条状态曲线"><h2>连续预测 RMSE（SI）</h2>
<table><tr><th>状态</th><th>原 7DOF</th><th>学习补偿</th></tr>{''.join(rows)}</table>
<h2>认知不确定性代理与回退</h2><img src="uncertainty.svg" alt="分歧与回退">
<p>完整指标见 <a href="evaluation.json">evaluation.json</a>。低网络分歧不保证低真实误差，尚未完成不确定性校准。</p></html>'''
    (out/'comparison.html').write_text(html, encoding='utf-8')
    print('Learned comparison:', out/'comparison.html', '|', role, flush=True)
    return report


def shadow_replay(checkpoint, capture, output, *, adapt=False, update_every=200):
    capture, output = Path(capture).resolve(), Path(output).resolve()
    if output.exists() or capture == output or capture in output.parents:
        raise ValueError('Shadow output must be new and outside source')
    _, info = prepare_capture(capture)
    shadow = OnlineShadow(checkpoint, adapt=adapt, update_every=update_every)
    role = source_role(shadow.model, info)
    if role == 'TRAINING_EPISODE':
        raise ValueError('Chronological validation cannot replay an episode already used to train the initial weights')
    meta, data = load_capture(capture)
    x = np.column_stack([data[k] for k in STATE_NAMES])
    for i, time in enumerate(data['solver_time_s']):
        shadow.observe(time, x[i], controls_at(data, i), meta['scenario_mu_declared'],
                       valid=data['return_code'][i] == 0 and not data['solver_done'][i])
        if i and i % 1000 == 0:
            print(f'Shadow replay: t={time:.3f}, labels={shadow.labels}, version={shadow.version}', flush=True)
    for name, digest in info['source_sha256'].items():
        if sha(capture/name) != digest:
            raise RuntimeError('Capture changed during shadow replay')
    report = shadow.save(output, {'capture': str(capture), 'role': role, 'replayed_not_live_CarSim': True})
    print('Shadow report:', output/'online_report.json', flush=True)
    return report
