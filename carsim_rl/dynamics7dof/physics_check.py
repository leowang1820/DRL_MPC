"""Offline engineering validation: observed timing, signed wheel torque, memory.

No training, no CarSim DLL, no modification of the reference model or captures.
Optional VS conversion operates on NEW local copies using the installed official
ERDConverter (-b <vs> -a <all.par> -c; CarSim 2022.1 manual pp11-12).
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .audit import audit_capture
from .model import Controls, LoadMemory, Reference7DOF, STATE_NAMES, VehicleParameters


WHEELS = ('fl', 'fr', 'rl', 'rr')
VS_WHEELS = ('L1', 'R1', 'L2', 'R2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_capture(path):
    root = Path(path).resolve()
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    with (root/'signals_si.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Capture contains no samples')
    excluded = {'phase', 'quality_flags', 'solver_time_source'}
    data = {k: np.array([float(row[k]) if row[k] else np.nan for row in rows])
            for k in rows[0] if k not in excluded}
    return meta, data


def convert_results(vs_path, output, converter):
    """Snapshot before converting; never let the converter overwrite CarSim files."""
    source = Path(vs_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    required = [source, source.with_suffix('.vsb'), source.with_name(source.stem+'_all.par'),
                source.with_name(source.stem+'_echo.par'), source.with_name(source.stem+'_log.txt')]
    manifest = {}
    for p in required:
        payload = p.read_bytes()
        (output/p.name).write_bytes(payload)
        manifest[p.name] = {'source': str(p), 'sha256': hashlib.sha256(payload).hexdigest()}
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    args = [str(converter), '-b', str(output/source.name), '-a',
            str(output/(source.stem+'_all.par')), '-c']
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        result = subprocess.run(args, capture_output=True, timeout=45, creationflags=flags, cwd=output)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Official converter timed out; snapshots retained, no simulator was launched') from exc
    (output/'converter.log').write_bytes(result.stdout+b'\n'+result.stderr)
    for p in required:
        if sha(p) != manifest[p.name]['sha256']:
            raise RuntimeError('Solver results changed while being copied/converted')
    csv_path = output/source.with_suffix('.csv').name
    if result.returncode != 0 or not csv_path.is_file():
        raise RuntimeError(f'Converter failed ({result.returncode}); see {output / "converter.log"}')
    return csv_path


def read_echo_parameters(path):
    """Read numeric installed/calculated values; units are checked at use sites."""
    text = Path(path).read_bytes().decode('cp936', errors='replace')
    values = {}
    for line in text.splitlines():
        match = re.match(r'^\s*!?\s*([A-Z][A-Z0-9_]*(?:\([\d,]+\))?)\s+([-+\d.eE]+)(?:\s|;|$)', line)
        if match:
            try:
                values[match[1]] = float(match[2])
            except ValueError:
                pass
    return values


def rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x)**2, axis=axis))


def wheel_torque_imbalance(drive, signed_brake, fx, alpha, *, radius, inertia,
                           brake_multiplier=1):
    """T_act - R*Fx - I*omega_dot; excludes unexported rolling/driveline moments."""
    if not np.isfinite(radius) or not np.isfinite(inertia) or radius <= 0 or inertia <= 0:
        raise ValueError('Wheel radius and inertia must be finite and positive')
    return (np.asarray(drive) + brake_multiplier*np.asarray(signed_brake)
            - radius*np.asarray(fx) - inertia*np.asarray(alpha))


def initial_total_vehicle_profile(echo):
    """An explicit comparison profile, NOT an in-place correction/calibration."""
    front = echo['LX_AXLE(1)']/1000
    rear = echo['LX_AXLE(2)']/1000
    cg = echo['LX_CG_TL']/1000
    inertia = [echo[f'ISPIN({axle},{side})'] for axle, side in ((1,1),(1,2),(2,1),(2,2))]
    if not np.allclose(inertia, inertia[0]):
        raise ValueError('Reference model has one common wheel inertia; unequal actual inertias need a different profile')
    return replace(VehicleParameters(), mass_kg=echo['M_TL'], lf_m=cg-front, lr_m=rear-cg,
                   cg_height_m=echo['H_CG_TL']/1000, yaw_inertia_kg_m2=echo['IZZ_TL'],
                   wheel_inertia_kg_m2=inertia[0])


def timing_and_torque(meta, data, evidence):
    evidence = Path(evidence)
    manifest = json.loads((evidence/'source_manifest.json').read_text(encoding='utf-8'))
    for name, item in manifest.items():
        if Path(name).name != name or sha(evidence/name) != item['sha256']:
            raise ValueError('Solver evidence snapshot hash mismatch')
    if sha(evidence/'LastRun_all.par') != meta['contract']['parameter_sha256']:
        raise ValueError('Solver parameters and capture parameters differ')
    header = json.loads((evidence/'LastRun.vs').read_text(encoding='utf-8'))['VsChannelGroup']
    units = {alias: ch['Units'] for ch in header['Channels'] for alias in ch['Name Aliases']}
    for name, unit in {'Vx':'km/h','Xo':'m','Steer_L1':'deg',**{f'Fx_{w}':'N' for w in VS_WHEELS}}.items():
        if units.get(name) != unit:
            raise ValueError(f'Unsupported evidence unit for {name}: {units.get(name)}')
    with (evidence/'LastRun.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    saved = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if not all(np.all(np.isfinite(a)) for a in saved.values()):
        raise ValueError('Nonfinite converted solver data')
    t = data.get('solver_time_s')
    if t is None or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):
        raise ValueError('Time/torque evidence validation requires a fully observed solver clock')
    ts = saved['Time']
    if not np.all(np.diff(ts)>0):
        raise ValueError('Converted solver time is not increasing')
    mask = (t>.05)&(t<ts[-1]-.003)&(data['return_code']==0)&(data['solver_done']==0)
    if mask.sum()<50:
        raise ValueError('Too few overlapping solver/capture samples')
    alignment = {}
    for key, name, scale, tolerance in (('vx_m_s','Vx',3.6,1e-3), ('x_origin_m','Xo',1,1e-4),
                                       ('front_steer_fl_rad','Steer_L1',180/np.pi,1e-4)):
        shifts = {str(shift): float(rms(np.interp(t[mask]+shift,ts,saved[name])-scale*data[key][mask]))
                  for shift in (-.001,-.0005,0,.0005,.001)}
        alignment[name] = {'rms_by_shift_s': shifts, 'units': units[name], 'zero_shift_tolerance': tolerance}
        # Constant channels cannot identify a lag. A numerical tie is not failure.
        if shifts['0']>tolerance or not np.isclose(shifts['0'],min(shifts.values()),rtol=.01,atol=1e-9):
            raise ValueError(f'Solver-output/capture pairing or timing is inconsistent for {name}: {shifts}')
    echo = read_echo_parameters(evidence/'LastRun_echo.par')
    matched = initial_total_vehicle_profile(echo)
    torque = {}
    for w, vw, axis, side in zip(WHEELS,VS_WHEELS,(1,1,2,2),(1,2,1,2)):
        omega = data[f'omega_{w}_rad_s']
        alpha = np.gradient(omega,t)
        drive, brake = data[f'drive_{w}_nm'], data[f'brake_{w}_nm']
        radius = echo[f'RRE({axis},{side})']/1000
        inertia = echo[f'ISPIN({axis},{side})']
        active = mask&(data['brake_pressure_command_mpa']>.05)&(omega>0)
        if active.sum()<20:
            raise ValueError('Not enough active braking samples for torque sign discrimination')
        alternatives = {}
        for name, multiplier in (('drive_plus_brake',1),('drive_minus_brake',-1),('drive_only',0)):
            by_shift = {}
            for shift in (-.0005,0,.0005):
                fx = np.interp(t+shift,ts,saved[f'Fx_{vw}'])
                imbalance = wheel_torque_imbalance(drive,brake,fx,alpha,radius=radius,
                                                   inertia=inertia,brake_multiplier=multiplier)
                by_shift[str(shift)] = {'rms_nm':float(rms(imbalance[active])),
                                        'mean_nm':float(np.mean(imbalance[active]))}
            alternatives[name] = by_shift
        torque[w] = {'braking_samples':int(active.sum()), 'radius_m':radius, 'inertia_kg_m2':inertia,
                     'balance_I_domega_equals_T_minus_RFx':alternatives}
        robust_ratios = [min(alternatives[k][str(s)]['rms_nm'] for k in ('drive_minus_brake','drive_only'))
                         / max(alternatives['drive_plus_brake'][str(s)]['rms_nm'],1e-12)
                         for s in (-.0005,0,.0005)]
        torque[w]['alternative_to_plus_min_rmse_ratio'] = min(robust_ratios)
        if min(robust_ratios) < 5:
            raise ValueError(f'{w}: torque sign is not strongly separated; do not adopt a mapping automatically')
    return {'alignment':alignment, 'clock_first_interval_s':float(t[1]-t[0]),
            'clock_subsequent_dt_range_s':[float(np.diff(t)[1:].min()),float(np.diff(t)[1:].max())],
            'wrapper_minus_solver_range_s':[float((data['wrapper_time_s']-t)[1:].min()),
                                            float((data['wrapper_time_s']-t)[1:].max())],
            'torques':torque, 'profile_reference':asdict(VehicleParameters()),
            'actuator_torque_mapping':'drive_plus_signed_brake_supported_in_this_forward_braking_run',
            'profile_initial_total_vehicle_proxy':asdict(matched),
            'parameter_sha256':meta['contract']['parameter_sha256'],
            'scope':'Straight forward braking in this Run; file data linearly interpolated, no extrapolated points used.',
            'unmodeled_terms':'Torque balance omits rolling resistance, detailed radius/axis and drivetrain coupling effects. Residual is not required to be zero.'}, matched


def shadow_diagnostics(meta, data, vehicle, *, stride=10, warmup=.1):
    """Teacher-forced 1 ms predictions; 50 ms measured-input replay cross-check.

    Only the carried policy propagates its own last force memory; it is never
    reset at each measured snapshot. Sensor and zero policies are sensitivity
    comparisons at the same evaluation indices, NOT assertions of equivalence.
    """
    actual = data.get('solver_time_s')
    observed_clock = actual is not None and np.all(np.isfinite(actual))
    if actual is not None and not observed_clock:
        raise ValueError('Partial/missing v2 solver clock: do not silently substitute nominal time')
    t = actual if observed_clock else data['wrapper_time_s']
    if len(t)<5 or not np.all(np.isfinite(t)):
        raise ValueError('Need at least five finite timestamps')
    if not isinstance(stride,int) or stride<1 or not np.isfinite(warmup) or warmup<0:
        raise ValueError('Stride must be positive integer and warmup nonnegative')
    if not np.all(np.diff(t)>0):
        raise ValueError('Shadow prediction requires strictly increasing timestamps')
    if not np.allclose(np.diff(t)[1:],.001,atol=1e-8):
        raise ValueError('This initial batch is scoped to the recorded 1 ms solver-step datasets')
    x = np.column_stack([data[k] for k in STATE_NAMES])
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite state samples')
    for key in ('ax_carsim_m_s2','ay_carsim_m_s2','return_code','solver_done'):
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f'Nonfinite {key}')
    controls = [Controls(np.array([data['front_steer_fl_rad'][i],data['front_steer_fr_rad'][i]]),
                         np.array([data[f'drive_{w}_nm'][i]+data[f'brake_{w}_nm'][i] for w in WHEELS]))
                for i in range(len(t))]
    mu = float(meta['scenario_mu_declared'])
    model = Reference7DOF(vehicle=vehicle)
    derivative = np.gradient(x,t,axis=0)
    mem = LoadMemory(0,0)  # explicit, ONCE per episode
    results = {name: {'state':[], 'derivative':[], 'flags':0, 'fz_difference':[]}
               for name in ('carried','sensor_lag','zero_each_step')}
    persistence = []
    memory_difference = []
    memory_seeds = {}
    window_starts = set(range(int(np.searchsorted(t,t[0]+warmup)),len(t)-51,1000))
    for i in range(len(t)-1):
        if i in window_starts:
            memory_seeds[i] = mem
        dt = float(t[i+1]-t[i])
        # The first observed interval may be half a step. No constant shift is applied.
        substeps = max(1,int(np.ceil(dt/.0005-1e-8)))
        carried = model.predict(x[i],controls[i],mu=mu,load_memory=mem,horizon_s=dt,substeps=substeps)
        good = (i>=2 and i<len(t)-2 and t[i]-t[0]>=warmup and x[i,0]>=1
                and np.all(data['return_code'][i-1:i+3]==0) and not np.any(data['solver_done'][i-1:i+3]))
        if good and i%stride==0:
            policies={'carried':mem, 'sensor_lag':LoadMemory(data['ax_carsim_m_s2'][i-1],data['ay_carsim_m_s2'][i-1]),
                      'zero_each_step':LoadMemory(0,0)}
            for name, seed in policies.items():
                prediction = carried if name=='carried' else model.predict(x[i],controls[i],mu=mu,load_memory=seed,
                                                                          horizon_s=dt,substeps=substeps)
                rhs = model.evaluate(x[i],controls[i],mu=mu,load_memory=seed)
                if name=='carried':
                    carried_fz=rhs.raw_fz_n
                results[name]['fz_difference'].append(rhs.raw_fz_n-carried_fz)
                results[name]['state'].append(prediction.state-x[i+1])
                results[name]['derivative'].append(rhs.derivative-derivative[i])
                results[name]['flags'] += int(bool(prediction.flags))
            persistence.append(x[i]-x[i+1])
            memory_difference.append([mem.ax_m_s2-data['ax_carsim_m_s2'][i-1],
                                      mem.ay_m_s2-data['ay_carsim_m_s2'][i-1]])
        mem=carried.load_memory
    if not persistence:
        raise ValueError('No eligible shadow samples')
    scores={name:{'sample_count':len(v['state']), 'state_rmse':dict(zip(STATE_NAMES,map(float,rms(v['state'],axis=0)))),
                  'derivative_rmse':dict(zip(STATE_NAMES,map(float,rms(v['derivative'],axis=0)))),
                  'raw_fz_difference_vs_carried_rmse_n':dict(zip(WHEELS,map(float,rms(v['fz_difference'],axis=0)))),
                  'flagged_predictions':v['flags']} for name,v in results.items()}
    # Keep input truth replay separate from a deployable 50 ms prediction:
    # actual future actuators are available offline, not to an online controller.
    rollouts=[]
    for start, seed in memory_seeds.items():
        if t[start]-t[0]<warmup or start+50>=len(t):
            continue
        window=slice(start,start+51)
        if (np.any(data['return_code'][window]!=0) or np.any(data['solver_done'][window])
                or np.any(x[window,0]<1)):
            continue
        endpoints=[]
        flags=set()
        for max_step in (.0005,.00025):
            state=x[start].copy(); memory=seed
            for i in range(start,start+50):
                dt=float(t[i+1]-t[i])
                pred=model.predict(state,controls[i],mu=mu,load_memory=memory,horizon_s=dt,
                                   substeps=max(1,int(np.ceil(dt/max_step-1e-8))))
                state,memory=pred.state,pred.load_memory; flags.update(pred.flags)
            endpoints.append(state)
        rollouts.append({'start':float(t[start]), 'error':endpoints[0]-x[start+50],
                         'refinement_difference':endpoints[1]-endpoints[0],
                         'persistence':x[start]-x[start+50], 'flags':sorted(flags)})
    multi={}
    if rollouts:
        multi={'windows':len(rollouts),'horizon_s':.05,
               'state_rmse':dict(zip(STATE_NAMES,map(float,rms([r['error'] for r in rollouts],axis=0)))),
               'persistence_rmse':dict(zip(STATE_NAMES,map(float,rms([r['persistence'] for r in rollouts],axis=0)))),
               'refinement_rms_0p5_to_0p25_ms':dict(zip(STATE_NAMES,map(float,rms([r['refinement_difference'] for r in rollouts],axis=0)))),
               'flagged_windows':sum(bool(r['flags']) for r in rollouts)}
    return {'time_basis':'observed_solver_T' if observed_clock else 'legacy_relative_intervals_nominal_no_absolute_clock_claim',
            'warmup_s':warmup,'score_stride':stride,'vehicle':asdict(vehicle),'policies':scores,
            'one_step_persistence_rmse':dict(zip(STATE_NAMES,map(float,rms(persistence,axis=0)))),
            'carried_minus_sensor_lag_memory_rmse_m_s2':list(map(float,rms(memory_difference,axis=0))),
            'fifty_ms_measured_future_input_replay':multi,
            'not_training_labels':True,
            'notes':['Carried memory is initialized once to zero, then generated by the unmodified physical model.',
                     'The sensor-lag case is a conditional proxy; it is not proof Ax/Ay equal tire-force-only LoadMemory.',
                     'Sensor lag is one export interval; carried memory is from the previous model microstep. They are not identical clocks or definitions.',
                     'Fz sensitivity compares model variants only. No measured Fz/Fy is available here for a physical load validation.',
                     '1 ms RMSE must be compared with persistence and derivative RMSE; tiny dt alone gives tiny errors.',
                     '50 ms replay uses measured future wheel inputs, not a deployable future-control forecast.']}


def render_markdown(report):
    """Human-readable engineering conclusions, with explicit evidence boundaries."""
    e=report['evidence']
    lines=['# 时间、轮端扭矩与载荷记忆：批量离线验证', '',
           '本报告不启动 CarSim、不训练网络、不修改原模型；证据来自已有采集及官方转换器导出的 CarSim 结果副本。', '',
           '## 1. 时间', '',
           f"实际 T 首间隔 {e['clock_first_interval_s']*1000:.6f} ms；随后间隔范围 "
           f"{e['clock_subsequent_dt_range_s'][0]*1000:.6f}–{e['clock_subsequent_dt_range_s'][1]*1000:.6f} ms。",
           '使用实际 T 对齐；不再额外平移 0.5 ms。此结论针对当前 Run/积分配置及下列已交叉核对的通道。', '',
           '| 通道 | 按实际 T 对齐 RMSE | 人为再加 0.5 ms 的 RMSE | 单位 |',
           '|---|---:|---:|---|']
    for name,a in e['alignment'].items():
        lines.append(f"| {name} | {a['rms_by_shift_s']['0']:.6g} | {a['rms_by_shift_s']['0.0005']:.6g} | {a['units']} |")
    lines += ['', '排除初始化、边界及异常样本。旧 v1 采集仅使用相对时间间隔作诊断，不能据此补写其绝对时间。', '',
              '## 2. 轮端扭矩', '',
              '`T_act = My_Dr + My_Bk`（保留制动负号）得到当前正向制动数据的强支持。',
              '独立核对公式：`imbalance = T_act - R*Fx_CarSim - Ispin*domega/dt`。其中 Fx 来自 CarSim 保存结果，不使用 7DOF 预测力来验证自身输入。', '',
              '| 车轮 | 相加 RMSE / N·m | 相减 RMSE / N·m | 仅驱动 RMSE / N·m |',
              '|---|---:|---:|---:|']
    for w,v in e['torques'].items():
        a=v['balance_I_domega_equals_T_minus_RFx']
        lines.append('| '+w.upper()+' | '+' | '.join(f"{a[k]['0']['rms_nm']:.4f}" for k in ('drive_plus_brake','drive_minus_brake','drive_only'))+' |')
    lines += ['', '±0.5 ms 的力信号对齐扰动未改变符号判断。剩余不平衡没有被归零：此简化平衡未包含滚阻力矩、详细有效力臂/坐标与传动系耦合。',
              '这里确定的是驱动/制动执行器合扭矩，不是所有外力矩总和；R*Fx 已由 7DOF 内部扣除，不能再次从输入扣除。结论不自动推广到倒车。', '',
              '## 3. 已知参数定义差异', '',
              '| 参数 | 用户原始参考值 | CarSim 初始整车对照值 |', '|---|---:|---:|']
    for label,key in [('总质量 / kg','mass_kg'),('横摆惯量 / kg·m²','yaw_inertia_kg_m2'),
                      ('车轮旋转惯量 / kg·m²','wheel_inertia_kg_m2'),('质心高度 / m','cg_height_m'),
                      ('质心到前轴 / m','lf_m'),('质心到后轴 / m','lr_m')]:
        lines.append(f"| {label} | {e['profile_reference'][key]:.9g} | {e['profile_initial_total_vehicle_proxy'][key]:.9g} |")
    lines += ['', '右列仅构成独立对照参数组，不代表完成标定，也未改写 model.py 默认值或轮胎拟合。',
              '整车初始参数与未加载簧载车身参数不能混用；单独替换部分参数不保证每项预测都改善。', '',
              '## 4. 载荷记忆及多步预测', '',
              '`memory = [上一模型微步 Sum_Fx/m, Sum_Fy/m]`：每回合初始化一次，之后连续传递。',
              'CarSim Ax/Ay 是传感输出对应的整车加速度；并非模型轮胎合力加速度的严格同义量。传感值滞后一整采样步与模型微步记忆的时刻也不同。',
              '下表是相同评分时刻的状态导数误差，不是网络训练结果。Fz 差异仅表示模型对记忆选择的敏感性，不是实测轮荷误差。', '']
    for name,c in report['captures'].items():
        lines += [f'### {name}', '', f"源目录：`{c['source']}`", '']
        for label,v in c['profiles'].items():
            lines += [f'参数组：`{label}`；时间依据：`{v["time_basis"]}`。', '',
                      '| 记忆策略 | 样本数 | dvx/dt RMSE / m/s² | dvy/dt RMSE / m/s² | dr/dt RMSE / rad/s² | 左前轮角加速度 RMSE / rad/s² | 左前 Fz 相对连续记忆 RMSE / N |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for policy,q in v['policies'].items():
                d=q['derivative_rmse']
                lines.append(f"| {policy} | {q['sample_count']} | {d['vx_m_s']:.5g} | {d['vy_m_s']:.5g} | {d['yaw_rate_rad_s']:.5g} | {d['omega_fl_rad_s']:.5g} | {q['raw_fz_difference_vs_carried_rmse_n']['fl']:.5g} |")
            z=v['fifty_ms_measured_future_input_replay']
            lines += ['', f"连续记忆 − 上一行传感加速度 RMSE [ax, ay] = {v['carried_minus_sensor_lag_memory_rmse_m_s2']} m/s²。", '']
            if z:
                lines += [f"50 ms 实测未来输入回放：{z['windows']} 个窗口；这不是在线可提前获知未来输入的预测。", '',
                          '| 状态（SI） | 7DOF RMSE | 保持当前值 RMSE | 子步 0.5→0.25 ms 的端点差 RMS |',
                          '|---|---:|---:|---:|']
                for k in STATE_NAMES:
                    lines.append(f"| {k} | {z['state_rmse'][k]:.6g} | {z['persistence_rmse'][k]:.6g} | {z['refinement_rms_0p5_to_0p25_ms'][k]:.6g} |")
                lines += ['', f"限幅/保护窗口数：{z['flagged_windows']}。", '']
    lines += ['## 5. 下一阶段采用的约定', '',
              '1. 新数据按实际 solver_time_s 构造时间间隔；全程保留原始值和证据。',
              '2. 输入用实际前轮角、四轮带符号驱动＋制动扭矩，不拿方向盘指令或制动压力代替。',
              '3. 载荷记忆连续传递；残差特征应显式包含这两个记忆量及 mu，不能假装模型只依赖七维状态。',
              '4. 原参数组和整车对照组版本分开。原参数保持用户定义；不要通过神经网络掩盖参数来源。',
              '5. 当前可以进入离线残差原型；仍非生产训练/控制接管验收。首次导数标签需检查差分窗口与噪声，不将当前诊断文件直接当监督标签。',
              '6. 按完整 episode/工况隔离验证数据；最新 2 s 是旧减速回合的重复前缀，不能一份训练、一份验证。',
              '7. Ensemble 分歧只是认知不确定性的代理量，需在留出工况上校准；低分歧不保证真实误差小。', '',
              '暂未验证：实测四轮 Fz/Fy、一切侧倾/俯仰耦合、全速度/附着范围、闭环残差控制安全性。',
              '完整数值和源文件哈希见同目录 physics_check.json。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',action='append',required=True,type=Path)
    p.add_argument('--evidence-dir',required=True,type=Path)
    p.add_argument('--evidence-capture',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path,help='New report directory only')
    args=p.parse_args(argv)
    output=args.output_dir.resolve()
    sources=[c.resolve() for c in args.capture]
    evidence_dir=args.evidence_dir.resolve()
    if output.exists() or any(output==c or c in output.parents for c in sources+[evidence_dir]):
        p.error('Output must be a new directory outside the source captures')
    if len({s.name for s in sources}) != len(sources):
        p.error('Capture names must be unique within a report')
    if args.evidence_capture.resolve() not in sources:
        p.error('--evidence-capture must also be in --capture')
    evidence_hashes={f.name:sha(f) for f in evidence_dir.iterdir() if f.is_file()}
    # Check and snapshot all source hashes before computation. Fail before saving
    # a misleading completion report if any capture fails its structural audit.
    audits={}
    for source in sources:
        a=audit_capture(source)
        if a['integrity']!='PASS':
            raise ValueError(f'{source.name}: {a["errors"]}')
        audits[source.name]=a
    target_meta,target_data=load_capture(args.evidence_capture)
    evidence,matched=timing_and_torque(target_meta,target_data,args.evidence_dir)
    report={'format':'physics_check_v2','evidence':evidence,'captures':{},'training_started':False,
            'evidence_directory':str(evidence_dir),'evidence_sha256':evidence_hashes,
            'implementation_sha256':{f:sha(Path(__file__).parent/f) for f in ('physics_check.py','model.py','audit.py','signals.py')},
            'ready_for_production_training':False,'original_model_changed':False}
    for source in sources:
        meta,data=load_capture(source)
        if meta['contract']['parameter_sha256']!=evidence['parameter_sha256']:
            raise ValueError(f'{source}: parameter fingerprint differs from evidence vehicle')
        profiles={}
        for label,vehicle in [('original_reference',VehicleParameters()),('initial_total_vehicle_proxy',matched)]:
            print(f'Checking {source.name}: {label}',flush=True)
            profiles[label]=shadow_diagnostics(meta,data,vehicle)
        report['captures'][source.name]={'source':str(source),'source_sha256':audits[source.name]['source_sha256'],
                                         'final_info':meta.get('final_info',{}),'profiles':profiles}
        for name,digest in audits[source.name]['source_sha256'].items():
            if sha(source/name)!=digest:
                raise RuntimeError('Capture changed during verification')
    for name,digest in evidence_hashes.items():
        if sha(evidence_dir/name)!=digest:
            raise RuntimeError('Solver evidence changed during verification')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'physics_check.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (output/'physics_check.md').open('x',encoding='utf-8') as f:
        f.write(render_markdown(report))
    print('Verification report:',output/'physics_check.json',flush=True)
    print('Readable summary:',output/'physics_check.md',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
