"""Causal feature stream with delayed, centered derivative supervision."""

from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..audit import audit_capture
from ..model import Controls, LoadMemory, Reference7DOF, STATE_NAMES, VehicleParameters
from ..physics_check import load_capture, sha


FEATURE_NAMES = (*STATE_NAMES, 'delta_fl_rad', 'delta_fr_rad', 'torque_fl_nm',
                 'torque_fr_nm', 'torque_rl_nm', 'torque_rr_nm', 'mu',
                 'memory_ax_m_s2', 'memory_ay_m_s2')
X_FLOOR = np.array([.1, .02, .01, .2, .2, .2, .2, .001, .001, 10, 10, 10, 10, .02, .1, .1])
Y_FLOOR = np.array([.05, .05, .05, .5, .5, .5, .5])
CONTRACT = {'feature_names': list(FEATURE_NAMES), 'target_names': list(STATE_NAMES),
            'model_sha256': sha(Path(__file__).resolve().parents[1]/'model.py'),
            'vehicle': asdict(VehicleParameters()), 'memory': 'nominal_force_only_carried',
            'label': 'centered_quadratic_dxdt_minus_instantaneous_nominal_rhs',
            'window_radius': 5, 'substep_s': .0005, 'warmup_s': .1}


def fingerprint(contract=CONTRACT):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def feature_vector(state, controls, mu, memory):
    z = np.r_[np.asarray(state).reshape(7), controls.front_steer_rad,
              controls.wheel_net_torque_nm, float(mu), memory.ax_m_s2, memory.ay_m_s2]
    if z.shape != (16,) or not np.all(np.isfinite(z)) or mu < 0:
        raise ValueError('Feature contract requires 16 finite SI values and mu >= 0')
    return z


def controls_at(data, i):
    return Controls([data['front_steer_fl_rad'][i], data['front_steer_fr_rad'][i]],
                    [data[f'drive_{w}_nm'][i]+data[f'brake_{w}_nm'][i]
                     for w in ('fl', 'fr', 'rl', 'rr')])


class LabelStream:
    """Observed x/u at t creates a feature immediately; label arrives 5 ticks later.

    No future values enter the feature or nominal load memory. A local quadratic
    fit estimates dx/dt at the central timestamp, not at the arrival timestamp.
    Reject boundaries, incomplete clocks, low speed and actuator discontinuities.
    """
    def __init__(self, radius=5, warmup=.1, model=None):
        if not isinstance(radius, int) or radius < 2 or warmup < 0:
            raise ValueError('Derivative radius >= 2 and warmup >= 0 required')
        self.radius, self.warmup = radius, warmup
        self.model = model or Reference7DOF()
        self.buffer = deque(maxlen=2*radius+1)
        self.memory = LoadMemory(0, 0)
        self.previous = None
        self.start = None
        self.index = 0

    def observe(self, t, state, controls, mu, valid=True):
        x = np.asarray(state, dtype=float).reshape(7).copy()
        if not np.isfinite(t) or not np.all(np.isfinite(x)):
            raise ValueError('Nonfinite stream timestamp/state')
        if self.start is None:
            self.start = float(t)
        if self.previous is not None:
            prev_t, prev_x, prev_u, prev_mu = self.previous
            dt = float(t-prev_t)
            if not 0 < dt <= .002:
                raise ValueError('Stream needs continuous solver samples, at most 2 ms apart')
            p = self.model.predict(prev_x, prev_u, mu=prev_mu, load_memory=self.memory,
                                   horizon_s=dt, substeps=max(1, int(np.ceil(dt/.0005-1e-8))))
            self.memory = p.load_memory
            valid = valid and not p.flags
        evaluation = self.model.evaluate(x, controls, mu=mu, load_memory=self.memory)
        z = feature_vector(x, controls, mu, self.memory)
        valid = bool(valid and not evaluation.flags and x[0] >= 2)
        current = {'time': float(t), 'state': x, 'features': z,
                   'base': evaluation.derivative.copy(), 'valid': valid, 'index': self.index}
        self.buffer.append(current)
        self.previous = (float(t), x, controls, mu)
        self.index += 1
        label = None
        if len(self.buffer) == self.buffer.maxlen:
            window = list(self.buffer)
            center = window[self.radius]
            times = np.array([s['time'] for s in window])
            features = np.array([s['features'] for s in window])
            dt = np.diff(times)
            clean = (all(s['valid'] for s in window) and center['time']-self.start >= self.warmup
                     and np.allclose(dt, dt[0], rtol=1e-5, atol=1e-9)
                     and np.max(np.ptp(features[:, 7:9], axis=0)) <= .006
                     and np.max(np.ptp(features[:, 9:13], axis=0)) <= 100
                     and np.ptp(features[:, 13]) < 1e-9)
            if clean:
                scale = times[-1]-times[0]
                tau = (times-center['time'])/scale
                design = np.column_stack([np.ones_like(tau), tau, tau*tau])
                coeff = np.linalg.lstsq(design, np.array([s['state'] for s in window]), rcond=None)[0]
                derivative = coeff[1]/scale
                label = dict(center, derivative=derivative, target=derivative-center['base'],
                             available_at=float(t))
        return current, label


def prepare_capture(path, stride=5):
    audit = audit_capture(path)
    if audit['integrity'] != 'PASS':
        raise ValueError(f'Capture integrity failed: {audit["errors"]}')
    meta, data = load_capture(path)
    t = data.get('solver_time_s')
    if t is None or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Learning requires actual solver time on every row; no nominal-time fallback')
    if not np.allclose(np.diff(t)[1:], .001, atol=1e-8):
        raise ValueError('Initial learning contract is for 1 ms exports')
    mu = float(meta['scenario_mu_declared'])
    configured = meta['contract']['settings'].get('MU_ROAD_CONSTANT')
    if configured is None or not np.isclose(mu, configured):
        raise ValueError('Explicit mu must match the constant road configuration')
    x = np.column_stack([data[k] for k in STATE_NAMES])
    stream, labels = LabelStream(), []
    for i, time in enumerate(t):
        _, label = stream.observe(time, x[i], controls_at(data, i), mu,
                                  valid=data['return_code'][i] == 0 and not data['solver_done'][i])
        if label is not None and label['index'] % stride == 0:
            labels.append(label)
    if len(labels) < 30:
        raise ValueError('Too few clean residual samples; need a longer forward-driving episode')
    # Detect same physical recording copied into different folders, independent of CSV path/comments.
    physical = np.column_stack([t-t[0], x, data['front_steer_fl_rad'], data['front_steer_fr_rad'],
                                *[data[f'{kind}_{w}_nm'] for kind in ('drive', 'brake') for w in ('fl', 'fr', 'rl', 'rr')]])
    signature = hashlib.sha256(np.round(physical, 9).tobytes()).hexdigest()
    for name, digest in audit['source_sha256'].items():
        if sha(Path(path)/name) != digest:
            raise RuntimeError('Capture changed while preparing data')
    arrays = {'features': np.array([s['features'] for s in labels]),
              'targets': np.array([s['target'] for s in labels]),
              'base': np.array([s['base'] for s in labels]),
              'derivative': np.array([s['derivative'] for s in labels]),
              'time': np.array([s['time']-t[0] for s in labels]),
              'available_at': np.array([s['available_at']-t[0] for s in labels]),
              'sample_index': np.array([s['index'] for s in labels])}
    info = {'capture': str(Path(path).resolve()), 'source_sha256': audit['source_sha256'],
            'physical_signature': signature, 'rows': len(t), 'labels': len(labels),
            'parameter_sha256': meta['contract']['parameter_sha256'], 'excitation': meta.get('excitation')}
    return arrays, info


def build_dataset(train, validation, output, stride=5):
    out = Path(output).resolve()
    sources = [Path(p).resolve() for p in (*train, *validation)]
    if out.exists() or any(out == p or p in out.parents for p in sources):
        raise ValueError('Dataset output must be a new directory outside captures')
    if not train or not validation or len(set(sources)) != len(sources):
        raise ValueError('Distinct train and validation capture directories are required')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('Stride must be a positive integer')
    arrays, episodes, signatures = {}, [], set()
    for split, paths in (('train', train), ('validation', validation)):
        groups = []
        for path in paths:
            print('Preparing', split, path, flush=True)
            a, info = prepare_capture(path, stride)
            if info['physical_signature'] in signatures:
                raise ValueError('Duplicate physical recording: train/validation leakage')
            signatures.add(info['physical_signature'])
            a['episode'] = np.full(len(a['time']), len(episodes), dtype=int)
            groups.append(a)
            episodes.append(dict(info, split=split, id=len(episodes)))
        for key in groups[0]:
            arrays[f'{split}_{key}'] = np.concatenate([g[key] for g in groups])
    # Conservatively reject near-identical label windows shared across splits,
    # e.g. an old 2-second prefix repackaged as a separate validation episode.
    def row_keys(split):
        joined = np.c_[arrays[f'{split}_features'], arrays[f'{split}_derivative']]
        return {row.tobytes() for row in np.round(joined, 7)}
    common = len(row_keys('train') & row_keys('validation'))
    overlap = common/max(1, min(len(arrays['train_time']), len(arrays['validation_time'])))
    if overlap > .1:
        raise ValueError(f'{overlap:.1%} near-identical windows shared across train/validation; choose another episode')
    manifest = {'format': 'residual_dataset_v1', 'contract': CONTRACT, 'contract_fingerprint': fingerprint(),
                'episodes': episodes, 'stride': stride, 'near_identical_window_fraction': overlap,
                'notes': ['Experimental derivative labels, not a physical ground truth certificate.',
                          'Whole episodes separated; validation never used for normalization/gradients.',
                          'Centered derivative labels arrive 5 solver ticks later; features are causal.',
                          'No Risk labels or controller takeover.']}
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'samples.npz', **arrays)
    manifest['samples_sha256'] = sha(out/'samples.npz')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Dataset:', out, flush=True)
    return manifest


def load_dataset(path):
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['contract_fingerprint'] != fingerprint():
        raise ValueError('Dataset physics/feature contract differs from current code')
    if sha(root/'samples.npz') != manifest['samples_sha256']:
        raise ValueError('Dataset file checksum mismatch')
    with np.load(root/'samples.npz', allow_pickle=False) as file:
        arrays = {k: file[k] for k in file.files}
    return arrays, manifest
