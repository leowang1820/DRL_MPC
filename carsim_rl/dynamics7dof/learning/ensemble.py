"""Bootstrapped deterministic residual ensemble; disagreement is a proxy, not Risk."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CONTRACT, X_FLOOR, Y_FLOOR, fingerprint, load_dataset
from .symmetry import MODES, Y_PERM, Y_SIGN, fit_population, mirror_features, mirror_targets


def make_members(count=5, hidden=64, seed=42):
    if count < 2 or hidden < 4:
        raise ValueError('Need >=2 independent members and hidden width >=4')
    members = []
    # Do not disturb the caller's global RNG state (important for later SAC coexistence).
    with torch.random.fork_rng(devices=[]):
        for i in range(count):
            torch.default_generator.manual_seed(seed+997*i)
            members.append(nn.Sequential(nn.Linear(16, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                         nn.Tanh(), nn.Linear(hidden, 7)))
    return nn.ModuleList(members)


def anchor_masks(time):
    """Internal replay guard: whole 0.5 s blocks, with 20 ms purge at boundaries.

    This anchor is NOT the independent validation set. It guards adaptation;
    epochs are fixed by CLI, not selected using the independent validation set.
    """
    t = np.asarray(time)
    block = np.floor(t/.5).astype(int)
    anchor = block % 5 == 4
    train = ~anchor
    for edge in np.unique(block):
        if edge % 5 in (0, 4):
            train &= np.abs(t-edge*.5) > .02
    if train.sum() < 30 or anchor.sum() < 20:
        raise ValueError('Need enough training data spanning held-out 0.5 s anchor blocks (>=2.6 s recommended)')
    return train, anchor


def fit_scaler(features, targets):
    x, y = np.asarray(features), np.asarray(targets)
    if x.ndim != 2 or len(x) < 1 or x.shape[1] != 16 or y.shape != (len(x), 7) or not np.all(np.isfinite(np.c_[x, y])):
        raise ValueError('Expected finite [N,16] features and [N,7] targets')
    return {'x_mean': x.mean(0), 'x_scale': np.maximum(x.std(0), X_FLOOR),
            'y_scale': np.maximum(y.std(0), Y_FLOOR),
            'lower': x.min(0)-np.maximum(.1*np.ptp(x, axis=0), 2*X_FLOOR),
            'upper': x.max(0)+np.maximum(.1*np.ptp(x, axis=0), 2*X_FLOOR),
            'correction_limit': np.maximum(1.5*np.quantile(np.abs(y), .995, axis=0), 2*Y_FLOOR)}


class ResidualEnsemble:
    """Small-batch NumPy inference from Torch weights; no GPU round-trip each ms."""
    def __init__(self, members, scaler, uncertainty_limit=.5, metadata=None, symmetry='none'):
        if symmetry not in MODES:
            raise ValueError('Unknown residual symmetry mode')
        self.symmetry = symmetry
        self.members = members.cpu().eval()
        self.scaler = {k: np.asarray(v, dtype=float) for k, v in scaler.items()}
        self.uncertainty_limit = float(uncertainty_limit)
        for name, size in [('x_mean', 16), ('x_scale', 16), ('y_scale', 7),
                           ('lower', 16), ('upper', 16), ('correction_limit', 7)]:
            if name not in self.scaler or self.scaler[name].shape != (size,) or not np.all(np.isfinite(self.scaler[name])):
                raise ValueError(f'Invalid scaler field: {name}')
        if (any(np.any(self.scaler[k] <= 0) for k in ('x_scale', 'y_scale', 'correction_limit'))
                or np.any(self.scaler['lower'] > self.scaler['upper'])
                or not np.isfinite(self.uncertainty_limit) or self.uncertainty_limit <= 0):
            raise ValueError('Invalid scale, bounds or uncertainty limit')
        if symmetry == 'mirror':
            for name in ('y_scale', 'correction_limit'):
                if not np.allclose(self.scaler[name], self.scaler[name][Y_PERM], rtol=1e-10, atol=1e-12):
                    raise ValueError('Mirror mode requires equal left/right output scales and limits')
        self.metadata = metadata or {}
        self.refresh()

    def refresh(self):
        self.weights = [[(layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy())
                         for layer in member if isinstance(layer, nn.Linear)] for member in self.members]

    def predict(self, features, gated=True):
        z = np.atleast_2d(np.asarray(features, dtype=float))
        if z.shape[1] != 16 or not np.all(np.isfinite(z)):
            raise ValueError('Nonfinite or incompatible residual feature')
        mirrored_z = mirror_features(z) if self.symmetry == 'mirror' else None
        inputs = np.concatenate([z, mirrored_z]) if mirrored_z is not None else z
        a = (inputs-self.scaler['x_mean'])/self.scaler['x_scale']
        outputs = []
        for layers in self.weights:
            value = a
            for i, (w, b) in enumerate(layers):
                value = value@w.T+b
                if i != len(layers)-1:
                    value = np.tanh(value)
            value = value*self.scaler['y_scale']
            if self.symmetry == 'mirror':
                value = .5*(value[:len(z)]+mirror_targets(value[len(z):]))
            outputs.append(value)
        predictions = np.asarray(outputs)
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError('Nonfinite ensemble prediction')
        mean, std = predictions.mean(0), predictions.std(0, ddof=1)
        uncertainty = np.sqrt(np.mean((std/self.scaler['y_scale'])**2, axis=1))
        ood = np.any((z < self.scaler['lower']) | (z > self.scaler['upper']), axis=1)
        if mirrored_z is not None:
            # Both reflection partners must be supported; gating itself is symmetric.
            ood |= np.any((mirrored_z < self.scaler['lower']) | (mirrored_z > self.scaler['upper']), axis=1)
        rejected = ood | (uncertainty > self.uncertainty_limit)
        clipped = np.any(np.abs(mean) > self.scaler['correction_limit'], axis=1)
        correction = np.clip(mean, -self.scaler['correction_limit'], self.scaler['correction_limit'])
        if gated:
            correction[rejected] = 0
        return {'members': predictions, 'mean': mean, 'std': std, 'uncertainty': uncertainty,
                'ood': ood, 'rejected': rejected, 'clipped': clipped, 'correction': correction}

    def save(self, path, anchor_x=None, anchor_y=None, history_x=None, history_y=None):
        payload = {'format': 'residual_ensemble_v2', 'symmetry': self.symmetry,
                   'fingerprint': fingerprint(), 'contract': CONTRACT,
                   'count': len(self.members), 'hidden': self.members[0][0].out_features,
                   'weights': self.members.state_dict(), 'scaler': {k: v.tolist() for k, v in self.scaler.items()},
                   'uncertainty_limit': self.uncertainty_limit, 'metadata': self.metadata}
        for k, a in [('anchor_x', anchor_x), ('anchor_y', anchor_y), ('history_x', history_x), ('history_y', history_y)]:
            if a is not None:
                payload[k] = torch.as_tensor(np.asarray(a), dtype=torch.float32)
        with Path(path).open('xb') as file:
            torch.save(payload, file)

    @classmethod
    def load(cls, path):
        # Only load trusted, locally generated files. weights_only narrows unpickling.
        p = torch.load(path, map_location='cpu', weights_only=True)
        if p['format'] not in ('residual_ensemble_v1', 'residual_ensemble_v2') or p['fingerprint'] != fingerprint():
            raise ValueError('Checkpoint feature/physics contract mismatch')
        members = make_members(p['count'], p['hidden'])
        members.load_state_dict(p['weights'])
        symmetry = p['symmetry'] if p['format'] == 'residual_ensemble_v2' else 'none'
        obj = cls(members, p['scaler'], p['uncertainty_limit'], p['metadata'], symmetry=symmetry)
        buffers = {k: p[k].numpy().copy() for k in ('anchor_x', 'anchor_y', 'history_x', 'history_y') if k in p}
        return obj, buffers


def score_model(model, x, y):
    p = model.predict(x)
    scale = model.scaler['y_scale']
    error = p['correction']-y
    base = np.sqrt(np.mean((y/scale)**2))
    corrected = np.sqrt(np.mean((error/scale)**2))
    return {'baseline_rmse_si': np.sqrt(np.mean(y*y, axis=0)).tolist(),
            'raw_ensemble_rmse_si': np.sqrt(np.mean((p['mean']-y)**2, axis=0)).tolist(),
            'gated_rmse_si': np.sqrt(np.mean(error**2, axis=0)).tolist(),
            'normalized_baseline_rmse': float(base), 'normalized_gated_rmse': float(corrected),
            'rejection_fraction': float(np.mean(p['rejected'])), 'ood_fraction': float(np.mean(p['ood'])),
            'clipped_fraction': float(np.mean(p['clipped'])),
            'disagreement_p50_p95': np.quantile(p['uncertainty'], [.5, .95]).tolist(),
            'uncertainty_is_calibrated': False}


def normalized_forward(member, inputs, reflected_inputs, output_scale, symmetry):
    """Same SI reflection projection as NumPy inference, with autograd intact."""
    if symmetry not in MODES:
        raise ValueError('Unknown residual symmetry mode')
    value = member(inputs)
    if symmetry == 'mirror':
        other = member(reflected_inputs)*output_scale
        reflected = other[:, Y_PERM]*other.new_tensor(Y_SIGN)
        value = .5*(value+reflected/output_scale)
    return value


def episode_loss_weights(targets, episodes, output_scale):
    """Balance physical channels per FIT episode without using validation errors.

    Weight the normalized error by inverse episode RMS, with physical noise
    floors and a capped gain. Small-steer dynamics must not be overwhelmed by
    the much larger derivative residuals of the large-steer recording.
    """
    y, ids = np.asarray(targets), np.asarray(episodes)
    weights, summary = np.ones_like(y), {}
    for episode in np.unique(ids):
        mask = ids == episode
        rms = np.maximum(np.sqrt(np.mean(y[mask]**2, axis=0)), Y_FLOOR)
        weight = np.clip(np.asarray(output_scale)/rms, .5, 4.)
        weights[mask] = weight
        summary[str(int(episode))] = weight.tolist()
    return weights, summary


def optimize(members, scaler, x, y, *, epochs=1, batch_size=256, lr=.001, seed=42,
             device='cpu', blocks=None, report=None, symmetry='none', sample_weights=None):
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('Positive epochs, batch size, and learning rate required')
    rng = np.random.default_rng(seed)
    members.to(device).train()
    inputs = torch.as_tensor((x-scaler['x_mean'])/scaler['x_scale'], dtype=torch.float32, device=device)
    mirrored = torch.as_tensor((mirror_features(x)-scaler['x_mean'])/scaler['x_scale'],
                               dtype=torch.float32, device=device) if symmetry == 'mirror' else None
    output_scale = torch.as_tensor(scaler['y_scale'], dtype=torch.float32, device=device)
    targets = torch.as_tensor(y/scaler['y_scale'], dtype=torch.float32, device=device)
    if sample_weights is not None:
        if np.shape(sample_weights) != np.shape(y) or not np.all(np.isfinite(sample_weights)) or np.any(sample_weights <= 0):
            raise ValueError('Loss weights must be finite positive [N,7]')
        loss_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    else:
        loss_weights = None
    optimizer = torch.optim.Adam(members.parameters(), lr=lr)
    boots = []
    for _ in members:
        if blocks is None:
            boots.append(rng.integers(len(x), size=len(x)))
        else:
            unique = np.unique(blocks)
            selected = rng.choice(unique, size=len(unique), replace=True)
            boots.append(np.concatenate([np.flatnonzero(blocks == b) for b in selected]))
    history = []
    for epoch in range(epochs):
        losses = []
        orders = [rng.permutation(b) for b in boots]
        for offset in range(0, max(map(len, orders)), batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = None
            for member, order in zip(members, orders):
                # Circular batches keep all bootstrap members training independently.
                idx = order[np.arange(offset, offset+batch_size) % len(order)]
                idx = torch.as_tensor(idx, dtype=torch.long, device=device)
                prediction = normalized_forward(member, inputs[idx], mirrored[idx] if mirrored is not None else None,
                                                output_scale, symmetry)
                error = prediction-targets[idx]
                if loss_weights is not None:
                    error = error*loss_weights[idx]
                part = torch.mean(error**2)
                loss = part if loss is None else loss+part
            loss = loss/len(members)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(members.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {'epoch': epoch+1, 'train_scaled_mse': float(np.mean(losses))}
        history.append(row)
        if report:
            report(row)
    members.cpu().eval()
    return history


def train_dataset(dataset, output, *, epochs=100, hidden=64, count=5, device='cpu', seed=42,
                  symmetry='mirror', loss_weighting='episode-rms'):
    output = Path(output).resolve()
    if output.exists() or Path(dataset).resolve() in output.parents:
        raise ValueError('Model output must be a new directory outside dataset')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    arrays, manifest = load_dataset(dataset)
    fit, anchor = anchor_masks(arrays['train_time'])
    x, y = arrays['train_features'][fit], arrays['train_targets'][fit]
    ax, ay = arrays['train_features'][anchor], arrays['train_targets'][anchor]
    scaler = fit_scaler(*fit_population(x, y, symmetry))  # fit rows + their mirrors ONLY
    if loss_weighting not in ('global', 'episode-rms'):
        raise ValueError('Unknown loss weighting')
    loss_weights, weight_report = (episode_loss_weights(y, arrays['train_episode'][fit], scaler['y_scale'])
                                   if loss_weighting == 'episode-rms' else (None, {}))
    members = make_members(count, hidden, seed)
    blocks = arrays['train_episode'][fit]*1000000+np.floor(arrays['train_time'][fit]/.05).astype(int)
    history = optimize(members, scaler, x, y, epochs=epochs, device=device, seed=seed, blocks=blocks, symmetry=symmetry,
                       sample_weights=loss_weights,
                       report=lambda row: print(row, flush=True) if row['epoch'] % 10 == 0 else None)
    model = ResidualEnsemble(members, scaler, metadata={'dataset': str(Path(dataset).resolve()),
                            'dataset_sha256': manifest['samples_sha256'], 'episodes': manifest['episodes'],
                            'seed': seed, 'epochs': epochs, 'independent_validation_used_for_gradients': False,
                            'validation_role': manifest.get('validation_role', 'held_out'),
                            'loss_weighting': loss_weighting, 'fit_episode_error_weights': weight_report,
                            'symmetry_assumption': 'flat_uniform_road_left_right_symmetric_vehicle' if symmetry == 'mirror' else 'none',
                            'stage': 'experimental_first_layer_only'}, symmetry=symmetry)
    # Heuristic scale from TRAIN only; this is not calibration or a safety certificate.
    model.uncertainty_limit = max(.2, 3*float(np.quantile(model.predict(x)['uncertainty'], .95)))
    train_score, anchor_score = score_model(model, x, y), score_model(model, ax, ay)
    val_score = score_model(model, arrays['validation_features'], arrays['validation_targets'])
    validation_key = 'development_validation_not_independent' if manifest.get('validation_role') == 'development' else 'independent_validation'
    report = {'train': train_score, 'internal_anchor_not_independent': anchor_score,
              validation_key: val_score, 'history': history, 'symmetry': symmetry,
              'loss_weighting': loss_weighting, 'fit_episode_error_weights': weight_report,
              'fit_rows': int(fit.sum()), 'anchor_rows': int(anchor.sum()),
              'validation_rows': len(arrays['validation_time']),
              'automatic_controller_deployment': False, 'risk_labels_created': False,
              'uncertainty_note': 'Uncalibrated normalized ensemble disagreement; low disagreement is not proof of low error.'}
    output.mkdir(parents=True, exist_ok=False)
    history_idx = np.linspace(0, len(x)-1, min(2048, len(x))).astype(int)
    model.save(output/'ensemble.pt', ax, ay, x[history_idx], y[history_idx])
    (output/'training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Model:', output/'ensemble.pt', '| validation:', val_score, flush=True)
    return report
