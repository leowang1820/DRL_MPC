"""Literature-guided truncated simulation-error fine-tuning, offline only.

Automatica 2025, doi:10.1016/j.automatica.2025.112409 (hybrid multi-step model)
Automatica 2023, doi:10.1016/j.automatica.2023.111210, Sec.3.1 (truncated loss)
This is a measured-state adaptation, NOT a replication of sparse identification
or SUBNET's latent encoder/innovation model, and inherits none of their proofs.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from ..model import STATE_NAMES
from ..physics_check import load_capture, sha
from .data import LabelStream, X_FLOOR, controls_at, load_dataset
from .ensemble import ResidualEnsemble
from .symmetry import X_PERM, X_SIGN, Y_PERM, Y_SIGN
from .torch_physics import euler_step, rhs


def window_indices(t, valid, horizon, stride, *, anchor=False):
    """Whole windows in one split, including final target; never cross a boundary."""
    t = np.asarray(t)
    if horizon < 2 or stride < 1:
        raise ValueError('horizon >=2 and stride >=1 required')
    blocks = np.floor((t-t[0])/.5).astype(int)
    mask = (blocks % 5 == 4) if anchor else (blocks % 5 != 4)
    mask = mask & valid & (t-t[0] >= .1)
    if not anchor:
        for edge in np.unique(blocks):
            if edge % 5 in (0, 4):
                mask &= np.abs(t-t[0]-edge*.5) > .02
    return np.array([i for i in range(0, len(t)-horizon, stride)
                     if mask[i:i+horizon+1].all()
                     and np.allclose(np.diff(t[i:i+horizon+1]), .001, rtol=0, atol=1e-9)], dtype=int)


def prepare_windows(model, *, horizon=100, stride=25):
    """Use only manifest TRAIN recordings; measured initial state + causal memory.

    No derivative labels are needed. Every original 1 ms actual input is retained.
    The 2 ms/5 ms decimated-label arrays must NEVER be used as the integrator grid.
    """
    _, manifest = load_dataset(model.metadata['dataset'])
    if manifest['samples_sha256'] != model.metadata['dataset_sha256']:
        raise ValueError('Checkpoint and dataset do not match')
    groups, anchors, provenance = [], [], []
    for episode in manifest['episodes']:
        if episode['split'] != 'train':
            continue
        capture = Path(episode['capture'])
        for name, digest in episode['source_sha256'].items():
            if sha(capture/name) != digest:
                raise ValueError('Original training capture changed')
        meta, data = load_capture(capture)
        mu, t = float(meta['scenario_mu_declared']), data['solver_time_s']
        configured = meta['contract']['settings'].get('MU_ROAD_CONSTANT')
        if configured is None or not np.isclose(mu, configured):
            raise ValueError('Actual road mu does not match recording')
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError('Actual monotone solver time required')
        states = np.column_stack([data[k] for k in STATE_NAMES])
        controls, memories, valid = [], [], []
        stream = LabelStream()
        for i, time in enumerate(t):
            u = controls_at(data, i)
            current, _ = stream.observe(time, states[i], u, mu,
                                        valid=data['return_code'][i] == 0 and not data['solver_done'][i])
            controls.append(np.r_[u.front_steer_rad, u.wheel_net_torque_nm])
            memories.append(current['features'][-2:])
            valid.append(current['valid'])
        controls, memories = np.asarray(controls), np.asarray(memories)
        fit = window_indices(t, np.asarray(valid), horizon, stride)
        hold = window_indices(t, np.asarray(valid), horizon, stride, anchor=True)
        if len(fit) < 10 or len(hold) < 3:
            raise ValueError('Need >=10 fit and >=3 internal anchor windows per training episode')
        # Fit-only normalization; include mirror to preserve parity in scales.
        fit_rows = np.unique((fit[:, None]+np.arange(horizon+1)).ravel())
        pop = states[fit_rows]
        reflected = pop[:, Y_PERM]*Y_SIGN
        scale = np.maximum(np.concatenate([pop, reflected]).std(0), X_FLOOR[:7])
        for starts, target in ((fit, groups), (hold, anchors)):
            idx = starts[:, None]+np.arange(horizon)
            target.append(dict(x=states[starts], memory=memories[starts], u=controls[idx],
                               truth=states[idx+1], mu=np.full(len(starts), mu),
                               scale=np.tile(scale, (len(starts), 1)),
                               block=episode['id']*1000000+np.floor(t[starts]/.1).astype(int)))
        provenance.append(dict(capture=str(capture), source_sha256=episode['source_sha256'],
                               fit_windows=len(fit), anchor_windows=len(hold),
                               state_scale_si=scale.tolist()))
        for name, digest in episode['source_sha256'].items():
            if sha(capture/name) != digest:
                raise ValueError('Training recording changed during window preparation')
        print('Windows:', capture.name, provenance[-1], flush=True)
    if not groups:
        raise ValueError('No training episodes in checkpoint manifest')
    concatenate = lambda entries: {k: np.concatenate([e[k] for e in entries]) for k in entries[0]}
    return concatenate(groups), concatenate(anchors), provenance


class TorchRollout:
    """K independent member trajectories, batch of B windows, exact 0.5 ms Euler.

    No teacher forcing inside a window. Physics and load memory stay in autograd.
    Fixed support and amplitude limits retained. Inter-member uncertainty gating
    is omitted during fitting (members follow different states); final NumPy
    ensemble-mean evaluation still uses ALL original gates.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.members = model.members.to(device=device, dtype=torch.float64)
        self.s = {k: torch.tensor(v, dtype=torch.float64, device=device) for k, v in model.scaler.items()}
        self.device = device

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def stacked_weights(self):
        return [(torch.stack([m[i].weight for m in self.members]),
                 torch.stack([m[i].bias for m in self.members])) for i in (0, 2, 4)]

    def residual(self, z, weights):
        n = z.shape[1]
        mirrored = z[..., X_PERM]*z.new_tensor(X_SIGN)
        value = torch.cat([z, mirrored], dim=1) if self.model.symmetry == 'mirror' else z
        value = (value-self.s['x_mean'])/self.s['x_scale']
        for i, (w, b) in enumerate(weights):
            value = torch.bmm(value, w.transpose(1, 2))+b[:, None, :]
            if i < 2:
                value = torch.tanh(value)
        value = value*self.s['y_scale']
        if self.model.symmetry == 'mirror':
            value = .5*(value[:, :n]+value[:, n:, Y_PERM]*value.new_tensor(Y_SIGN))
        outside = ((z < self.s['lower']) | (z > self.s['upper'])).any(-1)
        if self.model.symmetry == 'mirror':
            outside |= ((mirrored < self.s['lower']) | (mirrored > self.s['upper'])).any(-1)
        clipped = torch.maximum(torch.minimum(value, self.s['correction_limit']), -self.s['correction_limit'])
        return torch.where(outside[..., None], torch.zeros_like(clipped), clipped)

    def predict(self, x, memory, u, mu):
        # x [K,B,7]; u [K,B,H,6]. Only initial x and known controls accepted.
        weights, trajectory = self.stacked_weights(), []
        invalid = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for time in range(u.shape[2]):
            control = u[:, :, time]
            for _ in range(2):
                derivative, next_memory, bad = rhs(x, control, mu, memory)
                z = torch.cat([x, control, mu[..., None], memory], dim=-1)
                residual = self.residual(z, weights)
                residual = torch.where(bad[..., None], torch.zeros_like(residual), residual)
                x, clamp = euler_step(x, derivative+residual, .0005)
                memory = next_memory
                invalid = invalid | bad | clamp
            trajectory.append(x)
        return torch.stack(trajectory, dim=2), invalid

    def loss(self, batch):
        predicted, invalid = self.predict(batch['x'], batch['memory'], batch['u'], batch['mu'])
        error = (predicted-batch['truth'])/batch['scale'][:, :, None, :]
        return error.square().mean(), invalid


def batch_for(engine, windows, indices):
    return {key: engine.tensor(windows[key][indices]) for key in ('x', 'memory', 'u', 'truth', 'mu', 'scale')}


def anchor_loss(engine, windows, batch_size=16):
    total, count, invalid = 0., 0, 0
    with torch.no_grad():
        for start in range(0, len(windows['x']), batch_size):
            ix = np.arange(start, min(start+batch_size, len(windows['x'])))
            indices = np.tile(ix, (len(engine.members), 1))
            loss, bad = engine.loss(batch_for(engine, windows, indices))
            total += float(loss)*len(ix)
            count += len(ix)
            invalid += int(bad.sum())
    return {'normalized_state_mse': total/max(count, 1), 'invalid_member_windows': invalid,
            'windows': count, 'independent_validation': False}


def horizon_profile(checkpoint, output, *, horizon=400, stride=25):
    """Internal-anchor k-step diagnostic (SUBNET Sec.5.4), NOT independent testing.

    Report per-member free predictions against truth and nominal, at common
    initial windows. It is not the fully gated ensemble-mean rollout metric.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Horizon profile output must be new')
    model, _ = ResidualEnsemble.load(checkpoint)
    _, anchor, provenance = prepare_windows(model, horizon=horizon, stride=stride)
    engine = TorchRollout(model)
    nominal = deepcopy(model)
    with torch.no_grad():
        for parameter in nominal.members.parameters():
            parameter.zero_()
    base_engine = TorchRollout(nominal)
    steps = [k for k in (10, 20, 50, 100, 200, 400, 500) if k <= horizon]
    if horizon not in steps:
        steps.append(horizon)
    sums, counts, invalid = {}, {}, 0
    with torch.no_grad():
        for start in range(0, len(anchor['x']), 16):
            ix = np.arange(start, min(start+16, len(anchor['x'])))
            indices = np.tile(ix, (len(engine.members), 1))
            batch = batch_for(engine, anchor, indices)
            args = [batch[k] for k in ('x','memory','u','mu')]
            learned, bad = engine.predict(*args)
            base, bad_base = base_engine.predict(*args)
            invalid += int(bad.sum()+bad_base.sum())
            for slot, index in enumerate(ix):
                episode = int(anchor['block'][index]//1000000)
                for step in steps:
                    key = (episode, step)
                    errors = torch.stack([base[:,slot,step-1]-batch['truth'][:,slot,step-1],
                                          learned[:,slot,step-1]-batch['truth'][:,slot,step-1]])
                    sums[key] = sums.get(key, 0)+errors.square().sum(1).numpy()
                    counts[key] = counts.get(key, 0)+len(engine.members)
    rows = [dict(episode=k[0], horizon_s=k[1]*.001, member_window_count=counts[k],
                 nominal_rmse_si=np.sqrt(value[0]/counts[k]).tolist(),
                 learned_member_rmse_si=np.sqrt(value[1]/counts[k]).tolist()) for k,value in sorted(sums.items())]
    report = dict(checkpoint=str(Path(checkpoint).resolve()), source_sha256=sha(checkpoint),
                  max_horizon_s=horizon*.001, invalid_member_windows=invalid,
                  sources=provenance, rows=rows,
                  notes=['Training internal anchor only, not independent test.',
                         'Member trajectories, not deployed ensemble-mean trajectories.',
                         'k-step errors measured at horizon endpoints, no teacher forcing inside windows.',
                         'Initial memories are causal nominal estimates, not oracle hidden states.'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'horizon_profile.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Horizon profile:', output/'horizon_profile.json', flush=True)
    return report


def train_rollout(checkpoint, output, *, horizon=100, stride=25, updates=40, batch_size=8,
                  lr=1e-4, device='cpu', seed=42):
    if not 2 <= horizon <= 500 or updates < 1 or batch_size < 1 or stride < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('horizon 2..500, positive updates/batch/stride/lr required')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Rollout training output must be new')
    model, buffers = ResidualEnsemble.load(checkpoint)
    source_checkpoint_sha = sha(checkpoint)
    train, anchor, provenance = prepare_windows(model, horizon=horizon, stride=stride)
    engine = TorchRollout(model, device)
    initial = anchor_loss(engine, anchor)
    if initial['invalid_member_windows'] or not np.isfinite(initial['normalized_state_mse']):
        raise ValueError('Initial model has invalid internal-anchor rollouts')
    optimizer = torch.optim.Adam(engine.members.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    # Independent member bootstrap of temporal groups (overlap is not extra evidence).
    blocks = np.unique(train['block'])
    bootstraps = []
    for _ in engine.members:
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        bootstraps.append(np.concatenate([np.flatnonzero(train['block'] == b) for b in sampled]))
    best, best_score, best_update = deepcopy(engine.members.state_dict()), initial['normalized_state_mse'], 0
    history = []
    for update in range(1, updates+1):
        indices = np.stack([rng.choice(boot, size=batch_size, replace=True) for boot in bootstraps])
        optimizer.zero_grad(set_to_none=True)
        loss, invalid = engine.loss(batch_for(engine, train, indices))
        if invalid.any() or not torch.isfinite(loss):
            # Fail visibly, never silently omit a difficult window from the loss.
            raise FloatingPointError('Invalid training rollout; shorten horizon or review physics/support')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(engine.members.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': update, 'train_multistep_state_mse': float(loss.detach()), 'gradient_norm_before_clip': float(norm)}
        if update % 10 == 0 or update == updates:
            guard = anchor_loss(engine, anchor)
            row['internal_anchor'] = guard
            if not guard['invalid_member_windows'] and guard['normalized_state_mse'] < best_score:
                best, best_score, best_update = deepcopy(engine.members.state_dict()), guard['normalized_state_mse'], update
            print(row, flush=True)
        history.append(row)
    engine.members.load_state_dict(best)
    # Standard checkpoint/inference precision retained; conversion assessed explicitly.
    engine.members.cpu().float().eval()
    model.refresh()
    model.metadata = dict(model.metadata, training_objective='truncated_simulation_error',
                          rollout_source_checkpoint_sha256=source_checkpoint_sha,
                          rollout_horizon_s=horizon*.001, rollout_updates=updates, selected_update=best_update,
                          rollout_selection='internal_training_anchor_not_independent',
                          derivative_only_online_adaptation_allowed=False)
    report = dict(method='measured_state_truncated_simulation_error', horizon_s=horizon*.001,
                  integration_step_s=.0005, controls_sample_s=.001, updates=updates, batch_size=batch_size,
                  learning_rate=lr, device=device, seed=seed, source_checkpoint=str(Path(checkpoint).resolve()),
                  source_checkpoint_sha256=source_checkpoint_sha, training_sources=provenance,
                  initial_internal_anchor=initial, selected_update=best_update, selected_internal_anchor_mse=best_score,
                  history=history, optimization_uses_uncertainty_gate=False,
                  continuous_ensemble_rollout_required=True, automatic_deployment=False,
                  notes=['No validation or development samples used for gradients, normalization or checkpoint selection.',
                         'Initial state measured; initial load memory causal, then evolves with predicted states.',
                         'Targets are recorded states, not finite-difference derivative labels.',
                         'Each member trains its own trajectory; deployment uses ensemble mean and full gates.',
                         'Short-window improvement is not a long-rollout or closed-loop guarantee.',
                         'Not a replication of SUBNET or sparse off-white identification; no inherited theorem.'])
    if sha(checkpoint) != source_checkpoint_sha:
        raise RuntimeError('Source checkpoint changed during training')
    output.mkdir(parents=True, exist_ok=False)
    model.save(output/'ensemble.pt', **buffers)
    (output/'rollout_training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Rollout candidate:', output/'ensemble.pt', '| selected update:', best_update, flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon-steps', type=int, default=100)
    p.add_argument('--stride', type=int, default=25)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--profile-only', action='store_true', help='Only compute k-step errors on internal training anchor windows')
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    if args.profile_only:
        horizon_profile(args.checkpoint, args.output, horizon=args.horizon_steps, stride=args.stride)
    else:
        train_rollout(args.checkpoint, args.output, horizon=args.horizon_steps, stride=args.stride,
                      updates=args.updates, batch_size=args.batch_size, lr=args.lr, device=args.device, seed=args.seed)


if __name__ == '__main__':
    main()
