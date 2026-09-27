"""Bounded, delayed-label online adaptation in SHADOW mode only.

No solver import, no commands returned, no Risk labels. Candidate weights never
replace active weights before a historical-anchor and recent-holdout check.
"""

from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..model import STATE_NAMES
from .data import LabelStream, X_FLOOR
from .ensemble import ResidualEnsemble, optimize, score_model


def promotion_check(old_anchor, new_anchor, old_recent, new_recent):
    anchor_ok = new_anchor['normalized_gated_rmse'] <= old_anchor['normalized_gated_rmse']*1.02+1e-6
    recent_ok = new_recent['normalized_gated_rmse'] < old_recent['normalized_gated_rmse']*.99
    # Aggregate gains cannot hide a large regression of a single physical channel.
    channel_ok = np.all(np.asarray(new_recent['gated_rmse_si']) <=
                        np.maximum(1.1*np.asarray(old_recent['gated_rmse_si']), 1e-5))
    return bool(anchor_ok and recent_ok and channel_ok)


class OnlineShadow:
    def __init__(self, checkpoint, *, adapt=False, update_every=200, capacity=4096):
        if update_every < 20 or capacity < 250:
            raise ValueError('Update spacing >=20 and buffer capacity >=250 required')
        self.model, self.buffers = ResidualEnsemble.load(checkpoint)
        if set(self.buffers) != {'anchor_x', 'anchor_y', 'history_x', 'history_y'}:
            raise ValueError('Online updates require a checkpoint with frozen anchor and replay buffers')
        self.stream = LabelStream()
        self.recent = deque(maxlen=capacity)
        self.pending = {}
        self.rows, self.events = deque(maxlen=10000), deque(maxlen=2000)
        self.sum_base_sq, self.sum_error_sq, self.rejected_count = np.zeros(7), np.zeros(7), 0
        self.adapt, self.update_every = adapt, update_every
        self.labels, self.version, self.attempt = 0, 0, 0

    def observe(self, time, state, controls, mu, valid=True):
        current, label = self.stream.observe(time, state, controls, mu, valid)
        # Prediction occurs before revealing a delayed target or performing any update.
        pred = self.model.predict(current['features'])
        self.pending[current['index']] = {'base': current['base'], 'correction': pred['correction'][0].copy(),
                                          'uncertainty': float(pred['uncertainty'][0]),
                                          'rejected': bool(pred['rejected'][0]), 'version': self.version}
        if label is not None and label['index'] % 5 == 0:
            past = self.pending[label['index']]
            self.recent.append(label)
            self.labels += 1
            error = past['correction']-label['target']
            self.sum_base_sq += label['target']**2
            self.sum_error_sq += error**2
            self.rejected_count += int(past['rejected'])
            self.rows.append({'time': label['time'], 'available_at': label['available_at'],
                              'version_at_prediction': past['version'], 'uncertainty': past['uncertainty'],
                              'rejected': past['rejected'], 'target': label['target'].tolist(),
                              'error_before_update': error.tolist()})
            if self.adapt and len(self.recent) >= 200 and self.labels % self.update_every == 0:
                self.try_update()
        cutoff = current['index']-2*self.stream.radius-1
        self.pending = {k: v for k, v in self.pending.items() if k > cutoff}
        return pred

    def try_update(self):
        self.attempt += 1
        recent = list(self.recent)
        split = int(.8*len(recent))
        # Future labels within the current centered derivative window are purged.
        boundary = recent[split]['time']
        fit = [s for s in recent[:split] if s['available_at'] < boundary-.01]
        hold = recent[split:]
        if len(fit) < 30 or len(hold) < 20:
            return False
        rx, ry = np.array([s['features'] for s in fit]), np.array([s['target'] for s in fit])
        hx, hy = np.array([s['features'] for s in hold]), np.array([s['target'] for s in hold])
        rng = np.random.default_rng(9000+self.attempt)
        old_idx = rng.integers(len(self.buffers['history_x']), size=len(rx))
        x = np.r_[rx, self.buffers['history_x'][old_idx]]
        y = np.r_[ry, self.buffers['history_y'][old_idx]]
        candidate = deepcopy(self.model)
        # Normalization/output bounds remain frozen; extend INPUT coverage only to
        # observed fit rows, subject to the promotion gate. Never use holdout bounds.
        candidate.scaler['lower'] = np.minimum(candidate.scaler['lower'], rx.min(0)-2*X_FLOOR)
        candidate.scaler['upper'] = np.maximum(candidate.scaler['upper'], rx.max(0)+2*X_FLOOR)
        try:
            optimize(candidate.members, candidate.scaler, x, y, epochs=3, lr=1e-4,
                     seed=9000+self.attempt, device='cpu')
            candidate.refresh()
            anchor = self.buffers['anchor_x'], self.buffers['anchor_y']
            old_a, new_a = score_model(self.model, *anchor), score_model(candidate, *anchor)
            old_r, new_r = score_model(self.model, hx, hy), score_model(candidate, hx, hy)
            accepted = promotion_check(old_a, new_a, old_r, new_r)
            if accepted:
                self.model = candidate
                self.version += 1
            event = {'attempt': self.attempt, 'accepted': accepted, 'version': self.version,
                     'available_at': recent[-1]['available_at'], 'fit_rows': len(fit), 'holdout_rows': len(hold),
                     'old_anchor': old_a, 'new_anchor': new_a, 'old_recent': old_r, 'new_recent': new_r}
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            event = {'attempt': self.attempt, 'accepted': False, 'version': self.version,
                     'error': str(exc)}
        self.events.append(event)
        return event['accepted']

    def save(self, output, context=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        report = {'mode': 'shadow_only_no_control', 'adapt_enabled': self.adapt,
                  'version': self.version, 'labels': self.labels, 'events': list(self.events),
                  'logged_prediction_tail_rows': len(self.rows), 'prediction_log_capacity': self.rows.maxlen,
                  'context': context or {}, 'uncertainty_is_calibrated': False,
                  'notes': ['Errors saved at prediction time BEFORE delayed targets/updates.',
                            'Frozen replay anchor is an adaptation guard, not independent validation.',
                            'No hard real-time guarantee; wall time may increase.',
                            'No action commands or Risk labels are generated.']}
        if self.rows:
            report['prequential_baseline_rmse_si'] = dict(zip(STATE_NAMES, np.sqrt(self.sum_base_sq/self.labels).tolist()))
            report['prequential_corrected_rmse_si'] = dict(zip(STATE_NAMES, np.sqrt(self.sum_error_sq/self.labels).tolist()))
            report['rejection_fraction'] = self.rejected_count/self.labels
        (output/'online_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        with (output/'predictions.jsonl').open('x', encoding='utf-8') as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.model.metadata = dict(self.model.metadata, online_versions=self.version,
                                   online_context=context or {}, deployment='shadow_only')
        self.model.save(output/'shadow_ensemble.pt', **self.buffers)
        return report
