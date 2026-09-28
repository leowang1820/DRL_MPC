"""Exact reflection, straight preservation, compatibility and provenance tests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dynamics7dof.learning.data import build_dataset, feature_vector
from dynamics7dof.learning.dynamics import Learned7DOF
from dynamics7dof.learning.ensemble import (ResidualEnsemble, fit_scaler, make_members,
                                          episode_loss_weights, normalized_forward, optimize)
from dynamics7dof.learning.online import OnlineShadow
from dynamics7dof.learning.symmetry import (X_PERM, Y_PERM, fit_population,
                                          mirror_features, mirror_targets)
from dynamics7dof.learning.workflow import phase_metrics, source_role
from dynamics7dof.model import Controls, LoadMemory, Reference7DOF


class SymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='symmetry_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(73)
        self.x = self.rng.normal(size=(300, 16))
        self.y = self.x[:, :7]*.1  # This synthetic function obeys the reflection.

    def model(self):
        return ResidualEnsemble(make_members(3, 12), fit_scaler(*fit_population(self.x, self.y, 'mirror')),
                                uncertainty_limit=100., symmetry='mirror')

    def test_reflection_mapping_involution_and_no_mutation(self):
        x, y = np.arange(16.), np.arange(7.)
        np.testing.assert_equal(mirror_features(x), [0, -1, -2, 4, 3, 6, 5, -8, -7, 10, 9, 12, 11, 13, 14, -15])
        np.testing.assert_equal(mirror_targets(y), [0, -1, -2, 4, 3, 6, 5])
        np.testing.assert_array_equal(mirror_features(mirror_features(x)), x)
        np.testing.assert_array_equal(mirror_targets(mirror_targets(y)), y)
        np.testing.assert_array_equal(x, np.arange(16.))

    def test_physics_rhs_uses_same_reflection(self):
        state = np.array([11., .2, .03, 31., 32., 30., 33.])
        control = Controls([.02, .025], [10., 20., -30., -40.])
        z = feature_vector(state, control, .5, LoadMemory(.2, -.4))
        mz = mirror_features(z)
        model = Reference7DOF()
        a = model.evaluate(state, control, mu=.5, load_memory=LoadMemory(.2, -.4))
        b = model.evaluate(mz[:7], Controls(mz[7:9], mz[9:13]), mu=mz[13],
                           load_memory=LoadMemory(*mz[14:]))
        np.testing.assert_allclose(b.derivative, mirror_targets(a.derivative), atol=1e-10)

    def test_every_member_and_gate_are_equivariant(self):
        m = self.model()
        z = np.concatenate([self.x[:30], self.x[:3]*100])
        a, b = m.predict(z), m.predict(mirror_features(z))
        for key in ('members', 'mean', 'correction'):
            np.testing.assert_allclose(b[key], mirror_targets(a[key]), atol=1e-12)
        np.testing.assert_allclose(b['std'], a['std'][:, Y_PERM], atol=1e-12)
        np.testing.assert_allclose(b['uncertainty'], a['uncertainty'], atol=1e-12)
        np.testing.assert_array_equal(b['rejected'], a['rejected'])

    def test_straight_is_exact_for_every_member_and_longitudinal_is_retained(self):
        m = self.model()
        z = .5*(self.x+mirror_features(self.x))
        p = m.predict(z)
        np.testing.assert_array_equal(p['members'][..., 1:3], 0.)
        self.assertGreater(np.max(np.abs(p['mean'][:, 0])), 1e-5)
        np.testing.assert_array_equal(p['members'][..., 3], p['members'][..., 4])
        np.testing.assert_array_equal(p['members'][..., 5], p['members'][..., 6])

    def test_torch_projection_matches_numpy_and_backpropagates(self):
        m = self.model()
        x = self.x[:10]
        a = torch.tensor((x-m.scaler['x_mean'])/m.scaler['x_scale'], dtype=torch.float32)
        b = torch.tensor((mirror_features(x)-m.scaler['x_mean'])/m.scaler['x_scale'], dtype=torch.float32)
        scale = torch.tensor(m.scaler['y_scale'], dtype=torch.float32)
        predictions = [normalized_forward(member, a, b, scale, 'mirror')*scale for member in m.members]
        expected = np.array([p.detach().numpy() for p in predictions])
        np.testing.assert_allclose(expected, m.predict(x)['members'], atol=1e-7, rtol=1e-5)
        sum(p.square().mean() for p in predictions).backward()
        self.assertTrue(all(member[0].weight.grad.abs().sum() > 0 for member in m.members))

    def test_symmetry_training_reduces_error(self):
        m = self.model()
        before = np.mean((m.predict(self.x)['mean']-self.y)**2)
        optimize(m.members, m.scaler, self.x, self.y, epochs=15, batch_size=64, lr=.003, symmetry='mirror')
        m.refresh()
        self.assertLess(np.mean((m.predict(self.x)['mean']-self.y)**2), before*.5)
        np.testing.assert_allclose(m.predict(mirror_features(self.x))['mean'],
                                   mirror_targets(m.predict(self.x)['mean']), atol=1e-12)

    def test_episode_weighting_is_bounded_and_exposes_small_signal(self):
        y = np.r_[np.full((30, 7), .1), np.ones((30, 7))]
        ids = np.r_[np.zeros(30), np.ones(30)]
        w, report = episode_loss_weights(y, ids, np.ones(7))
        self.assertGreater(w[0, 1], w[-1, 1])
        self.assertTrue(np.all((w >= .5) & (w <= 4)))
        self.assertEqual(set(report), {'0', '1'})
        m = self.model()
        with self.assertRaisesRegex(ValueError, 'Loss weights'):
            optimize(m.members, m.scaler, self.x, self.y, sample_weights=np.zeros_like(self.y))

    def test_straight_augmented_integration_preserves_lateral_equilibrium(self):
        m = self.model()
        # Widen synthetic test support only; production bounds come from fit data.
        m.scaler['lower'][:] = -100
        m.scaler['upper'][:] = 100
        state = np.r_[10., 0., 0., np.full(4, 10/.353)]
        p = Learned7DOF(m).predict(state, Controls([0., 0.], [0.]*4), mu=.5,
                                   load_memory=LoadMemory(0., 0.), horizon_s=.05, substeps=100)
        np.testing.assert_allclose(p.state[1:3], 0., atol=1e-13)
        self.assertNotAlmostEqual(p.state[0], state[0], places=7)

    def test_checkpoint_preserves_mode_and_legacy_stays_legacy(self):
        m = self.model(); path = self.root/'mirror.pt'; m.save(path)
        loaded, _ = ResidualEnsemble.load(path)
        self.assertEqual(loaded.symmetry, 'mirror')
        np.testing.assert_array_equal(loaded.predict(self.x)['mean'], m.predict(self.x)['mean'])
        payload = torch.load(path, weights_only=True)
        payload['format'] = 'residual_ensemble_v1'; payload.pop('symmetry')
        torch.save(payload, self.root/'legacy.pt')
        legacy, _ = ResidualEnsemble.load(self.root/'legacy.pt')
        self.assertEqual(legacy.symmetry, 'none')
        ordinary = ResidualEnsemble(m.members, m.scaler, uncertainty_limit=m.uncertainty_limit)
        np.testing.assert_array_equal(legacy.predict(self.x)['mean'], ordinary.predict(self.x)['mean'])

    def test_bad_mode_and_nonsymmetric_limits_rejected(self):
        m = self.model()
        with self.assertRaises(ValueError):
            ResidualEnsemble(m.members, m.scaler, symmetry='unknown')
        m.scaler['correction_limit'][3] *= 2
        with self.assertRaisesRegex(ValueError, 'equal left/right'):
            ResidualEnsemble(m.members, m.scaler, symmetry='mirror')

    def test_online_candidate_keeps_projection_and_mirrored_bounds(self):
        m = self.model(); path = self.root/'online.pt'
        m.save(path, self.x[:30], self.y[:30], self.x[30:100], self.y[30:100])
        shadow = OnlineShadow(path, adapt=True)
        for i in range(220):
            x = self.x[i].copy(); x[1] += 10
            shadow.recent.append({'time': i*.005, 'available_at': i*.005+.005,
                                  'features': x, 'target': self.y[i]})
        with patch('dynamics7dof.learning.online.optimize', wraps=optimize) as train, \
                patch('dynamics7dof.learning.online.promotion_check', return_value=True):
            self.assertTrue(shadow.try_update())
            self.assertEqual(train.call_args.kwargs['symmetry'], 'mirror')
        self.assertEqual(shadow.model.symmetry, 'mirror')
        self.assertGreater(shadow.model.scaler['upper'][1], 10)
        self.assertLess(shadow.model.scaler['lower'][1], -10)
        np.testing.assert_allclose(shadow.model.predict(mirror_features(self.x))['mean'],
                                   mirror_targets(shadow.model.predict(self.x)['mean']), atol=1e-12)

    def test_development_provenance_includes_shared_startup_without_independence_claim(self):
        arrays = {'features': self.x, 'targets': self.y, 'derivative': self.y,
                  'time': np.arange(len(self.x))*.005}
        def prepare(path, stride):
            return {k: v.copy() for k, v in arrays.items()}, {
                'capture': str(path), 'physical_signature': str(path),
                'source_sha256': {'signals_si.csv': str(path)}}
        with patch('dynamics7dof.learning.data.prepare_capture', side_effect=prepare), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'near-identical'):
                build_dataset([self.root/'a'], [self.root/'b'], self.root/'held')
            report = build_dataset([self.root/'a'], [self.root/'b'], self.root/'dev', validation_role='development')
        self.assertEqual(report['validation_role'], 'development')
        self.assertEqual(report['near_identical_window_fraction'], 1.)
        m = self.model(); m.metadata['episodes'] = report['episodes']
        self.assertEqual(source_role(m, report['episodes'][1]), 'DEVELOPMENT_EPISODE_NOT_INDEPENDENT')

    def test_phase_report_exposes_straight_bias(self):
        t = np.arange(80)*.1
        truth = np.zeros((80, 7)); base = truth.copy(); aug = truth.copy()
        aug[t < 1, 2] = .01
        phases = phase_metrics(t, truth, base, aug, t >= .1,
                               dict(steer_start_s=1., steer_cycles=2., frequency_hz=.5))
        self.assertAlmostEqual(phases['before_steering']['augmented']['yaw_rate_rad_s']['rmse'], .01)
        self.assertEqual(phases['before_steering']['baseline']['yaw_rate_rad_s']['rmse'], 0.)
        self.assertEqual(phases['after_steering_plus_0p5s']['samples'], 25)


if __name__ == '__main__':
    unittest.main()
