"""Residual contract/causality/gates/checkpoint tests. No CarSim DLL."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF
from dynamics7dof.learning.data import CONTRACT, LabelStream, feature_vector, fingerprint, load_dataset
from dynamics7dof.learning.dynamics import Learned7DOF
from dynamics7dof.learning.ensemble import (ResidualEnsemble, anchor_masks, fit_scaler, make_members,
                                          optimize, score_model, train_dataset)
from dynamics7dof.learning.online import OnlineShadow, promotion_check
from dynamics7dof.learning.workflow import source_role
from dynamics7dof.physics_check import sha


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='learning_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(size=(300, 16))
        self.y = self.x[:, :7]*.1
        self.state = np.r_[10., 0., 0., np.full(4, 10/.353)]
        self.controls = Controls([0., 0.], [0.]*4)

    def model(self, zero=False):
        model = ResidualEnsemble(make_members(3, 8, 5), fit_scaler(self.x, self.y))
        if zero:
            with torch.no_grad():
                for p in model.members.parameters():
                    p.zero_()
            model.refresh()
        return model

    def checkpoint(self, model=None):
        path = self.root/'model.pt'
        (model or self.model()).save(path, self.x[:30], self.y[:30], self.x[30:100], self.y[30:100])
        return path

    def test_feature_order_and_units(self):
        c = Controls([.1, .2], [1, 2, -3, -4])
        z = feature_vector(self.state, c, .5, LoadMemory(6, 7))
        self.assertEqual(z.shape, (16,))
        np.testing.assert_equal(z[7:], [.1, .2, 1, 2, -3, -4, .5, 6, 7])

    def test_delayed_quadratic_label_is_centered_not_arrival_time(self):
        stream = LabelStream(warmup=.01)
        labels = []
        for t in np.arange(40)*.001:
            x = self.state.copy(); x[0] = 10+.1*t+.2*t*t; x[3:] = x[0]/.353
            _, label = stream.observe(t, x, self.controls, .5)
            if label:
                labels.append(label)
        self.assertGreater(len(labels), 10)
        for label in labels:
            self.assertAlmostEqual(label['available_at']-label['time'], .005)
            self.assertAlmostEqual(label['derivative'][0], .1+.4*label['time'], places=8)
            self.assertAlmostEqual(label['features'][0], 10+.1*label['time']+.2*label['time']**2)
            np.testing.assert_allclose(label['target']+label['base'], label['derivative'])

    def test_no_future_samples_in_features(self):
        a, b = LabelStream(warmup=0), LabelStream(warmup=0)
        za = zb = None
        for i in range(26):
            t = i*.001
            x = self.state.copy()
            _, la = a.observe(t, x, self.controls, .5)
            if i > 20:
                x[0] += (i-20)*.001
            _, lb = b.observe(t, x, self.controls, .5)
            if i == 25:
                za, zb = la, lb
        np.testing.assert_array_equal(za['features'], zb['features'])
        self.assertNotAlmostEqual(za['derivative'][0], zb['derivative'][0])

    def test_stream_rejects_time_reset_and_gap(self):
        s = LabelStream()
        s.observe(0, self.state, self.controls, .5)
        with self.assertRaises(ValueError):
            s.observe(0, self.state, self.controls, .5)
        with self.assertRaises(ValueError):
            s.observe(.01, self.state, self.controls, .5)

    def test_invalid_window_has_no_label(self):
        s = LabelStream(warmup=0)
        for i in range(11):
            _, label = s.observe(i*.001, self.state, self.controls, .5, valid=i != 5)
        self.assertIsNone(label)

    def test_anchor_is_disjoint_and_purged(self):
        t = np.arange(600)*.005
        fit, anchor = anchor_masks(t)
        self.assertFalse(np.any(fit & anchor))
        self.assertFalse(fit[np.argmin(abs(t-2))])
        self.assertTrue(anchor[np.argmin(abs(t-2.2))])

    def test_scaling_floors_and_nonfinite_rejection(self):
        s = fit_scaler(np.zeros((10, 16)), np.zeros((10, 7)))
        self.assertTrue(np.all(s['x_scale'] > 0))
        self.assertTrue(np.all(s['y_scale'] > 0))
        with self.assertRaises(ValueError):
            fit_scaler(np.full((10, 16), np.nan), np.zeros((10, 7)))

    def test_bootstrap_members_are_independent(self):
        members = make_members(3, 8, 5)
        self.assertFalse(torch.equal(members[0][0].weight, members[1][0].weight))

    def test_numpy_matches_torch_members(self):
        m = self.model()
        x = self.x[:4]
        p = m.predict(x)
        with torch.no_grad():
            z = torch.tensor((x-m.scaler['x_mean'])/m.scaler['x_scale'], dtype=torch.float32)
            expected = np.array([member(z).numpy()*m.scaler['y_scale'] for member in m.members])
        np.testing.assert_allclose(p['members'], expected, rtol=1e-5, atol=1e-7)

    def test_ood_falls_back_and_reports_raw_prediction(self):
        m = self.model()
        p = m.predict(np.full(16, 1e6))
        self.assertTrue(p['ood'][0])
        np.testing.assert_array_equal(p['correction'], 0)
        self.assertTrue(np.all(np.isfinite(p['mean'])))

    def test_high_disagreement_rejects(self):
        m = self.model(zero=True)
        with torch.no_grad():
            for i, member in enumerate(m.members):
                member[-1].bias.fill_(i*10)
        m.refresh()
        p = m.predict(self.x[:3])
        self.assertTrue(np.all(p['rejected']))
        np.testing.assert_equal(p['correction'], 0)

    def test_zero_residual_exactly_recovers_reference_integrator(self):
        m = Learned7DOF(self.model(zero=True))
        c = Controls([.01, .01], [20]*4)
        kwargs = dict(mu=.5, load_memory=LoadMemory(0, 0), horizon_s=.05, substeps=100)
        a = Reference7DOF().predict(self.state, c, **kwargs)
        b = m.predict(self.state, c, **kwargs)
        np.testing.assert_allclose(a.state, b.state, rtol=0, atol=1e-12)
        self.assertEqual(a.load_memory, b.load_memory)

    def test_checkpoint_roundtrip_and_buffers(self):
        m = self.model()
        path = self.checkpoint(m)
        loaded, buffers = ResidualEnsemble.load(path)
        np.testing.assert_allclose(m.predict(self.x[:5])['mean'], loaded.predict(self.x[:5])['mean'])
        self.assertEqual(buffers['anchor_x'].shape, (30, 16))
        with self.assertRaises(FileExistsError):
            m.save(path)

    def test_checkpoint_contract_mismatch_fails(self):
        path = self.checkpoint()
        payload = torch.load(path, weights_only=True)
        payload['fingerprint'] = 'different model'
        torch.save(payload, self.root/'wrong.pt')
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            ResidualEnsemble.load(self.root/'wrong.pt')

    def test_training_smoke_reduces_synthetic_loss(self):
        m = self.model()
        before = np.mean((m.predict(self.x)['mean']-self.y)**2)
        optimize(m.members, m.scaler, self.x, self.y, epochs=15, batch_size=64, lr=.003, seed=2)
        m.refresh()
        after = np.mean((m.predict(self.x)['mean']-self.y)**2)
        self.assertLess(after, before)

    def test_promotion_needs_anchor_and_all_channels(self):
        old = dict(normalized_gated_rmse=1., gated_rmse_si=[1.]*7)
        better = dict(normalized_gated_rmse=.8, gated_rmse_si=[.8]*7)
        worse = dict(normalized_gated_rmse=2., gated_rmse_si=[2.]*7)
        self.assertTrue(promotion_check(old, old, old, better))
        self.assertFalse(promotion_check(old, worse, old, better))
        hidden = dict(better, gated_rmse_si=[2, .1, .1, .1, .1, .1, .1])
        self.assertFalse(promotion_check(old, old, old, hidden))
        self.assertFalse(promotion_check(old, hidden, old, better))

    def test_online_default_does_not_train_and_records_preupdate_version(self):
        shadow = OnlineShadow(self.checkpoint(), adapt=False)
        for i in range(400):
            shadow.observe(i*.001, self.state, self.controls, .5)
        self.assertGreater(shadow.labels, 30)
        self.assertEqual(shadow.version, 0)
        self.assertEqual(len(shadow.events), 0)
        self.assertLessEqual(len(shadow.pending), 12)
        self.assertTrue(all(row['available_at'] > row['time'] for row in shadow.rows))
        self.assertTrue(all(row['version_at_prediction'] == 0 for row in shadow.rows))
        report = shadow.save(self.root/'online')
        self.assertFalse(report['adapt_enabled'])

    def test_dataset_checksum_blocks_tamper(self):
        root = self.root/'dataset'; root.mkdir()
        np.savez_compressed(root/'samples.npz', train_features=self.x)
        m = {'contract_fingerprint': fingerprint(), 'samples_sha256': 'wrong'}
        (root/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_dataset(root)

    def test_adapted_episode_no_longer_claimed_held_out(self):
        m = self.model()
        info = {'capture': str(self.root/'capture'), 'source_sha256': {'signals_si.csv': 'abc'}}
        m.metadata = {'episodes': [dict(info, split='validation')]}
        self.assertEqual(source_role(m, info), 'HELD_OUT_EPISODE')
        m.metadata['online_adaptation_sources'] = [dict(info)]
        self.assertEqual(source_role(m, info), 'ADAPTATION_EPISODE_NOT_HELD_OUT')
        copied = dict(info, capture=str(self.root/'copied_capture'))
        self.assertEqual(source_role(m, copied), 'ADAPTATION_EPISODE_NOT_HELD_OUT')

    def test_live_observer_leaves_applied_commands_unchanged(self):
        from contextlib import redirect_stdout
        import io
        from dynamics7dof.open_loop import Excitation, capture_open_loop, parser
        from dynamics7dof.signals import EXPORT_NAMES, IMPORT_NAMES, inspect_configuration
        from dynamics7dof.tests.test_open_loop import FakeEnv
        sim, par = self.root/'simfile.sim', self.root/'Run_all.par'
        sim.write_text(f'INPUT {par}\nPORTS_IMP 1,3\nPORTS_EXP 1,25\n')
        par.write_text('\n'.join([*(f'IMPORT {k} REPLACE 0' for k in IMPORT_NAMES),
                                  *(f'EXPORT {k}' for k in EXPORT_NAMES), 'MU_ROAD_CONSTANT 0.5', 'TSTEP 0.001']))
        args = parser().parse_args(['--sim', str(sim), '--mu', '.5', '--seconds', '.2'])
        plain, observed = FakeEnv(), FakeEnv()
        with redirect_stdout(io.StringIO()):
            capture_open_loop(args, inspect_configuration(sim), Excitation(), self.root/'plain', env_factory=lambda _: plain)
            args.learning_model = self.checkpoint()
            capture_open_loop(args, inspect_configuration(sim), Excitation(), self.root/'observed', env_factory=lambda _: observed)
        self.assertEqual(plain.calls, observed.calls)
        self.assertTrue(observed.closed)
        report = json.loads((self.root/'learning_shadow'/'online_report.json').read_text())
        self.assertTrue(report['context']['controls_unchanged_by_learning'])
        self.assertFalse(report['adapt_enabled'])

    def test_different_online_version_is_retained_when_loading(self):
        m = self.model(); m.metadata['online_versions'] = 3
        shadow = OnlineShadow(self.checkpoint(m))
        self.assertEqual(shadow.version, 3)


if __name__ == '__main__':
    unittest.main()
