"""No real DLL: schedules, lifecycle, continuous replay, metrics and SVG output."""

from contextlib import redirect_stdout
import ctypes
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from dynamics7dof.comparison import error_metrics, replay_states, svg_panels, write_comparison
from dynamics7dof.model import Controls, LoadMemory, Reference7DOF, STATE_NAMES
from dynamics7dof.open_loop import Excitation, capture_open_loop, guard_reason, main, parser
from dynamics7dof.signals import EXPORT_NAMES, IMPORT_NAMES, inspect_configuration


class Getter:
    def __init__(self, clock):
        self.clock = clock

    def __call__(self, name):
        return ctypes.pointer(self.clock)


class Solver:
    def __init__(self):
        self.clock = ctypes.c_double(0)
        self.dll_handle = self
        self.vs_get_var_ptr = Getter(self.clock)
        self.closed = False

    def integrate_io_inplace(self, t, *args):
        self.clock.value = t+.0005
        return 0

    def terminate_run(self, t):
        self.closed = True
        self.clock.value = float('nan')


class FakeEnv:
    def __init__(self, sim=None, terminal_at=None, failure_at=None):
        self.solver = Solver()
        self.n_import, self.n_export = 3, 25
        self.t_current, self.t_step = 0., .001
        self.config = dict(n_import=3, n_export=25, t_start=0., t_step=.001, t_stop=40.)
        self.terminal_at, self.failure_at = terminal_at, failure_at
        self.calls = []
        self.closed = False
        self.done = False

    def exports(self):
        v = np.zeros(25)
        v[0] = max(0., self.t_current-.0005)*10
        v[6] = 36
        v[7] = v[0]
        v[9:13] = (10/.353)*60/(2*np.pi)
        if self.calls:
            v[13:15] = self.calls[-1][2]/16
        return v

    def reset(self):
        return self.exports()

    def control_step(self, action, inner_steps):
        if self.done:
            raise RuntimeError('stepped after done')
        self.calls.append(list(action))
        if len(self.calls) == self.failure_at:
            raise RuntimeError('synthetic solver failure')
        self.solver.integrate_io_inplace(self.t_current)
        self.t_current += self.t_step
        obs = self.exports()
        if len(self.calls) == self.terminal_at:
            self.done = True
            self.solver.terminate_run(self.t_current)
        return obs, 0., self.done, {'return_code': 0}

    def close(self):
        self.closed = True
        if not self.done:
            self.solver.terminate_run(self.t_current)
        self.done = True


class OpenLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='open_loop_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sim, self.par = self.root/'simfile.sim', self.root/'Run_all.par'
        self.sim.write_text(f'INPUT {self.par}\nPORTS_IMP 1,3\nPORTS_EXP 1,25\n', encoding='utf-8')
        self.par.write_text('\n'.join([*(f'IMPORT {k} REPLACE 0' for k in IMPORT_NAMES),
                                       *(f'EXPORT {k}' for k in EXPORT_NAMES),
                                       'MU_ROAD_CONSTANT 0.5', 'TSTEP 0.001']), encoding='utf-8')
        self.args = parser().parse_args(['--sim', str(self.sim), '--mu', '.5', '--seconds', '.2',
                                        '--metric-warmup-s', '.01'])

    def make_capture(self, env=None):
        env = env or FakeEnv()
        with redirect_stdout(io.StringIO()):
            status, info = capture_open_loop(self.args, inspect_configuration(self.sim), Excitation(),
                                            self.root/'capture', env_factory=lambda sim: env)
        return env, status, info

    def synthetic(self):
        t = np.r_[0., .0005+np.arange(200)*.001]
        x = np.tile(np.r_[10., 0., 0., np.full(4, 10/.353)], (len(t), 1))
        data = {k: x[:, i].copy() for i, k in enumerate(STATE_NAMES)}
        data.update(solver_time_s=t, wrapper_time_s=np.arange(len(t))*.001,
                    return_code=np.zeros(len(t)), solver_done=np.zeros(len(t)),
                    front_steer_fl_rad=np.zeros(len(t)), front_steer_fr_rad=np.zeros(len(t)))
        for w in ('fl', 'fr', 'rl', 'rr'):
            data[f'drive_{w}_nm'] = np.zeros(len(t))
            data[f'brake_{w}_nm'] = np.zeros(len(t))
        return data

    def test_sine_schedule_and_wheel_ratio(self):
        s = Excitation()
        self.assertEqual(s.front_angle(s.steer_start_s), 0)
        self.assertEqual(s.front_angle(s.steer_start_s+s.steer_cycles/s.frequency_hz), 0)
        t = s.steer_start_s+1/(4*s.frequency_hz)
        self.assertAlmostEqual(s.front_angle(t), s.front_amplitude_deg)
        self.assertAlmostEqual(s.command(t)[2], s.front_amplitude_deg*16)
        self.assertEqual(s.command(0)[:2], [.12, 0])
        self.assertEqual(s.command(4)[:2], [0, 0])
        self.assertEqual(s.command(5.5)[:2], [0, .2])
        self.assertEqual(s.command(6)[:2], [0, 0])

    def test_invalid_schedule_rejected(self):
        for kwargs in ({'frequency_hz': 0}, {'throttle': 2}, {'brake_mpa': -1},
                       {'drive_until_s': 6}, {'ramp_s': 100}, {'front_amplitude_deg': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Excitation(**kwargs)

    def test_default_mode_never_loads_simulator_or_creates_output(self):
        with patch('dynamics7dof.open_loop.capture_open_loop') as run, redirect_stdout(io.StringIO()):
            main(['--sim', str(self.sim), '--output-dir', str(self.root/'unused')])
        run.assert_not_called()
        self.assertFalse((self.root/'unused').exists())

    def test_missing_mu_refuses_live_run(self):
        with patch('dynamics7dof.open_loop.capture_open_loop') as run, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--run', '--sim', str(self.sim)])
        run.assert_not_called()

    def test_capture_lifecycle_and_observed_time(self):
        env, status, _ = self.make_capture()
        self.assertEqual(status, 'captured_requested_duration')
        self.assertTrue(env.closed)
        meta = json.loads((self.root/'capture'/'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['rows'], 201)
        self.assertEqual(meta['solver_clock_rows'], 201)
        self.assertAlmostEqual(meta['last_solver_time_s'], .1995)
        self.assertEqual(meta['control_updates'], 200)

    def test_schedule_uses_preintegration_actual_time(self):
        env = FakeEnv()
        s = Excitation(steer_start_s=0, ramp_s=0)
        with redirect_stdout(io.StringIO()):
            capture_open_loop(self.args, inspect_configuration(self.sim), s, self.root/'capture',
                              env_factory=lambda sim: env)
        self.assertEqual(env.calls[1], s.command(.0005))
        self.assertNotEqual(env.calls[1], s.command(.001))

    def test_terminal_snapshot_saved_and_no_posttermination_read(self):
        env, status, _ = self.make_capture(FakeEnv(terminal_at=20))
        self.assertEqual(status, 'episode_ended')
        self.assertEqual(len(env.calls), 20)
        meta = json.loads((self.root/'capture'/'metadata.json').read_text())
        self.assertTrue(np.isfinite(meta['last_solver_time_s']))

    def test_failure_preserves_capture_and_closes(self):
        env = FakeEnv(failure_at=10)
        with self.assertRaisesRegex(RuntimeError, 'synthetic solver failure'):
            self.make_capture(env)
        self.assertTrue(env.closed)
        meta = json.loads((self.root/'capture'/'metadata.json').read_text())
        self.assertEqual(meta['status'], 'failed')
        self.assertEqual(meta['rows'], 10)

    def test_guard_transforms_to_initial_heading(self):
        initial = dict(vx_m_s=10, vy_m_s=0, yaw_rate_rad_s=0, x_origin_m=0, y_origin_m=0, yaw_rad=np.pi/2)
        later = dict(initial, y_origin_m=20)
        self.assertIsNone(guard_reason(later, initial, self.args))
        later['x_origin_m'] = 10
        self.assertEqual(guard_reason(later, initial, self.args), 'lateral_displacement_limit')

    def test_free_rollout_does_not_reset_to_measurement(self):
        d = self.synthetic(); d['vy_m_s'] = .2*d['solver_time_s']
        r = replay_states(d, .5)
        np.testing.assert_allclose(r['predicted'][:, 1], 0, atol=1e-10)
        self.assertGreater(r['one_step'][-1, 1], .03)
        self.assertGreater(abs(r['predicted'][-1, 1]-d['vy_m_s'][-1]), .03)

    def test_generated_dynamics_free_rollout_matches_with_memory(self):
        d = self.synthetic()
        d['front_steer_fl_rad'].fill(.01); d['front_steer_fr_rad'].fill(.01)
        x = np.array([d[k][0] for k in STATE_NAMES]); memory = LoadMemory(0, 0)
        m = Reference7DOF(); c = Controls([.01, .01], [0]*4)
        for i, dt in enumerate(np.diff(d['solver_time_s'])):
            q = m.predict(x, c, mu=.5, load_memory=memory, horizon_s=float(dt),
                          substeps=max(1, int(np.ceil(dt/.0005-1e-8))))
            x, memory = q.state, q.load_memory
            for k, v in zip(STATE_NAMES, x):
                d[k][i+1] = v
        r = replay_states(d, .5)
        np.testing.assert_allclose(r['predicted'], r['truth'], atol=1e-11)

    def test_legacy_and_partial_clocks(self):
        d = self.synthetic(); d['solver_time_s'][5] = np.nan
        with self.assertRaises(ValueError):
            replay_states(d, .5, allow_nominal=True)
        del d['solver_time_s']
        with self.assertRaises(ValueError):
            replay_states(d, .5)
        self.assertEqual(replay_states(d, .5, allow_nominal=True)['time_basis'], 'NOMINAL_TIME_UNVERIFIED')

    def test_terminal_samples_excluded(self):
        d = self.synthetic(); d['solver_done'][30] = 1
        r = replay_states(d, .5)
        self.assertTrue(np.all(np.isnan(r['predicted'][30:])))
        self.assertEqual(r['rollout_stop']['reason'], 'solver_terminal_or_error')

    def test_metrics_and_zero_variance(self):
        result = error_metrics([1, 1, 1], [2, 1, 0])
        self.assertAlmostEqual(result['rmse'], np.sqrt(2/3))
        self.assertIsNone(result['nrmse_by_truth_std'])
        self.assertEqual(error_metrics([1], [np.nan])['samples'], 0)

    def test_svg_is_valid_and_omits_nan_segments(self):
        svg = svg_panels(np.arange(4), [('a & b', [('x', np.array([1., np.nan, 2., 3.]))])], '<test>', 'demo')
        ET.fromstring(svg)
        self.assertNotIn('nan,', svg)
        self.assertIn('&lt;test&gt;', svg)

    def test_full_fake_capture_report_is_read_only_and_rejects_overwrite(self):
        self.make_capture()
        files = {p.name: p.read_bytes() for p in (self.root/'capture').iterdir()}
        with redirect_stdout(io.StringIO()):
            report = write_comparison(self.root/'capture', self.root/'report', warmup_s=.01)
        self.assertEqual(report['rollout_coverage_fraction'], 1)
        for name in ('states.svg', 'errors.svg', 'inputs.svg'):
            ET.parse(self.root/'report'/name)
        self.assertTrue((self.root/'report'/'comparison.html').is_file())
        self.assertEqual(files, {p.name: p.read_bytes() for p in (self.root/'capture').iterdir()})
        with self.assertRaises(ValueError):
            write_comparison(self.root/'capture', self.root/'report')
        with self.assertRaises(ValueError):
            write_comparison(self.root/'capture', self.root/'capture'/'nested_report')


if __name__ == '__main__':
    unittest.main()
