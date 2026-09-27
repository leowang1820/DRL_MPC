"""Offline regression tests; synthetic DLL-shaped objects, no CarSim process."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import ctypes
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dynamics7dof.audit import audit_capture, main as audit_main
from dynamics7dof.collect import CaptureWriter, IntegrationRecorder
from dynamics7dof.signals import EXPORT_NAMES, IMPORT_NAMES, NATIVE_UNITS
from dynamics7dof.timing import ObservedSolver, SolverClock, TIME_SOURCE


class Getter:
    def __init__(self, value):
        self.value = value
        self.args = []

    def __call__(self, key):
        self.args.append(key)
        return ctypes.pointer(self.value)


class DummyDLL:
    def __init__(self):
        self.time = ctypes.c_double(0)
        self.vs_get_var_ptr = Getter(self.time)


class DummySolver:
    def __init__(self):
        self.dll_handle = DummyDLL()
        self.calls = []
        self.terminated = False

    def integrate_io_inplace(self, t, imports, exports, n):
        self.calls.append((t, tuple(imports), n))
        self.dll_handle.time.value = t + 0.0005
        return 0

    def terminate_run(self, t):
        self.terminated = True
        # Simulate invalidating internal state; observer must never read it now.
        self.dll_handle.time.value = float('nan')
        return 0


class DummyEnv:
    def __init__(self, terminal_at=None):
        self.solver = DummySolver()
        self.config = dict(n_import=3, n_export=25, t_step=0.001, t_start=0, t_stop=40)
        self.n_import, self.n_export = 3, 25
        self.t_step, self.t_current = 0.001, 0.0
        self.done = False
        self.terminal_at = terminal_at
        self.count = 0

    def exports(self):
        a = np.zeros(25)
        a[0] = max(0, self.t_current-0.0005)*10
        a[6] = 36
        a[9:13] = 270
        a[21:] = [-100, -100, -50, -50]
        return tuple(a)

    def reset(self):
        return self.exports()

    def control_step(self, action, inner_steps):
        for _ in range(inner_steps):
            self.solver.integrate_io_inplace(self.t_current, action, None, 25)
            self.t_current += self.t_step
            self.count += 1
            if self.count == self.terminal_at:
                self.done = True
                self.solver.terminate_run(self.t_current)
                break
        return self.exports(), 0, self.done, {'return_code': 0}

    def close(self):
        if not self.done:
            self.solver.terminate_run(self.t_current)
        self.done = True


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dynamics7dof_audit_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def capture(self, terminal_at=None):
        manifest = {
            'format': 'carsim_25_signal_capture_v2',
            'contract': {'imports': list(IMPORT_NAMES), 'exports': list(EXPORT_NAMES),
                         'native_units': list(NATIVE_UNITS), 'settings': {'OPT_INT_METHOD': 2}},
            'control_period_s': 0.05, 'reference_path': str(self.root/'missing.par'),
        }
        sink = CaptureWriter(self.root/'capture', manifest)
        self.addCleanup(sink.close_files)
        env = DummyEnv(terminal_at)
        original = env.solver
        rec = IntegrationRecorder(env, sink, require_solver_time=True)
        rec.reset()
        rec.control_step([0, 1, 0], 50)
        rec.close()
        self.assertIs(env.solver, original)
        sink.finish('episode_ended' if terminal_at else 'captured_requested_duration', {'control_updates': 1})
        return sink.output

    def edit_csv(self, root, name, change):
        p = root/name
        with p.open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            fields, rows = reader.fieldnames, list(reader)
        change(fields, rows)
        with p.open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_v2_clock_half_first_step_and_no_auto_training_readiness(self):
        root = self.capture()
        result = audit_capture(root)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['integrity'], 'PASS')
        self.assertEqual(result['timing']['clock_rows'], 51)
        self.assertAlmostEqual(result['timing']['first_interval_s'], 0.0005)
        self.assertAlmostEqual(result['timing']['wrapper_minus_solver_s']['mean'], 0.0005)
        self.assertEqual(result['torques']['forward_braking_rows'], 50)
        self.assertEqual(result['torques']['wheels']['fl']['negative_brake_fraction'], 1)
        self.assertFalse(result['ready_for_residual_training'])

    def test_terminal_clock_read_occurs_before_termination(self):
        root = self.capture(terminal_at=7)
        result = audit_capture(root)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['timing']['clock_rows'], 8)
        self.assertAlmostEqual(result['timing']['subsequent_dt_s']['mean'], 0.001)

    def test_legacy_clock_is_not_invented(self):
        root = self.capture()
        def remove(fields, rows):
            for key in ('solver_time_before_s', 'solver_time_s', 'solver_time_source'):
                fields.remove(key)
                for row in rows:
                    del row[key]
        for name in ('raw_exports.csv', 'signals_si.csv'):
            self.edit_csv(root, name, remove)
        p = root/'metadata.json'
        m = json.loads(p.read_text())
        m['format'] = 'carsim_25_signal_capture_v1'
        p.write_text(json.dumps(m))
        result = audit_capture(root)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['timing']['diagnostic_time_basis'], 'wrapper_nominal_unverified')
        self.assertEqual(result['timing']['clock_rows'], 0)
        self.assertNotIn('first_interval_s', result['timing'])

    def test_conversion_corruption_fails_and_original_not_modified(self):
        root = self.capture()
        self.edit_csv(root, 'signals_si.csv', lambda f, rows: rows[10].update(vx_m_s='99'))
        before = {p.name:p.read_bytes() for p in root.iterdir()}
        result = audit_capture(root)
        self.assertEqual(result['integrity'], 'FAIL')
        self.assertTrue(any('conversion' in s for s in result['errors']))
        self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})

    def test_missing_sample_is_detected(self):
        root = self.capture()
        for name in ('raw_exports.csv', 'signals_si.csv'):
            self.edit_csv(root, name, lambda f, rows: rows.pop(10))
        result = audit_capture(root)
        self.assertEqual(result['integrity'], 'FAIL')
        self.assertTrue(any('indices' in s for s in result['errors']))

    def test_nonfinite_input_fails_closed(self):
        root = self.capture()
        self.edit_csv(root, 'signals_si.csv', lambda f, rows: rows[10].update(vy_m_s='nan'))
        result = audit_capture(root)
        self.assertEqual(result['integrity'], 'FAIL')
        self.assertTrue(any('non-finite' in s for s in result['errors']))

    def test_stuck_solver_clock_fails(self):
        root = self.capture()
        for name in ('raw_exports.csv', 'signals_si.csv'):
            self.edit_csv(root, name, lambda f, rows: rows[10].update(solver_time_s=rows[9]['solver_time_s']))
        result = audit_capture(root)
        self.assertEqual(result['integrity'], 'FAIL')
        self.assertTrue(any('increasing' in s for s in result['errors']))

    def test_partial_clock_is_explicit_not_synthesized(self):
        root = self.capture()
        for name in ('raw_exports.csv', 'signals_si.csv'):
            self.edit_csv(root, name, lambda f, rows: rows[10].update(solver_time_s='', solver_time_source=''))
        p = root/'metadata.json'
        m = json.loads(p.read_text())
        m['solver_clock_rows'] = 50
        p.write_text(json.dumps(m))
        result = audit_capture(root)
        self.assertEqual(result['timing']['diagnostic_time_basis'], 'wrapper_nominal_unverified')
        self.assertTrue(any('partial' in s for s in result['warnings']))
        self.assertFalse(result['ready_for_residual_training'])

    def test_failed_metadata_cannot_pass(self):
        root = self.capture()
        p = root/'metadata.json'
        m = json.loads(p.read_text())
        m['status'] = 'failed'
        p.write_text(json.dumps(m))
        self.assertEqual(audit_capture(root)['integrity'], 'FAIL')

    def test_cli_read_only_and_new_reports_refuse_overwrite(self):
        root = self.capture()
        before = {p.name:p.read_bytes() for p in root.iterdir()}
        report_dir = self.root/'report'
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(audit_main(['--capture', str(root)]), 0)
            self.assertEqual(audit_main(['--capture', str(root), '--report-dir', str(report_dir)]), 0)
            with self.assertRaises(SystemExit):
                audit_main(['--capture', str(root), '--report-dir', str(report_dir)])
            with self.assertRaises(SystemExit):
                audit_main(['--capture', str(root), '--report-dir', str(root/'nested')])
        self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})
        self.assertFalse(json.loads((report_dir/'audit.json').read_text())['ready_for_residual_training'])

    def test_ragged_csv_is_rejected(self):
        root = self.capture()
        p = root/'signals_si.csv'
        with p.open('a', encoding='utf-8') as file:
            file.write('1,52,2\n')
        # Matching row counts ensures this exercises shape validation, not pairing.
        result = audit_capture(root)
        self.assertEqual(result['integrity'], 'FAIL')
        self.assertTrue(any('ragged' in s for s in result['errors']))

    def test_no_files_cli_bad_source_and_bad_warmup(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(audit_main(['--capture', str(self.root/'missing')]), 2)
            with self.assertRaises(SystemExit):
                audit_main(['--capture', str(self.root), '--warmup-seconds', '-1'])
        self.assertFalse((self.root/'missing').exists())

    def test_binding_requires_known_initial_time(self):
        dll = DummyDLL()
        clock = SolverClock()
        self.assertEqual(clock.bind(dll, 0), 0)
        self.assertEqual(dll.vs_get_var_ptr.args, [b'T'])
        self.assertEqual(dll.vs_get_var_ptr.argtypes, [ctypes.c_char_p])
        self.assertIs(dll.vs_get_var_ptr.restype, ctypes.POINTER(ctypes.c_double))
        clock.discard()
        self.assertIsNone(clock.read())
        dll.time.value = 1
        self.assertIsNone(clock.bind(dll, 0))
        self.assertIn('differs', clock.reason)

    def test_no_getter_never_uses_wrapper_time_as_fallback(self):
        clock = SolverClock()
        self.assertIsNone(clock.bind(object(), 0))
        self.assertIsNone(clock.read())
        self.assertIn('unavailable', clock.reason)

    def test_null_and_nonfinite_clock(self):
        class NullGetter(Getter):
            def __call__(self, key):
                return ctypes.POINTER(ctypes.c_double)()
        dll = DummyDLL()
        dll.vs_get_var_ptr = NullGetter(dll.time)
        clock = SolverClock()
        self.assertIsNone(clock.bind(dll, 0))
        dll = DummyDLL()
        self.assertEqual(clock.bind(dll, 0), 0)
        dll.time.value = np.nan
        self.assertIsNone(clock.read())

    def test_require_clock_stops_after_initial_snapshot(self):
        env = DummyEnv()
        env.solver.dll_handle = None
        sink = CaptureWriter(self.root/'failed', {})
        self.addCleanup(sink.close_files)
        rec = IntegrationRecorder(env, sink, require_solver_time=True)
        with self.assertRaisesRegex(RuntimeError, 'required'):
            rec.reset()
        self.assertEqual(sink.count, 1)
        self.assertEqual(env.count, 0)

    def test_observer_forwards_identical_calls_and_results(self):
        base = DummySolver()
        clock = SolverClock()
        clock.bind(base.dll_handle, 0)
        observed = ObservedSolver(base, clock)
        self.assertEqual(observed.integrate_io_inplace(0, [0.1,0,2], None, 25), 0)
        self.assertEqual(base.calls, [(0, (0.1,0,2), 25)])
        self.assertEqual(observed.before, 0)
        self.assertEqual(observed.after, 0.0005)
        observed.terminate_run(0.001)
        self.assertIsNone(clock.read())
        self.assertTrue(base.terminated)


if __name__ == '__main__':
    unittest.main()
