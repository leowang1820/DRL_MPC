"""Capture contract and fake-solver tests. Never loads a CarSim DLL."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from dynamics7dof.collect import CaptureWriter, IntegrationRecorder, main
from dynamics7dof.signals import EXPORT_NAMES, IMPORT_NAMES, decode_exports, inspect_configuration


def sample_exports():
    values = np.zeros(25)
    values[0:4] = [1.0, 2.0, 90.0, 180.0]
    values[4:9] = [0.1, 0.2, 36.0, 3.0, 3.6]
    values[9:13] = [60.0, 120.0, 180.0, 240.0]
    values[13:17] = [1.0, -2.0, 0.5, -0.25]
    values[17:25] = [10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0]
    return values


class FakeEnv:
    def __init__(self, stop_after=None):
        self.n_import, self.n_export = 3, 25
        self.t_current, self.t_step, self.t_stop = 0.0, 0.001, 40.0
        self.config = {"n_import": 3, "n_export": 25, "t_step": 0.001}
        self.closed = False
        self.calls = []
        self.stop_after = stop_after
        self.done = False

    def reset(self):
        self.t_current = 0.0
        self.done = False
        return tuple(sample_exports())

    def control_step(self, action, inner_steps):
        for _ in range(inner_steps):
            if self.done:
                raise RuntimeError("Cannot step a terminated fake solver")
            self.calls.append((self.t_current, tuple(action)))
            self.t_current += self.t_step
            if self.stop_after is not None and len(self.calls) >= self.stop_after:
                self.done = True
                break
        exports = sample_exports()
        exports[0] = self.t_current * 10.0
        return tuple(exports), 0.0, self.done, {"return_code": int(self.done)}

    def close(self):
        self.closed = True


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dynamics7dof_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sim = self.root / "simfile.sim"
        self.par = self.root / "Run_all.par"
        self.sim.write_text(
            f"SIMFILE\nSET_MACRO $(DIR)$ {self.root}\n"
            "INPUT $(DIR)$/Run_all.par\nPORTS_IMP 1,3\nPORTS_EXP 1,25\nEND\n",
            encoding="utf-8",
        )
        self.par_text = "\n".join([
            *(f"IMPORT {name} REPLACE 0" for name in IMPORT_NAMES),
            *(f"EXPORT {name}" for name in EXPORT_NAMES),
            "OPT_IO_SYNC_FM 0", "MU_ROAD_CONSTANT 0.5", "TSTOP 40",
        ])
        self.par.write_text(self.par_text, encoding="utf-8")

    def new_sink(self):
        sink = CaptureWriter(self.root / "capture", {"test_only": True})
        self.addCleanup(sink.close_files)
        return sink

    def test_all_si_conversions_and_separate_signed_torques(self):
        si = decode_exports(sample_exports())
        self.assertEqual(si["vx_m_s"], 10.0)
        self.assertEqual(si["vy_m_s"], 1.0)
        self.assertAlmostEqual(si["yaw_rate_rad_s"], np.pi)
        for wheel, expected in zip(("fl", "fr", "rl", "rr"), (2, 4, 6, 8)):
            self.assertAlmostEqual(si[f"omega_{wheel}_rad_s"], expected*np.pi)
        self.assertAlmostEqual(si["front_steer_fl_rad"], np.pi/180)
        self.assertAlmostEqual(si["front_steer_fr_rad"], -2*np.pi/180)
        self.assertAlmostEqual(si["ax_carsim_m_s2"], 0.5*9.81)
        self.assertAlmostEqual(si["ay_carsim_m_s2"], -0.25*9.81)
        self.assertEqual(si["drive_fr_nm"], 20.0)
        self.assertEqual(si["brake_fr_nm"], -2.0)
        self.assertFalse(any("net_torque" in name for name in si))

    def test_conversion_rejects_wrong_length_or_nonfinite(self):
        for values in (np.zeros(24), np.zeros((25, 1)), np.full(25, np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                decode_exports(values)

    def test_contract_resolves_macros_and_reads_actual_order(self):
        result = inspect_configuration(self.sim)
        self.assertEqual(result["parameter_path"], str(self.par.resolve()))
        self.assertEqual(result["exports"], list(EXPORT_NAMES))
        self.assertEqual(result["settings"]["MU_ROAD_CONSTANT"], 0.5)
        self.assertEqual(len(result["parameter_sha256"]), 64)

    def test_same_count_wrong_order_is_rejected(self):
        self.par.write_text(self.par_text.replace("EXPORT AVy_L1", "EXPORT AVy_R1", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "#10"):
            inspect_configuration(self.sim)

    def test_windows_ansi_parameter_comments_are_supported(self):
        self.par.write_bytes((self.par_text + "\n! 中文注释：参数\n").encode("gbk"))
        with patch("dynamics7dof.signals.locale.getpreferredencoding", return_value="gbk"):
            result = inspect_configuration(self.sim)
        self.assertEqual(result["parameter_encoding"], "gbk")
        self.assertEqual(result["exports"], list(EXPORT_NAMES))

    def test_incorrect_port_count_and_import_mode_are_rejected(self):
        initial_sim = self.sim.read_text(encoding="utf-8")
        self.sim.write_text(initial_sim.replace("1,25", "1,8"), encoding="utf-8")
        with self.assertRaises(ValueError):
            inspect_configuration(self.sim)
        self.sim.write_text(initial_sim, encoding="utf-8")
        self.par.write_text(self.par_text.replace("REPLACE", "ADD", 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            inspect_configuration(self.sim)

    def test_unknown_macro_and_nested_input_fail_closed(self):
        initial_sim = self.sim.read_text(encoding="utf-8")
        self.sim.write_text(initial_sim.replace("INPUT $(DIR)$", "INPUT $(UNKNOWN)$"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            inspect_configuration(self.sim)
        self.sim.write_text(initial_sim, encoding="utf-8")
        self.par.write_text(self.par_text + "\nINPUT external.par", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nested"):
            inspect_configuration(self.sim)

    def test_read_only_preflight_does_not_create_a_capture(self):
        with patch("dynamics7dof.collect.CaptureWriter") as writer, redirect_stdout(io.StringIO()) as output:
            main(["--sim", str(self.sim), "--check-config-only"])
        writer.assert_not_called()
        self.assertIn("Config check: PASS", output.getvalue())
        self.assertFalse((self.root / "capture").exists())

    def test_adapter_preserves_held_commands_and_end_state(self):
        sink = self.new_sink()
        base = FakeEnv()
        recorder = IntegrationRecorder(base, sink)
        recorder.reset()
        command = [0.1, 0.0, 2.0]
        captured = recorder.control_step(command, inner_steps=50)
        plain = FakeEnv()
        plain.reset()
        uncaptured = plain.control_step(command, inner_steps=50)
        self.assertEqual(captured, uncaptured)
        self.assertEqual(base.calls, plain.calls)
        self.assertEqual(sink.count, 51)
        sink.finish("test_completed")
        with (sink.output / "raw_exports.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["phase"], "initial")
        self.assertEqual(rows[0]["api_time_argument_s"], "")
        self.assertEqual(float(rows[1]["api_time_argument_s"]), 0.0)
        self.assertAlmostEqual(float(rows[1]["wrapper_time_s"]), 0.001)
        self.assertIn("first_integrate", rows[1]["quality_flags"])
        self.assertTrue(all(float(row["steering_wheel_command_deg"]) == 2.0 for row in rows[1:]))

    def test_adapter_stops_at_solver_end_and_keeps_terminal_row(self):
        sink = self.new_sink()
        base = FakeEnv(stop_after=3)
        recorder = IntegrationRecorder(base, sink)
        recorder.reset()
        _, _, done, _ = recorder.control_step([0, 0, 0], 50)
        self.assertTrue(done)
        self.assertEqual(len(base.calls), 3)
        self.assertEqual(sink.count, 4)
        recorder.close()
        self.assertTrue(base.closed)
        sink.finish("test_terminal")

    def test_live_counts_are_checked(self):
        sink = self.new_sink()
        base = FakeEnv()
        base.n_export = 8
        with self.assertRaisesRegex(RuntimeError, "counts"):
            IntegrationRecorder(base, sink).reset()
        self.assertEqual(sink.count, 0)

    def test_no_cross_episode_reset_and_no_overwrite(self):
        sink = self.new_sink()
        recorder = IntegrationRecorder(FakeEnv(), sink)
        recorder.reset()
        with self.assertRaisesRegex(RuntimeError, "one episode"):
            recorder.reset()
        with self.assertRaises(FileExistsError):
            CaptureWriter(sink.output, {})

    def test_bad_time_spacing_is_rejected(self):
        sink = self.new_sink()
        recorder = IntegrationRecorder(FakeEnv(), sink)
        recorder.reset()
        with self.assertRaisesRegex(ValueError, "interval"):
            sink.append(sample_exports(), wrapper_time=0.003, api_time=0, dt=0.001,
                        control_index=1, phase="after_integrate", action=[0, 0, 0], info={}, done=False)
        self.assertEqual(sink.count, 1)

    def test_metadata_never_claims_training_ready(self):
        sink = self.new_sink()
        recorder = IntegrationRecorder(FakeEnv(), sink)
        recorder.reset()
        recorder.control_step([0, 0, 0], 3)
        sink.finish("test_completed")
        metadata = json.loads((sink.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["ready_for_residual_training"])
        self.assertEqual(metadata["rows"], 4)
        self.assertEqual(len(metadata["pending_checks"]), 3)
        self.assertAlmostEqual(metadata["wrapper_dt_min_s"], 0.001)
        with (sink.output / "signals_si.csv").open(encoding="utf-8-sig", newline="") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 4)

    def _run_fake_capture(self, failing=False):
        solver = FakeEnv()

        class FakeWrapper:
            def __init__(self, *args, **kwargs):
                self._env = solver
                self.steps = 0

            def reset(self):
                return self._env.reset()

            def step(self, action):
                self.steps += 1
                if failing and self.steps == 2:
                    raise RuntimeError("injected controller failure")
                obs, reward, done, info = self._env.control_step([0.1, 0, 0], 50)
                info.update(station=self._env.t_current*10, speed_kmh=36.0, lateral_error=0.0)
                return obs, reward, done, info

            def close(self):
                self._env.close()

        modules = {
            "carsim_wrapper": types.SimpleNamespace(CONTROL_DT_S=0.05, CarSimGymWrapper=FakeWrapper),
            "run_dlc_baseline": types.SimpleNamespace(pure_pursuit_action=lambda *args: [0.1, 0]),
        }
        self.addCleanup(lambda: self.assertTrue(solver.closed))
        with patch.dict("sys.modules", modules), redirect_stdout(io.StringIO()):
            main(["--sim", str(self.sim), "--path", str(self.par), "--mu", "0.5",
                  "--seconds", "0.1", "--output-dir", str(self.root / "run")])

    def test_full_cli_with_fake_solver(self):
        self._run_fake_capture()
        metadata = json.loads((self.root / "run" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "captured_requested_duration")
        self.assertEqual(metadata["rows"], 101)
        self.assertEqual(metadata["control_updates"], 2)
        self.assertFalse(metadata["ready_for_residual_training"])

    def test_failed_capture_keeps_partial_data_and_closes_solver(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self._run_fake_capture(failing=True)
        metadata = json.loads((self.root / "run" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["rows"], 51)
        self.assertIn("injected", metadata["error"])


if __name__ == "__main__":
    unittest.main()
