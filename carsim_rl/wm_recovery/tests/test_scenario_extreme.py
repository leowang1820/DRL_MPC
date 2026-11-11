# -*- coding: utf-8 -*-
"""新版操纵极限场景的无 DLL 单元测试（只验证代码逻辑）。"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from wm_recovery.scenario_config import (
    DT_S, SCENARIOS, NOMINAL_MU, Scenario,
    command_at, roadwheel_delta_for_ay, validate_scenarios,
    steering_amplitude_deg,
)
from wm_recovery import scenario_runner


class FakeData:
    def __init__(self, speed=80, steer=0):
        self.Vx = speed
        self.Ax = 0.0
        # 仿真测试用线性侧向响应：角度 1.2 度 => 0.3g。
        self.Ay = 0.25 * steer
        self.AVz = 8.0 * steer
        self.Steer_L1 = steer
        self.Steer_R1 = steer - 0.003


class FakeSnapshot:
    def __init__(self, t=0.0, steer=0.0):
        self.time_s = t
        self.data = FakeData(80, steer)


class FakeEnv:
    def __init__(self):
        self.config = {"t_step": 0.001, "t_stop": 6.0}


class FakeBridge:
    instances = []

    def __init__(self, sim_path, prediction_dt_s):
        self.sim_path = sim_path
        self.dt = prediction_dt_s
        self.latest = FakeSnapshot()
        self._env = FakeEnv()
        self._ended = False
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def step(self, command):
        if len(command) != 5:
            raise ValueError("Five controls required")
        self.calls.append(list(command))
        angle = math.degrees(command[4])
        self.latest = FakeSnapshot(len(self.calls) * self.dt, angle)
        return self.latest

    def save_raw(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "mock_marker.txt").write_text(
            "No actual CarSim data", encoding="utf-8"
        )


def fake_candidate_creator(raw, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    return {"torque_echo_matches": True}


class ExtremeScenarioTests(unittest.TestCase):
    def test_splits_have_independent_conditions(self):
        scenes = validate_scenarios()
        self.assertEqual(len(scenes), 18)
        self.assertEqual(
            [s.split for s in scenes].count("train"), 12
        )
        self.assertEqual(
            [s.split for s in scenes].count("validation"), 3
        )
        self.assertEqual(
            [s.split for s in scenes].count("test"), 3
        )
        self.assertEqual(len({s.name for s in scenes}), len(scenes))
        self.assertEqual(
            len({s.condition_id for s in scenes}), len(scenes)
        )

    def test_invalid_duplicate_name_and_split(self):
        with self.assertRaises(ValueError):
            validate_scenarios((
                SCENARIOS[0],
                replace(SCENARIOS[1], name=SCENARIOS[0].name),
                *SCENARIOS[2:],
            ))
        with self.assertRaises(ValueError):
            validate_scenarios((
                replace(SCENARIOS[0], split="wrong"),
                *SCENARIOS[1:],
            ))

    def test_roadwheel_angle_decreases_with_speed(self):
        self.assertGreater(
            roadwheel_delta_for_ay(0.6, 40),
            roadwheel_delta_for_ay(0.6, 80),
        )
        with self.assertRaises(ValueError):
            roadwheel_delta_for_ay(.7, 0)

    def test_all_commands_are_five_finite_values(self):
        for scene in SCENARIOS:
            for t in [0, 0.85, 1.7, 2.4, 3.0, 4.3, 5.99]:
                command = command_at(scene, t, a03_deg=1.0)
                self.assertEqual(len(command), 5)
                self.assertTrue(all(math.isfinite(x) for x in command))
                self.assertLessEqual(
                    abs(math.degrees(command[-1])), 12.0 + 1e-8
                )
        with self.assertRaises(ValueError):
            command_at(SCENARIOS[0], 6.0, 1.0)

    def test_sine_with_dwell_has_half_second_second_peak_hold(self):
        swd = next(s for s in SCENARIOS if s.mode == "swd")
        onset = 1.0
        s = 0.75 / 0.7
        values = [
            math.degrees(command_at(swd, onset + s + offset, 1.0)[4])
            for offset in [0.02, 0.20, 0.49]
        ]
        self.assertAlmostEqual(values[0], values[1], places=6)
        self.assertAlmostEqual(values[1], values[2], places=6)

    def test_torque_excites_both_signs(self):
        torque_scene = next(s for s in SCENARIOS if s.mode == "torque")
        pos = command_at(torque_scene, 2.7, 1.0)
        neg = command_at(torque_scene, 4.5, 1.0)
        self.assertTrue(all(x > 0 for x in pos[:4]))
        self.assertTrue(all(x < 0 for x in neg[:4]))

    def test_mock_steer_calibration_reaches_03g(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = scenario_runner.calibrate_steer(
                "mock.sim", root, bridge_type=FakeBridge
            )
            self.assertEqual(
                result["format"], scenario_runner.CALIBRATION_FORMAT
            )
            self.assertTrue(
                0.1 < result["positive"]["reference_roadwheel_angle_deg"] < 12.0
            )
            self.assertTrue(
                0.1 < result["negative"]["reference_roadwheel_angle_deg"] < 12.0
            )
            self.assertTrue(
                (root / "steer_calibration.json").is_file()
            )

    def test_mock_capture_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calib = {
                "positive":{"reference_roadwheel_angle_deg":1.2,
                            "initial_speed_kmh":80},
                "negative":{"reference_roadwheel_angle_deg":1.2,
                            "initial_speed_kmh":80}
            }
            candidate, report = scenario_runner.capture_one(
                SCENARIOS[2], "mock.sim", root, calib,
                bridge_type=FakeBridge,
                candidate_creator=fake_candidate_creator
            )
            self.assertTrue(candidate.is_dir())
            self.assertEqual(len(FakeBridge.instances[-1].calls), 600)
            self.assertTrue(FakeBridge.instances[-1].closed)
            self.assertGreater(report["peak_abs_ay_g"], 0.3)
            self.assertEqual(report["split"], "train")

    def test_coverage_gate_detects_insufficient_data(self):
        r = [{
            "split":s.split,
            "sign":s.sign,
            "peak_abs_ay_g":0.03,
        } for s in SCENARIOS]
        result = scenario_runner.summarize_coverage(r)
        self.assertFalse(result["passed"])
        self.assertFalse(all(result["checks"].values()))

    def test_no_overwrite_of_existing_batch(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                scenario_runner.capture_all(
                    "mock.sim", d, bridge_type=FakeBridge
                )

    def test_plan_does_not_call_car_sim(self):
        with patch(
            "wm_recovery.scenario_runner._open_bridge",
            side_effect=AssertionError("Solver must not start"),
        ):
            scenario_runner.print_plan()

    def test_opt_in_required_for_training(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                scenario_runner.require_opt_in()


if __name__ == "__main__":
    unittest.main()
