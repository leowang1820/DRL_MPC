# -*- coding: utf-8 -*-
"""统一场景配置/场景运行入口的无 DLL 单元测试。"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wm_recovery.scenario_config import (
    DT_S, SCENARIOS, command_at, validate_scenarios
)
from wm_recovery import scenario_runner


class FakeEnvironment:
    def __init__(self):
        self.config = {"t_step": 0.001, "t_stop": 10.0}


class FakeSnapshot:
    def __init__(self, time_s):
        self.time_s = time_s


class FakeBridge:
    """只模拟数据交互形状，不调用真实 CarSim DLL。"""

    instances = []

    def __init__(self, sim_path, prediction_dt_s):
        self.sim_path = sim_path
        self.dt = prediction_dt_s
        self._env = FakeEnvironment()
        self._ended = False
        self.latest = FakeSnapshot(0.0)
        self.commands = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def step(self, command):
        if len(command) != 5:
            raise AssertionError("Expected five control values")
        self.commands.append(list(command))
        self.latest = FakeSnapshot(len(self.commands) * self.dt)
        return self.latest

    def save_raw(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        # This is a mock marker, not a usable CarSim raw_capture.npz.
        (folder / "mock_marker.txt").write_text(
            "mock", encoding="utf-8"
        )


def fake_candidate_creator(raw_dir, candidate_dir):
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    (candidate_dir / "mock_marker.txt").write_text(
        "mock", encoding="utf-8"
    )
    return {"torque_echo_matches": True}


class ScenarioRefactorTests(unittest.TestCase):
    def test_four_independent_splits(self):
        scenes = validate_scenarios()
        self.assertEqual(len(scenes), 4)
        self.assertEqual(
            [s.split for s in scenes],
            ["train", "train", "validation", "test"],
        )
        self.assertEqual(len({s.condition_id for s in scenes}), 4)

    def test_five_controls_and_steer_in_radians(self):
        for scene in SCENARIOS:
            for t in (0.0, 0.1, 1.5, scene.duration_s - DT_S):
                command = command_at(scene, t)
                self.assertEqual(len(command), 5)
                self.assertTrue(all(abs(float(x)) < 1e6 for x in command))
        self.assertEqual(command_at(SCENARIOS[0], 0.5)[4], 0.0)

    def test_modes_distinct_and_invalid_time_rejected(self):
        commands = [
            command_at(scene, 1.1)
            for scene in SCENARIOS
        ]
        self.assertNotEqual(commands[1], commands[2])
        self.assertNotEqual(commands[2], commands[3])
        with self.assertRaises(ValueError):
            command_at(SCENARIOS[0], SCENARIOS[0].duration_s)

    def test_invalid_split_or_duplicate_condition_rejected(self):
        wrong_split = (
            replace(SCENARIOS[0], split="test"),
            *SCENARIOS[1:],
        )
        with self.assertRaises(ValueError):
            validate_scenarios(wrong_split)

        duplicated = (
            SCENARIOS[0],
            replace(SCENARIOS[1], condition_id=SCENARIOS[0].condition_id),
            *SCENARIOS[2:],
        )
        with self.assertRaises(ValueError):
            validate_scenarios(duplicated)

    def test_single_capture_with_mock_bridge(self):
        # Reduces a test trajectory to 3 s; no real Solver is started.
        scene = replace(SCENARIOS[1], duration_s=3.0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = scenario_runner.capture_one(
                scene,
                sim_path="mock.sim",
                root=Path(tmp),
                bridge_type=FakeBridge,
                candidate_creator=fake_candidate_creator,
            )
            self.assertTrue(folder.is_dir())
            self.assertEqual(len(FakeBridge.instances[-1].commands), 300)
            self.assertTrue(FakeBridge.instances[-1].closed)

    def test_experiment_gate_rejects_training_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                scenario_runner.require_experimental_opt_in()

    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                scenario_runner.capture_all("mock.sim", existing)

    def test_plan_no_solver_or_writes(self):
        # plan() only prints, doesn't call the Bridge.
        with patch("wm_recovery.scenario_runner.capture_one",
                   side_effect=AssertionError("Must not run DLL")):
            scenario_runner.print_plan()


if __name__ == "__main__":
    unittest.main()
