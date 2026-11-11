# -*- coding: utf-8 -*-
"""兼容性回归：新版场景不再限制为恰好四条轨迹。

本文件取代旧版“4 条固定轨迹”测试，避免重构升级后的误判。
不调用 CarSim DLL。
"""
import math
import tempfile
import unittest
from pathlib import Path

from wm_recovery.scenario_config import (
    SCENARIOS, command_at, validate_scenarios
)
from wm_recovery import scenario_runner


class ScenarioRefactorCompatibilityTests(unittest.TestCase):

    def test_unique_partitions(self):
        scenes = validate_scenarios()
        self.assertGreaterEqual(len(scenes), 4)
        self.assertGreaterEqual(
            sum(s.split == "train" for s in scenes), 2
        )
        self.assertGreaterEqual(
            sum(s.split == "validation" for s in scenes), 1
        )
        self.assertGreaterEqual(
            sum(s.split == "test" for s in scenes), 1
        )

    def test_five_finite_controls_with_a03_reference(self):
        for scene in SCENARIOS:
            action = command_at(scene, 1.0, a03_deg=1.0)
            self.assertEqual(len(action), 5)
            self.assertTrue(all(math.isfinite(x) for x in action))

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                scenario_runner.capture_all("mock.sim", temp)


if __name__ == "__main__":
    unittest.main()
