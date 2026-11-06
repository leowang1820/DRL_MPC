# -*- coding: utf-8 -*-
"""真实轨迹候选数据契约测试（不启动 CarSim）。"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_recovery.carsim_candidate import create_candidate
from wm_recovery.carsim_export_schema import EXPORT_NAMES
from wm_recovery.controls import DIRECT_IMPORT_NAMES


def fixture(folder):
    """构造一段含真实左右转角差的两帧原始记录。"""
    folder = Path(folder)
    names = tuple(EXPORT_NAMES)
    exports = np.zeros((2, 61), dtype=float)

    def assign(name, values):
        exports[:, names.index(name)] = values

    assign("Vx", [40.0, 40.1])
    assign("AVy_L1", [300, 300.5])
    assign("AVy_R1", [300, 300.5])
    assign("AVy_L2", [300, 300.5])
    assign("AVy_R2", [300, 300.5])
    assign("Steer_L1", [0, 0.19335])
    assign("Steer_R1", [0, 0.19313])
    assign("Ax", [0, 0.001])
    assign("My_US_L1", [0, 20])

    np.savez_compressed(
        folder / "raw_capture.npz",
        time_s=[0.0, 0.01],
        raw_exports=exports,
        command=[[20.0, 0, 0, 0, math.radians(0.2)]],
        import_values=[[20.0, 0, 0, 0, 0.2, 0.2]],
    )
    (folder / "metadata.json").write_text(
        json.dumps({
            "format": "wm_recovery_carsim_raw_capture_v1",
            "source_kind": "mock_or_injected",
            "export_names": list(EXPORT_NAMES),
            "import_names": list(DIRECT_IMPORT_NAMES),
        }),
        encoding="utf-8",
    )


class CandidateTests(unittest.TestCase):
    def test_three_input_concepts_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            report = create_candidate(folder, folder / "candidate")

            with np.load(folder / "candidate" / "candidate.npz") as data:
                self.assertEqual(data["state_si"].shape, (2, 7))
                self.assertEqual(
                    data["nominal_7dof_input_si"].shape, (1, 6)
                )
                self.assertEqual(data["measured_steer_rad"].shape, (2, 2))
                self.assertTrue(np.array_equal(
                    data["nominal_7dof_input_si"][:, 0],
                    data["nominal_7dof_input_si"][:, 1],
                ))
                self.assertNotEqual(
                    data["measured_steer_rad"][1, 0],
                    data["measured_steer_rad"][1, 1],
                )
                self.assertAlmostEqual(data["state_si"][0, 0], 40 / 3.6)
                self.assertAlmostEqual(
                    data["nominal_7dof_input_si"][0, 0],
                    math.radians(0.2),
                )

            self.assertTrue(report["torque_echo_matches"])
            self.assertFalse(report["ready_for_wm_training"])
            self.assertFalse(
                (folder / "candidate" / "trajectory.npz").exists()
            )

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            target = folder / "candidate"
            create_candidate(folder, target)
            with self.assertRaises(FileExistsError):
                create_candidate(folder, target)

    def test_bad_import_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            original = folder / "raw_capture.npz"
            with np.load(original) as file:
                data = {k: file[k].copy() for k in file.files}
            data["import_values"][0, 5] = 1.7
            np.savez_compressed(original, **data)
            with self.assertRaises(ValueError):
                create_candidate(folder, folder / "candidate")

    def test_torque_mismatch_does_not_claim_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fixture(folder)
            original = folder / "raw_capture.npz"
            with np.load(original) as file:
                data = {k: file[k].copy() for k in file.files}
            name_index = tuple(EXPORT_NAMES).index("My_US_L1")
            data["raw_exports"][1, name_index] = 19.5
            np.savez_compressed(original, **data)
            report = create_candidate(folder, folder / "candidate")
            self.assertFalse(report["torque_echo_matches"])
            self.assertFalse(report["ready_for_wm_training"])


if __name__ == "__main__":
    unittest.main()
