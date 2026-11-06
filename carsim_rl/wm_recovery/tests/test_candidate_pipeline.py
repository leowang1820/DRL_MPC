# -*- coding: utf-8 -*-
"""检查 CarSim 候选轨迹只读入口与旧 WM 训练路径隔离（不启动 CarSim）。"""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from dynamics7dof.model import LoadMemory
from wm_recovery.carsim_candidate import create_candidate
from wm_recovery.carsim_export_schema import EXPORT_NAMES
from wm_recovery.dataset import audit_carsim_candidate
from wm_recovery.offline import CandidateEpisode, Episode
from wm_recovery.tests.test_carsim_candidate import fixture


class CandidatePipelineTests(unittest.TestCase):

    def test_load_preserves_real_steer_and_nominal_command(self):
        """测量角度可以不等，名义同角输入必须仍与 Python 指令一致。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            create_candidate(root, root / "candidate")

            ep = CandidateEpisode.load(root / "candidate")
            self.assertNotEqual(ep.measured_steer[1, 0],
                                ep.measured_steer[1, 1])
            self.assertEqual(ep.nominal_input.shape, (1, 6))
            self.assertAlmostEqual(ep.nominal_input[0, 0],
                                   math.radians(.2))
            self.assertAlmostEqual(ep.nominal_input[0, 1],
                                   math.radians(.2))
            self.assertEqual(ep.initial_memory, LoadMemory(0, 0))
            self.assertFalse(ep.metadata["ready_for_wm_training"])
            # 不能误用旧版训练所需的 applied 字段。
            self.assertFalse(hasattr(ep, "applied"))
            with self.assertRaises((ValueError, FileNotFoundError)):
                Episode.load(root / "candidate")

    def test_changed_candidate_array_rejected(self):
        """改动数组后不更新内容指纹，必须被拒绝。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            target = root / "candidate"
            create_candidate(root, target)
            file = target / "candidate.npz"
            with np.load(file, allow_pickle=False) as archive:
                values = {
                    name: archive[name].copy()
                    for name in archive.files
                }
            values["measured_steer_rad"][1, 0] += 0.01
            np.savez_compressed(file, **values)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                CandidateEpisode.load(target)

    def test_causal_load_memory_from_physics_not_carsim(self):
        """模型记忆来自上个微步模型预测，不来自已记录的 Ax/Ay。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            file = root / "raw_capture.npz"
            with np.load(file) as archive:
                values = {
                    name: archive[name].copy()
                    for name in archive.files
                }
            # 将两帧扩展到三帧，以检查跨两个控制区间的记忆传递。
            values["time_s"] = np.array([0.00, 0.01, 0.02])
            values["raw_exports"] = np.vstack((
                values["raw_exports"], values["raw_exports"][1]
            ))
            values["command"] = np.vstack((
                values["command"], values["command"]
            ))
            values["import_values"] = np.vstack((
                values["import_values"], values["import_values"]
            ))
            # 把 CarSim Ax 改成夸张的值，确保模型不会读取它初始化记忆。
            idx_ax = tuple(EXPORT_NAMES).index("Ax")
            values["raw_exports"][:, idx_ax] = 3.0
            np.savez_compressed(file, **values)

            target = root / "candidate"
            create_candidate(root, target)
            ep = CandidateEpisode.load(target)

            calls = []

            class FakeModel:
                def predict(self, state, controls, *, mu, load_memory,
                            horizon_s, substeps):
                    calls.append((
                        np.asarray(controls.front_steer_rad).copy(),
                        np.array([
                            load_memory.ax_m_s2,
                            load_memory.ay_m_s2,
                        ]),
                        mu,
                        horizon_s,
                        substeps,
                    ))
                    return SimpleNamespace(
                        state=np.asarray(state) + .01,
                        flags=(),
                        load_memory=LoadMemory(
                            load_memory.ax_m_s2 + .1,
                            load_memory.ay_m_s2 + .2,
                        ),
                    )

            with patch(
                "wm_recovery.dataset.Reference7DOF",
                return_value=FakeModel(),
            ):
                result = audit_carsim_candidate(
                    ep, max_substep_s=.0005
                )

            self.assertEqual(len(calls), 2)
            np.testing.assert_allclose(
                calls[0][0], [math.radians(.2), math.radians(.2)]
            )
            np.testing.assert_allclose(calls[0][1], [0, 0])
            np.testing.assert_allclose(calls[1][1], [.1, .2])
            self.assertAlmostEqual(calls[0][2], .8)
            self.assertAlmostEqual(calls[0][3], .01)
            self.assertEqual(calls[0][4], 20)
            self.assertTrue(result.valid.all())
            self.assertFalse(
                result.report["carsim_ax_ay_used_in_load_memory"]
            )
            self.assertFalse(result.report["ready_for_wm_training"])

    def test_nonmatching_torque_blocks_nominal_audit(self):
        """如果外加转矩回显不一致，不允许进行名义控制对照。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            file = root / "raw_capture.npz"
            with np.load(file) as archive:
                values = {
                    name: archive[name].copy()
                    for name in archive.files
                }
            idx = tuple(EXPORT_NAMES).index("My_US_L1")
            values["raw_exports"][1, idx] = 19.5
            np.savez_compressed(file, **values)

            target = root / "candidate"
            create_candidate(root, target)
            ep = CandidateEpisode.load(target)
            self.assertFalse(ep.metadata["torque_echo_matches"])
            with self.assertRaisesRegex(ValueError, "转矩回显"):
                audit_carsim_candidate(ep)


if __name__ == "__main__":
    unittest.main()
