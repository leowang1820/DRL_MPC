# -*- coding: utf-8 -*-
"""CarSim 实验性数据 -> 旧 WM 训练张量兼容性测试。

这里用人工构造的假轨迹测试数据契约，绝不作为真实车辆验证证据。
不启动 CarSim，也不训练神经网络。
"""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dynamics7dof.model import LoadMemory
from wm_recovery.carsim_candidate import create_candidate
from wm_recovery.carsim_export_schema import EXPORT_NAMES
from wm_recovery.controls import DIRECT_IMPORT_NAMES
from wm_recovery.carsim_experimental import (
    EXPERIMENT_ENV, build_workspace, pilot
)
from wm_recovery.offline import Episode
from wm_recovery.dataset import (
    load_partition, trajectory_bootstrap, fit_scaler
)


def make_real_format_mock_candidate(folder, flavor):
    """伪造不同数值轨迹以测试数据读取，不声称是真实 CarSim 记录。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)

    count = 130
    dt = 0.01
    t = np.arange(count + 1, dtype=float) * dt
    names = tuple(EXPORT_NAMES)
    raw = np.zeros((len(t), len(names)))
    command = np.zeros((count, 5))
    command[:, 0] = flavor + np.sin(2*np.pi*.7*t[:-1])
    command[:, 1] = (flavor + 1) * np.cos(2*np.pi*.5*t[:-1])
    command[:, 4] = math.radians(.1) * np.sin(t[:-1])
    imports = np.column_stack((
        command[:, :4], np.rad2deg(command[:, 4]),
        np.rad2deg(command[:, 4])
    ))

    # 状态真实保存于 N+1 个时刻，输入在 N 个区间内生效
    raw[:, names.index("Vx")] = 40 + flavor*.5 + .02*t
    for name in ("AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2"):
        raw[:, names.index(name)] = 300
    raw[:, names.index("Steer_L1")] = .01 + .03*np.sin(t)
    raw[:, names.index("Steer_R1")] = -.01 + .03*np.sin(t)
    for col, name in enumerate(
        ("My_US_L1", "My_US_R1", "My_US_L2", "My_US_R2")
    ):
        raw[1:, names.index(name)] = command[:, col]

    np.savez_compressed(
        folder / "raw_capture.npz",
        time_s=t, raw_exports=raw, command=command,
        import_values=imports,
    )
    (folder / "metadata.json").write_text(json.dumps({
        "format": "wm_recovery_carsim_raw_capture_v1",
        # 此标志只为了测试真实记录路径的格式门禁，不是真实证据
        "source_kind": "carsim_env",
        "export_names": list(EXPORT_NAMES),
        "import_names": list(DIRECT_IMPORT_NAMES),
    }), encoding="utf-8")

    candidate = folder.parent / (folder.name + "_candidate")
    create_candidate(folder, candidate)
    return candidate


def prepare_fixture(root):
    """创建四条不同内容的模拟候选轨迹及 train/val/test 清单。"""
    paths = {}
    for i, name in enumerate(("train1","train2","validation","test")):
        paths[name] = make_real_format_mock_candidate(
            root / f"raw_{name}", i + 1
        )
    output = root / "workspace"
    registry = build_workspace(paths, output)
    return output, registry


class ExperimentalDatasetTests(unittest.TestCase):

    def test_experimental_gate_rejects_default(self):
        """用户未主动开启时必须拒绝实验性数据进入正式管线。"""
        with tempfile.TemporaryDirectory() as temp:
            workspace, _ = prepare_fixture(Path(temp))
            train_path = workspace / "episodes" / "train1"
            with patch.dict("os.environ", {EXPERIMENT_ENV: "0"}):
                with self.assertRaisesRegex(ValueError, "实验性"):
                    Episode.load(train_path)

    def test_preserves_control_semantics_and_real_steer(self):
        """名义同角仅在兼容字段中存在，真实左右转角仍保留。"""
        with tempfile.TemporaryDirectory() as temp:
            workspace, _ = prepare_fixture(Path(temp))
            path = workspace / "episodes" / "train1"

            with patch.dict("os.environ", {EXPERIMENT_ENV: "1"}):
                ep = Episode.load(path)

            self.assertEqual(ep.state.shape, (131, 7))
            self.assertEqual(ep.applied.shape, (130, 6))
            self.assertEqual(ep.command.shape, (130, 5))
            self.assertTrue(np.array_equal(
                ep.applied[:, 0], ep.applied[:, 1]
            ))
            self.assertEqual(ep.initial_memory, LoadMemory(0, 0))
            self.assertFalse(ep.metadata["ready_for_wm_training"])
            self.assertTrue(ep.metadata["experimental_training_only"])
            self.assertEqual(ep.metadata["source_kind"], "carsim_experimental")

            from wm_recovery.offline import CandidateEpisode
            candidate_path = json.loads(
                (path / "experimental_episode.json").read_text(encoding="utf-8")
            )["candidate_path"]
            candidate = CandidateEpisode.load(candidate_path)
            self.assertFalse(np.array_equal(
                candidate.measured_steer[:, 0],
                candidate.measured_steer[:, 1],
            ))
            np.testing.assert_allclose(ep.applied,
                                       candidate.nominal_input)
            self.assertFalse(np.array_equal(
                ep.applied[:, 0],
                candidate.measured_steer[:-1, 0],
            ))

    def test_duplicate_raw_capture_rejected_for_splits(self):
        """重复使用同一真实原始记录，不能伪装成另一个分区。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cand = make_real_format_mock_candidate(root/"raw", 1)
            with self.assertRaisesRegex(ValueError, "多条轨迹"):
                build_workspace({
                    "train1": cand, "train2": cand,
                    "validation": cand, "test": cand,
                }, root / "bad_workspace")

    def test_pilot_builds_workspace_after_independent_captures(self):
        """模拟三次采集的返回值，验证 pilot 的分区编排。

        此测试绝不打开 CarSim，不能代替真实 Solver 测试。
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = {
                name: make_real_format_mock_candidate(
                    root / ("raw_" + name), i+1
                )
                for i, name in enumerate(
                    ("train1", "train2", "validation", "test")
                )
            }
            out = root / "pilot"
            with patch(
                "wm_recovery.carsim_experimental._capture_one",
                side_effect=[
                    candidates["train2"],
                    candidates["validation"],
                    candidates["test"],
                ],
            ) as recorder:
                registry = pilot(
                    candidates["train1"], out,
                    sim_path="mock-sim.sim", duration_s=6.0,
                )
            self.assertEqual(recorder.call_count, 3)
            self.assertEqual(registry["episode_count"], 4)
            self.assertTrue(
                (out/"workspace"/"episodes"/"train1"/
                 "experimental_episode.json").is_file()
            )
            self.assertTrue(
                (out/"workspace"/"episodes"/"test"/
                 "experimental_episode.json").is_file()
            )

    def test_manifest_hash_mismatch_rejected(self):
        """编辑清单中的候选来源指纹后，数据加载必须报错。"""
        with tempfile.TemporaryDirectory() as temp:
            workspace, _ = prepare_fixture(Path(temp))
            path = workspace / "episodes" / "train1"
            file = path / "experimental_episode.json"
            manifest = json.loads(file.read_text(encoding="utf-8"))
            manifest["candidate_sha256"] = "0"*64
            file.write_text(json.dumps(manifest), encoding="utf-8")

            with patch.dict("os.environ", {EXPERIMENT_ENV: "1"}):
                with self.assertRaisesRegex(ValueError, "指纹"):
                    Episode.load(path)

    def test_causal_windows_and_trajectory_bootstrap(self):
        """网络历史只含之前的状态、名义输入；按整条轨迹 Bootstrap。"""
        with tempfile.TemporaryDirectory() as temp:
            workspace, _ = prepare_fixture(Path(temp))
            train_paths = [
                workspace / "episodes" / "train1",
                workspace / "episodes" / "train2",
            ]

            from types import SimpleNamespace

            class FakeModel:
                def predict(self, state, controls, *, mu, load_memory,
                            horizon_s, substeps):
                    return SimpleNamespace(
                        state=np.asarray(state, dtype=float).copy(),
                        flags=(),
                        load_memory=LoadMemory(
                            load_memory.ax_m_s2+.01,
                            load_memory.ay_m_s2+.02
                        ),
                    )

            config = SimpleNamespace(
                prediction_dt_s=0.01,
                history_steps=10,
                rollout_steps=5,
                substeps=20,
            )

            with patch.dict("os.environ", {EXPERIMENT_ENV: "1"}):
                with patch(
                    "wm_recovery.dataset.Reference7DOF",
                    return_value=FakeModel()
                ):
                    prepared = load_partition(
                        train_paths, config, split="train"
                    )

            self.assertEqual(len(prepared), 2)
            anchor = int(prepared[0].starts[0])
            history = prepared[0].history_at(anchor, config.history_steps)
            np.testing.assert_allclose(
                history[:, :7],
                prepared[0].episode.state[
                    anchor-config.history_steps:anchor
                ]
            )
            np.testing.assert_allclose(
                history[:, 7:],
                prepared[0].episode.model_input[
                    anchor-config.history_steps:anchor
                ]
            )
            self.assertEqual(history.shape, (10, 13))
            self.assertNotIn(
                anchor, range(anchor-config.history_steps, anchor)
            )
            draws, pools = trajectory_bootstrap(
                prepared, member_count=5, seed=1234
            )
            self.assertEqual(len(draws), 5)
            self.assertEqual(len(pools), 5)
            self.assertTrue(all(x.shape[1] == 2 for x in pools))
            scales = fit_scaler(prepared, config)
            self.assertEqual(scales["feature_mean"].shape[0],
                             13*(1+10)+2)
            self.assertEqual(scales["residual_scale"].shape[0], 7)


if __name__ == "__main__":
    unittest.main()
