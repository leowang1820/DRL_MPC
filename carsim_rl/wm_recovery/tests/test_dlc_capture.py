# -*- coding: utf-8 -*-
"""DLC 只读实时 Export 采集：不需要 CarSim DLL 的回归测试。"""
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np

from wm_recovery.dlc_capture import capture_dlc, check_sampling_times


def make_inventory(imports=None, exports=None):
    return {
        "imports": imports if imports is not None else [],
        "exports": exports if exports is not None else
                   ["Vx", "Yaw", "Steer_SW", "Steer_SW"],
        "sim_sha256": "a" * 64,
        "parameter_path": "DLC_all.par",
        "parameter_sha256": "b" * 64,
    }


class FakeCarSimEnv:
    def __init__(self, sim_path, *, n_import=0, n_export=4,
                 t_step=0.005, n_outputs=5):
        self.sim_path = sim_path
        self.config = {
            "n_import": n_import,
            "n_export": n_export,
            "t_step": t_step,
            "t_start": 0.0,
            "t_stop": 0.05,
        }
        self.n_outputs = n_outputs
        self.i = 0
        self.t_current = 0.0
        self.done = False
        self.actions = []
        self.closed = False

    def reset(self):
        self.t_current = 0.0
        self.i = 0
        self.done = False
        return np.arange(self.config["n_export"], dtype=float)

    def control_step(self, action, inner_steps):
        values = np.asarray(action)
        assert values.shape == (0,), "不允许发送任何外部 DLC 控制量"
        assert inner_steps == 2
        self.actions.append(values.copy())
        self.i += 1
        self.t_current = self.i * 0.01
        self.done = self.i >= self.n_outputs
        out = np.arange(self.config["n_export"], dtype=float) + self.i * 10
        return out, 0.0, self.done, {"return_code": 0}

    def close(self):
        self.closed = True


class TestDLCCapture(unittest.TestCase):
    def test_capture_all_channels_duplicate_names_and_time(self):
        with tempfile.TemporaryDirectory() as td:
            sim = Path(td) / "simfile.sim"
            sim.write_text("SIMFILE\nEND\n", encoding="utf-8")
            envs = []
            def factory(path):
                env = FakeCarSimEnv(path)
                envs.append(env)
                return env
            out = Path(td) / "capture_001"
            result = capture_dlc(
                sim, out, sample_dt_s=0.01,
                env_factory=factory, inspect_fn=lambda _: make_inventory()
            )
            self.assertEqual(result["n_import"], 0)
            self.assertEqual(result["n_export"], 4)
            self.assertEqual(result["n_samples"], 6)
            self.assertTrue(envs[0].closed)
            self.assertEqual(len(envs[0].actions), 5)
            with np.load(out/"dlc_exports.npz", allow_pickle=False) as npz:
                self.assertEqual(npz["raw_exports"].shape, (6, 4))
                self.assertEqual(npz["export_names"].tolist(),
                                 ["Vx", "Yaw", "Steer_SW", "Steer_SW"])
                np.testing.assert_allclose(
                    npz["time_s"], np.linspace(0.0,0.05,6)
                )
                np.testing.assert_allclose(npz["raw_exports"][-1],
                                            [50,51,52,53])
            with self.assertRaises(FileExistsError):
                capture_dlc(sim,out,env_factory=factory,
                            inspect_fn=lambda _: make_inventory())

    def test_external_steering_import_rejected_before_env_load(self):
        with tempfile.TemporaryDirectory() as td:
            sim=Path(td)/"simfile.sim"
            sim.write_text("SIMFILE\nEND\n",encoding="utf-8")
            called=[]
            def fake_factory(path):
                called.append(path)
                return FakeCarSimEnv(path)
            with self.assertRaisesRegex(ValueError,"零个外部 Import"):
                capture_dlc(
                    sim, Path(td)/"bad",
                    env_factory=fake_factory,
                    inspect_fn=lambda _: make_inventory(
                        imports=[{"name":"IMP_STEER_L1",
                                  "mode":"REPLACE","initial":"0"}]
                    )
                )
            self.assertEqual(called,[])

    def test_zero_import_adapter_does_not_modify_third_party_carsim_env(self):
        # 模拟目前 CarSimEnv 的初始化约定；零 Import 由独立适配器支持。
        from wm_recovery.dlc_env import make_preview_env

        class FakeSolver:
            def read_configuration(self, sim_path):
                return {
                    "n_import": 0, "n_export": 4,
                    "t_start": 0.0, "t_stop": 2.0, "t_step": 0.005,
                }

            def copy_export_vars_into(self, raw_array, n_export,
                                      *, return_list):
                assert return_list is False
                for j in range(n_export):
                    raw_array[j] = float(j * 10)

        class ExistingCarSimEnv:
            def __init__(self, sim_path):
                self.sim_path = sim_path
                self.solver = FakeSolver()
                self.initialized = False
                self.done = True

            def close(self):
                self.done = True

        stub = types.SimpleNamespace(CarSimEnv=ExistingCarSimEnv)
        with patch.dict(sys.modules, {"carsim_env": stub}):
            env = make_preview_env("DLC_simfile.sim")
            first = env.reset()
            self.assertEqual(first, (0.0, 10.0, 20.0, 30.0))
            self.assertEqual(env.n_import, 0)
            self.assertEqual(env._import_c_array._length_, 1)
            self.assertEqual(env.import_np.shape, (0,))
            self.assertEqual(env.export_np.shape, (4,))

    def test_final_short_interval_split_without_losing_last_state(self):
        with tempfile.TemporaryDirectory() as td:
            sim=Path(td)/"simfile.sim"
            sim.write_text("SIMFILE\\nEND\\n",encoding="utf-8")

            class PartialEndingEnv(FakeCarSimEnv):
                def control_step(self, action, inner_steps):
                    values,reward,ended,info=super().control_step(
                        action,inner_steps,
                    )
                    if ended:
                        self.t_current -= 0.003  # 最后一步仅 7ms。
                    return values,reward,ended,info

            out=Path(td)/"short_end"
            result=capture_dlc(
                sim,out,sample_dt_s=.01,
                env_factory=lambda path:PartialEndingEnv(path),
                inspect_fn=lambda _:make_inventory(),
            )
            self.assertEqual(result["n_samples"],5)
            self.assertEqual(result["raw_solver_samples"],6)
            self.assertAlmostEqual(result["solver_final_time_s"],.047)
            self.assertAlmostEqual(result["actual_final_time_s"],.04)
            self.assertAlmostEqual(
                result["sampling_diagnostics"]["terminal_partial_dt_s"],
                .007,
            )
            self.assertTrue(result["time_grid_uniform_0p01s"])
            with np.load(out/"dlc_exports.npz",
                         allow_pickle=False) as data:
                np.testing.assert_allclose(
                    data["time_s"],np.arange(5)*.01,
                )
                self.assertEqual(data["raw_exports"].shape,(5,4))
            with np.load(out/"terminal_partial_step.npz",
                         allow_pickle=False) as tail:
                self.assertAlmostEqual(tail["time_s"][0],.047)
                self.assertEqual(tail["raw_exports"].shape,(1,4))
                np.testing.assert_allclose(tail["raw_exports"][0],
                                            [50,51,52,53])

    def test_middle_gap_never_discarded_or_silently_interpolated(self):
        with tempfile.TemporaryDirectory() as td:
            sim=Path(td)/"simfile.sim"
            sim.write_text("SIMFILE\\nEND\\n",encoding="utf-8")

            class SkippingEnv(FakeCarSimEnv):
                def control_step(self, action, inner_steps):
                    values,reward,ended,info=super().control_step(
                        action,inner_steps,
                    )
                    if self.i >= 3:
                        self.t_current += .005
                    return values,reward,ended,info

            out=Path(td)/"invalid"
            with self.assertRaisesRegex(ValueError,"异常数量"):
                capture_dlc(
                    sim,out,env_factory=lambda path:SkippingEnv(path),
                    inspect_fn=lambda _:make_inventory(),
                )
            # 即使中途采样异常，仍隔离保留原始 CarSim 数据供排查。
            self.assertTrue(out.exists())
            self.assertTrue(
                (out/"INVALID_TIMING_do_not_train.npz").exists()
            )
            self.assertTrue(
                (out/"INVALID_TIMING_details.txt").exists()
            )
            self.assertFalse((out/"dlc_exports.npz").exists())

    def test_diagnostics_explain_actual_time_intervals(self):
        with self.assertRaisesRegex(ValueError,"中位数"):
            check_sampling_times(
                np.asarray([0.0,.01,.025,.035,.045]),
                .01,solver_done=True,
            )
        with self.assertRaises(ValueError):
            check_sampling_times(
                np.asarray([0.0,.01,.017]),
                .01,solver_done=True,terminal_partial_policy="reject",
            )
        ok=check_sampling_times(
            np.asarray([0.0,.01,.02,.027]),
            .01,solver_done=True,terminal_partial_policy="split",
        )
        self.assertEqual(ok["uniform_sample_count"],3)
        self.assertTrue(ok["terminal_partial_detected"])

    def test_bad_solver_export_width_rejected_and_closed(self):
        with tempfile.TemporaryDirectory() as td:
            sim=Path(td)/"simfile.sim"
            sim.write_text("SIMFILE\nEND\n",encoding="utf-8")
            refs=[]
            def factory(path):
                env=FakeCarSimEnv(path,n_export=3)
                refs.append(env)
                return env
            with self.assertRaisesRegex(ValueError,"数量"):
                capture_dlc(sim,Path(td)/"bad",
                            env_factory=factory,
                            inspect_fn=lambda _: make_inventory())
            self.assertTrue(refs[0].closed)

if __name__ == "__main__":
    unittest.main()
