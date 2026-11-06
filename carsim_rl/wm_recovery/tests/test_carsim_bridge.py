"""仅使用模拟环境测试 CarSim 数据采集桥接，不加载 DLL。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_recovery.carsim_bridge import (
    CarSimBridge,
    verify_inventory,
)
from wm_recovery.controls import DIRECT_IMPORT_NAMES
from wm_recovery.carsim_export_schema import EXPORT_NAMES


def fake_inventory():
    """构造六输入、六十一输出的静态配置检查结果。"""
    return {
        "imports": [
            {
                "name": name,
                "mode": "ADD" if index < 4 else "REPLACE",
                "initial": "0",
            }
            for index, name in enumerate(DIRECT_IMPORT_NAMES)
        ],
        "exports": list(EXPORT_NAMES),
        "ports": {
            "PORTS_IMP": 6,
            "PORTS_EXP": 61,
        },
        "sim_path": "fake.sim",
        "parameter_path": "fake.par",
        "sim_sha256": "mock",
        "parameter_sha256": "mock",
    }


class FakeCarSimEnv:
    """模拟现有 CarSimEnv 的必要接口。"""

    instances = []

    def __init__(self, sim_path, t_step=0.001):
        self.sim_path = sim_path
        self.config = {
            "n_import": 6,
            "n_export": 61,
            "t_start": 0.0,
            "t_stop": 1.0,
            "t_step": t_step,
        }
        self.t_current = 0.0
        self.done = True
        self.closed = False
        self.received = []
        self.__class__.instances.append(self)

    def reset(self):
        self.done = False
        return tuple(np.arange(1, 62, dtype=float))

    def control_step(self, action, inner_steps):
        values = np.asarray(action, dtype=float)
        assert values.shape == (6,)
        self.received.append((values.copy(), inner_steps))
        self.t_current += inner_steps * self.config["t_step"]
        raw = np.arange(1, 62, dtype=float)
        raw[2] = 20.0 + self.t_current
        return tuple(raw), 0.0, False, {"return_code": 0}

    def close(self):
        self.closed = True
        self.done = True


class BridgeTests(unittest.TestCase):

    def make_bridge(self, env_factory=FakeCarSimEnv, inspect_fn=None):
        return CarSimBridge(
            "mock.sim",
            env_factory=env_factory,
            inspection_fn=inspect_fn or (lambda path: fake_inventory()),
        )

    def test_inventory_and_duplicate_exports(self):
        self.assertTrue(verify_inventory(fake_inventory()))
        altered = fake_inventory()
        altered["exports"][39] = "Vx"
        with self.assertRaises(ValueError):
            verify_inventory(altered)

    def test_mode_or_order_mismatch_rejected(self):
        altered = fake_inventory()
        altered["imports"][-1]["mode"] = "ADD"
        with self.assertRaises(ValueError):
            verify_inventory(altered)
        altered = fake_inventory()
        altered["imports"][0], altered["imports"][1] = (
            altered["imports"][1], altered["imports"][0]
        )
        with self.assertRaises(ValueError):
            verify_inventory(altered)

    def test_capture_and_save(self):
        bridge = self.make_bridge()
        start = bridge.reset()

        self.assertEqual(bridge.inner_steps, 10)
        self.assertEqual(start.data.Steer_SW_1, 40.0)
        self.assertEqual(start.data.Steer_SW_2, 49.0)

        step = bridge.step([100, -20, 30, -40, np.deg2rad(3)])
        self.assertAlmostEqual(step.time_s, 0.01)
        self.assertAlmostEqual(step.data.Vx, 20.01)

        action, nsteps = FakeCarSimEnv.instances[-1].received[-1]
        np.testing.assert_allclose(action, [100, -20, 30, -40, 3, 3])
        self.assertEqual(nsteps, 10)

        with tempfile.TemporaryDirectory() as temp:
            result = bridge.save_raw(Path(temp) / "capture")
            self.assertEqual(result["samples"], 2)

            with np.load(
                Path(temp) / "capture" / "raw_capture.npz",
                allow_pickle=False,
            ) as recorded:
                self.assertEqual(recorded["raw_exports"].shape, (2, 61))
                self.assertEqual(recorded["command"].shape, (1, 5))
                self.assertEqual(recorded["import_values"].shape, (1, 6))
                np.testing.assert_allclose(recorded["time_s"], [0, 0.01])

            meta = json.loads(
                (Path(temp) / "capture" / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(meta["ready_for_wm_training"])
            self.assertEqual(meta["source_kind"], "mock_or_injected")
            self.assertAlmostEqual(meta["solver_t_step_s"], 0.001)
            self.assertEqual(len(meta["export_names"]), 61)
            self.assertEqual(meta["export_names"][39], "Steer_SW")
            self.assertEqual(meta["export_names"][48], "Steer_SW")
            with self.assertRaises(FileExistsError):
                bridge.save_raw(Path(temp) / "capture")

        bridge.close()
        self.assertTrue(FakeCarSimEnv.instances[-1].closed)

    def test_invalid_control_rejected_before_env_step(self):
        bridge = self.make_bridge()
        bridge.reset()

        with self.assertRaises(ValueError):
            bridge.step([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            bridge.step([1, 2, 3, 4, float("nan")])

        self.assertEqual(
            len(FakeCarSimEnv.instances[-1].received), 0
        )
        bridge.close()

    def test_incompatible_solver_timestep_rejected(self):
        factory = lambda path: FakeCarSimEnv(path, t_step=0.003)
        bridge = self.make_bridge(env_factory=factory)

        with self.assertRaises(ValueError):
            bridge.reset()

        self.assertTrue(FakeCarSimEnv.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
