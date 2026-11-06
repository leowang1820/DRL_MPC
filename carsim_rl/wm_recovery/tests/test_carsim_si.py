"""仅测试 SI 转换，不启动 CarSim。"""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from wm_recovery.carsim_export_schema import EXPORT_NAMES
from wm_recovery.controls import DIRECT_IMPORT_NAMES
from wm_recovery.carsim_si import load_si_view, save_si_audit


def make_capture(root, unequal=True):
    """构造完整 61 通道记录，验证转换和同角审计。"""
    root = Path(root)
    raw = np.zeros((2, 61))
    def set_column(name, value):
        raw[:, tuple(EXPORT_NAMES).index(name)] = value
    set_column("Vx", 36)
    set_column("Vy", 3.6)
    set_column("AVz", 180)
    for name in ("AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2"):
        set_column(name, 60)
    set_column("Steer_L1", 1.0)
    set_column("Steer_R1", 1.1 if unequal else 1.0)
    set_column("Ax", 1.0)
    set_column("Ay", -0.5)
    raw[1, tuple(EXPORT_NAMES).index("My_US_L1")] = 20
    np.savez_compressed(
        root / "raw_capture.npz",
        time_s=[0, 0.01],
        raw_exports=raw,
        command=[[20, 0, 0, 0, math.radians(1)]],
        import_values=[[20, 0, 0, 0, 1, 1]],
    )
    (root / "metadata.json").write_text(json.dumps({
        "format": "wm_recovery_carsim_raw_capture_v1",
        "export_names": list(EXPORT_NAMES),
        "import_names": list(DIRECT_IMPORT_NAMES),
    }), encoding="utf-8")


class CarSimSIUnitTests(unittest.TestCase):
    def test_conversion_and_unequal_front_steer(self):
        with tempfile.TemporaryDirectory() as folder:
            make_capture(folder)
            result = load_si_view(folder)
            np.testing.assert_allclose(
                result["state_si"][0],
                [10, 1, math.pi, 2*math.pi, 2*math.pi,
                 2*math.pi, 2*math.pi],
            )
            np.testing.assert_allclose(
                result["reported_accel_mps2"][0],
                [9.80665, -4.903325],
            )
            self.assertFalse(result["report"]["steering_strictly_equal_for_7dof"])
            self.assertTrue(result["report"]["reported_torque_matches_import"])
            self.assertFalse(result["report"]["ready_for_wm_training"])

    def test_wrong_import_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            make_capture(folder)
            file = Path(folder) / "raw_capture.npz"
            with np.load(file) as record:
                values = {name: record[name] for name in record.files}
            values["import_values"][0, 5] = 9.0
            np.savez_compressed(file, **values)
            with self.assertRaises(ValueError):
                load_si_view(folder)

    def test_saved_audit_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            make_capture(folder, unequal=False)
            output = Path(folder) / "si_audit"
            report = save_si_audit(folder, output)
            self.assertTrue(report["steering_strictly_equal_for_7dof"])
            self.assertFalse(report["ready_for_wm_training"])
            self.assertTrue((output / "si_audit.npz").exists())
            self.assertTrue((output / "audit.json").exists())
            with self.assertRaises(FileExistsError):
                save_si_audit(folder, output)


if __name__ == "__main__":
    unittest.main()
