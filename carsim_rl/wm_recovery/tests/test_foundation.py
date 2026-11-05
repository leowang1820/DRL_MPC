import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dynamics7dof.model import STATE_NAMES
from wm_recovery.controls import APPLIED_INPUT_NAMES, COMMAND_NAMES, DIRECT_IMPORT_NAMES, Command
from wm_recovery.inspection import inspect_simfile
from wm_recovery.offline import FORMAT, Episode, audit_partition, evaluate_nominal


def write_episode(root, *, identifier="example", condition="straight", split="train", actual_mu=.8):
    root.mkdir()
    meta = {
        "format": FORMAT, "trajectory_id": identifier, "condition_id": condition,
        "source_kind": "synthetic", "split": split, "nominal_mu": .8, "actual_mu": actual_mu,
        "state_names": list(STATE_NAMES), "applied_input_names": list(APPLIED_INPUT_NAMES),
        "command_names": list(COMMAND_NAMES), "input_alignment": "held_over_interval_t_i_to_t_i_plus_1",
        "actuation_definition": "external_wheel_torque_without_friction_brake",
        "other_wheel_actuation_verified_zero": True, "parameter_alignment": "unverified",
        "initial_memory_source": "synthetic_equilibrium",
    }
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    time = np.linspace(0, 1.2, 13)
    state = np.tile([10, 0, 0, *([10 / .353] * 4)], (len(time), 1))
    np.savez(root / "trajectory.npz", time_s=time, state=state,
             applied_input=np.zeros((len(time) - 1, 6)), command=np.zeros((len(time) - 1, 5)),
             initial_load_memory_m_s2=np.zeros(2))
    return root


class FoundationTests(unittest.TestCase):
    def test_signed_torques_equal_angles_and_channel_order(self):
        command = Command.from_vector([10, -20, 30, -40, np.deg2rad(3)])
        np.testing.assert_allclose(command.to_imports(DIRECT_IMPORT_NAMES), [10, -20, 30, -40, 3, 3])
        np.testing.assert_allclose(command.to_imports(DIRECT_IMPORT_NAMES[::-1]), [3, 3, -40, 30, -20, 10])
        with self.assertRaises(ValueError):
            command.to_imports((*DIRECT_IMPORT_NAMES[:4], "IMP_STEER_SW"))

    def test_invalid_actions_are_never_padded_or_truncated(self):
        for bad in ([1] * 4, [1] * 6, [0, 0, float("nan"), 0, 0], [0, 0, 0, 0, np.pi]):
            with self.assertRaises(ValueError):
                Command.from_vector(bad)

    def test_configuration_inventory_preserves_modes_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.par").write_text("PARSFILE\nIMPORT IMP_MYUSM_L1 Add 0\nEXPORT Steer_SW\nEXPORT Steer_SW\nEND\n")
            (root / "run.sim").write_text("SIMFILE\nSET_MACRO $(ROOT)$ .\nINPUT $(ROOT)$/run.par\nPORTS_IMP 1,1\nPORTS_EXP 1,2\nEND\n")
            result = inspect_simfile(root / "run.sim")
            self.assertTrue(result["port_counts_match"])
            self.assertFalse(result["physical_interface_verified"])
            self.assertEqual(result["imports"][0]["mode"], "ADD")
            self.assertEqual(result["duplicate_exports"], {"STEER_SW": 2})

    def test_synthetic_free_rolling_and_exact_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = Episode.load(write_episode(Path(tmp) / "one"))
            report, arrays = evaluate_nominal(ep, max_substep_s=.0005)
            self.assertEqual(report["source_kind"], "synthetic")
            self.assertEqual(report["continuous"]["coverage"], 1)
            self.assertEqual(report["rolling_horizon"]["total_samples"], 3)
            self.assertLess(max(report["continuous"]["rmse"].values()), 1e-8)
            np.testing.assert_array_equal(arrays["horizon_end_indices"], [10, 11, 12])

    def test_unknown_mu_is_never_used_by_nominal_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = Episode.load(write_episode(root / "baseline"))
            unknown = Episode.load(write_episode(root / "unknown", split="unknown_mu_test", actual_mu=.3))
            expected, _ = evaluate_nominal(baseline, max_substep_s=.0005)
            actual, _ = evaluate_nominal(unknown, max_substep_s=.0005)
            self.assertEqual(actual["one_step"], expected["one_step"])
            self.assertEqual(actual["nominal_mu_used"], .8)
            self.assertEqual(actual["actual_mu_for_reporting_only"], .3)

    def test_non_nominal_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown_mu_test"):
                Episode.load(write_episode(Path(tmp) / "bad", actual_mu=.5))

    def test_duplicate_trajectory_and_cross_split_condition_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = Episode.load(write_episode(root / "one"))
            two = Episode.load(write_episode(root / "two", identifier="two", split="test"))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit_partition([one, two])
            two.source_sha256 = "a different physical recording"
            with self.assertRaisesRegex(ValueError, "condition_id"):
                audit_partition([one, two])

    def test_missing_actuation_verification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_episode(Path(tmp) / "one")
            meta = json.loads((root / "metadata.json").read_text())
            meta["other_wheel_actuation_verified_zero"] = False
            (root / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "drive/friction"):
                Episode.load(root)

    def test_actual_unequal_front_angles_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_episode(Path(tmp) / "one")
            with np.load(root / "trajectory.npz") as source:
                arrays = {key: source[key].copy() for key in source.files}
            arrays["applied_input"][:, 1] = .001
            np.savez(root / "trajectory.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "equal actual"):
                Episode.load(root)

    def test_recompressing_a_recording_does_not_make_an_independent_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = Episode.load(write_episode(root / "one"))
            second_path = write_episode(root / "two", identifier="two", condition="two", split="test")
            with np.load(second_path / "trajectory.npz") as source:
                arrays = {key: source[key].copy() for key in source.files}
            np.savez_compressed(second_path / "trajectory.npz", **arrays)
            second = Episode.load(second_path)
            self.assertEqual(first.source_sha256, second.source_sha256)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit_partition([first, second])

    def test_short_data_never_claim_one_second_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Episode.load(write_episode(Path(tmp) / "one"))
            episode.time = episode.time / 2
            report, _ = evaluate_nominal(episode, max_substep_s=.0005)
            self.assertEqual(report["rolling_horizon"]["total_samples"], 0)


if __name__ == "__main__":
    unittest.main()
