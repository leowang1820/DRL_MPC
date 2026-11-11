# -*- coding: utf-8 -*-
"""NMPC 等效输入映射：不调用 CarSim DLL 的数值与契约测试。"""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_recovery.carsim_export_schema import EXPORT_COUNT, get_export_index
from wm_recovery.nmpc_actuation import (
    NMPCActuationConfig,
    NMPC_INPUT_NAMES,
    derive_nmpc_actuation,
    load_from_candidate,
    audit_workspace,
    contiguous_true_runs,
)


def row(**values):
    x = np.zeros(EXPORT_COUNT, dtype=float)
    for name, value in values.items():
        x[get_export_index(name)] = value
    return x


class NMPCActuationTests(unittest.TestCase):

    def test_native_units_order_front_force_lambda_mz(self):
        raw = row(
            Vx=72.0, Vy=7.2, AVz=30.0, Steer_L1=0.0, Steer_R1=0.0,
            Fx_L1=100.0, Fx_R1=120.0, Fx_L2=100.0, Fx_R2=100.0,
        )
        r = derive_nmpc_actuation(raw)
        self.assertEqual(r.z_si.shape, (1,8))
        self.assertEqual(len(NMPC_INPUT_NAMES),8)
        self.assertAlmostEqual(r.z_si[0,0],2.0)  # Vy km/h -> m/s
        self.assertAlmostEqual(r.z_si[0,1],math.pi/6)  # deg/s -> rad/s
        self.assertAlmostEqual(r.z_si[0,2],20.0)
        self.assertAlmostEqual(r.z_si[0,3],0.0)
        self.assertAlmostEqual(r.z_si[0,4],16.05)  # B/2*(120-100)
        self.assertAlmostEqual(r.z_si[0,5],420.0)
        self.assertAlmostEqual(r.z_si[0,6],220.0/420.0)
        self.assertAlmostEqual(r.z_si[0,7],0.8)
        self.assertTrue(bool(r.valid_mask[0]))

    def test_front_steer_average_and_lateral_force_rotation(self):
        raw = row(
            Vx=54.0, Steer_L1=10.0, Steer_R1=14.0,
            Fx_L1=100.0, Fx_R1=120.0,
            Fy_L1=30.0, Fy_R1=40.0,
            Fx_L2=110.0, Fx_R2=90.0,
        )
        r = derive_nmpc_actuation(raw)
        flb=100*math.cos(math.radians(10))-30*math.sin(math.radians(10))
        frb=120*math.cos(math.radians(14))-40*math.sin(math.radians(14))
        expected=0.5*1.605*((frb-flb)+(90-110))
        self.assertAlmostEqual(r.z_si[0,3],math.radians(12.0))
        self.assertAlmostEqual(r.delta_left_right_diff_deg[0],-4.0)
        self.assertAlmostEqual(r.z_si[0,4],expected)

    def test_lamda_is_not_clipped_even_if_out_of_expected_band(self):
        raw = row(Fx_L1=500.0, Fx_R1=500.0, Fx_L2=0.0, Fx_R2=0.0)
        r=derive_nmpc_actuation(raw)
        self.assertAlmostEqual(r.z_si[0,6],1.0)
        self.assertTrue(bool(r.valid_mask[0]))
        self.assertFalse(bool(r.lamda_in_expected_band_mask[0]))
        self.assertEqual(r.as_summary()["lamda_outside_expected_band_samples"],1)

    def test_opposing_axles_near_zero_total_force_is_unobservable(self):
        raw = row(Fx_L1=100.0, Fx_R1=100.0, Fx_L2=-100.0, Fx_R2=-100.0)
        r=derive_nmpc_actuation(raw)
        self.assertTrue(math.isnan(r.z_si[0,6]))
        self.assertFalse(bool(r.lamda_observable_mask[0]))
        self.assertFalse(bool(r.valid_mask[0]))
        self.assertEqual(r.as_summary()["lamda_unobservable_samples"],1)

    def test_low_nonzero_total_force_uses_declared_neutral_value(self):
        raw = row(Fx_L1=30.0, Fx_R1=15.0, Fx_L2=2.0, Fx_R2=2.0)
        r = derive_nmpc_actuation(raw)
        self.assertTrue(bool(r.valid_mask[0]))
        self.assertTrue(bool(r.lamda_neutral_mask[0]))
        self.assertFalse(bool(r.lamda_observable_mask[0]))
        self.assertAlmostEqual(r.z_si[0, 6], 0.5)
        self.assertEqual(r.lamda_regime[0], "neutral")
        permissive = derive_nmpc_actuation(
            raw, NMPCActuationConfig(min_abs_fx_all_n=0.0)
        )
        self.assertTrue(bool(permissive.lamda_observable_mask[0]))
        self.assertAlmostEqual(permissive.z_si[0, 6], 45.0/49.0)

    def test_distinct_time_rows_are_not_future_shifted(self):
        raw=np.stack([
            row(Vx=36, Fx_L1=100, Fx_R1=100, Fx_L2=100, Fx_R2=100),
            row(Vx=72, Fx_L1=200, Fx_R1=200, Fx_L2=100, Fx_R2=100),
        ])
        r=derive_nmpc_actuation(raw)
        self.assertEqual(r.z_si.shape,(2,8))
        self.assertAlmostEqual(r.z_si[0,2],10)
        self.assertAlmostEqual(r.z_si[1,2],20)
        self.assertAlmostEqual(r.z_si[0,5],400)
        self.assertAlmostEqual(r.z_si[1,5],600)
        self.assertEqual(r.z_si[:-1].shape,(1,8))

    def test_invalid_shapes_and_missing_finiteness(self):
        with self.assertRaises(ValueError):
            derive_nmpc_actuation(np.zeros((3,60)))
        with self.assertRaises(ValueError):
            derive_nmpc_actuation(np.zeros((0,61)))
        with self.assertRaises(ValueError):
            derive_nmpc_actuation(np.zeros((2,61,1)))
        bad=np.zeros(61)
        bad[get_export_index("Vx")]=float('nan')
        with self.assertRaises(ValueError):
            derive_nmpc_actuation(bad)

    def test_parameters_are_explicit(self):
        with self.assertRaises(ValueError):
            NMPCActuationConfig(front_track_m=-1.0).validate()
        with self.assertRaises(ValueError):
            NMPCActuationConfig(mu_hat=0).validate()

    def test_clean_candidate_raw_source_and_time_alignment(self):
        """使用项目原有严格候选读取器，验证保留前缀和原始文件指纹。"""
        from wm_recovery.carsim_candidate import _fingerprint

        n=200
        raw=np.stack([
            row(Vx=72, Vy=0, AVz=0, Steer_L1=0, Steer_R1=0,
                Fx_L1=100, Fx_R1=110, Fx_L2=100, Fx_R2=110)
            for _ in range(n+1)
        ])
        # 模拟清洗后只保留到 1.25s；原始数据还包含更长的时长。
        keep=125
        timeline=np.arange(n+1)*0.01
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            raw_dir=root/"raw"
            raw_dir.mkdir()
            raw_file=raw_dir/"raw_capture.npz"
            np.savez_compressed(
                raw_file,time_s=timeline,raw_exports=raw,
                command=np.zeros((n,5)),
                import_values=np.zeros((n,6)),
            )
            raw_hash=hashlib.sha256(raw_file.read_bytes()).hexdigest()
            candidate_dir=root/"candidate"
            candidate_dir.mkdir()
            arrays={
                "time_s":timeline[:keep+1],
                "state_si":np.column_stack((
                    np.full(keep+1,20.0),
                    np.zeros((keep+1,6))
                )),
                "command_si":np.zeros((keep,5)),
                "nominal_7dof_input_si":np.zeros((keep,6)),
                "measured_steer_rad":np.zeros((keep+1,2)),
                "reported_external_torque_nm":np.zeros((keep+1,4)),
                "reported_accel_mps2":np.zeros((keep+1,2)),
                "import_native":np.zeros((keep,6)),
                "my_dr_native":np.zeros((keep+1,4)),
                "nominal_initial_load_memory_mps2":np.zeros(2),
            }
            report={
                "format":"wm_recovery_carsim_candidate_v1",
                "status":"candidate_created_not_training_ready",
                "source_kind":"carsim_env",
                "raw_source":str(raw_file),
                "raw_source_sha256":raw_hash,
                "canonical_physical_arrays_sha256":_fingerprint(arrays),
                "dt_s":0.01,
                "samples":keep+1,
                "intervals":keep,
                "split":"unassigned",
                "model_input_definition":
                    "equal_front_commanded_steer_plus_four_commanded_external_torques",
                "interval_alignment":
                    "command_i_holds_from_t_i_to_t_i_plus_1",
                "torque_echo_matches":True,
                "ready_for_wm_training":False,
            }
            np.savez_compressed(candidate_dir/"candidate.npz",**arrays)
            (candidate_dir/"candidate_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            time_s,r=load_from_candidate(candidate_dir)
            self.assertEqual(len(time_s),keep+1)
            self.assertEqual(r.z_si.shape,(keep+1,8))
            self.assertEqual(int(r.valid_mask.sum()), keep+1)
            self.assertAlmostEqual(r.z_si[0,4],16.05)
            # 改动原始文件即使数据仍然有 61 通道也应被拒绝。
            with raw_file.open("ab") as f:
                f.write(b"x")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                load_from_candidate(candidate_dir)

    def test_three_regimes_exhaustive_and_not_clipped(self):
        raw = np.stack([
            row(Fx_L1=100, Fx_R1=0, Fx_L2=100, Fx_R2=0),    # ratio .5
            row(Fx_L1=0, Fx_R1=0, Fx_L2=0, Fx_R2=0),        # neutral
            row(Fx_L1=100, Fx_R1=0, Fx_L2=-100, Fx_R2=0),  # cancel
            row(Fx_L1=200, Fx_R1=0, Fx_L2=-100, Fx_R2=0),  # ratio 2.0
            row(Fx_L1=-200, Fx_R1=0, Fx_L2=100, Fx_R2=0),  # ratio 2.0
        ])
        result = derive_nmpc_actuation(raw)
        np.testing.assert_array_equal(
            result.lamda_regime,
            ["ratio", "neutral", "cancellation", "ratio", "ratio"]
        )
        np.testing.assert_array_equal(
            result.valid_mask, [True, True, False, True, True]
        )
        self.assertTrue(np.isnan(result.z_si[2, 6]))
        self.assertAlmostEqual(result.z_si[0, 6], 0.5)
        self.assertAlmostEqual(result.z_si[1, 6], 0.5)
        self.assertAlmostEqual(result.z_si[3, 6], 2.0)
        self.assertAlmostEqual(result.z_si[4, 6], 2.0)
        summary = result.as_summary()
        self.assertEqual(summary["lamda_ratio_samples"], 3)
        self.assertEqual(summary["lamda_neutral_assumed_samples"], 1)
        self.assertEqual(summary["lamda_cancellation_samples"], 1)
        self.assertEqual(summary["valid_z_samples"], 4)
        self.assertEqual(summary["lamda_unobservable_samples"], 2)

    def test_cancellation_threshold_boundary_and_custom_neutral(self):
        # Each axle at exactly +/-50 N is classified neutral.
        r = derive_nmpc_actuation(
            row(Fx_L1=50.0, Fx_L2=-50.0),
        )
        self.assertTrue(bool(r.lamda_neutral_mask[0]))
        self.assertAlmostEqual(r.z_si[0, 6], 0.5)

        # Slightly over axle threshold with still-small net force -> cancellation.
        r = derive_nmpc_actuation(
            row(Fx_L1=50.1, Fx_L2=-50.1)
        )
        self.assertTrue(bool(r.lamda_cancellation_mask[0]))
        self.assertFalse(bool(r.valid_mask[0]))

        # Neutral convention can be changed and is not clipped to 0.4..0.6.
        custom = NMPCActuationConfig(neutral_lamda=0.9)
        r = derive_nmpc_actuation(row(), custom)
        self.assertAlmostEqual(r.z_si[0, 6], 0.9)
        self.assertFalse(bool(r.lamda_in_expected_band_mask[0]))

    def test_longest_contiguous_valid_interval(self):
        mask = np.array([True, True, False, True, True, True,
                         False, True, False, False], dtype=bool)
        times = np.arange(11, dtype=float) * 0.01
        runs = contiguous_true_runs(mask, times)
        self.assertEqual([r["intervals"] for r in runs], [2, 3, 1])
        self.assertEqual(runs[1]["start_index"], 3)
        self.assertEqual(runs[1]["end_index_exclusive"], 6)
        self.assertAlmostEqual(runs[1]["duration_s"], 0.03)
        self.assertEqual(contiguous_true_runs(np.zeros(3, bool), times[:4]), [])
        self.assertEqual(
            contiguous_true_runs(np.ones(3, bool), times[:4])[0]["intervals"], 3
        )
        with self.assertRaises(ValueError):
            contiguous_true_runs(mask.astype(int), times)

    def test_workspace_audit_by_split_speed_and_longest_run(self):
        import unittest.mock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = [
                ("v40_train_turn", "train",
                 [(100., 100.), (100., 100.), (100., -100.),
                  (0., 0.), (0., 0.), (0., 0.)]),
                ("v60_validation_turn", "validation",
                 [(0., 0.), (100., -100.), (100., 100.),
                  (100., 100.), (100., 100.), (100., 100.)]),
            ]
            rows = []
            mapping = {}
            for name, split, forces in inputs:
                ep_dir = root / "episodes" / name
                ep_dir.mkdir(parents=True)
                cand = root / "candidates" / name
                cand.mkdir(parents=True)
                (ep_dir/"experimental_episode.json").write_text(
                    json.dumps({"candidate_path": str(cand)}),
                    encoding="utf-8"
                )
                rows.append({"name": name, "split": split,
                             "episode_path": str(ep_dir)})
                exports = np.stack([
                    row(Fx_L1=front, Fx_L2=rear)
                    for front, rear in forces
                ])
                mapping[str(cand)] = (
                    np.arange(len(forces), dtype=float) * 0.01,
                    derive_nmpc_actuation(exports)
                )

            (root/"registry.json").write_text(
                json.dumps({
                    "format": "wm_recovery_experimental_registry_v1",
                    "episodes": rows,
                }), encoding="utf-8"
            )
            def fake_loader(path, config):
                return mapping[str(path)]

            with unittest.mock.patch(
                "wm_recovery.nmpc_actuation.load_from_candidate",
                side_effect=fake_loader
            ):
                report = audit_workspace(root)

            self.assertEqual(report["total_interval_inputs"], 10)
            self.assertEqual(report["regime_totals"]["ratio_interval_inputs"], 5)
            self.assertEqual(report["regime_totals"]["neutral_interval_inputs"], 3)
            self.assertEqual(report["regime_totals"]["cancellation_interval_inputs"], 2)
            self.assertEqual(report["valid_interval_inputs"], 8)
            self.assertEqual(report["by_speed"]["v40"]["episode_count"], 1)
            self.assertEqual(report["by_speed"]["v60"]["episode_count"], 1)
            self.assertEqual(report["by_split"]["train"]["episode_count"], 1)
            self.assertEqual(report["by_split"]["validation"]["episode_count"], 1)
            self.assertEqual(
                report["longest_contiguous_valid_run"]["scene"],
                "v60_validation_turn"
            )
            self.assertEqual(
                report["longest_contiguous_valid_run"]["intervals"], 3
            )
            self.assertEqual(
                report["episodes"][1]["longest_contiguous_ratio_run"]["intervals"],
                3
            )


if __name__ == "__main__":
    unittest.main()
