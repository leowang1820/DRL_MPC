# -*- coding: utf-8 -*-
"""Post-capture cleaning without CarSim DLL: guards, time alignment and provenance."""

import json
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from wm_recovery import data_cleaning as q


def fake_arrays(n=600, speed_kmh=60.0):
    time = np.arange(n + 1, dtype=float) * 0.01
    state = np.zeros((n + 1, 7), dtype=float)
    state[:, 0] = speed_kmh / 3.6
    steer = np.zeros((n + 1, 2), dtype=float)
    ext = np.zeros((n + 1, 4), dtype=float)
    accel = np.zeros((n + 1, 2), dtype=float)
    accel[:,1] = 0.78 * q.G
    command = np.zeros((n,5), dtype=float)
    nominal = np.zeros((n,6), dtype=float)
    imports = np.zeros((n,6), dtype=float)
    return {
        "time_s": time, "state_si": state,
        "command_si": command,
        "nominal_7dof_input_si": nominal,
        "measured_steer_rad": steer,
        "reported_external_torque_nm": ext,
        "reported_accel_mps2": accel,
        "import_native": imports,
        "my_dr_native": np.zeros((n+1,4), dtype=float),
        "nominal_initial_load_memory_mps2": np.zeros(2, dtype=float),
    }


class QualityTests(unittest.TestCase):
    def test_all_clean_retained(self):
        a = fake_arrays()
        res = q.analyze_candidate_arrays(a,q.QualityPolicy())
        self.assertTrue(res["kept"])
        self.assertEqual(res["keep_intervals"],600)
        self.assertIsNone(res["first_bad_interval"])

    def test_beta_violation_trims_prefix_not_stitching(self):
        a=fake_arrays()
        a["state_si"][301:, 1] = a["state_si"][301:, 0] * math.tan(math.radians(35))
        res=q.analyze_candidate_arrays(a,q.QualityPolicy(guard_s=0.05))
        self.assertTrue(res["kept"])
        self.assertEqual(res["first_bad_interval"],300)
        self.assertEqual(res["keep_intervals"],295)
        trim=q._trim_arrays(a,res["keep_intervals"])
        self.assertEqual(trim["state_si"].shape,(296,7))
        self.assertEqual(trim["command_si"].shape,(295,5))
        self.assertEqual(trim["measured_steer_rad"].shape,(296,2))
        self.assertAlmostEqual(trim["time_s"][-1],2.95)

    def test_vx_negative_trims_even_if_no_beta_limit(self):
        a=fake_arrays()
        a["state_si"][401:,0]=-2.0
        res=q.analyze_candidate_arrays(a,q.QualityPolicy(beta_limit_deg=89,guard_s=0))
        self.assertEqual(res["first_bad_interval"],400)
        self.assertEqual(res["keep_intervals"],400)
        self.assertGreater(res["reason_counts"]["vx_too_low"],0)

    def test_steering_rate_jitter_trims(self):
        a=fake_arrays()
        a["measured_steer_rad"][201, 0]=math.radians(20)
        res=q.analyze_candidate_arrays(a,q.QualityPolicy(guard_s=0))
        self.assertEqual(res["first_bad_interval"],200)
        self.assertGreater(res["reason_counts"]["steer_rate_too_large"],0)

    def test_steering_acceleration_trigger(self):
        a=fake_arrays()
        # 0.5° 单采样间隔速率 50deg/s，但是加速度达 5000deg/s²。
        a["measured_steer_rad"][201, 0]=math.radians(0.5)
        p=q.QualityPolicy(
            max_steer_rate_deg_s=1000,
            max_steer_accel_deg_s2=4000,
            guard_s=0,
        )
        res=q.analyze_candidate_arrays(a,p)
        self.assertGreater(res["reason_counts"]["steer_accel_too_large"],0)

    def test_torque_pulses_are_not_automatically_jitter(self):
        a=fake_arrays()
        a["reported_external_torque_nm"][300:,0]=1000
        a["import_native"][299:,0]=1000
        a["command_si"][299:,0]=1000
        a["nominal_7dof_input_si"][299:,2]=1000
        base=q.analyze_candidate_arrays(a,q.QualityPolicy())
        self.assertEqual(base["keep_intervals"],600)
        flagged=q.analyze_candidate_arrays(
            a,q.QualityPolicy(max_external_torque_rate_nm_s=1000,guard_s=0)
        )
        self.assertLess(flagged["keep_intervals"],600)

    def test_short_prefix_is_excluded_not_patched(self):
        a=fake_arrays()
        a["state_si"][40:,0]=-1.0
        res=q.analyze_candidate_arrays(a,q.QualityPolicy())
        self.assertFalse(res["kept"])
        self.assertLess(res["keep_intervals"],120)

    def test_invalid_time_grid_is_blocked(self):
        a=fake_arrays()
        a["time_s"][155]+=0.002
        with self.assertRaisesRegex(ValueError,"时间栅格"):
            q.analyze_candidate_arrays(a,q.QualityPolicy())

    def test_reject_negative_or_absurd_thresholds(self):
        for pol in (
            q.QualityPolicy(beta_limit_deg=95),
            q.QualityPolicy(max_steer_rate_deg_s=-10),
            q.QualityPolicy(min_keep_s=0.1),
        ):
            with self.assertRaises(ValueError):
                pol.validate()



    def test_end_to_end_clean_workspace_with_mock_candidate_contract(self):
        """模拟 merge 后四条轨迹，验证新 workspace 与来源签名没有被破坏。"""
        import sys
        import types
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original"
            output = Path(tmp) / "clean"
            (original / "episodes").mkdir(parents=True)

            scenenames = (
                ("v40_train_sis_left","train"),
                ("v40_train_brake_turn_left","train"),
                ("v40_validation_circle_right","validation"),
                ("v40_test_swd_right","test"),
            )
            rows=[]
            for idx,(name,split) in enumerate(scenenames):
                root=Path(tmp)/"sources"/name
                candidate=root/"candidate"
                candidate.mkdir(parents=True)
                a=fake_arrays()
                if "brake" in name:
                    a["state_si"][301:, 1] = (
                        a["state_si"][301:,0]*math.tan(math.radians(35))
                    )
                np.savez_compressed(candidate/"candidate.npz",**a)
                raw_hash=f"raw_record_{idx}"
                sha=f"source_sha_{idx}"
                meta={
                    "format":"wm_recovery_carsim_candidate_v1",
                    "status":"candidate_created_not_training_ready",
                    "source_kind":"carsim_env",
                    "raw_source_sha256":raw_hash,
                    "canonical_physical_arrays_sha256":sha,
                    "dt_s":0.01,
                    "samples":601,"intervals":600,
                    "ready_for_wm_training":False,
                    "torque_echo_matches":True,
                    "max_torque_echo_error_nm":0,
                }
                (candidate/"candidate_report.json").write_text(
                    json.dumps(meta),encoding="utf-8"
                )
                original_scene=name[len("v40_"):]
                (root/"coverage_report.json").write_text(
                    json.dumps({"scenarios":[{
                        "scenario":original_scene,
                        "split":split,
                        "sign":-1 if "right" in name else 1,
                    }]}),encoding="utf-8"
                )
                ep=(original/"episodes"/name)
                ep.mkdir()
                manifest={
                    "format":"wm_recovery_experimental_episode_manifest_v1",
                    "trajectory_id":f"carsim_experimental_{name}",
                    "condition_id":f"test_{name}",
                    "split":split,
                    "actual_mu":0.8,
                    "candidate_path":str(candidate),
                    "candidate_sha256":sha,
                    "raw_source_sha256":raw_hash,
                    "assumption_acknowledged":True,
                    "source_kind":"carsim_experimental",
                    "ready_for_wm_training":False,
                }
                (ep/"experimental_episode.json").write_text(
                    json.dumps(manifest),encoding="utf-8"
                )
                rows.append({
                    "name":name,"split":split,
                    "episode_path":str(ep),
                    "candidate_sha256":sha,
                    "raw_source_sha256":raw_hash,
                })
            (original/"registry.json").write_text(
                json.dumps({
                    "format":q.REGISTRY_FORMAT,
                    "episodes":rows,
                    "episode_count":len(rows),
                }),encoding="utf-8"
            )
            real_hashes={
                f"source_sha_{i}":name for i,(name,_) in enumerate(scenenames)
            }

            def fingerprint(arrays):
                h=hashlib.sha256()
                for name,arr in arrays.items():
                    h.update(name.encode("ascii"))
                    h.update(str(arr.shape).encode("ascii"))
                    h.update(
                        np.ascontiguousarray(arr,dtype="<f8").tobytes()
                    )
                return h.hexdigest()

            def fake_facts(path):
                p=Path(path)
                meta=json.loads(
                    (p/"candidate_report.json").read_text(encoding="utf-8")
                )
                result={
                    "candidate_sha256":meta["canonical_physical_arrays_sha256"],
                    "raw_source_sha256":meta["raw_source_sha256"],
                    "candidate":types.SimpleNamespace(metadata=meta),
                }
                return result

            fake_experimental=types.ModuleType(
                "wm_recovery.carsim_experimental"
            )
            fake_experimental._candidate_facts=fake_facts
            fake_candidate=types.ModuleType("wm_recovery.carsim_candidate")
            fake_candidate._fingerprint=fingerprint
            with patch.dict(sys.modules,{
                "wm_recovery.carsim_experimental":fake_experimental,
                "wm_recovery.carsim_candidate":fake_candidate,
            }):
                summary=q.audit_or_clean(
                    original,None,q.QualityPolicy(),write=False
                )
                self.assertEqual(summary["retained_episodes"],4)
                self.assertFalse(output.exists())
                result=q.audit_or_clean(
                    original,output,q.QualityPolicy(),write=True
                )
                self.assertTrue((output/"registry.json").is_file())
                self.assertTrue((output/"quality_report.json").is_file())
                self.assertTrue((
                    output/"multispeed_coverage_summary.json"
                ).is_file())
                reg=json.loads(
                    (output/"registry.json").read_text(encoding="utf-8")
                )
                self.assertEqual(reg["episode_count"],4)
                trimmed=[
                    row for row in reg["episodes"]
                    if "brake" in row["name"]
                ][0]
                new_manifest=json.loads(
                    (Path(trimmed["episode_path"])/"experimental_episode.json")
                    .read_text(encoding="utf-8")
                )
                new_candidate=Path(new_manifest["candidate_path"])
                self.assertTrue(new_candidate.is_dir())
                with np.load(new_candidate/"candidate.npz") as z:
                    self.assertEqual(z["time_s"].shape,(296,))
                    self.assertEqual(z["command_si"].shape,(295,5))
                self.assertEqual(new_manifest["raw_source_sha256"],
                                 "raw_record_1")
                self.assertFalse(new_manifest["ready_for_wm_training"])

if __name__ == "__main__":
    unittest.main()
