# -*- coding: utf-8 -*-
"""NMPC 8→3 标签生成和严格来源/时间分区的无 DLL 测试。"""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_recovery.carsim_export_schema import EXPORT_COUNT, get_export_index
from wm_recovery.carsim_candidate import _fingerprint
from wm_recovery.nmpc_actuation import derive_nmpc_actuation
from wm_recovery.nmpc_reference import rk4_nominal_step
from wm_recovery.nmpc_residual_dataset import (
    ResidualDatasetConfig,
    iter_contiguous_windows,
    make_dataset,
    prepare_episode,
)


def make_export(q, force_mode):
    """模拟 CarSim 原生单位：Vx,Vy 是 km/h，AVz 是 deg/s。"""
    a=np.zeros(EXPORT_COUNT,dtype=np.float64)
    a[get_export_index("Vx")]=q[2]*3.6
    a[get_export_index("Vy")]=q[0]*3.6
    a[get_export_index("AVz")]=q[1]*180/math.pi
    if force_mode=="ratio":
        front,rear=100.0,100.0
    elif force_mode=="neutral":
        front,rear=0.0,0.0
    elif force_mode=="cancel":
        front,rear=100.0,-100.0
    else:
        raise ValueError(force_mode)
    a[get_export_index("Fx_L1")]=front
    a[get_export_index("Fx_L2")]=rear
    return a


def fixture_episode(root, name, split, speed_mps=15.0,
                    n=130, cancellation_indices=()):
    """生成符合项目原 CandidateEpisode.load 的原始+候选+manifest。

    此数据完全模拟，不作为真实车辆或 CarSim 物理验证使用。
    """
    q=np.zeros((n+1,3),dtype=float)
    q[0]=[0.0,0.0,speed_mps]
    time=np.arange(n+1,dtype=float)*0.01
    raw=np.zeros((n+1,EXPORT_COUNT),dtype=float)
    bias=np.array([0.23,0.012,0.08],dtype=float)
    cancelled=set(cancellation_indices)
    for k in range(n):
        kind="cancel" if k in cancelled else ("ratio" if k%4==0 else "neutral")
        raw[k]=make_export(q[k],kind)
        input_act=derive_nmpc_actuation(raw[k])
        if bool(input_act.valid_mask[0]):
            predicted=rk4_nominal_step(input_act.z_si[0],0.01)
            q[k+1]=predicted+0.01*bias
        else:
            q[k+1]=q[k]+0.01*bias
    raw[n]=make_export(q[n],"neutral")
    raw_dir=root/"sources"/name
    raw_dir.mkdir(parents=True)
    raw_path=raw_dir/"raw_capture.npz"
    np.savez_compressed(
        raw_path,
        time_s=time,raw_exports=raw,command=np.zeros((n,5)),
        import_values=np.zeros((n,6)),
    )
    raw_sha=hashlib.sha256(raw_path.read_bytes()).hexdigest()

    candidate=root/"candidates"/name
    candidate.mkdir(parents=True)
    state=np.zeros((n+1,7),dtype=float)
    state[:,0]=q[:,2]
    state[:,1]=q[:,0]
    state[:,2]=q[:,1]
    arrays={
        "time_s":time,
        "state_si":state,
        "command_si":np.zeros((n,5)),
        "nominal_7dof_input_si":np.zeros((n,6)),
        "measured_steer_rad":np.zeros((n+1,2)),
        "reported_external_torque_nm":np.zeros((n+1,4)),
        "reported_accel_mps2":np.zeros((n+1,2)),
        "import_native":np.zeros((n,6)),
        "my_dr_native":np.zeros((n+1,4)),
        "nominal_initial_load_memory_mps2":np.zeros(2),
    }
    candidate_sha=_fingerprint(arrays)
    np.savez_compressed(candidate/"candidate.npz",**arrays)
    meta={
        "format":"wm_recovery_carsim_candidate_v1",
        "status":"candidate_created_not_training_ready",
        "source_kind":"carsim_env",
        "raw_source":str(raw_path),
        "raw_source_sha256":raw_sha,
        "canonical_physical_arrays_sha256":candidate_sha,
        "samples":n+1,"intervals":n,"dt_s":0.01,
        "split":"unassigned",
        "model_input_definition":(
            "equal_front_commanded_steer_plus_four_commanded_external_torques"
        ),
        "interval_alignment":"command_i_holds_from_t_i_to_t_i_plus_1",
        "torque_echo_matches":True,
        "ready_for_wm_training":False,
    }
    (candidate/"candidate_report.json").write_text(
        json.dumps(meta,ensure_ascii=False),encoding="utf-8"
    )
    ep_dir=root/"workspace"/"episodes"/name
    ep_dir.mkdir(parents=True)
    manifest={
        "format":"wm_recovery_experimental_episode_manifest_v1",
        "trajectory_id":"carsim_experimental_"+name,
        "condition_id":"condition_"+name,
        "split":split,
        "actual_mu":0.8,
        "candidate_path":str(candidate),
        "candidate_sha256":candidate_sha,
        "raw_source_sha256":raw_sha,
        "assumption_acknowledged":True,
        "ready_for_wm_training":False,
    }
    (ep_dir/"experimental_episode.json").write_text(
        json.dumps(manifest),encoding="utf-8"
    )
    return {
        "name":name,"split":split,"condition_id":manifest["condition_id"],
        "episode_path":str(ep_dir),"candidate_sha256":candidate_sha,
        "raw_source_sha256":raw_sha,
    }, candidate, bias


def make_workspace(base):
    root=Path(base)/"workspace"
    specs=(
        ("v40_train_sis_left", "train", 11.1,()),
        ("v60_train_torque_sweep", "train", 16.6,()),
        ("v40_validation_swd_left", "validation", 12.2,()),
        ("v60_test_swd_right", "test", 17.1,(35,80)),
    )
    entries=[]
    candidates=[]
    expected_bias=None
    for name,split,speed,cancel in specs:
        entry,cand,bias=fixture_episode(Path(base),name,split,speed,130,cancel)
        entries.append(entry)
        candidates.append(cand)
        expected_bias=bias
    registry={
        "format":"wm_recovery_experimental_registry_v1",
        "episodes":entries,
        "episode_count":len(entries),
        "ready_for_wm_training":False,
        "quality_cleaned":True,
    }
    (root/"registry.json").write_text(json.dumps(registry),encoding="utf-8")
    return root,candidates,expected_bias


class NMPCResidualDatasetTests(unittest.TestCase):
    def test_candidate_one_step_labels_match_known_interval_bias(self):
        with tempfile.TemporaryDirectory() as td:
            _,cand,bias=fixture_episode(Path(td),"v40_train_sis_left","train")
            ep=prepare_episode("v40_train_sis_left","train",cand)
            self.assertEqual(ep["included_intervals"],130)
            self.assertEqual(len(ep["runs"]),1)
            np.testing.assert_allclose(
                ep["data"]["residual_y_si"],
                np.broadcast_to(bias,(130,3)), rtol=0,atol=2e-12
            )
            self.assertEqual(ep["data"]["z_si"].shape,(130,8))
            self.assertEqual(ep["data"]["nominal_accel_si"].shape,(130,3))
            np.testing.assert_array_equal(ep["data"]["interval_index"],np.arange(130))

    def test_three_splits_and_cancellation_do_not_stitch(self):
        with tempfile.TemporaryDirectory() as td:
            root,cands,bias=make_workspace(td)
            out=Path(td)/"nmpc_dataset"
            result=make_dataset(
                root,out,settings=ResidualDatasetConfig(neutral_train_weight=0.5)
            )
            self.assertEqual(result["episode_count"],4)
            self.assertEqual(result["split_counts"]["train"]["intervals"],260)
            self.assertEqual(result["split_counts"]["validation"]["intervals"],130)
            self.assertEqual(result["split_counts"]["test"]["intervals"],128)
            self.assertEqual(result["total_intervals"],518)
            self.assertEqual(result["split_counts"]["test"]["segments"],3)
            self.assertEqual(result["split_counts"]["test"]["max_contiguous_intervals"],49)
            self.assertTrue((out/"dataset_manifest.json").is_file())
            with np.load(out/"test.npz",allow_pickle=False) as z:
                rows={k:z[k].copy() for k in z.files}
            self.assertEqual(rows["z_si"].shape,(128,8))
            self.assertEqual(rows["residual_y_si"].shape,(128,3))
            self.assertEqual(set(rows["segment_id"]),set(np.unique(rows["segment_id"])))
            for st, en in iter_contiguous_windows(rows["segment_id"],10):
                self.assertEqual(len(set(rows["segment_id"][st:en])),1)
                self.assertTrue(np.all(np.diff(rows["interval_index"][st:en])==1))
            self.assertNotIn(35,rows["interval_index"].tolist())
            self.assertNotIn(80,rows["interval_index"].tolist())
            self.assertEqual(len(iter_contiguous_windows(rows["segment_id"],100)),0)
            with np.load(out/"train.npz",allow_pickle=False) as z:
                neutral=z["lamda_regime_code"]==1
                self.assertTrue(np.all(z["sample_weight"][neutral]==0.5))
                self.assertTrue(np.all(z["sample_weight"][~neutral]==1.0))
            with np.load(out/"validation.npz",allow_pickle=False) as z:
                self.assertTrue(np.all(z["sample_weight"]==1.0))
            self.assertFalse(result["training_ready_for_closed_loop_nmpc"])
            with self.assertRaises(FileExistsError):
                make_dataset(root,out)

    def test_audit_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as td:
            root,_,_=make_workspace(td)
            dest=Path(td)/"not_created"
            report=make_dataset(root,output=None)
            self.assertEqual(report["total_intervals"],518)
            self.assertFalse(dest.exists())

    def test_reject_workspace_without_quality_clean_marker(self):
        """避免把未经清洗的 Brake-turn 严重失稳区间混入新数据集。"""
        with tempfile.TemporaryDirectory() as td:
            root, _, _ = make_workspace(td)
            registry_file = root / "registry.json"
            reg = json.loads(registry_file.read_text(encoding="utf-8"))
            reg.pop("quality_cleaned", None)
            registry_file.write_text(json.dumps(reg), encoding="utf-8")
            dest = Path(td) / "not_created"
            with self.assertRaisesRegex(ValueError, "quality_cleaned=True"):
                make_dataset(root, output=dest)
            self.assertFalse(dest.exists())

    def test_reject_source_fingerprint_before_creating_output(self):
        with tempfile.TemporaryDirectory() as td:
            root,_,_=make_workspace(td)
            registry_path=root/"registry.json"
            registry=json.loads(registry_path.read_text(encoding="utf-8"))
            registry["episodes"][0]["candidate_sha256"]="WRONG_HASH"
            registry_path.write_text(json.dumps(registry),encoding="utf-8")
            dest=Path(td)/"must_not_exist"
            with self.assertRaisesRegex(ValueError,"candidate_sha256"):
                make_dataset(root,output=dest)
            self.assertFalse(dest.exists())

    def test_invalid_segment_window_contract(self):
        ids=np.array([10,10,11,11,11,12],dtype=np.int64)
        self.assertEqual(iter_contiguous_windows(ids,2),[(0,2),(2,4),(3,5)])
        self.assertEqual(iter_contiguous_windows(ids,4),[])
        self.assertEqual(iter_contiguous_windows(np.empty(0,dtype=np.int64),3),[])
        with self.assertRaisesRegex(ValueError,"不相邻"):
            iter_contiguous_windows(np.array([1,2,1],dtype=np.int64),1)
        with self.assertRaises(ValueError):
            iter_contiguous_windows(np.array([1,1],dtype=np.int64),0)

    def test_settings_validation(self):
        for settings in (
            ResidualDatasetConfig(neutral_train_weight=0),
            ResidualDatasetConfig(neutral_train_weight=1.1),
            ResidualDatasetConfig(min_segment_intervals=0),
            ResidualDatasetConfig(rk4_substeps=0),
        ):
            with self.assertRaises(ValueError):
                settings.validate()


if __name__ == "__main__":
    unittest.main()
