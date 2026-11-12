# -*- coding: utf-8 -*-
"""DLC A/B/C/D 消融接口测试，无需 CarSim DLL 或用户真实轨迹。"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wm_recovery.dlc_si import REQUIRED, load_dlc_si
from wm_recovery.dlc_actuation import build_dlc_inputs
from wm_recovery.dlc_data_ablation import (
    make_abcd_indices,
    make_fixed_anchor_indices,
    _shared_state_scale,
    _batched_rollout,
    _truth_for_windows,
    _rollout_metrics,
    _share_of_validation_outside_training,
    _train_one_arm,
    run_ablation,
)
from wm_recovery.nmpc_reference import NMPCReferenceParams
from wm_recovery.nmpc_rollout import rollout_known_future
from wm_recovery.nmpc_residual_model import load_model_checkpoint


def make_synthetic_dlc(path, *, steps=120, dt=0.01):
    """静态 SI 车辆状态 + 合法的 61 个 CarSim 名称（仅测试流转）。"""
    t = np.arange(steps+1)*dt
    names = list(REQUIRED)+["Steer_SW","Steer_SW"]
    names.extend(f"Unused_{i}" for i in range(61-len(names)))
    raw = np.zeros((steps+1, 61), dtype=np.float64)
    s = {
        "Vx":np.full_like(t, 90.0),
        "Vy":np.full_like(t, .15*3.6),
        "AVz":np.full_like(t, .012*180/np.pi),
        "Yaw":np.rad2deg(.012*t),
        "Steer_L1":np.full_like(t, .01*180/np.pi),
        "Steer_R1":np.full_like(t, .01*180/np.pi),
        "Xo":25.0*t,
        "Yo":.15*t,
        "Ax":np.zeros_like(t),
        "Ay":np.full_like(t,(.012*25)/9.81),
        "Fx_L1":np.zeros_like(t),
        "Fx_R1":np.zeros_like(t),
        "Fx_L2":np.zeros_like(t),
        "Fx_R2":np.zeros_like(t),
    }
    for key, value in s.items():
        raw[:, names.index(key)] = value
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        p,time_s=t,raw_exports=raw,
        export_names=np.asarray(names,dtype=str)
    )
    return p


class TestDLCAbcd(unittest.TestCase):

    def test_design_counts_nested_and_reproducible(self):
        groups=make_abcd_indices(1000)
        again=make_abcd_indices(1000)
        self.assertEqual(
            {k:len(v) for k,v in groups.items()},
            {"A":500,"B":500,"C":750,"D":1000}
        )
        np.testing.assert_array_equal(groups["A"],np.arange(500))
        self.assertEqual(groups["B"][0],0)
        self.assertEqual(groups["B"][-1],999)
        self.assertTrue(set(groups["B"]).issubset(set(groups["C"])))
        self.assertTrue(set(groups["C"]).issubset(set(groups["D"])))
        self.assertFalse(np.array_equal(groups["A"],groups["B"]))
        for key in "ABCD":
            np.testing.assert_array_equal(groups[key],again[key])
            self.assertEqual(len(groups[key]),len(np.unique(groups[key])))

    def test_anchor_counts_and_identical_fixed_coverage(self):
        a3=make_fixed_anchor_indices(1000,300,50)
        a5=make_fixed_anchor_indices(1000,500,50)
        self.assertEqual(len(a3),15)
        self.assertEqual(len(a5),11)
        self.assertEqual(a3[0],0)
        self.assertEqual(a3[-1],700)
        self.assertEqual(a5[0],0)
        self.assertEqual(a5[-1],500)
        with self.assertRaises(ValueError):
            make_fixed_anchor_indices(1000,1001,50)

    def test_invalid_sampling_design_rejected(self):
        with self.assertRaises(ValueError):
            make_abcd_indices(1000,a_size=500,b_size=499,c_size=750)
        with self.assertRaises(ValueError):
            make_abcd_indices(1000,a_size=500,b_size=500,c_size=1001)

    def test_scale_and_coverage_report_stays_finite(self):
        z=np.column_stack((
            np.arange(100,dtype=float),
            np.arange(100,dtype=float)/100,
            np.arange(100,dtype=float)*.1,
            np.zeros((100,5)),
        ))
        q=np.ones((101,3))
        sc=_shared_state_scale(q)
        np.testing.assert_allclose(sc,np.ones(3))
        cover=_share_of_validation_outside_training(z,np.arange(50))
        self.assertAlmostEqual(
            cover["vy_mps"]["evaluation_outside_train_range_pct"],
            (49/99)*100
        )

    def test_batched_nominal_matches_scalar_rollout(self):
        with tempfile.TemporaryDirectory() as td:
            f=make_synthetic_dlc(Path(td)/"dlc_exports.npz",steps=120)
            dlc=load_dlc_si(f)
            z=build_dlc_inputs(dlc).z_si
            starts=np.asarray([0,20,40])
            h=30
            batched=_batched_rollout(
                dlc.q_si,z,np.diff(dlc.time_s),starts,h,
                model=None,reference=NMPCReferenceParams(),
            )
            self.assertEqual(batched.shape,(h+1,3,3))
            for k,start in enumerate(starts):
                single=rollout_known_future(
                    dlc.q_si[start],z[start:start+h],
                    dt_s=np.diff(dlc.time_s[start:start+h+1]),
                )
                np.testing.assert_allclose(
                    batched[:,k],single,atol=1e-9,rtol=1e-9
                )

    def test_fixed_seed_produces_equal_weight_init_and_training(self):
        rng=np.random.default_rng(101)
        z=rng.normal(size=(50,8))
        z[:,7]=.8
        labels=rng.normal(size=(50,3))
        idx=np.arange(50,dtype=np.int64)
        model1,log1=_train_one_arm(
            z,labels,idx,seed=42,updates=5,batch_size=16,
            lr=.001,weight_decay=1e-5,log_every=5
        )
        model2,log2=_train_one_arm(
            z,labels,idx,seed=42,updates=5,batch_size=16,
            lr=.001,weight_decay=1e-5,log_every=5
        )
        for a,b in zip(model1.parameters(),model2.parameters()):
            np.testing.assert_allclose(
                a.detach().numpy(),b.detach().numpy(),atol=0,rtol=0
            )
        self.assertEqual(log1,log2)

    def test_rollout_metric_reports_deterioration(self):
        true=np.zeros((11,3,3))
        pure=np.ones_like(true)
        wrong=np.full_like(true,10)
        scale=np.asarray([1.,1.,1.])
        p=_rollout_metrics(pure,true,scale)
        r=_rollout_metrics(wrong,true,scale)
        self.assertGreater(
            r["path_normalized_rmse"],p["path_normalized_rmse"]
        )
        self.assertEqual(len(r["per_window_path_normalized_rmse"]),3)

    def test_end_to_end_four_arms_and_shared_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            file=make_synthetic_dlc(
                Path(td)/"capture"/"dlc_exports.npz",steps=120
            )
            out=Path(td)/"result"
            summary=run_ablation(
                file,out,
                a_size=50,b_size=50,c_size=75,
                updates=6,batch_size=16,lr=.002,
                seeds=(42,),horizons=(20,50),stride=10,
                make_plots=False,verbose=False,
            )
            self.assertTrue(out.joinpath("summary.json").is_file())
            self.assertTrue(out.joinpath("experiment_config.json").is_file())
            self.assertTrue(out.joinpath("training_indices.npz").is_file())
            self.assertTrue(out.joinpath("metrics_by_seed.csv").is_file())
            self.assertTrue(out.joinpath("summary_table.csv").is_file())
            self.assertTrue(out.joinpath("per_window_metrics.csv").is_file())
            self.assertEqual(summary["baseline"]["20"]["n_windows"],11)
            self.assertEqual(summary["baseline"]["50"]["n_windows"],8)
            self.assertFalse(summary["genuine_independent_validation"])
            self.assertEqual(
                [summary["groups"][key]["20"]["n_train_intervals"]
                 for key in "ABCD"],
                [50,50,75,120],
            )
            with np.load(out/"training_indices.npz",allow_pickle=False) as d:
                np.testing.assert_array_equal(
                    d["A_interval_indices"],np.arange(50)
                )
                self.assertTrue(
                    set(d["B_interval_indices"]).issubset(
                        set(d["C_interval_indices"])
                    )
                )
            for key in "ABCD":
                path=out/"models"/f"{key}_seed_42.pt"
                self.assertTrue(path.is_file())
                loaded,_=load_model_checkpoint(path)
                self.assertEqual(
                    loaded.predict_numpy(
                        np.asarray([0.15,.012,25.,.01,0.,0.,.5,.8])
                    ).shape,(3,)
                )
                for h in (20,50):
                    data=out/"rollouts"/f"{key}_seed_42_{h}steps.npz"
                    self.assertTrue(data.is_file())
                    with np.load(data,allow_pickle=False) as d:
                        self.assertEqual(d["truth_si"].shape[0],h+1)
                        self.assertEqual(d["pure_nmpc_si"].shape,
                                         d["residual_nmpc_si"].shape)
            with (out/"metrics_by_seed.csv").open(
                encoding="utf-8-sig",newline=""
            ) as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),8)
            for h in (20,50):
                records=[r for r in rows if float(r["horizon_s"])==h*.01]
                self.assertEqual(len(records),4)
                baseline={float(r["nominal_rmse"]) for r in records}
                self.assertEqual(len(baseline),1)

    def test_reusing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=make_synthetic_dlc(Path(td)/"capture"/"dlc_exports.npz")
            out=Path(td)/"already_exists"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                run_ablation(
                    f,out,updates=1,
                    a_size=50,b_size=50,c_size=75,
                    horizons=(20,),make_plots=False,
                    verbose=False
                )


if __name__=="__main__":
    unittest.main()
