# -*- coding: utf-8 -*-
"""DLC 单场景测试：使用合成 CarSim Export，不依赖 CarSim DLL。

测试的目标是检查数据通路和拟合能力；合成实验中的改善比例不代表
真实 DLC 90km/h 数据的改善比例。
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wm_recovery.dlc_si import REQUIRED, load_dlc_si
from wm_recovery.dlc_actuation import build_dlc_inputs
from wm_recovery.dlc_single_scene_experiment import (
    _split_index, make_interval_targets, run_experiment,
    evaluate_windows,
)
from wm_recovery.nmpc_reference import rk4_nominal_step
from wm_recovery.nmpc_residual_model import load_model_checkpoint


def write_synthetic_dlc(path, *, steps=200, dt=.01,
                        residual=(.07, .002, .03), steady=False):
    """单条带已知恒定残差的合成轨迹，保留 61 个原生通道。"""
    t=np.arange(steps+1,dtype=float)*dt
    names=list(REQUIRED)+["Steer_SW", "Steer_SW"]
    names += [f"Unused_{i}" for i in range(61-len(names))]
    raw=np.zeros((len(t),len(names)),dtype=float)
    q=np.zeros((len(t),3),dtype=float)
    q[0]=[0.,0.,25.]
    y=np.asarray(residual,dtype=float)
    delta=np.zeros(len(t),dtype=float)
    if steady:
        # 非零稳态，纯名义模型不能保持该状态，MLP 可学习常值修正。
        q[:]=[.15,.015,25.]
        delta[:]=.015
    else:
        for i in range(steps):
            z=np.r_[q[i],delta[i],0.,0.,0.5,0.8]
            q[i+1]=rk4_nominal_step(z,dt_s=dt)+dt*y

    vx=q[:,2]; vy=q[:,0]; r=q[:,1]
    yaw=np.zeros(len(t),dtype=float)
    yaw[1:]=np.cumsum(r[:-1]*dt)
    x=np.zeros(len(t),dtype=float)
    y_pos=np.zeros(len(t),dtype=float)
    x[1:]=np.cumsum(
        (vx[:-1]*np.cos(yaw[:-1])-vy[:-1]*np.sin(yaw[:-1]))*dt
    )
    y_pos[1:]=np.cumsum(
        (vx[:-1]*np.sin(yaw[:-1])+vy[:-1]*np.cos(yaw[:-1]))*dt
    )
    signals={
        "Xo":x, "Yo":y_pos, "Vx":vx*3.6, "Vy":vy*3.6,
        "Yaw":np.rad2deg(yaw), "AVz":np.rad2deg(r),
        "Ax":(np.gradient(vx,t)-r*vy)/9.81,
        "Ay":(np.gradient(vy,t)+r*vx)/9.81,
        "Steer_L1":np.rad2deg(delta),
        "Steer_R1":np.rad2deg(delta),
    }
    for k,v in signals.items():
        raw[:,names.index(k)]=v
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,time_s=t,raw_exports=raw,
                        export_names=np.asarray(names,dtype=str))
    return q,t


class TestDLCSingleScene(unittest.TestCase):
    def test_si_mapping_and_duplicate_export(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            true_q,t=write_synthetic_dlc(file)
            data=load_dlc_si(file)
            self.assertEqual(data.raw.shape,(201,61))
            self.assertEqual(data.q_si.shape,(201,3))
            np.testing.assert_allclose(data.q_si,true_q,atol=1e-12)
            self.assertEqual(len(data.diagnostics["duplicates"]["Steer_SW"]),2)
            self.assertLess(data.diagnostics["yaw_rate_consistency_rmse_radps"],
                            1e-3)

    def test_nominal_discrete_target_and_default_actuation(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            write_synthetic_dlc(file,residual=(.07,.002,.03))
            dlc=load_dlc_si(file)
            ext=build_dlc_inputs(dlc,mu_hat=.8)
            z=ext.z_si
            self.assertEqual(z.shape,(201,8))
            np.testing.assert_allclose(z[:,4],0.)
            np.testing.assert_allclose(z[:,5],0.)
            np.testing.assert_allclose(z[:,6],.5)
            np.testing.assert_allclose(z[:,7],.8)
            next_nominal,y=make_interval_targets(dlc,z)
            self.assertEqual(next_nominal.shape,(200,3))
            np.testing.assert_allclose(
                y,np.tile([.07,.002,.03],(200,1)),atol=1e-9
            )

    def test_time_split_nonoverlap_and_short_training_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            write_synthetic_dlc(file)
            dlc=load_dlc_si(file)
            self.assertEqual(_split_index(dlc.time_s,1.0,(50,100)),100)
            with self.assertRaises(ValueError):
                _split_index(dlc.time_s,.5,(50,100))

    def test_ratio_strict_rejects_force_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            write_synthetic_dlc(file)
            dlc=load_dlc_si(file)
            dlc.fx_tire_n[:,0]=100
            dlc.fx_tire_n[:,2]=-100
            with self.assertRaisesRegex(ValueError,"不可辨识"):
                build_dlc_inputs(dlc,lambda_mode="ratio-strict")
            alternate=build_dlc_inputs(
                dlc,lambda_mode="fixed-half",
                mz_mode="force-diff-diagnostic",
            )
            self.assertTrue(np.isfinite(alternate.z_si).all())

    def test_timing_anomaly_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            q,t=write_synthetic_dlc(file)
            with np.load(file,allow_pickle=False) as npz:
                names=npz["export_names"]
                raw=npz["raw_exports"]
            t[100]=t[99]+.004
            np.savez_compressed(
                file,time_s=t,raw_exports=raw,export_names=names
            )
            with self.assertRaisesRegex(ValueError,"时间轴异常"):
                load_dlc_si(file)

    def test_rollout_can_truthfully_report_negative_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"dlc_exports.npz"
            write_synthetic_dlc(file,steady=True)
            dlc=load_dlc_si(file)
            z=build_dlc_inputs(dlc).z_si

            class WrongResidual:
                def predict_numpy(self, z_si):
                    return np.tile([10.,0.,0.],(len(z_si),1))

            result,_=evaluate_windows(
                dlc,z,WrongResidual(),
                start_interval=100,stop_interval=200,
                horizon=50,stride=50,
                q_scale=np.ones(3),
            )
            self.assertLess(
                result["normalized_path_improvement_pct"],0.
            )
            self.assertGreater(
                result["residual_normalized_path_rmse"],
                result["nominal_normalized_path_rmse"],
            )

    def test_full_data_mode_uses_all_intervals_for_train_and_validation(self):
        """整条轨迹即训练数据又是评价数据，报告必须明确标注重叠。"""
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "capture" / "dlc_exports.npz"
            write_synthetic_dlc(file, steps=200, steady=True)
            out = Path(temp) / "all_data_results"
            report = run_experiment(
                file, out,
                full_data=True, train_end_s=0.2,  # full_data 忽略时间分割
                horizons=(50, 100),
                anchor_stride=50,
                epochs=40, batch_size=32, lr=.003,
                make_plots=False, verbose=False,
            )

            self.assertTrue(report["full_data"])
            self.assertEqual(
                report["experiment_type"], "in_sample_full_data_reconstruction"
            )
            self.assertEqual(report["train_intervals"], 200)
            self.assertEqual(report["validation_intervals"], 200)
            self.assertEqual(
                report["train_interval_time_range_s"], [0., 2.]
            )
            self.assertEqual(
                report["validation_interval_time_range_s"], [0., 2.]
            )
            self.assertTrue(report["train_and_validation_same_intervals"])
            self.assertFalse(report["train_validation_no_interval_overlap"])
            self.assertFalse(report["independent_validation"])
            self.assertTrue(
                report["normalization_includes_evaluation_data"]
            )
            self.assertTrue(
                report["training"][
                    "used_validation_labels_for_training_or_checkpoint_selection"
                ]
            )
            for horizon, window_count in ((50, 4), (100, 3)):
                label = f"{horizon}_steps"
                train_result = report["rollouts"]["train"][label]
                valid_result = report["rollouts"]["validation"][label]
                self.assertEqual(train_result, valid_result)
                self.assertEqual(valid_result["n_windows"], window_count)
                self.assertEqual(valid_result["anchor_times_s"][0], 0.)
            with np.load(out / "dlc_residual_dataset.npz",
                         allow_pickle=False) as dataset:
                self.assertEqual(len(dataset["residual_target_si"]), 200)
                self.assertEqual(int(dataset["train_end_interval"]), 200)
                self.assertEqual(int(dataset["validation_start_interval"]), 0)
                self.assertTrue(
                    bool(dataset["train_and_evaluation_same_data"])
                )
            self.assertTrue((out / "validation_0.5s_rollout.npz").is_file())
            self.assertTrue((out / "validation_1s_rollout.npz").is_file())
            self.assertTrue((out / "final.pt").is_file())

    def test_full_data_rejects_horizon_longer_than_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "dlc_exports.npz"
            write_synthetic_dlc(file, steps=200, steady=True)
            with self.assertRaisesRegex(ValueError, "不足最大"):
                run_experiment(
                    file, Path(temp) / "should_not_exist",
                    full_data=True, horizons=(300,),
                    epochs=1, make_plots=False, verbose=False,
                )

    def test_end_to_end_training_and_rollout_on_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"capture"/"dlc_exports.npz"
            write_synthetic_dlc(file,steady=True)
            out=Path(temp)/"output"
            report=run_experiment(
                file,out,
                train_end_s=1.0,
                horizons=(50,100),
                anchor_stride=50,
                epochs=160,batch_size=32,lr=.003,
                make_plots=False,verbose=False,
            )
            self.assertEqual(report["train_intervals"],100)
            self.assertEqual(report["validation_intervals"],100)
            self.assertTrue(
                report["train_validation_no_interval_overlap"]
            )
            self.assertFalse(
                report["training"][
                    "used_validation_labels_for_training_or_checkpoint_selection"
                ]
            )
            valid=report["rollouts"]["validation"]["100_steps"]
            self.assertEqual(valid["n_windows"],1)
            # 人工产生的恒定加速度残差可由 MLP 学习；
            # 验证整段误差应小于纯名义模型。
            self.assertLess(
                valid["residual_normalized_path_rmse"],
                valid["nominal_normalized_path_rmse"],
            )
            self.assertTrue((out/"report.json").is_file())
            self.assertTrue((out/"final.pt").is_file())
            self.assertTrue((out/"dlc_residual_dataset.npz").is_file())
            restored,_=load_model_checkpoint(out/"final.pt")
            self.assertEqual(
                restored.predict_numpy(np.zeros((1,8),dtype=float)
                    +[0.,0.,25.,0.,0.,0.,.5,.8]).shape,(1,3)
            )


if __name__ == "__main__":
    unittest.main()
