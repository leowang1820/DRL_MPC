# -*- coding: utf-8 -*-
"""NMPC Residual MLP / Trainer / Rollout / Evaluator 的离线合成数据回归测试。

无需 CarSim DLL / FORCESPRO，只需 PyTorch + NumPy。
测试生成一个临时合成数据集，保持与真实 dataset_manifest v1 一致的最小契约。
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from wm_recovery.nmpc_reference import rk4_nominal_step
from wm_recovery.nmpc_residual_model import (
    fit_normalization, NMPCResidualMLP, load_model_checkpoint,
)
from wm_recovery.nmpc_rollout import (
    rk4_free_step, rollout_known_future, rollout_pair_known_future,
)
from wm_recovery.nmpc_rollout_evaluate import (
    evaluate_split, _segment_slices, _make_anchors,
)
from wm_recovery.nmpc_residual_train import run_training, load_split_npz


class _KnownResidual:
    """一个不依赖未来真值的确定性残差，用于检验滚动器。"""

    def predict_numpy(self, z_si):
        x = np.asarray(z_si)
        if x.ndim == 1:
            return np.array([0.3, 0.02, -0.1])
        return np.broadcast_to([0.3, 0.02, -0.1], (len(x), 3)).copy()


def _synthetic_split(n, episode_index, segment_id, *, true_residual):
    """生成没有数据泄漏、物理连续的 CarSim 替身数据集。"""
    q = np.array([0.2, 0.02, 11.2 + episode_index], dtype=float)
    zs, ys, qk, qnext, qnom, times = [], [], [], [], [], []
    zexternal = np.array([0.012, 35.0, 300.0, 0.5, 0.8], dtype=float)
    for k in range(n):
        external = zexternal.copy()
        external[0] = 0.012 + 0.005 * np.sin(0.02*k)
        z = np.concatenate((q, external))
        nominal = rk4_nominal_step(z, dt_s=0.01)
        true_next = rk4_free_step(
            q, external, dt_s=0.01, residual_model=true_residual
        )
        zs.append(z)
        ys.append((true_next - nominal)/0.01)
        qk.append(q.copy())
        qnext.append(true_next)
        qnom.append(nominal)
        times.append(k*0.01)
        q = true_next

    return {
        "z_si": np.asarray(zs),
        "residual_y_si": np.asarray(ys),
        "q_k_si": np.asarray(qk),
        "q_true_next_si": np.asarray(qnext),
        "q_nominal_next_si": np.asarray(qnom),
        "dt_s": np.full(n, 0.01, dtype=float),
        "time_k_s": np.asarray(times),
        "sample_weight": np.ones(n, dtype=float),
        "lamda_regime_code": np.zeros(n, dtype=np.int64),
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "segment_id": np.full(n, segment_id, dtype=np.int64),
        "segment_step": np.arange(n, dtype=np.int64),
        "interval_index": np.arange(n, dtype=np.int64),
    }


def _create_dataset(root):
    from wm_recovery.nmpc_reference import NMPCReferenceParams
    from dataclasses import asdict
    root.mkdir(parents=True, exist_ok=True)
    synth = _KnownResidual()
    files = {
        "train": _synthetic_split(550, 0, 0, true_residual=synth),
        "validation": _synthetic_split(220, 1, 1, true_residual=synth),
        "test": _synthetic_split(560, 2, 2, true_residual=synth),
    }
    for name, data in files.items():
        np.savez_compressed(root / f"{name}.npz", **data)
    manifest = {
        "format":"wm_recovery_nmpc_residual_dataset_v1",
        "dataset_settings":{"expected_dt_s":0.01,"rk4_substeps":1},
        "reference_parameters":asdict(NMPCReferenceParams()),
        "input_names":[
            "vy_mps","yaw_rate_radps","vx_mps","delta_front_rad",
            "mz_tv_estimated_nm","fx_all_n","lamda","mu_hat",
        ],
        "target_names":[
            "delta_ay_mps2","delta_yaw_accel_radps2","delta_ax_mps2",
        ],
        "split_counts":{
            name:{"intervals":len(data["z_si"])}
            for name, data in files.items()
        },
        "episodes":[
            {"name":"v40_train_synthetic","split":"train"},
            {"name":"v60_validation_synthetic","split":"validation"},
            {"name":"v40_test_synthetic","split":"test"},
        ],
        "training_ready_for_closed_loop_nmpc":False,
    }
    (root/"dataset_manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    return files


class NMPCResidualPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_normalization_constant_mu_no_train_test_leakage(self):
        train_z = np.array(
            [[0.0, 0.01, 10.0, 0.0, 1.0, 40.0, 0.5, 0.8],
             [1.0, 0.02, 12.0, 0.1, 2.0, 60.0, 0.7, 0.8]]
        )
        train_y=np.array([[1.0, 0.1, 0.02],[2.0, 0.2, 0.04]])
        s=fit_normalization(train_z,train_y)
        self.assertEqual(s["x_scale"][-1],1.0)
        self.assertTrue(bool(s["x_constant_mask"][-1]))
        model=NMPCResidualMLP(s)
        r=model.predict_numpy(train_z)
        self.assertEqual(r.shape,(2,3))
        self.assertTrue(np.isfinite(r).all())
        with self.assertRaises(ValueError):
            fit_normalization(train_z,np.zeros((3,3)))

    def test_nominal_zero_residual_equals_reference_nominal_step(self):
        z=np.array([0.3, 0.02, 12.0, 0.04, 80.0, 200.0, 0.5, 0.8])
        ref=rk4_nominal_step(z,0.01)
        observed=rk4_free_step(z[:3],z[3:],dt_s=0.01)
        np.testing.assert_allclose(ref,observed,rtol=0,atol=1e-12)

    def test_future_true_vehicle_states_are_ignored_in_free_rollout(self):
        q0=np.array([0.3,0.02,12.0])
        z=np.tile([9999.0,9999.0,9999.0,0.03,60.0,200.0,0.5,0.8],(250,1))
        a=rollout_known_future(q0,z)
        z2=z.copy()
        z2[:, :3]=-9999.0
        b=rollout_known_future(q0,z2)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(251,3))
        self.assertFalse(np.allclose(a[-1],q0))

    def test_known_residual_produces_real_improvement_at_5s(self):
        q0=np.array([0.1,0.02,12.0])
        z=np.tile([0.,0.,0.,0.02,30.0,250.0,0.5,0.8],(500,1))
        true=rollout_known_future(q0,z,residual_model=_KnownResidual())
        baseline=rollout_known_future(q0,z,residual_model=None)
        corrected=rollout_known_future(q0,z,residual_model=_KnownResidual())
        self.assertGreater(np.linalg.norm(baseline[-1]-true[-1]),1e-4)
        np.testing.assert_allclose(corrected,true,atol=1e-10,rtol=0)

    def test_bad_segment_gaps_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            data = _create_dataset(Path(td))["train"]
            data["segment_step"][100] = 999
            with self.assertRaisesRegex(ValueError,"segment_step"):
                _segment_slices(data)

    def test_full_trainer_and_evaluator_generate_report_and_5s_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            ds=root/"dataset"
            _create_dataset(ds)
            output=root/"model"
            report=run_training(
                ds,output,epochs=3,batch_size=64,learning_rate=0.001,
                patience=5,seed=4,
            )
            self.assertTrue((output/"best.pt").exists())
            self.assertTrue((output/"normalization.json").exists())
            self.assertTrue((output/"training_report.json").exists())
            self.assertEqual(report["train_samples"],550)
            self.assertEqual(report["validation_samples"],220)
            self.assertFalse(report["closed_loop_nmpc_validated"])
            model,payload=load_model_checkpoint(output/"best.pt")
            self.assertEqual(model.predict_numpy(
                np.array([0.0,0.0,12.0,0.0,30.0,0.0,0.5,0.8])
            ).shape,(3,))

            # 注意：评价仅用 test 数据计算误差，训练没有读取 test.npz。
            evaluation=evaluate_split(
                ds,output/"best.pt",split="test",
                anchor_stride=50,max_horizon=500,
                output_dir=root/"test_evaluation",
            )
            self.assertEqual(evaluation["split"],"test")
            self.assertIn("5.00s",evaluation["horizons"])
            self.assertTrue(evaluation["horizons"]["5.00s"]["windows"] >= 1)
            self.assertEqual(evaluation["horizons"]["5.00s"]["nominal"]["endpoint_rmse_q_si"].__len__(),3)
            example_path=root/"test_evaluation"/"example_5s.npz"
            self.assertTrue(example_path.exists())
            with np.load(example_path,allow_pickle=False) as e:
                self.assertEqual(e["q_nominal_si"].shape,(501,3))
                self.assertEqual(e["q_residual_si"].shape,(501,3))
                self.assertEqual(e["known_external_si"].shape,(500,5))
            with self.assertRaises(FileExistsError):
                run_training(ds,output,epochs=1)
            with self.assertRaises(FileExistsError):
                evaluate_split(
                    ds,output/"best.pt",split="test",
                    output_dir=root/"test_evaluation"
                )
            # 修改数据集清单后，旧模型不能静默用于改变版本的数据集。
            manifest_path=ds/"dataset_manifest.json"
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tampered"]=True
            manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                evaluate_split(ds,output/"best.pt",split="test")


if __name__ == "__main__":
    unittest.main()
