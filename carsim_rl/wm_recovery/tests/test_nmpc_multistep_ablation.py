# -*- coding: utf-8 -*-
"""新增：固定窗口消融、可微 NMPC 公式、多步微调回归测试。

用现有合成 CarSim 替身数据保证不依赖 DLL/CarSim/真实 NPZ。
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wm_recovery.nmpc_reference import (
    NMPCReferenceParams, nmpc_accelerations, rk4_nominal_step,
)
from wm_recovery.nmpc_reference_torch import (
    nmpc_force_balance_torch, nmpc_accelerations_torch,
    rk4_nominal_step_torch, hybrid_rk4_step_torch,
)
from wm_recovery.nmpc_residual_model import (
    NMPCResidualMLP, fit_normalization, load_model_checkpoint,
)
from wm_recovery.nmpc_residual_train import run_training, load_split_npz
from wm_recovery.nmpc_residual_multistep_train import (
    contiguous_window_starts,window_batch_torch,differentiable_rollout_loss,
    fine_tune_multistep,
)
from wm_recovery.nmpc_rollout_ablation import (
    parse_gamma_grid,make_variants,select_fixed_cohort,run_fixed_ablation_on_arrays,
    evaluate_fixed_ablation,
)
from wm_recovery.tests.test_nmpc_residual_pipeline import (
    _create_dataset, _KnownResidual,
)


class _DifferentiableResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff=torch.nn.Parameter(
            torch.tensor([0.09,0.03,-0.02],dtype=torch.float64)
        )

    def forward(self,z):
        return z[:, :3] * self.coeff[None, :]


class _MeasuredKnownResidual(_KnownResidual):
    x_scale=torch.ones(8,dtype=torch.float32)


class ExtendedNMPCPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_force_balance_parity_random_and_strong_forces(self):
        rng=np.random.default_rng(33)
        z=np.column_stack((
            rng.uniform(-2,2,100),
            rng.uniform(-0.3,0.3,100),
            rng.uniform(0.5,26.0,100),
            rng.uniform(-0.25,0.25,100),
            rng.uniform(-1200,1200,100),
            rng.uniform(-11000,11000,100),
            rng.uniform(-0.5,1.5,100),
            rng.uniform(0.35,1.1,100),
        ))
        t=torch.tensor(z,dtype=torch.float64,requires_grad=True)
        result=nmpc_accelerations_torch(t)
        oracle=nmpc_accelerations(z)
        np.testing.assert_allclose(
            result.detach().numpy(),oracle,rtol=1e-9,atol=2e-10
        )
        grads=torch.autograd.grad(result.square().mean(),t)[0]
        self.assertTrue(bool(torch.isfinite(grads).all()))
        self.assertGreater(float(grads.abs().sum()),1e-8)

    def test_rk4_parity_and_single_input(self):
        rng=np.random.default_rng(99)
        z=np.column_stack((
            rng.normal(0,0.5,16),rng.normal(0,0.05,16),
            rng.uniform(7,24,16),rng.normal(0,0.025,16),
            rng.uniform(-200,200,16),rng.uniform(-600,900,16),
            rng.uniform(0.35,0.65,16),np.full(16,0.8),
        ))
        h=np.linspace(0.005,0.02,len(z))
        np.testing.assert_allclose(
            rk4_nominal_step_torch(
                torch.tensor(z,dtype=torch.float64),
                torch.tensor(h,dtype=torch.float64),substeps=2
            ).detach().numpy(),
            rk4_nominal_step(z,h,substeps=2),
            rtol=1e-13,atol=1e-12
        )
        self.assertEqual(
            rk4_nominal_step_torch(
                torch.tensor(z[0],dtype=torch.float64),.01
            ).shape,(3,)
        )

    def test_discrete_integrator_formula_and_autograd_through_50ms(self):
        q=torch.tensor([[0.4,0.06,14.0]],dtype=torch.float64,requires_grad=True)
        ext=torch.tensor([[0.02,100.0,340.0,0.5,0.8]],dtype=torch.float64)
        m=_DifferentiableResidual()
        q_disc=hybrid_rk4_step_torch(
            q,ext,.01,residual_model=m,integration="discrete"
        )
        q_nom=hybrid_rk4_step_torch(q,ext,.01)
        torch.testing.assert_close(
            q_disc,q_nom+.01*m(torch.cat((q,ext),dim=1)),
            rtol=1e-13,atol=1e-14
        )
        q_cont=q
        for _ in range(5):
            q_cont=hybrid_rk4_step_torch(
                q_cont,ext,.01,residual_model=m,integration="continuous"
            )
        val=q_cont.square().sum()
        val.backward()
        self.assertTrue(torch.isfinite(q.grad).all().item())
        self.assertTrue(torch.isfinite(m.coeff.grad).all().item())
        self.assertGreater(float(m.coeff.grad.abs().sum()),1e-6)
        # 两种积分方式在状态相关残差时不必完全一样。
        self.assertGreater(
            float(torch.linalg.vector_norm(q_disc-
                hybrid_rk4_step_torch(
                    q,ext,.01,residual_model=m,integration="continuous"
                )).item()),1e-10
        )

    def test_fixed_cohort_identical_numerators_and_truth_is_never_fed_back(self):
        with tempfile.TemporaryDirectory() as d:
            ds=Path(d)/"data"
            files=_create_dataset(ds)
            data=files["validation"]
            manifest=json.loads(
                (ds/"dataset_manifest.json").read_text(encoding="utf-8")
            )
            model=_MeasuredKnownResidual()
            report,example=run_fixed_ablation_on_arrays(
                data,manifest,model,
                fixed_horizon=100,anchor_stride=50,
                gamma_grid=(0.0,0.5,1.0),
            )
            self.assertTrue(report["fixed_cohort"])
            self.assertEqual(report["window_count"],3) # starts 0,50,100
            for k,terms in report["horizons"].items():
                self.assertEqual(len(terms),6)
                self.assertEqual(terms["continuous_g0"]["windows"],3)
                self.assertEqual(terms["discrete_g0"]["windows"],3)
            self.assertEqual(example["q_variants_si"].shape,(6,101,3))
            self.assertEqual(example["q_true_si"].shape,(101,3))
            np.testing.assert_allclose(
                example["q_variants_si"][0],
                example["q_variants_si"][3],
                rtol=0,atol=1e-10,
            )
            # 新纯物理基线必须等于旧 rollout_known_future() 的 100 步轨迹。
            from wm_recovery.nmpc_rollout import rollout_known_future
            original=rollout_known_future(
                data["q_k_si"][0],data["z_si"][:100],dt_s=0.01
            )
            np.testing.assert_allclose(
                original,example["q_variants_si"][0],rtol=0,atol=1e-12
            )
            self.assertEqual(len(make_variants((0.0,1.0),True)),10)
            self.assertLess(
                report["horizons"]["1.00s"]["continuous_g1"]
                ["endpoint_normalized_rmse"],1e-9
            )
            self.assertGreater(
                report["horizons"]["1.00s"]["continuous_g0"]
                ["endpoint_normalized_rmse"],1e-4
            )
            # 插入伪造的未来 CarSim 车辆状态但保持执行器不变，
            # 滚动结果必须逐位一致。
            altered={key:v.copy() for key,v in data.items()}
            altered["z_si"][:, :3] = -123456.0
            report2,example2=run_fixed_ablation_on_arrays(
                altered,manifest,model,
                fixed_horizon=100,anchor_stride=50,
                gamma_grid=(0.0,0.5,1.0),
            )
            np.testing.assert_array_equal(
                example["q_variants_si"],example2["q_variants_si"]
            )
            self.assertEqual(
                report["horizons"]["1.00s"],
                report2["horizons"]["1.00s"]
            )

    def test_fixed_cohort_rejects_gaps_and_no_eligible_window(self):
        with tempfile.TemporaryDirectory() as d:
            data=_create_dataset(Path(d))["validation"]
            with self.assertRaises(ValueError):
                select_fixed_cohort(data,500,50)
            # segment_step 断裂不得默默并接。
            data["segment_step"][75]=1000
            with self.assertRaises(ValueError):
                select_fixed_cohort(data,100,50)
            with self.assertRaises(ValueError):
                parse_gamma_grid("0.5,1")
            with self.assertRaises(ValueError):
                parse_gamma_grid("0,1,1")

    def test_multistep_windows_not_cross_episode_and_real_rollout_gradient(self):
        with tempfile.TemporaryDirectory() as d:
            data=_create_dataset(Path(d))["train"]
            starts=contiguous_window_starts(data,20,stride=30)
            self.assertGreater(len(starts),0)
            batch=window_batch_torch(data,starts[:2],20)
            qscale=torch.ones(3,dtype=torch.float64)
            model=_DifferentiableResidual()
            loss,end=differentiable_rollout_loss(
                model,batch,qscale,params=NMPCReferenceParams()
            )
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(torch.isfinite(model.coeff.grad).all().item())
            self.assertGreater(float(model.coeff.grad.abs().sum()),0.0)

    def test_two_epoch_finetuner_checkpoint_compatible_with_old_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ds=root/"dataset"
            _create_dataset(ds)
            initial=root/"pretrain"
            run_training(ds,initial,epochs=1,batch_size=96,seed=14)
            finetuned=root/"multistep"
            with patch(
                "wm_recovery.nmpc_residual_multistep_train.load_split_npz",
                wraps=load_split_npz
            ) as reader:
                report=fine_tune_multistep(
                    ds,initial/"best.pt",finetuned,
                    horizon=10,epochs=2,batch_size=4,
                    steps_per_epoch=2,
                    train_stride=50,validation_stride=50,
                    learning_rate=1e-4,patience=3,seed=2,
                )
                actual_splits=[call.args[1] for call in reader.call_args_list]
                self.assertEqual(set(actual_splits),{"train","validation"})
            self.assertTrue(report["test_not_used"])
            self.assertTrue((finetuned/"best.pt").exists())
            self.assertEqual(report["horizon_intervals"],10)
            self.assertTrue(np.isfinite(report["best_validation_rollout_normalized_mse"]))
            model,payload=load_model_checkpoint(finetuned/"best.pt")
            self.assertEqual(payload["training_mode"],"multistep_rollout_finetune")
            self.assertEqual(model.predict_numpy(
                np.array([0.0,0.0,12.,0.02,30.,200.,0.5,0.8])
            ).shape,(3,))
            # 继续使用同一个新格式 checkpoint 做固定窗口消融评价。
            output=root/"ablation"
            abl=evaluate_fixed_ablation(
                ds,finetuned/"best.pt",
                split="validation",output_dir=output,
                fixed_horizon=100,anchor_stride=50,
                gamma_grid=(0.0,1.0),
            )
            self.assertEqual(abl["window_count"],3)
            self.assertTrue((output/"ablation_report.json").exists())
            self.assertTrue((output/"example_fixed.npz").exists())
            # 禁止静默覆盖实验产物。
            with self.assertRaises(FileExistsError):
                fine_tune_multistep(
                    ds,initial/"best.pt",finetuned,
                    horizon=10,epochs=1,
                )


if __name__=="__main__":
    unittest.main()
