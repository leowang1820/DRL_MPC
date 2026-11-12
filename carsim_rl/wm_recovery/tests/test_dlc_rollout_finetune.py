# -*- coding: utf-8 -*-
"""DLC多步微调离线回归测试：无 DLL、无真实 CarSim 轨迹。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from wm_recovery.tests.test_dlc_single_scene import write_synthetic_dlc
from wm_recovery.dlc_si import load_dlc_si
from wm_recovery.dlc_actuation import build_dlc_inputs
from wm_recovery.nmpc_reference import NMPCReferenceParams
from wm_recovery.nmpc_reference_torch import hybrid_rk4_step_torch
from wm_recovery.nmpc_residual_model import load_model_checkpoint
from wm_recovery.dlc_single_scene_experiment import run_experiment
from wm_recovery.dlc_rollout_finetune import (
    parse_stages,run_finetune,_focus_diagnostic,
    verify_nominal_torch_numpy,
)


class TestDLCMultistepFineTune(unittest.TestCase):

    def test_stage_syntax(self):
        self.assertEqual(parse_stages("50:30,100:30,200:20"),
                         ((50,30),(100,30),(200,20)))
        for bad in ("","50","50:0","0:10","50:10,20:10","xx:2"):
            with self.assertRaises(ValueError,msg=bad):
                parse_stages(bad)

    def test_torch_nominal_consistent_with_numpy(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=120,steady=True)
            dlc=load_dlc_si(source)
            z=build_dlc_inputs(dlc).z_si
            diagnostics=verify_nominal_torch_numpy(
                z[:-1],np.diff(dlc.time_s),NMPCReferenceParams(),
            )
            self.assertLess(
                diagnostics["nominal_rk4_numpy_torch_max_abs_si"],1e-3
            )

    def test_rollout_training_gradients_through_all_states(self):
        from wm_recovery.nmpc_residual_model import (
            NMPCResidualMLP,fit_normalization
        )
        q0=torch.tensor([[.3,.02,25.]],dtype=torch.float32)
        x=torch.tensor([[.3,.02,25.,.1,0.,0.,.5,.8]],
                       dtype=torch.float32)
        stats=fit_normalization(
            np.tile(x.numpy(),(15,1)),
            np.zeros((15,3))
        )
        model=NMPCResidualMLP(stats)
        model.train()
        before=model.net[0].weight.detach().clone()
        q=q0
        for _ in range(10):
            q=hybrid_rk4_step_torch(
                q,x[:,3:],
                torch.tensor([.01],dtype=torch.float32),
                residual_model=model,
                params=NMPCReferenceParams(),
                integration="continuous",
            )
        loss=(q**2).mean()
        loss.backward()
        norms=[
            float(torch.linalg.vector_norm(p.grad))
            for p in model.parameters()
            if p.grad is not None
        ]
        self.assertTrue(all(np.isfinite(norms)))
        self.assertGreater(sum(norms),0.)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        optimizer.step()
        self.assertFalse(torch.equal(before,model.net[0].weight))

    def test_end_to_end_no_forced_improvement_and_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"capture"/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=140,steady=True)
            original=Path(td)/"pretrained"
            old=run_experiment(
                source,original,full_data=True,
                horizons=(20,40),anchor_stride=20,
                epochs=8,batch_size=32,lr=.001,
                make_plots=False,verbose=False,
            )
            new=Path(td)/"finetune"
            report=run_finetune(
                source,original/"final.pt",new,
                stages=((5,2),(10,2)),batch_size=2,
                single_batch_size=12,lr=2e-4,
                horizons=(20,40),stride=20,
                focus_start_s=.3,focus_horizon_steps=40,
                show_figures=False,
            )
            self.assertEqual(
                report["format"],"wm_dlc_multistep_finetune_full_data_v1"
            )
            self.assertTrue(report["train_and_evaluation_same_data"])
            self.assertFalse(report["independent_generalization_test"])
            self.assertTrue((new/"report.json").is_file())
            self.assertTrue((new/"history.json").is_file())
            self.assertTrue((new/"fine_tuned.pt").is_file())
            self.assertEqual(len(report["training"]["history"]),4)
            new_model,payload=load_model_checkpoint(new/"fine_tuned.pt")
            self.assertEqual(payload["training_mode"],
                             "same_DLC_multistep_continuous_RK4_finetune")
            a=report["all_windows_before"]["40_steps"]
            b=report["all_windows_after"]["40_steps"]
            self.assertEqual(a["n_windows"],b["n_windows"])
            self.assertEqual(a["anchor_times_s"],b["anchor_times_s"])
            self.assertAlmostEqual(
                a["pure_normalized_path_rmse"],
                b["pure_normalized_path_rmse"],
                places=10
            )
            self.assertTrue(
                np.isfinite(b["mlp_normalized_path_rmse"])
            )
            # 如果结果变差，必须真实报告；不保证训练会提升。
            self.assertEqual(
                len(report["focus_window_after"]
                    ["residual_nmpc"]["terminal_abs_error_si_by_state"]),3
            )
            with self.assertRaises(FileExistsError):
                run_finetune(
                    source,original/"final.pt",new,
                    stages=((5,1),),
                    horizons=(20,),focus_start_s=.3,
                    focus_horizon_steps=40,
                )

    def test_rejects_foreign_pretrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"capture"/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=140,steady=True)
            original=Path(td)/"pretrained"
            run_experiment(
                source,original,full_data=True,horizons=(20,),
                epochs=2,make_plots=False,verbose=False,
            )
            model=original/"final.pt"
            dlc=load_dlc_si(source)
            _,payload=load_model_checkpoint(model)
            from wm_recovery.dlc_rollout_finetune import (
                _validate_pretraining_metadata,
            )
            act=build_dlc_inputs(dlc)
            payload["dataset_sha256"]="wrong"
            with self.assertRaisesRegex(ValueError,"SHA256"):
                _validate_pretraining_metadata(payload,dlc,act)


if __name__=="__main__":
    unittest.main()
