# -*- coding: utf-8 -*-
"""DLC 全窗口交互式查看器离线测试；使用合成 CarSim 数据，无 DLL。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import types

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wm_recovery.tests.test_dlc_single_scene import write_synthetic_dlc
from wm_recovery.dlc_si import load_dlc_si
from wm_recovery.dlc_actuation import build_dlc_inputs
from wm_recovery.nmpc_residual_model import load_model_checkpoint
from wm_recovery.nmpc_rollout import rollout_pair_known_future
from wm_recovery.dlc_single_scene_experiment import (
    evaluate_windows,run_experiment,
)
from wm_recovery.dlc_all_windows_viewer import (
    compute_all_window_batches,
    batches_from_recorded_windows,
    build_window_figure,
    build_overview_heatmap,
    window_error_arrays,
    open_from_checkpoint,
    _selected_window_title,
)


class ZeroResidual:
    def predict_numpy(self, z):
        a=np.asarray(z)
        return np.zeros((a.shape[0],3),dtype=float)


class TestDLCAllWindowsViewer(unittest.TestCase):

    def test_batch_rollout_every_window_matches_original_free_rollout(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=200,steady=True)
            dlc=load_dlc_si(source)
            z=build_dlc_inputs(dlc).z_si
            model=ZeroResidual()
            batches=compute_all_window_batches(
                dlc,z,model,horizons=(30,50),
                anchor_stride=25
            )
            self.assertEqual(len(batches[30].anchors),7)
            self.assertEqual(len(batches[50].anchors),7)
            for h,batch in batches.items():
                self.assertEqual(batch.true.shape,
                                 (h+1,len(batch.anchors),3))
                self.assertEqual(batch.time_s.shape,
                                 (h+1,len(batch.anchors)))
                for j in (0,len(batch.anchors)//2,len(batch.anchors)-1):
                    start=int(batch.anchors[j])
                    nom,res=rollout_pair_known_future(
                        dlc.q_si[start],z[start:start+h],
                        model=model,
                        dt_s=np.diff(dlc.time_s[start:start+h+1]),
                    )
                    np.testing.assert_allclose(
                        batch.nominal[:,j],nom,atol=1e-10,rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        batch.residual[:,j],res,atol=1e-10,rtol=1e-10
                    )

    def test_collector_preserves_all_nonrandom_anchors(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=200,steady=True)
            dlc=load_dlc_si(source)
            z=build_dlc_inputs(dlc).z_si
            collection=[]
            scores,example=evaluate_windows(
                dlc,z,ZeroResidual(),
                start_interval=0,stop_interval=200,
                horizon=50,stride=25,
                q_scale=np.ones(3),
                collect_windows=collection,
            )
            self.assertEqual(scores["n_windows"],7)
            self.assertEqual(len(collection),7)
            batches=batches_from_recorded_windows({50:collection})
            batch=batches[50]
            np.testing.assert_array_equal(
                batch.anchors,np.arange(0,151,25)
            )
            np.testing.assert_allclose(
                batch.true[:,0],example["true"]
            )
            np.testing.assert_allclose(
                batch.residual[:,0],example["residual"]
            )

    def test_metrics_and_index_nav_under_headless_agg(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"dlc_exports.npz"
            write_synthetic_dlc(source,steps=200,steady=True)
            dlc=load_dlc_si(source)
            z=build_dlc_inputs(dlc).z_si
            batches=compute_all_window_batches(
                dlc,z,ZeroResidual(),horizons=(20,50),
                anchor_stride=20,
            )
            scale=np.ones(3)
            try:
                controllers={}
                for h,batch in batches.items():
                    errors=window_error_arrays(batch,scale)
                    self.assertEqual(
                        errors["nominal"]["time_window_matrix"].shape,
                        (len(batch.anchors),h+1)
                    )
                    np.testing.assert_allclose(
                        errors["nominal"]["per_window_rmse"],
                        errors["residual"]["per_window_rmse"]
                    )
                    fig,controller=build_window_figure(
                        batch,dlc.time_s,dlc.y_m,scale
                    )
                    controllers[h]=controller
                    self.assertEqual(controller["selected_index"](),0)
                    controller["set_index"](len(batch.anchors)-1)
                    self.assertEqual(
                        controller["selected_index"](),
                        len(batch.anchors)-1
                    )
                    self.assertIn("Window",fig._suptitle.get_text())
                    index=controller["selected_index"]()
                    for k in range(3):
                        np.testing.assert_allclose(
                            controller["state_lines"][k][0].get_ydata(),
                            batch.true[:,index,k]
                        )
                heatmap=build_overview_heatmap(
                    batches,scale,controllers=controllers
                )
                self.assertGreaterEqual(len(heatmap.axes),4)
            finally:
                plt.close("all")

    def test_run_experiment_auto_displays_all_windows_without_png(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"capture"/"dlc_exports.npz"
            write_synthetic_dlc(path,steps=200,steady=True)
            out=Path(td)/"train_results"
            recorded={}
            def fake_show(batches, **kwargs):
                recorded["batches"]=batches
                recorded["kwargs"]=kwargs
            with patch(
                "wm_recovery.dlc_all_windows_viewer.show_all_windows",
                new=fake_show,
            ):
                report=run_experiment(
                    path,out,
                    full_data=True,show_all_windows=True,
                    horizons=(20,50),anchor_stride=25,
                    epochs=10,batch_size=32,lr=.003,
                    make_plots=True,verbose=False,
                )
            self.assertTrue((out/"final.pt").is_file())
            self.assertTrue((out/"report.json").is_file())
            self.assertEqual(len(list(out.glob("*.png"))),0)
            self.assertIn(20,recorded["batches"])
            self.assertIn(50,recorded["batches"])
            self.assertEqual(
                len(recorded["batches"][20].anchors),
                report["rollouts"]["validation"]["20_steps"]["n_windows"],
            )
            self.assertEqual(
                len(recorded["batches"][50].anchors),
                report["rollouts"]["validation"]["50_steps"]["n_windows"],
            )
            self.assertEqual(recorded["kwargs"]["backend"],"TkAgg")

            # 已训练 checkpoint 重新加载完整全部窗口，不重新训练。
            batches=open_from_checkpoint(
                path,out/"final.pt",
                horizons=(20,50),anchor_stride=25,
                display=False,
            )
            for h in (20,50):
                np.testing.assert_allclose(
                    recorded["batches"][h].nominal,
                    batches[h].nominal,
                    rtol=1e-9,atol=1e-9
                )
                np.testing.assert_allclose(
                    recorded["batches"][h].residual,
                    batches[h].residual,
                    rtol=1e-6,atol=1e-6
                )
            with self.assertRaisesRegex(ValueError,"mu_hat"):
                open_from_checkpoint(
                    path,out/"final.pt",
                    horizons=(20,),mu_hat=0.9,
                    display=False,
                )

    def test_missing_horizon_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"dlc_exports.npz"
            write_synthetic_dlc(path,steps=100,steady=True)
            dlc=load_dlc_si(path)
            z=build_dlc_inputs(dlc).z_si
            with self.assertRaisesRegex(ValueError,"数据不足"):
                compute_all_window_batches(
                    dlc,z,ZeroResidual(),
                    horizons=(200,),anchor_stride=50,
                )


if __name__=="__main__":
    unittest.main()
