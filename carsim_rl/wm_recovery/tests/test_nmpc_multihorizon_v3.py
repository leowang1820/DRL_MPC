# -*- coding: utf-8 -*-
"""NMPC v3 多时域 + 终点损失 + 场景均衡采样的离线回归测试。"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wm_recovery.nmpc_reference import NMPCReferenceParams, rk4_nominal_step
from wm_recovery.nmpc_rollout import rk4_free_step
from wm_recovery.nmpc_residual_model import (
    INPUT_NAMES, TARGET_NAMES, load_model_checkpoint,
)
from wm_recovery.nmpc_residual_train import run_training, load_split_npz, sha256
from wm_recovery.nmpc_residual_multistep_train import (
    parse_horizon_configuration, _scenario_family,
    BalancedIndexSampler, contiguous_window_starts, window_batch_torch,
    differentiable_rollout_components, differentiable_rollout_loss,
    validation_joint_metrics, fine_tune_multistep,
)


class KnownResidual:
    def predict_numpy(self, z):
        x = np.asarray(z)
        output = np.array([0.3, 0.02, -0.1])
        return np.broadcast_to(
            output, (len(x), 3) if x.ndim == 2 else (3,)
        ).copy()


def _one_scene(length, episode_index, name):
    q = np.array([0.2+0.02*episode_index, 0.02, 11.0+episode_index],
                 dtype=np.float64)
    z_list, labels, q0s, truths, nominals = [], [], [], [], []
    for i in range(length):
        ext = np.array([
            0.014 + 0.003*np.sin(i/11.0), 25.0, 200.0, 0.5, 0.8
        ], dtype=np.float64)
        z = np.concatenate((q, ext))
        nominal = rk4_nominal_step(z, dt_s=0.01)
        actual = rk4_free_step(q, ext, dt_s=0.01,
                               residual_model=KnownResidual())
        z_list.append(z)
        labels.append((actual - nominal)/0.01)
        q0s.append(q.copy())
        truths.append(actual.copy())
        nominals.append(nominal.copy())
        q = actual
    n = length
    return {
        "z_si": np.asarray(z_list),
        "residual_y_si": np.asarray(labels),
        "q_k_si": np.asarray(q0s),
        "q_true_next_si": np.asarray(truths),
        "q_nominal_next_si": np.asarray(nominals),
        "dt_s": np.full(n,0.01),
        "time_k_s": np.arange(n, dtype=float) * 0.01,
        "sample_weight": np.ones(n, dtype=float),
        "lamda_regime_code": np.zeros(n,dtype=np.int64),
        "episode_index": np.full(n,episode_index,dtype=np.int64),
        "segment_id": np.full(n,episode_index,dtype=np.int64),
        "segment_step": np.arange(n,dtype=np.int64),
        "interval_index": np.arange(n,dtype=np.int64),
    }


def _dataset(root):
    root.mkdir(parents=True, exist_ok=False)
    specs = [
        ("v40_train_circle_left", "train", 75),
        ("v40_train_brake_turn_left", "train", 65),
        ("v60_train_swd_left", "train", 75),
        ("v60_train_brake_turn_right", "train", 55),
        ("v40_validation_circle_right", "validation", 65),
        ("v60_validation_brake_turn", "validation", 60),
        ("v60_test_swd_right", "test", 55),
    ]
    per_split = {"train":[], "validation":[], "test":[]}
    for ep, (name, split, length) in enumerate(specs):
        per_split[split].append(_one_scene(length, ep, name))
    data = {}
    for split, parts in per_split.items():
        assembled = {
            key:np.concatenate([episode[key] for episode in parts], axis=0)
            for key in parts[0]
        }
        data[split] = assembled
        np.savez_compressed(root/f"{split}.npz", **assembled)

    manifest = {
        "format": "wm_recovery_nmpc_residual_dataset_v1",
        "input_names": list(INPUT_NAMES),
        "target_names": list(TARGET_NAMES),
        "dataset_settings": {"expected_dt_s":0.01, "rk4_substeps":1},
        "reference_parameters": asdict(NMPCReferenceParams()),
        "split_counts": {
            key: {"intervals":int(len(value["z_si"]))}
            for key, value in data.items()
        },
        "episodes": [
            {"name":name, "split":split}
            for name, split, _ in specs
        ],
    }
    (root/"dataset_manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2), encoding="utf-8"
    )
    return data, manifest


class TrainableResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor([0.05,0.03,-0.01],dtype=torch.float64)
        )

    def forward(self,z):
        return z[:, :3] * self.scale[None,:]


class MultihorizonV3Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parse_and_validation(self):
        h,w=parse_horizon_configuration(horizons="50,100,200",
                                        horizon_weights="4,3.5,2.5")
        self.assertEqual(h,(50,100,200))
        self.assertAlmostEqual(w[50],0.4)
        self.assertAlmostEqual(w[100],0.35)
        self.assertAlmostEqual(w[200],0.25)
        self.assertEqual(parse_horizon_configuration(horizon=10)[0],(10,))
        for horizons,weights in [
            ("50,50",None),("50,0",None),("50,501",None),
            ("50,100","1"),("50,100","0,1"),
            ("50,100","nan,1"),("1.5,100",None),
        ]:
            with self.subTest(horizons=horizons,weights=weights):
                with self.assertRaises(ValueError):
                    parse_horizon_configuration(horizons=horizons,
                                                horizon_weights=weights)

    def test_scenario_family(self):
        self.assertEqual(
            _scenario_family("v60_train_brake_turn_left"), "v60:brake_turn"
        )
        self.assertEqual(
            _scenario_family("v40_train_swd_right"), "v40:swd"
        )
        self.assertEqual(
            _scenario_family("v40_train_dlc_proxy"),"v40:dlc_proxy"
        )

    def test_balanced_sampler_round_robin_across_train_families(self):
        with tempfile.TemporaryDirectory() as td:
            data,manifest=_dataset(Path(td)/"data")
            train=data["train"]
            starts=contiguous_window_starts(train,30,stride=10)
            sampler=BalancedIndexSampler(
                starts, train, manifest, mode="family", seed=1
            )
            indices=sampler.draw(120)
            self.assertTrue(set(indices).issubset(set(starts)))
            summary=sampler.summary()
            self.assertEqual(summary["group_count"],4)
            counts=list(summary["observed_sampled_counts"].values())
            self.assertEqual(max(counts)-min(counts),0)
            # 均衡采样只来自 Train 中已存在的原始时间索引。
            self.assertTrue(np.all(train["episode_index"][indices]<4))
            raw=BalancedIndexSampler(
                starts,train,manifest,mode="none",seed=2
            )
            self.assertEqual(len(raw.draw(5)),5)

    def test_multihorizon_terminal_has_gradient(self):
        with tempfile.TemporaryDirectory() as td:
            data,_=_dataset(Path(td)/"data")
            starts=contiguous_window_starts(data["train"],10,stride=20)[:2]
            batch=window_batch_torch(data["train"],starts,10)
            model=TrainableResidual()
            result=differentiable_rollout_components(
                model,batch,torch.ones(3,dtype=torch.float64),
                params=NMPCReferenceParams(),
            )
            self.assertTrue(result["path"].isfinite().item())
            self.assertTrue(result["terminal"].isfinite().item())
            (result["path"]+0.4*result["terminal"]).backward()
            self.assertGreater(float(model.scale.grad.abs().sum()),0)
            self.assertTrue(model.scale.grad.isfinite().all().item())
            old=differentiable_rollout_loss(
                model,batch,torch.ones(3,dtype=torch.float64),
                params=NMPCReferenceParams(),
            )
            self.assertAlmostEqual(old[0].detach().item(),result["path"].detach().item())
            self.assertAlmostEqual(old[1].detach().item(),result["terminal"].detach().item())

    def test_two_hundred_step_rk4_autograd_smoke(self):
        # 覆盖真正的 2 秒训练反向传播，不调用 CarSim DLL。
        # 监督轨迹为合成常量，仅验证计算图与梯度数值稳定。
        model=TrainableResidual()
        batch={
            "q0":torch.tensor([[0.2,0.03,13.0]],dtype=torch.float64),
            "external":torch.tensor(
                [[[0.02,35.0,200.0,0.5,0.8]]*200],
                dtype=torch.float64,
            ),
            "dt":torch.full((1,200),0.01,dtype=torch.float64),
            "true_next":torch.zeros((1,200,3),dtype=torch.float64),
            "weights":torch.ones((1,200),dtype=torch.float64),
        }
        losses=differentiable_rollout_components(
            model,batch,torch.tensor([1.,1.,10.],dtype=torch.float64),
            params=NMPCReferenceParams(),
        )
        loss=losses["path"] + 0.3*losses["terminal"]
        loss.backward()
        self.assertTrue(model.scale.grad.isfinite().all().item())
        self.assertGreater(float(model.scale.grad.abs().sum()),1e-7)

    def test_windows_refuse_time_gaps_and_cross_episode(self):
        with tempfile.TemporaryDirectory() as td:
            data,_=_dataset(Path(td)/"data")
            valid=data["validation"]
            starts=contiguous_window_starts(valid,15,stride=10)
            self.assertTrue(len(starts)>0)
            bad={key:value.copy() for key,value in valid.items()}
            bad["time_k_s"][20] += 0.02
            with self.assertRaises(ValueError):
                window_batch_torch(bad,np.array([10]),20,
                                   use_train_weights=False)
            # 模拟跨越真实轨迹边界。
            with self.assertRaises(ValueError):
                window_batch_torch(valid,np.array([63]),10)

    def test_end_to_end_joint_and_mixed_and_no_test_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            ds=root/"data"
            _,manifest=_dataset(ds)
            initial=root/"pretrained"
            run_training(ds,initial,epochs=1,batch_size=64,seed=16)
            before_hash=sha256(initial/"best.pt")
            from wm_recovery import nmpc_residual_multistep_train as trainer
            original_loader=trainer.load_split_npz
            calls=[]
            def safe_read(folder, split):
                calls.append(split)
                if split=="test":
                    raise AssertionError("绝不允许加载 Test 数据")
                return original_loader(folder,split)

            with patch.object(trainer,"load_split_npz",side_effect=safe_read):
                report=fine_tune_multistep(
                    ds, initial/"best.pt",root/"joint",
                    horizons="10,20",horizon_weights="0.6,0.4",
                    horizon_strategy="joint",
                    terminal_weight=0.3,
                    scene_balance="family",validation_scene_weight=0.5,
                    epochs=2,batch_size=2,steps_per_epoch=2,
                    learning_rate=0.00005,
                    train_stride=10,validation_stride=20,
                    seed=7,
                )
            self.assertEqual(set(calls),{"train","validation"})
            self.assertTrue(report["test_not_used"])
            self.assertEqual(report["horizons_intervals"],[10,20])
            self.assertIn("10",report["initial_validation"]["per_horizon"])
            self.assertIn("20",report["initial_validation"]["per_horizon"])
            self.assertTrue(np.isfinite(
                report["best_validation_joint_normalized_mse"]
            ))
            self.assertEqual(sha256(initial/"best.pt"),before_hash)
            self.assertTrue((root/"joint"/"history.json").exists())
            model,payload=load_model_checkpoint(root/"joint"/"best.pt")
            self.assertEqual(payload["training_version"],"v3_joint_multihorizon")
            self.assertEqual(model.predict_numpy(
                np.array([0.0,0.0,12.0,0.03,50.0,200.0,0.5,0.8])
            ).shape,(3,))
            val=load_split_npz(ds,"validation")
            fixed={
                h:contiguous_window_starts(val,h,stride=20) for h in (10,20)
            }
            selected=validation_joint_metrics(
                model.double(),val,fixed,(10,20),{10:0.6,20:0.4},
                manifest=manifest,params=NMPCReferenceParams(),
                q_scale=model.x_scale[:3].double(),device="cpu",
                batch_size=2,integration="continuous",rk4_substeps=1,
                terminal_weight=0.3,validation_scene_weight=0.5,
            )
            self.assertTrue(np.isfinite(
                selected["validation_joint_normalized_mse"]
            ))

            # 随机混合时域模式，仍用完整的联合 Validation 选模。
            mixed=fine_tune_multistep(
                ds,initial/"best.pt",root/"mixed",
                horizons=(10,20),horizon_weights=(0.5,0.5),
                horizon_strategy="mixed",terminal_weight=0.2,
                scene_balance="scene",validation_scene_weight=0.0,
                epochs=1,batch_size=2,steps_per_epoch=4,
                train_stride=10,validation_stride=20,seed=13,
            )
            self.assertTrue(mixed["test_not_used"])
            history=json.loads((root/"mixed"/"history.json").read_text(
                encoding="utf-8"
            ))
            chosen=history[0]["sampled_horizon_updates"]
            self.assertEqual(sum(chosen.values()),4)
            self.assertFalse((root/"joint"/"test.npz").exists())

            # 旧 --horizon 形式的 API 保持工作。
            legacy=fine_tune_multistep(
                ds, initial/"best.pt", root/"legacy",
                horizon=10,epochs=1,batch_size=2,
                steps_per_epoch=1,train_stride=10,
                validation_stride=20,seed=17,
            )
            self.assertEqual(legacy["horizons_intervals"],[10])
            self.assertTrue((root/"legacy"/"best.pt").exists())

            with self.assertRaises(FileExistsError):
                fine_tune_multistep(
                    ds,initial/"best.pt",root/"joint",epochs=1,
                )


if __name__=="__main__":
    unittest.main()
