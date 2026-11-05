from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF
from wm_recovery.confidence import ConfidenceMonitor, GateSettings, calibrate
from wm_recovery.config import WorldModelConfig
from wm_recovery.dataset import fit_scaler, load_partition, trajectory_bootstrap
from wm_recovery.evaluation import one_step_metrics
from wm_recovery.synthetic import generate
from wm_recovery.training import train
from wm_recovery.world_model import WorldModel


class WorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return WorldModelConfig(history_steps=2, hidden_width=8, rollout_steps=2)

    def model(self, zero=False):
        config = self.config()
        model = WorldModel(config, {"feature_mean": np.zeros(config.input_width),
                                   "feature_scale": np.ones(config.input_width),
                                   "residual_scale": np.ones(7) * .001, "state_scale": np.ones(7)})
        if zero:
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
        return model

    def sample(self, model):
        return tuple(map(model.tensor, (
            [10., .02, .01, 28.3, 28.4, 28.3, 28.4], [.002, .002, 5., -2., 4., -1.],
            np.zeros((model.config.history_steps, 13)), [.1, -.2])))

    def test_zero_residual_matches_original_7dof_with_carried_memory(self):
        model = self.model(zero=True)
        x, u, h, m = self.sample(model)
        expected = Reference7DOF().predict(x.numpy(), Controls(u[:2].numpy(), u[2:].numpy()), mu=.8,
                    load_memory=LoadMemory(*m.numpy()), horizon_s=.01, substeps=20)
        actual, memory, bad = model.member_step(0, x, u, h, m)
        np.testing.assert_allclose(actual.detach(), expected.state, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(memory.detach(), [expected.load_memory.ax_m_s2, expected.load_memory.ay_m_s2], atol=1e-10)
        self.assertFalse(bad)

    def test_rollout_uses_predicted_state_and_history(self):
        model = self.model()
        x, u, h, m = self.sample(model)
        expected = []
        for _ in range(3):
            value, next_memory, _ = model.member_step(1, x, u, h, m)
            h = model.advance_history(h, x, u)
            x, m = value, next_memory
            expected.append(x)
        x, u, h, m = self.sample(model)
        actual, _ = model.member_rollout(1, x, u.repeat(3, 1), h, m)
        torch.testing.assert_close(actual, torch.stack(expected))
        actual[-1, 0].backward()
        self.assertTrue(torch.isfinite(model.members[1][-1].bias.grad).all())
        self.assertGreater(abs(float(model.members[1][-1].bias.grad[0])), 0)

    def test_independent_initializations_and_unbiased_variance(self):
        model = self.model()
        result = model.predict(*(v.numpy() for v in self.sample(model)))
        self.assertEqual(result["members"].shape, (5, 7))
        self.assertFalse(np.array_equal(result["members"][0], result["members"][1]))
        np.testing.assert_allclose(result["variance"], np.var(result["members"], axis=0, ddof=1))
        self.assertAlmostEqual(result["uncertainty"], np.var(result["members"], axis=0, ddof=1).sum())

    def test_checkpoint_roundtrip_and_refuse_overwrite(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            restored = WorldModel.load(path)
            expected = model.predict(*(v.numpy() for v in self.sample(model)))
            actual = restored.predict(*(v.numpy() for v in self.sample(restored)))
            np.testing.assert_array_equal(actual["members"], expected["members"])
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_config_rejects_noninteger_physics_substeps(self):
        with self.assertRaises(ValueError):
            WorldModelConfig(physics_substep_s=.003)

    def test_training_bootstraps_whole_trajectories_and_excludes_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config = Path(tmp), self.config()
            paths = [generate(root / str(i), config, identifier=f"train_{i}", split="train", speed=10 + i, seconds=.12) for i in range(2)]
            prepared = load_partition(paths, config, "train", allow_synthetic=True)
            draws, pools = trajectory_bootstrap(prepared, 5, 42)
            for draw, pool in zip(draws, pools):
                for ep in set(draw):
                    counts = [np.count_nonzero((pool[:, 0] == ep) & (pool[:, 1] == i)) for i in prepared[ep].starts]
                    self.assertEqual(counts, [draw.count(ep)] * len(counts))
            report = train(paths, root / "trained", config, epochs=1, batch_size=16, allow_synthetic=True)
            self.assertEqual(report["metadata"]["bootstrap_unit"], "whole_trajectory")
            self.assertTrue(report["metadata"]["synthetic_demo_only"])
            model = WorldModel.load(root / "trained" / "world_model.pt")
            metrics = one_step_metrics(model, prepared[0])
            self.assertTrue(np.all(metrics["valid"]))
            validation = generate(root / "val", config, identifier="val", split="validation", speed=12, seconds=.12)
            with self.assertRaisesRegex(ValueError, "Expected train"):
                load_partition([validation], config, "train", allow_synthetic=True)
            with self.assertRaisesRegex(ValueError, "requires"):
                load_partition(paths, config, "train")

    def test_heldout_cannot_reuse_training_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config = Path(tmp), self.config()
            path = generate(root / "test", config, identifier="seen", split="test", speed=12, seconds=.12)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_partition([path], config, "test", allow_synthetic=True, excluded_ids=["seen"])

    def test_history_contains_only_past_measured_state_and_applied_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config = Path(tmp), self.config()
            path = generate(root / "one", config, identifier="one", split="train", speed=10, seconds=.12)
            data = load_partition([path], config, "train", allow_synthetic=True)[0]
            original = data.history_at(4, 2)
            data.episode.state[4:] += 100
            data.episode.applied[4:] += 100
            np.testing.assert_array_equal(data.history_at(4, 2), original)


class ConfidenceTests(unittest.TestCase):
    def calibration(self):
        return {"settings": {"window_steps": 2, "exceed_steps": 2, "recover_steps": 2, "quantile": .99},
                "uncertainty_threshold": 1., "residual_threshold": 1., "residual_scale": [1.] * 7}

    def test_warmup_persistence_and_recovery(self):
        monitor = ConfidenceMonitor(self.calibration())
        x = np.zeros(7)
        self.assertFalse(monitor.observe(0, x, x)["trusted"])
        self.assertFalse(monitor.observe(0, x, x)["trusted"])
        self.assertTrue(monitor.observe(0, x, x)["trusted"])
        monitor.observe(4, x, x)
        self.assertEqual(monitor.observe(4, x, x)["state"], "untrusted")
        for _ in range(4):
            status = monitor.observe(0, x, x)
        self.assertTrue(status["trusted"])

    def test_residual_catches_shared_error_when_variance_is_zero(self):
        joint = ConfidenceMonitor(self.calibration())
        ensemble = ConfidenceMonitor(self.calibration(), "ensemble_only")
        for _ in range(6):
            a = joint.observe(0, np.zeros(7), np.ones(7))
            b = ensemble.observe(0, np.zeros(7), np.ones(7))
        self.assertEqual(a["state"], "untrusted")
        self.assertEqual(b["state"], "trusted")

    def test_invalid_prediction_resets_window(self):
        monitor = ConfidenceMonitor(self.calibration())
        for _ in range(5):
            monitor.observe(0, np.zeros(7), np.zeros(7))
        status = monitor.observe(0, np.zeros(7), np.zeros(7), prediction_valid=False)
        self.assertEqual(status["state"], "untrusted")
        self.assertFalse(monitor.observe(0, np.zeros(7), np.zeros(7))["trusted"])

    def test_calibration_never_spans_episode_boundary(self):
        episodes = [{"valid": np.ones(12, dtype=bool), "uncertainty": np.zeros(12), "error": np.zeros((12, 7))} for _ in range(2)]
        result = calibrate(episodes, GateSettings(window_steps=2), np.ones(7))
        self.assertEqual(result["calibration_windows"], 22)
        self.assertGreater(result["residual_threshold"], 0)
        episodes[0]["valid"][3] = False
        with self.assertRaisesRegex(ValueError, "invalid"):
            calibrate(episodes, GateSettings(window_steps=2), np.ones(7))


if __name__ == "__main__":
    unittest.main()
