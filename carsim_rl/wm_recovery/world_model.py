"""Five independent history-MLP residuals around the original 7DOF map."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dynamics7dof.learning.torch_physics import euler_step, rhs

from .config import WorldModelConfig, contract_digest, physics_contract


class WorldModel(nn.Module):
    def __init__(self, config, scaler):
        super().__init__()
        self.config = config
        expected = {"feature_mean": config.input_width, "feature_scale": config.input_width,
                    "residual_scale": 7, "state_scale": 7}
        for key, size in expected.items():
            value = np.asarray(scaler[key], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all() or (key != "feature_mean" and (value <= 0).any()):
                raise ValueError(f"Invalid training scaler: {key}")
            self.register_buffer(key, torch.tensor(value, dtype=torch.float64))
        members = []
        with torch.random.fork_rng(devices=[]):
            for k in range(config.ensemble_size):
                torch.manual_seed(config.seed + 997 * k)
                member = nn.Sequential(nn.Linear(config.input_width, config.hidden_width), nn.Tanh(),
                                       nn.Linear(config.hidden_width, config.hidden_width), nn.Tanh(),
                                       nn.Linear(config.hidden_width, 7)).double()
                with torch.no_grad():
                    member[-1].weight.mul_(.01)
                    member[-1].bias.zero_()
                members.append(member)
        self.members = nn.ModuleList(members)
        self.metadata = {}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.feature_mean.dtype, device=self.feature_mean.device)

    def nominal(self, state, applied, memory):
        invalid = torch.zeros(state.shape[:-1], dtype=torch.bool, device=state.device)
        mu = state.new_full(state.shape[:-1], .8)
        for _ in range(self.config.substeps):
            derivative, memory, bad = rhs(state, applied, mu, memory)
            state, clamped = euler_step(state, derivative, self.config.physics_substep_s)
            invalid = invalid | bad | clamped
        return state, memory, invalid

    def member_step(self, k, state, applied, history, memory):
        features = torch.cat((state, applied, history.flatten(-2), memory), dim=-1)
        residual = self.members[k]((features - self.feature_mean) / self.feature_scale) * self.residual_scale
        nominal, next_memory, invalid = self.nominal(state, applied, memory)
        predicted = nominal + residual
        invalid = (invalid | ~torch.isfinite(predicted).all(-1) | (predicted[..., 0] < .1)
                   | (predicted[..., 3:] < -1).any(-1))
        return predicted, next_memory, invalid

    @staticmethod
    def advance_history(history, state, applied):
        return torch.cat((history[..., 1:, :], torch.cat((state, applied), dim=-1).unsqueeze(-2)), dim=-2)

    def member_rollout(self, k, state, controls, history, memory):
        predictions, invalid = [], torch.zeros(state.shape[:-1], dtype=torch.bool, device=state.device)
        for j in range(controls.shape[-2]):
            applied = controls[..., j, :]
            predicted, next_memory, bad = self.member_step(k, state, applied, history, memory)
            history = self.advance_history(history, state, applied)
            state, memory = predicted, next_memory
            invalid = invalid | bad
            predictions.append(state)
        return torch.stack(predictions, dim=-2), invalid

    @torch.no_grad()
    def predict(self, state, applied, history, memory):
        state, applied, history, memory = map(self.tensor, (state, applied, history, memory))
        if (state.shape != (7,) or applied.shape != (6,) or history.shape != (self.config.history_steps, 13)
                or memory.shape != (2,) or not all(torch.isfinite(v).all() for v in (state, applied, history, memory))):
            raise ValueError("predict requires one finite SI state/input/history/load-memory sample")
        predictions, memories, invalid = [], [], []
        for k in range(self.config.ensemble_size):
            x, m, bad = self.member_step(k, state, applied, history, memory)
            predictions.append(x)
            memories.append(m)
            invalid.append(bad)
        values = torch.stack(predictions)
        mean = values.mean(0)
        # W = diag(1/state_scale^2), unbiased K-1 variance.
        uncertainty = (((values - mean) / self.state_scale).square().sum(-1).sum()
                       / (self.config.ensemble_size - 1))
        return {"members": values.cpu().numpy(), "mean": mean.cpu().numpy(),
                "variance": values.var(0, unbiased=True).cpu().numpy(), "uncertainty": float(uncertainty),
                "next_load_memory": torch.stack(memories).mean(0).cpu().numpy(),
                "valid": not bool(torch.stack(invalid).any())}

    def save(self, path):
        payload = {"format": "wm_recovery_ensemble_v1", "config": asdict(self.config),
                   "physics_digest": contract_digest(), "physics_contract": physics_contract(),
                   "metadata": self.metadata, "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "wm_recovery_ensemble_v1" or payload.get("physics_digest") != contract_digest():
            raise ValueError("World Model format or original-physics fingerprint differs")
        config = WorldModelConfig(**payload["config"])
        scaler = {key: payload["state_dict"][key].numpy() for key in ("feature_mean", "feature_scale", "state_scale", "residual_scale")}
        model = cls(config, scaler)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.metadata = payload["metadata"]
        return model.to(device).eval()
