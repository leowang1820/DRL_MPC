"""Extract a deterministic deployable SAC actor from a full checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from SACModule import PolicyNetwork
from carsim_wrapper import OBS_DIM


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    BASE_DIR
    / "saves"
    / "carsim_sac_complete_dlc_40_seed42"
    / "best.pth"
)
DEFAULT_OUTPUT = BASE_DIR / "exports" / "dlc_40_best_actor.pt"


class DeterministicActor(nn.Module):
    """TorchScript-friendly mean action of the squashed Gaussian actor."""

    def __init__(self, policy: PolicyNetwork):
        super().__init__()
        self.net = policy.net
        self.mu_layer = policy.mu_layer
        self.register_buffer("action_scale", policy.action_scale.detach().clone())
        self.register_buffer("action_bias", policy.action_bias.detach().clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.net(observation)
        mu = self.mu_layer(hidden)
        return torch.tanh(mu) * self.action_scale + self.action_bias


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_actor(checkpoint_path: Path, output_path: Path):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    obs_dim = int(config.get("obs_dim", OBS_DIM))
    act_dim = int(config.get("act_dim", 2))
    hidden_size = int(config.get("hidden_size", 256))
    if obs_dim != OBS_DIM or act_dim != 2:
        raise ValueError(
            f"Unsupported checkpoint dimensions: obs_dim={obs_dim}, act_dim={act_dim}"
        )

    policy = PolicyNetwork(
        obs_dim,
        act_dim,
        hidden_size,
        np.ones(act_dim, dtype=np.float32),
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    actor = DeterministicActor(policy).eval()
    scripted = torch.jit.script(actor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.jit.save(scripted, temporary)
    temporary.replace(output_path)

    test_observations = torch.zeros((4, obs_dim), dtype=torch.float32)
    with torch.inference_mode():
        expected = torch.tanh(policy(test_observations)[0]) * policy.action_scale
        actual = torch.jit.load(str(output_path))(test_observations)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    metadata = {
        "format": "carsim_sac_deterministic_actor_v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "actor_sha256": sha256(output_path),
        "episode": int(checkpoint.get("episode", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_eval_reward": float(checkpoint.get("best_eval_reward", float("nan"))),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_size": hidden_size,
        "target_speed_kmh": float(config.get("target_speed", 40.0)),
        "scenario": config.get("scenario", "unknown"),
        "reference_path": config.get("path", ""),
        "observation_order": [
            "lateral_error/5m",
            "heading_error/45deg",
            "yaw_rate/60deg_s",
            "speed_0_160kmh_to_minus1_plus1",
            "target_speed_0_160kmh_to_minus1_plus1",
            "previous_longitudinal_action",
            "previous_steering_action",
        ],
        "action_order": [
            "longitudinal: positive=throttle, negative=brake",
            "steering_wheel: normalized, multiply by 90deg",
        ],
        "control_period_s": 0.05,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, metadata_path, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Export SAC actor to TorchScript")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    actor_path, metadata_path, metadata = export_actor(args.checkpoint, args.output)
    print("Actor exported:", actor_path)
    print("Metadata:", metadata_path)
    print("Source episode:", metadata["episode"])
    print("Best evaluation reward:", metadata["best_eval_reward"])
    print("Actor SHA-256:", metadata["actor_sha256"])
    print("TorchScript verification: PASS")


if __name__ == "__main__":
    main()
