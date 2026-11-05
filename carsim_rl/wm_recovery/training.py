"""Offline trajectory-bootstrap training with explicit one/multi-step losses."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import fit_scaler, load_partition, provenance, trajectory_bootstrap
from .world_model import WorldModel


def make_batch(model, prepared, pairs):
    states, controls, histories, memories, truth = [], [], [], [], []
    h, steps = model.config.history_steps, model.config.rollout_steps
    for ep, i in pairs:
        data = prepared[ep]
        states.append(data.episode.state[i])
        controls.append(data.episode.applied[i:i + steps])
        histories.append(data.history_at(i, h))
        memories.append(data.memory[i])
        truth.append(data.episode.state[i + 1:i + steps + 1])
    return tuple(model.tensor(np.asarray(value)) for value in (states, controls, histories, memories, truth))


def train(paths, output, config, *, epochs=20, batch_size=32, learning_rate=1e-3,
          device="cpu", allow_synthetic=False):
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive epochs, batch_size, and learning_rate required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training output must be a new directory")
    prepared = load_partition(paths, config, "train", allow_synthetic=allow_synthetic)
    draws, pools = trajectory_bootstrap(prepared, config.ensemble_size, config.seed)
    model = WorldModel(config, fit_scaler(prepared, config)).to(device)
    optimizers = [torch.optim.Adam(member.parameters(), lr=learning_rate) for member in model.members]
    generators = [np.random.default_rng(config.seed + 99991 * (k + 1)) for k in range(config.ensemble_size)]
    history = []
    for epoch in range(epochs):
        losses = []
        for k, pool in enumerate(pools):
            shuffled = generators[k].permutation(len(pool))
            total, count = 0., 0
            for start in range(0, len(pool), batch_size):
                batch = make_batch(model, prepared, pool[shuffled[start:start + batch_size]])
                state, controls, past, memory, truth = batch
                predicted, invalid = model.member_rollout(k, state, controls, past, memory)
                if invalid.any():
                    raise RuntimeError("Training rollout left the original model domain; inspect data and training settings")
                one_step = ((predicted[:, 0] - truth[:, 0]) / model.residual_scale).square().mean()
                multi_step = ((predicted - truth) / model.state_scale).square().mean()
                loss = one_step + config.rollout_loss_weight * multi_step
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                optimizers[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.members[k].parameters(), 10., error_if_nonfinite=True)
                optimizers[k].step()
                total += float(loss.detach()) * len(truth)
                count += len(truth)
            losses.append(total / count)
        history.append({"epoch": epoch + 1, "member_loss": losses})
    sources = provenance(prepared)
    model.metadata = {
        "training_trajectories": sources,
        "bootstrap_trajectory_ids": [[sources[i]["trajectory_id"] for i in draw] for draw in draws],
        "bootstrap_unit": "whole_trajectory", "scaler_source": "training_partition_only",
        "synthetic_demo_only": any(p["source_kind"] == "synthetic" for p in sources),
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "multi_step_teacher_forcing": False,
        "loss": "mean(square(first_state_error / residual_scale)) + rollout_loss_weight * mean(square(rollout_state_error / state_scale))",
    }
    output.mkdir(parents=True, exist_ok=False)
    model.eval().save(output / "world_model.pt")
    report = {"config": asdict(config), "metadata": model.metadata, "history": history,
              "status": "trained_offline_not_validated_for_control"}
    (output / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
