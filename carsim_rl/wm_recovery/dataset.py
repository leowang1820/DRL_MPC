"""Causal windows and trajectory bootstrap; no cross-trajectory histories."""

from dataclasses import dataclass

import numpy as np

from dynamics7dof.model import Reference7DOF

from .offline import Episode, NOMINAL_MU, audit_partition, controls_at


FEATURE_FLOOR = np.array([.1, .02, .01, .2, .2, .2, .2, .001, .001, 10, 10, 10, 10])
STATE_FLOOR = FEATURE_FLOOR[:7]
RESIDUAL_FLOOR = np.array([.001, .001, .0001, .02, .02, .02, .02])


def feature_vector(state, applied, history, memory):
    return np.concatenate((state, applied, history.reshape(-1), memory), axis=-1)


@dataclass
class PreparedEpisode:
    episode: Episode
    memory: np.ndarray
    nominal_next: np.ndarray
    valid: np.ndarray
    starts: np.ndarray

    def history_at(self, index, steps):
        return np.concatenate((self.episode.state[index - steps:index], self.episode.applied[index - steps:index]), axis=1)


def prepare_episode(episode, config):
    if not np.allclose(np.diff(episode.time), config.prediction_dt_s, rtol=0, atol=1e-8):
        raise ValueError("Trajectory grid differs from prediction_dt_s; resampling requires an explicit input-timing policy")
    model = Reference7DOF()
    memory = episode.initial_memory
    memories, nominal, valid = [], [], []
    for i in range(len(episode.time) - 1):
        memories.append([memory.ax_m_s2, memory.ay_m_s2])
        p = model.predict(episode.state[i], controls_at(episode, i), mu=NOMINAL_MU,
                          load_memory=memory, horizon_s=config.prediction_dt_s, substeps=config.substeps)
        nominal.append(p.state)
        valid.append(not p.flags)
        memory = p.load_memory
    valid = np.asarray(valid, dtype=bool)
    # Once nominal load memory is invalid, subsequent anchors are also unverified.
    valid = np.logical_and.accumulate(valid)
    starts = np.array([i for i in range(config.history_steps, len(valid) - config.rollout_steps + 1)
                       if valid[i - config.history_steps:i + config.rollout_steps].all()], dtype=int)
    if not len(starts):
        raise ValueError(f"No valid causal windows in trajectory {episode.metadata['trajectory_id']}")
    return PreparedEpisode(episode, np.asarray(memories), np.asarray(nominal), valid, starts)


def load_partition(paths, config, split, *, allow_synthetic=False, excluded_ids=(), excluded_hashes=(), excluded_conditions=()):
    episodes = [Episode.load(path) for path in paths]
    if not episodes:
        raise ValueError("At least one trajectory is required")
    audit_partition(episodes)
    for episode in episodes:
        meta = episode.metadata
        if meta["split"] != split:
            raise ValueError(f"Expected {split} trajectory, got {meta['split']}")
        if (meta["trajectory_id"] in excluded_ids or episode.source_sha256 in excluded_hashes
                or meta["condition_id"] in excluded_conditions):
            raise ValueError("Evaluation/calibration trajectory overlaps a previously used partition")
        if meta["source_kind"] == "synthetic":
            if not allow_synthetic:
                raise ValueError("Synthetic data requires --allow-synthetic and is never CarSim evidence")
        elif meta["parameter_alignment"] != "verified":
            raise ValueError("CarSim/paper parameter alignment is unverified; complete it before nominal training")
    return [prepare_episode(episode, config) for episode in episodes]


def fit_scaler(prepared, config):
    features, residuals, states = [], [], []
    for data in prepared:
        for i in data.starts:
            features.append(feature_vector(data.episode.state[i], data.episode.applied[i],
                                           data.history_at(i, config.history_steps), data.memory[i]))
            residuals.append(data.episode.state[i + 1] - data.nominal_next[i])
            states.append(data.episode.state[i + 1:i + config.rollout_steps + 1])
    features, residuals, states = np.asarray(features), np.asarray(residuals), np.concatenate(states)
    floor = np.r_[FEATURE_FLOOR, np.tile(FEATURE_FLOOR, config.history_steps), [.1, .1]]
    return {"feature_mean": features.mean(0), "feature_scale": np.maximum(features.std(0), floor),
            "residual_scale": np.maximum(residuals.std(0), RESIDUAL_FLOOR),
            "state_scale": np.maximum(states.std(0), STATE_FLOOR)}


def trajectory_bootstrap(prepared, member_count, seed):
    """Sample entire trajectories with replacement, retaining all their windows."""
    if len(prepared) < 2:
        raise ValueError("Trajectory-level bootstrap needs at least two distinct training trajectories")
    draws, pools = [], []
    for k in range(member_count):
        rng = np.random.default_rng(seed + 104729 * (k + 1))
        sampled = rng.integers(0, len(prepared), size=len(prepared))
        draws.append(sampled.tolist())
        pools.append(np.array([(int(ep), int(i)) for ep in sampled for i in prepared[ep].starts], dtype=int))
    return draws, pools


def provenance(prepared):
    return [{"trajectory_id": p.episode.metadata["trajectory_id"], "condition_id": p.episode.metadata["condition_id"],
            "sha256": p.episode.source_sha256, "hash_basis": "canonical_physical_arrays",
            "source_kind": p.episode.metadata["source_kind"],
             "split": p.episode.metadata["split"], "windows": len(p.starts)} for p in prepared]
