"""Explicit trajectory format and offline nominal-model validation.

No hidden road-friction input, learned model, or CarSim command is used here.
Inputs are actual applied interval inputs, not direction-wheel commands.
"""

from dataclasses import dataclass
import json
import hashlib
import math
from pathlib import Path

import numpy as np

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF, STATE_NAMES

from .controls import APPLIED_INPUT_NAMES, COMMAND_NAMES, finite_array


FORMAT = "wm_recovery_trajectory_v1"
NOMINAL_MU = 0.8
SPLITS = ("train", "validation", "test", "unknown_mu_test")


@dataclass
class Episode:
    time: np.ndarray
    state: np.ndarray
    applied: np.ndarray
    command: np.ndarray
    initial_memory: LoadMemory
    metadata: dict
    source_sha256: str

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError("Unsupported trajectory format; old captures need an explicit converter")
        for key in ("trajectory_id", "condition_id", "source_kind"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                raise ValueError(f"Missing {key}")
        if meta["source_kind"] not in ("carsim", "synthetic") or meta.get("split") not in SPLITS:
            raise ValueError("Invalid source_kind or split")
        for key, names in (("state_names", STATE_NAMES), ("applied_input_names", APPLIED_INPUT_NAMES),
                           ("command_names", COMMAND_NAMES)):
            if meta.get(key) != list(names):
                raise ValueError(f"Explicit SI column contract mismatch: {key}")
        if meta.get("input_alignment") != "held_over_interval_t_i_to_t_i_plus_1":
            raise ValueError("Input timing must be explicit; endpoint input samples are not interval inputs")
        if meta.get("nominal_mu") != NOMINAL_MU:
            raise ValueError("First version requires the confirmed nominal mu=0.8")
        actual_mu = meta.get("actual_mu")
        if (isinstance(actual_mu, bool) or not isinstance(actual_mu, (int, float))
                or not math.isfinite(actual_mu) or actual_mu <= 0):
            raise ValueError("actual_mu must be positive and finite scenario metadata")
        if meta["split"] != "unknown_mu_test" and actual_mu != NOMINAL_MU:
            raise ValueError("Other friction conditions belong exclusively to unknown_mu_test")
        if meta.get("actuation_definition") != "external_wheel_torque_without_friction_brake":
            raise ValueError("External moment definition must be explicit")
        if meta.get("other_wheel_actuation_verified_zero") is not True:
            raise ValueError("Verify internal drive/friction-brake contributions before nominal replay")
        if meta.get("parameter_alignment") not in ("unverified", "verified"):
            raise ValueError("Parameter alignment status is required")
        if not isinstance(meta.get("initial_memory_source"), str) or not meta["initial_memory_source"].strip():
            raise ValueError("initial_memory_source must explain the explicit load-memory initialization")
        path = directory / "trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            time = np.asarray(data["time_s"], dtype=float)
            if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
                raise ValueError("At least two strictly increasing observed solver timestamps required")
            n = len(time)
            state = finite_array(data["state"], (n, 7), "state")
            applied = finite_array(data["applied_input"], (n - 1, 6), "applied_input")
            command = finite_array(data["command"], (n - 1, 5), "command")
            memory = finite_array(data["initial_load_memory_m_s2"], (2,), "initial_load_memory_m_s2")
        if np.any(np.abs(applied[:, :2]) >= np.pi / 2) or np.any(np.abs(command[:, 4]) >= np.pi / 2):
            raise ValueError("Steering columns must contain radians inside (-pi/2, pi/2)")
        if not np.allclose(applied[:, 0], applied[:, 1], rtol=0, atol=1e-10):
            raise ValueError("Confirmed first-stage vehicle requires equal actual left/right front angles")
        if "mismatch_start_s" in meta:
            onset = meta["mismatch_start_s"]
            if (isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset)
                    or not time[0] <= onset <= time[-1]):
                raise ValueError("mismatch_start_s must be a finite time inside the trajectory")
        # Hash physical arrays, not ZIP bytes: recompression or renaming must not
        # disguise the same recording as independent validation/test data.
        digest = hashlib.sha256()
        for array in (time, state, applied, command, memory):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return cls(time.copy(), state, applied, command, LoadMemory(*memory), meta, digest.hexdigest())


def audit_partition(episodes):
    """No trajectory/content duplication; a condition must belong to one split."""
    ids, hashes, conditions = set(), set(), {}
    for episode in episodes:
        meta = episode.metadata
        if meta["trajectory_id"] in ids or episode.source_sha256 in hashes:
            raise ValueError("Duplicate trajectory or recording content")
        ids.add(meta["trajectory_id"])
        hashes.add(episode.source_sha256)
        previous = conditions.setdefault(meta["condition_id"], meta["split"])
        if previous != meta["split"]:
            raise ValueError("One condition_id cannot span dataset splits")
    return {"trajectories": len(ids), "conditions": len(conditions), "split_unit": "whole_trajectory_and_condition"}


def controls_at(episode, index):
    return Controls(episode.applied[index, :2], episode.applied[index, 2:])


def _scores(predicted, truth, valid):
    selected = predicted[valid] - truth[valid]
    return {
        "valid_samples": int(np.count_nonzero(valid)), "total_samples": len(valid),
        "coverage": float(np.mean(valid)),
        "rmse": dict(zip(STATE_NAMES, map(float, np.sqrt(np.mean(selected ** 2, axis=0))))) if len(selected) else None,
        "mae": dict(zip(STATE_NAMES, map(float, np.mean(np.abs(selected), axis=0)))) if len(selected) else None,
    }


def evaluate_nominal(episode, *, max_substep_s, model=None, horizon_s=1.0):
    """One-step, uninterrupted replay, and exact-horizon rolling predictions.

    The reference model retains its explicit load memory. A causal observer
    carries nominal memory while resetting only the measured seven-state vector
    for one-step scores. Continuous replay never resets either. Rolling windows
    start from the corresponding causal memory, not from future accelerations.
    """
    if not math.isfinite(max_substep_s) or max_substep_s <= 0:
        raise ValueError("max_substep_s must be positive and finite")
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("horizon_s must be positive and finite")
    model = Reference7DOF() if model is None else model
    n = len(episode.time)
    one = np.full((n - 1, 7), np.nan)
    free = np.full((n - 1, 7), np.nan)
    one_valid = np.zeros(n - 1, dtype=bool)
    free_valid = np.zeros(n - 1, dtype=bool)
    memories = [episode.initial_memory]
    free_memory, free_state = episode.initial_memory, episode.state[0].copy()
    observer_alive = free_alive = True
    flags = set()

    def step(state, memory, index):
        dt = float(episode.time[index + 1] - episode.time[index])
        return model.predict(state, controls_at(episode, index), mu=NOMINAL_MU,
                             load_memory=memory, horizon_s=dt,
                             substeps=max(1, math.ceil(dt / max_substep_s)))

    for i in range(n - 1):
        if observer_alive:
            p = step(episode.state[i], memories[-1], i)
            flags.update(p.flags)
            observer_alive = not p.flags
            if observer_alive:
                one[i], one_valid[i] = p.state, True
                memories.append(p.load_memory)
        if free_alive:
            p = step(free_state, free_memory, i)
            flags.update(p.flags)
            free_alive = not p.flags
            if free_alive:
                free[i], free_valid[i], free_state, free_memory = p.state, True, p.state, p.load_memory

    endpoints, starts, predicted, targets, horizon_valid = [], [], [], [], []
    for i in range(n - 1):
        target_time = episode.time[i] + horizon_s
        j = int(np.searchsorted(episode.time, target_time - 1e-9))
        if j >= n or abs(episode.time[j] - target_time) > 1e-8:
            continue  # Never relabel a shorter or longer prediction as exactly 1 s.
        good = i < len(memories)
        x = episode.state[i].copy()
        memory = memories[i] if good else None
        if good:
            for index in range(i, j):
                p = step(x, memory, index)
                flags.update(p.flags)
                if p.flags:
                    good = False
                    break
                x, memory = p.state, p.load_memory
        starts.append(i)
        endpoints.append(j)
        predicted.append(x if good else np.full(7, np.nan))
        targets.append(episode.state[j])
        horizon_valid.append(good)

    persistence = episode.state[:-1]
    rollout_summary = (_scores(np.asarray(predicted), np.asarray(targets), np.array(horizon_valid, dtype=bool))
                       if predicted else {"valid_samples": 0, "total_samples": 0, "coverage": 0., "rmse": None, "mae": None})
    if starts:
        rollout_summary["persistence"] = _scores(episode.state[starts], np.asarray(targets), np.array(horizon_valid, dtype=bool))
    report = {
        "format": "wm_recovery_nominal_evaluation_v1", "trajectory_id": episode.metadata["trajectory_id"],
        "split": episode.metadata["split"], "source_kind": episode.metadata["source_kind"],
        "source_sha256": episode.source_sha256,
        "nominal_mu_used": NOMINAL_MU, "actual_mu_for_reporting_only": episode.metadata["actual_mu"],
        "parameter_alignment": episode.metadata["parameter_alignment"],
        "max_substep_s": max_substep_s, "horizon_s": horizon_s,
        "one_step": _scores(one, episode.state[1:], one_valid),
        "persistence_same_one_step_mask": _scores(persistence, episode.state[1:], one_valid),
        "continuous": _scores(free, episode.state[1:], free_valid),
        "rolling_horizon": rollout_summary, "model_flags": sorted(flags),
        "notes": ["Nominal reference only; no World Model has been trained.",
                  "Actual mu is scenario metadata and never enters nominal prediction.",
                  "Model-domain flags stop the affected prediction stream; coverage is reported.",
                  "Exact-horizon endpoints are used without interpolation.",
                  "Numerical agreement on synthetic data is not CarSim validation."],
    }
    arrays = {"time_s": episode.time[1:], "truth": episode.state[1:], "one_step": one,
              "continuous": free, "one_step_valid": one_valid, "continuous_valid": free_valid,
              "horizon_start_indices": np.array(starts, dtype=int),
              "horizon_end_indices": np.array(endpoints, dtype=int),
              "horizon_prediction": np.asarray(predicted).reshape(-1, 7)}
    return report, arrays
