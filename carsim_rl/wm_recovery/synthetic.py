"""Explicitly synthetic workflow fixtures; never a substitute for CarSim data."""

import json
from pathlib import Path

import numpy as np

from dynamics7dof.model import Controls, LoadMemory, Reference7DOF, STATE_NAMES

from .controls import APPLIED_INPUT_NAMES, COMMAND_NAMES
from .offline import FORMAT


def generate(directory, config, *, identifier, split, speed, seconds=1.3, actual_mu=.8):
    count = int(round(seconds / config.prediction_dt_s))
    if count < config.history_steps + config.rollout_steps + 1:
        raise ValueError("Synthetic duration is too short for the configured history and rollout")
    root = Path(directory)
    if root.exists():
        raise FileExistsError("Synthetic episode directory already exists")
    time = np.arange(count + 1) * config.prediction_dt_s
    states = [np.array([speed, 0, 0, *([speed / .353] * 4)])]
    inputs, commands, memory = [], [], LoadMemory(0, 0)
    physics = Reference7DOF()
    for t in time[:-1]:
        angle = .002 * np.sin(2 * np.pi * .6 * t)
        torques = np.full(4, 8 * np.sin(2 * np.pi * .4 * t))
        p = physics.predict(states[-1], Controls([angle, angle], torques), mu=actual_mu,
                            load_memory=memory, horizon_s=config.prediction_dt_s, substeps=config.substeps)
        if p.flags:
            raise RuntimeError(f"Synthetic fixture left nominal model domain: {p.flags}")
        state = p.state.copy()
        # Deliberate artificial mismatch for code-path testing, not a vehicle law.
        state[0] -= .02 * config.prediction_dt_s
        states.append(state)
        memory = p.load_memory
        inputs.append([angle, angle, *torques])
        commands.append([*torques, angle])
    meta = {
        "format": FORMAT, "trajectory_id": identifier, "condition_id": identifier,
        "source_kind": "synthetic", "split": split, "nominal_mu": .8, "actual_mu": actual_mu,
        "state_names": list(STATE_NAMES), "applied_input_names": list(APPLIED_INPUT_NAMES),
        "command_names": list(COMMAND_NAMES), "input_alignment": "held_over_interval_t_i_to_t_i_plus_1",
        "actuation_definition": "external_wheel_torque_without_friction_brake",
        "other_wheel_actuation_verified_zero": True, "parameter_alignment": "unverified",
        "initial_memory_source": "explicit_synthetic_initial_condition",
        "warning": "SYNTHETIC ONLY: original physics plus artificial longitudinal increment; no CarSim validation",
    }
    if split == "unknown_mu_test":
        meta["mismatch_start_s"] = 0.
    root.mkdir(parents=True, exist_ok=False)
    (root / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(root / "trajectory.npz", time_s=time, state=np.array(states),
                        applied_input=np.array(inputs), command=np.array(commands), initial_load_memory_m_s2=np.zeros(2))
    return root


def run_demo(output, config, *, epochs=1):
    from .training import train
    from .evaluation import calibrate_model, evaluate

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, split, speed, mu in (
        ("train_a", "train", 10., .8), ("train_b", "train", 11., .8),
        ("validation", "validation", 12., .8), ("test", "test", 10.5, .8),
        ("unknown_mu", "unknown_mu_test", 10.7, .4),
    ):
        paths[name] = generate(root / name, config, identifier="synthetic_" + name, split=split, speed=speed, actual_mu=mu)
    train([paths["train_a"], paths["train_b"]], root / "model", config, epochs=epochs, allow_synthetic=True)
    checkpoint = root / "model" / "world_model.pt"
    calibrate_model(checkpoint, [paths["validation"]], root / "confidence.json", allow_synthetic=True)
    for name in ("test", "unknown_mu"):
        evaluate(checkpoint, paths[name], root / (name + "_evaluation"), calibration_path=root / "confidence.json", allow_synthetic=True)
    report = {"status": "synthetic_workflow_completed", "output": str(root.resolve()),
              "claim": "Only data/training/calibration/evaluation code paths were exercised. No CarSim validation."}
    (root / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
