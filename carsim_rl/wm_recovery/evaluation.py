"""Independent offline predictions, nominal calibration, and gate ablations."""

import json
from pathlib import Path

import numpy as np
import torch

from .confidence import ConfidenceMonitor, GateSettings, calibrate
from .dataset import load_partition, provenance
from .inspection import file_sha256
from .offline import Episode, _scores
from .world_model import WorldModel


def exclusions(records):
    return {"excluded_ids": [p["trajectory_id"] for p in records],
            "excluded_hashes": [p["sha256"] for p in records],
            "excluded_conditions": [p["condition_id"] for p in records]}


@torch.no_grad()
def one_step_metrics(model, data):
    h = model.config.history_steps
    indices = np.arange(h, len(data.episode.time) - 1)
    state = model.tensor(data.episode.state[indices])
    controls = model.tensor(data.episode.applied[indices])
    history = model.tensor(np.array([data.history_at(i, h) for i in indices]))
    memory = model.tensor(data.memory[indices])
    predictions, validity = [], []
    for k in range(model.config.ensemble_size):
        p, _, bad = model.member_step(k, state, controls, history, memory)
        predictions.append(p.cpu().numpy())
        validity.append(~bad.cpu().numpy())
    predictions = np.array(predictions)
    mean = predictions.mean(0)
    variance = predictions.var(0, ddof=1)
    valid = np.array(validity).all(0) & data.valid[indices]
    uncertainty = (variance / model.state_scale.cpu().numpy() ** 2).sum(-1)
    return {"indices": indices, "time_s": data.episode.time[indices + 1],
            "mean": mean, "members": predictions, "truth": data.episode.state[indices + 1],
            "nominal": data.nominal_next[indices], "valid": valid, "uncertainty": uncertainty,
            "error": data.episode.state[indices + 1] - mean}


def calibrate_model(checkpoint, paths, output, settings=None, *, allow_synthetic=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration output must be a new file")
    model = WorldModel.load(checkpoint)
    prepared = load_partition(paths, model.config, "validation", allow_synthetic=allow_synthetic,
                              **exclusions(model.metadata["training_trajectories"]))
    result = calibrate([one_step_metrics(model, data) for data in prepared], settings or GateSettings(),
                       model.residual_scale.cpu().numpy())
    result.update({"format": "wm_recovery_confidence_v1", "checkpoint_sha256": file_sha256(checkpoint),
                   "calibration_trajectories": provenance(prepared),
                   "synthetic_demo_only": model.metadata["synthetic_demo_only"] or any(p.episode.metadata["source_kind"] == "synthetic" for p in prepared)})
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return result


@torch.no_grad()
def rollout_metrics(model, data, horizon_s, stride):
    length = horizon_s / model.config.prediction_dt_s
    if not np.isfinite(length) or length < 1 or not np.isclose(length, round(length), rtol=0, atol=1e-8) or stride < 1:
        raise ValueError("Horizon must be a positive multiple of prediction dt, and stride positive")
    length = int(round(length))
    starts = np.arange(model.config.history_steps, len(data.episode.time) - length, stride, dtype=int)
    if not len(starts):
        return {"horizon_s": horizon_s, "windows": 0, "status": "insufficient_duration"}, {}
    x = model.tensor(data.episode.state[starts])
    controls = model.tensor(np.array([data.episode.applied[i:i + length] for i in starts]))
    history = model.tensor(np.array([data.history_at(i, model.config.history_steps) for i in starts]))
    memory = model.tensor(data.memory[starts])
    predicted, masks = [], []
    for k in range(model.config.ensemble_size):
        values, bad = model.member_rollout(k, x, controls, history, memory)
        predicted.append(values[:, -1].cpu().numpy())
        masks.append(~bad.cpu().numpy())
    nominal, nominal_good = x.clone(), data.valid[starts].copy()
    for j in range(length):
        nominal, memory, bad = model.nominal(nominal, controls[:, j], memory)
        nominal_good &= ~bad.cpu().numpy()
    mean = np.mean(predicted, axis=0)
    good = np.array(masks).all(0) & data.valid[starts]
    common = good & nominal_good
    truth = data.episode.state[starts + length]
    report = {"horizon_s": horizon_s, "windows": len(starts),
              "world_model": _scores(mean, truth, good), "nominal": _scores(nominal.cpu().numpy(), truth, nominal_good),
              "world_model_common_mask": _scores(mean, truth, common),
              "nominal_common_mask": _scores(nominal.cpu().numpy(), truth, common),
              "persistence_common_mask": _scores(data.episode.state[starts], truth, common)}
    return report, {"horizon_starts": starts, "horizon_truth": truth, "horizon_mean": mean,
                    "horizon_members": np.array(predicted), "horizon_valid": good}


@torch.no_grad()
def continuous_metrics(model, data):
    start = model.config.history_steps
    state = model.tensor(data.episode.state[start])
    inputs = model.tensor(data.episode.applied[start:])
    history = model.tensor(data.history_at(start, start))
    memory = model.tensor(data.memory[start])
    member_predictions, member_valid = [], []
    for k in range(model.config.ensemble_size):
        x, m, past = state.clone(), memory.clone(), history.clone()
        values, mask, alive = [], [], True
        for applied in inputs:
            if alive:
                p, m, bad = model.member_step(k, x, applied, past, m)
                alive = not bool(bad)
                past = model.advance_history(past, x, applied)
                x = p
            values.append(x.cpu().numpy() if alive else np.full(7, np.nan))
            mask.append(alive)
        member_predictions.append(values)
        member_valid.append(mask)
    values = np.array(member_predictions)
    good = np.array(member_valid).all(0) & data.valid[start]
    mean = values.mean(0)
    nominal, nominal_valid, alive = [], [], True
    x, m = state.clone(), memory.clone()
    for applied in inputs:
        if alive:
            x, m, bad = model.nominal(x, applied, m)
            alive = not bool(bad)
        nominal.append(x.cpu().numpy() if alive else np.full(7, np.nan))
        nominal_valid.append(alive)
    nominal, nominal_valid = np.array(nominal), np.array(nominal_valid) & data.valid[start]
    truth, common = data.episode.state[start + 1:], good & nominal_valid
    report = {"world_model": _scores(mean, truth, good), "nominal": _scores(nominal, truth, nominal_valid),
              "world_model_common_mask": _scores(mean, truth, common), "nominal_common_mask": _scores(nominal, truth, common),
              "reset_after_initialization": False}
    return report, {"continuous_time_s": data.episode.time[start + 1:], "continuous_mean": mean,
                    "continuous_nominal": nominal, "continuous_truth": truth, "continuous_valid": good}


def uncertainty_relationship(metrics, state_scale):
    valid = metrics["valid"] & np.isfinite(metrics["uncertainty"])
    u = metrics["uncertainty"][valid]
    error = np.linalg.norm(metrics["error"][valid] / state_scale, axis=1)
    correlation = float(np.corrcoef(u, error)[0, 1]) if len(u) > 2 and np.std(u) > 1e-12 and np.std(error) > 1e-12 else None
    bins = []
    for indices in np.array_split(np.argsort(u), 4):
        if len(indices):
            bins.append({"samples": len(indices), "mean_uncertainty": float(np.mean(u[indices])),
                         "mean_normalized_prediction_error": float(np.mean(error[indices]))})
    return {"pearson_correlation": correlation, "increasing_uncertainty_bins": bins,
            "interpretation": "descriptive association; low disagreement does not guarantee low error"}


def gate_metrics(metrics, calibration, metadata):
    result = {}
    for mode in ("ensemble_only", "residual_only", "joint"):
        monitor = ConfidenceMonitor(calibration, mode)
        records = [monitor.observe(u, p, y, prediction_valid=bool(v))
                   for u, p, y, v in zip(metrics["uncertainty"], metrics["mean"], metrics["truth"], metrics["valid"])]
        flags = np.array([r["state"] == "untrusted" for r in records])
        ready = np.array(["window_uncertainty" in r for r in records])
        summary = {"untrusted_samples": int(flags.sum()), "unavailable_window_samples": int(np.count_nonzero(~ready)),
                   "untrusted_fraction_after_window": float(flags[ready].mean()) if ready.any() else None}
        onset = metadata.get("mismatch_start_s")
        if onset is not None:
            times = metrics["time_s"]
            triggered = times[flags & (times >= onset)]
            summary["detection_delay_s"] = float(triggered[0] - onset) if len(triggered) else None
            summary["detected_within_observed_episode"] = bool(len(triggered))
            before = ready & (times < onset)
            summary["pre_mismatch_alarm_fraction"] = float(flags[before].mean()) if before.any() else None
        if metadata["split"] == "test":
            summary["nominal_alarm_fraction"] = summary["untrusted_fraction_after_window"]
        result[mode] = summary
    return result


def evaluate(checkpoint, episode_path, output, *, calibration_path=None, horizon_s=1., stride=10,
             allow_synthetic=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    model = WorldModel.load(checkpoint)
    calibration, excluded = None, list(model.metadata["training_trajectories"])
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if calibration.get("format") != "wm_recovery_confidence_v1" or calibration["checkpoint_sha256"] != file_sha256(checkpoint):
            raise ValueError("Calibration belongs to a different World Model")
        excluded += calibration["calibration_trajectories"]
    split = Episode.load(episode_path).metadata["split"]
    if split not in ("test", "unknown_mu_test"):
        raise ValueError("Final evaluation accepts only independent test or unknown_mu_test trajectories")
    data = load_partition([episode_path], model.config, split, allow_synthetic=allow_synthetic,
                          **exclusions(excluded))[0]
    metrics = one_step_metrics(model, data)
    rollout, rollout_arrays = rollout_metrics(model, data, horizon_s, stride)
    continuous, continuous_arrays = continuous_metrics(model, data)
    valid = metrics["valid"]
    report = {"format": "wm_recovery_evaluation_v1", "source": provenance([data])[0],
              "checkpoint_sha256": file_sha256(checkpoint),
              "synthetic_demo_only": model.metadata["synthetic_demo_only"] or data.episode.metadata["source_kind"] == "synthetic",
              "nominal_mu_used": .8, "actual_mu_metadata_only": data.episode.metadata["actual_mu"],
              "one_step": {"world_model": _scores(metrics["mean"], metrics["truth"], valid),
                           "nominal_common_mask": _scores(metrics["nominal"], metrics["truth"], valid),
                           "persistence_common_mask": _scores(data.episode.state[metrics["indices"]], metrics["truth"], valid)},
              "rolling_horizon": rollout, "continuous": continuous,
              "uncertainty_vs_error": uncertainty_relationship(metrics, model.state_scale.cpu().numpy()),
              "confidence": gate_metrics(metrics, calibration, data.episode.metadata) if calibration else None,
              "notes": ["Risk/SE/recovery labels are not part of this first-stage evaluation.",
                        "No real CarSim accuracy or unknown-friction detection claim follows from synthetic data.",
                        "Residual gate uses the next observed state and is available at that observation time."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **metrics, **rollout_arrays, **continuous_arrays)
    from .report import write_report
    write_report(output, report, metrics, continuous_arrays, model.residual_scale.cpu().numpy(), calibration)
    return report
