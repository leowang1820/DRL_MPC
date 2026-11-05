"""Run from carsim_rl: python -m wm_recovery --help."""

import argparse
import json
from pathlib import Path

import numpy as np

from .inspection import inspect_simfile
from .offline import Episode, audit_partition, evaluate_nominal


def config_arguments(parser):
    parser.add_argument("--prediction-dt-s", type=float, default=.01)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--physics-substep-s", type=float, default=.0005)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)


def model_config(args):
    from .config import WorldModelConfig
    return WorldModelConfig(**{key: getattr(args, key) for key in (
        "prediction_dt_s", "history_steps", "physics_substep_s", "hidden_width",
        "rollout_steps", "rollout_loss_weight", "seed")})


def main():
    parser = argparse.ArgumentParser(description="First-stage distributed-drive interface and offline checks")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read a target simfile; never starts CarSim")
    inspect.add_argument("--sim", required=True, type=Path)
    inspect.add_argument("--output", type=Path, help="Optional new JSON file")
    check = commands.add_parser("check-data", help="Check complete trajectories and dataset separation")
    check.add_argument("episodes", nargs="+", type=Path)
    evaluate = commands.add_parser("evaluate-nominal", help="Offline 7DOF baseline, with mu fixed at 0.8")
    evaluate.add_argument("--episode", required=True, type=Path)
    evaluate.add_argument("--max-substep-s", required=True, type=float)
    evaluate.add_argument("--horizon-s", type=float, default=1.0)
    evaluate.add_argument("--output", required=True, type=Path)
    training = commands.add_parser("train", help="Train five history MLPs using complete-trajectory bootstrap")
    training.add_argument("--episodes", nargs="+", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=.001)
    training.add_argument("--device", default="cpu")
    training.add_argument("--allow-synthetic", action="store_true")
    config_arguments(training)
    calibration = commands.add_parser("calibrate", help="Calibrate confidence thresholds on separate nominal validation trajectories")
    calibration.add_argument("--checkpoint", required=True, type=Path)
    calibration.add_argument("--episodes", nargs="+", required=True, type=Path)
    calibration.add_argument("--output", required=True, type=Path)
    calibration.add_argument("--window-steps", type=int, default=10)
    calibration.add_argument("--exceed-steps", type=int, default=3)
    calibration.add_argument("--recover-steps", type=int, default=10)
    calibration.add_argument("--quantile", type=float, default=.99)
    calibration.add_argument("--allow-synthetic", action="store_true")
    wm_eval = commands.add_parser("evaluate", help="Independent one-step, 1 s, continuous, and uncertainty evaluation")
    wm_eval.add_argument("--checkpoint", required=True, type=Path)
    wm_eval.add_argument("--episode", required=True, type=Path)
    wm_eval.add_argument("--calibration", type=Path)
    wm_eval.add_argument("--output", required=True, type=Path)
    wm_eval.add_argument("--horizon-s", type=float, default=1.)
    wm_eval.add_argument("--stride", type=int, default=10)
    wm_eval.add_argument("--allow-synthetic", action="store_true")
    demo = commands.add_parser("demo", help="Run the entire workflow on explicitly SYNTHETIC data; never starts CarSim")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--epochs", type=int, default=1)
    config_arguments(demo)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_simfile(args.sim)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    elif args.command == "check-data":
        result = audit_partition([Episode.load(path) for path in args.episodes])
    elif args.command == "evaluate-nominal":
        if args.output.exists():
            parser.error("Output directory already exists; choose a new directory")
        result, arrays = evaluate_nominal(Episode.load(args.episode), max_substep_s=args.max_substep_s,
                                          horizon_s=args.horizon_s)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(args.output / "predictions.npz", **arrays)
    else:
        import torch
        torch.set_num_threads(1)
        if args.command == "train":
            from .training import train
            result = train(args.episodes, args.output, model_config(args), epochs=args.epochs,
                           batch_size=args.batch_size, learning_rate=args.learning_rate,
                           device=args.device, allow_synthetic=args.allow_synthetic)
        elif args.command == "calibrate":
            from .confidence import GateSettings
            from .evaluation import calibrate_model
            settings = GateSettings(args.window_steps, args.exceed_steps, args.recover_steps, args.quantile)
            result = calibrate_model(args.checkpoint, args.episodes, args.output, settings,
                                     allow_synthetic=args.allow_synthetic)
        elif args.command == "evaluate":
            from .evaluation import evaluate as evaluate_wm
            result = evaluate_wm(args.checkpoint, args.episode, args.output, calibration_path=args.calibration,
                                 horizon_s=args.horizon_s, stride=args.stride, allow_synthetic=args.allow_synthetic)
        else:
            from .synthetic import run_demo
            result = run_demo(args.output, model_config(args), epochs=args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
