"""Deploy an exported deterministic SAC actor in the CarSim DLL loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_carsim_sac_complete import DEFAULT_DLC_PATH_FILE, DEFAULT_SIM_FILE
from carsim_wrapper import CarSimGymWrapper
from trajectory_visualization import row_from_state, save_trajectory_report


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ACTOR = BASE_DIR / "exports" / "dlc_40_best_actor.pt"


def select_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return device


def load_metadata(actor_path: Path):
    metadata_path = actor_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Actor metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run exported SAC actor with CarSim")
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--sim", default=DEFAULT_SIM_FILE)
    parser.add_argument("--path", default=DEFAULT_DLC_PATH_FILE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "artifacts" / "dlc_actor_control"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actor_path = args.actor.expanduser().resolve()
    if not actor_path.is_file():
        raise FileNotFoundError(
            f"Actor not found: {actor_path}. Run export_sac_actor.py first."
        )
    metadata = load_metadata(actor_path)
    if int(metadata.get("obs_dim", 0)) != 7 or int(metadata.get("act_dim", 0)) != 2:
        raise ValueError("The exported actor is incompatible with the current wrapper")

    device = select_device(args.device)
    actor = torch.jit.load(str(actor_path), map_location=device).eval()
    target_speed = float(metadata.get("target_speed_kmh", 40.0))
    env = CarSimGymWrapper(
        args.sim,
        target_speed=target_speed,
        path_file=args.path,
    )

    rows = []
    total_reward = 0.0
    info = {}
    try:
        observation = env.reset()
        rows.append(row_from_state(env._env.t_current, env.current_state))
        print("Actor:", actor_path)
        print("Device:", device)
        print("Target speed:", target_speed, "km/h")
        print("CarSim configuration:", env._env.config)

        for step in range(1, args.max_steps + 1):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.inference_mode():
                action = actor(observation_tensor).squeeze(0).cpu().numpy()
            if not np.all(np.isfinite(action)):
                raise RuntimeError(f"Actor returned non-finite action: {action}")
            action = np.clip(action, -1.0, 1.0)

            observation, reward, done, info = env.step(action)
            total_reward += reward
            rows.append(row_from_state(env._env.t_current, env.current_state, reward, info))

            if step == 1 or step % 20 == 0 or done:
                executed = info["executed_action"]
                print(
                    f"step={step:3d} t={env._env.t_current:6.2f}s "
                    f"station={info['station']:7.2f}m "
                    f"lat={info['lateral_error']:+6.3f}m "
                    f"speed={info['speed_kmh']:6.2f}km/h "
                    f"action=({executed[0]:+.3f},{executed[1]:+.3f}) "
                    f"progress={100.0 * info['path_progress']:5.1f}%"
                )
            if done:
                break

        summary = {
            "steps": len(rows) - 1,
            "total_reward": float(total_reward),
            "path_progress": float(info.get("path_progress", 0.0)),
            "completed": bool(info.get("completed", False)),
            "off_track": bool(info.get("off_track", False)),
        }
        csv_path, svg_path = save_trajectory_report(
            args.output,
            env,
            rows,
            summary,
        )
        print("Control summary:", summary)
        print("Trajectory CSV:", csv_path)
        print("Trajectory SVG:", svg_path)
    finally:
        env.close()
        print("CarSim environment closed safely.")


if __name__ == "__main__":
    main()
