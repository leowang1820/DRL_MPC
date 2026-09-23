"""Run a pure-pursuit DLC baseline before reinforcement-learning training."""

import argparse
from pathlib import Path

import numpy as np

from train_carsim_sac_complete import DEFAULT_DLC_PATH_FILE, DEFAULT_SIM_FILE
from carsim_wrapper import CarSimGymWrapper
from trajectory_visualization import row_from_state, save_trajectory_report


BASE_DIR = Path(__file__).resolve().parent


def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def pure_pursuit_action(env, target_speed, lookahead, wheelbase, steering_ratio):
    state = env.current_state
    target_station = min(state["station"] + lookahead, env._path.path_length)
    target_x = float(np.interp(target_station, env._path.s, env._path.x))
    target_y = float(np.interp(target_station, env._path.s, env._path.y))

    bearing_deg = np.degrees(
        np.arctan2(target_y - state["y"], target_x - state["x"])
    )
    alpha = np.radians(wrap_angle_deg(bearing_deg - state["yaw_deg"]))
    road_wheel_rad = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead)
    steering_wheel_deg = np.degrees(road_wheel_rad) * steering_ratio
    steer_action = float(np.clip(steering_wheel_deg / 90.0, -1.0, 1.0))

    speed_error = target_speed - state["speed"]
    longitudinal_action = float(np.clip(0.12 + 0.03 * speed_error, -0.3, 0.5))
    return np.array([longitudinal_action, steer_action], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="CarSim DLC pure-pursuit baseline")
    parser.add_argument("--sim", default=DEFAULT_SIM_FILE)
    parser.add_argument("--path", default=DEFAULT_DLC_PATH_FILE)
    parser.add_argument("--target-speed", type=float, default=40.0)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--steering-ratio", type=float, default=16.0)
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "artifacts" / "dlc_baseline"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = CarSimGymWrapper(
        args.sim,
        target_speed=args.target_speed,
        path_file=args.path,
    )
    rows = []
    total_reward = 0.0
    info = {}
    try:
        env.reset()
        rows.append(row_from_state(env._env.t_current, env.current_state))
        for step in range(1, args.max_steps + 1):
            action = pure_pursuit_action(
                env,
                args.target_speed,
                args.lookahead,
                args.wheelbase,
                args.steering_ratio,
            )
            _, reward, done, info = env.step(action)
            total_reward += reward
            rows.append(row_from_state(env._env.t_current, env.current_state, reward, info))
            if step % 20 == 0 or done:
                print(
                    f"step={step:3d} station={info['station']:7.2f} m "
                    f"lat={info['lateral_error']:+6.3f} m "
                    f"speed={info['speed_kmh']:6.2f} km/h "
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
        print("Summary:", summary)
        print("Trajectory CSV:", csv_path)
        print("Trajectory SVG:", svg_path)
    finally:
        env.close()
        print("CarSim environment closed safely.")


if __name__ == "__main__":
    main()
