"""Production training entry point for SAC + CarSim co-simulation."""

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from SACModule import SACAgent, ReplayBuffer
from carsim_wrapper import (
    DEFAULT_PATH_FILE,
    DEFAULT_TARGET_SPEED_KMH,
    OBS_DIM,
    WARMUP_HOLD_STEPS,
    WARMUP_LONGITUDINAL_LIMIT,
    WARMUP_STEER_LIMIT,
    CarSimGymWrapper,
)
from trajectory_visualization import (
    record_deterministic_episode,
    save_trajectory_report,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DLC_PATH_FILE = str(BASE_DIR / "paths" / "dlc_reference_path.par")
DEFAULT_SIM_FILE = (
    r"E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim"
)


def resolve_reference_path(args) -> str:
    if getattr(args, "path", ""):
        return str(Path(args.path).expanduser().resolve())
    if getattr(args, "scenario", "uturn") == "dlc":
        return DEFAULT_DLC_PATH_FILE
    return DEFAULT_PATH_FILE


def select_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_agent(args, device: torch.device, checkpoint=None) -> SACAgent:
    config = checkpoint.get("config", {}) if checkpoint else {}
    hidden_size = int(config.get("hidden_size", args.hidden_size))
    obs_dim = int(config.get("obs_dim", OBS_DIM))
    act_dim = int(config.get("act_dim", 2))

    if obs_dim != OBS_DIM or act_dim != 2:
        raise ValueError(
            f"Checkpoint dimensions ({obs_dim}, {act_dim}) do not match "
            f"the current environment ({OBS_DIM}, 2)."
        )

    return SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_scale=np.ones(act_dim, dtype=np.float32),
        device=device,
        gamma=args.gamma,
        alpha=args.alpha,
        auto_alpha=True,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        hidden_size=hidden_size,
        tau=args.tau,
    )


def agent_checkpoint_state(agent: SACAgent) -> dict:
    state = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "q1_state_dict": agent.q1_net.state_dict(),
        "q2_state_dict": agent.q2_net.state_dict(),
        "q1_target_state_dict": agent.q1_target_net.state_dict(),
        "q2_target_state_dict": agent.q2_target_net.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "q1_optimizer": agent.q1_optimizer.state_dict(),
        "q2_optimizer": agent.q2_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }
    if agent.alpha_optimizer is not None:
        state["alpha_optimizer"] = agent.alpha_optimizer.state_dict()
    return state


def load_agent_state(agent: SACAgent, checkpoint: dict) -> None:
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.q1_net.load_state_dict(checkpoint["q1_state_dict"])
    agent.q2_net.load_state_dict(checkpoint["q2_state_dict"])
    agent.q1_target_net.load_state_dict(checkpoint["q1_target_state_dict"])
    agent.q2_target_net.load_state_dict(checkpoint["q2_target_state_dict"])

    with torch.no_grad():
        agent.log_alpha.copy_(
            checkpoint["log_alpha"].to(agent.device)
        )

    if "actor_optimizer" in checkpoint:
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        agent.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        if agent.alpha_optimizer is not None and "alpha_optimizer" in checkpoint:
            agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


def pack_replay(buffer: ReplayBuffer, limit: int):
    if limit <= 0 or len(buffer) == 0:
        return None

    transitions = list(buffer.buffer)[-limit:]
    states, actions, rewards, next_states, dones = zip(*transitions)
    return {
        "states": torch.from_numpy(np.stack(states).astype(np.float32)),
        "actions": torch.from_numpy(np.stack(actions).astype(np.float32)),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_states": torch.from_numpy(np.stack(next_states).astype(np.float32)),
        "dones": torch.tensor(dones, dtype=torch.bool),
    }


def restore_replay(buffer: ReplayBuffer, replay_state) -> int:
    if not replay_state:
        return 0

    states = replay_state["states"].cpu().numpy()
    actions = replay_state["actions"].cpu().numpy()
    rewards = replay_state["rewards"].cpu().numpy()
    next_states = replay_state["next_states"].cpu().numpy()
    dones = replay_state["dones"].cpu().numpy()

    for transition in zip(states, actions, rewards, next_states, dones):
        state, action, reward, next_state, done = transition
        buffer.push(state, action, float(reward), next_state, bool(done))
    return len(states)


def save_checkpoint(
    path: Path,
    agent: SACAgent,
    buffer: ReplayBuffer,
    args,
    episode: int,
    global_step: int,
    best_eval_reward: float,
    replay_limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = agent_checkpoint_state(agent)
    checkpoint.update(
        {
            "format_version": 1,
            "episode": int(episode),
            "global_step": int(global_step),
            "best_eval_reward": float(best_eval_reward),
            "config": {
                **vars(args),
                "obs_dim": OBS_DIM,
                "act_dim": 2,
            },
            "replay": pack_replay(buffer, replay_limit),
        }
    )

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint_file(path: str, device: torch.device) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def sample_warmup_action() -> np.ndarray:
    return np.array(
        [
            np.random.uniform(
                -WARMUP_LONGITUDINAL_LIMIT,
                WARMUP_LONGITUDINAL_LIMIT,
            ),
            np.random.uniform(
                -WARMUP_STEER_LIMIT,
                WARMUP_STEER_LIMIT,
            ),
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def evaluate(env, agent, episodes: int, max_steps: int):
    rewards = []
    lengths = []
    path_progresses = []
    off_track_count = 0
    completion_count = 0

    for _ in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        info = {}

        for step in range(1, max_steps + 1):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        lengths.append(step)
        path_progresses.append(float(info.get("path_progress", 0.0)))
        off_track_count += int(bool(info.get("off_track", False)))
        completion_count += int(bool(info.get("completed", False)))

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_path_progress": float(np.mean(path_progresses)),
        "off_track_rate": off_track_count / max(episodes, 1),
        "completion_rate": completion_count / max(episodes, 1),
    }


def train(args) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = (
        load_checkpoint_file(args.resume, device)
        if args.resume
        else None
    )
    agent = make_agent(args, device, checkpoint)
    buffer = ReplayBuffer(args.replay_size)

    start_episode = 1
    global_step = 0
    best_eval_reward = -float("inf")

    if checkpoint:
        load_agent_state(agent, checkpoint)
        start_episode = int(checkpoint.get("episode", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_eval_reward = float(
            checkpoint.get("best_eval_reward", -float("inf"))
        )
        restored = restore_replay(buffer, checkpoint.get("replay"))
        print(
            f"Resumed episode={start_episode}, global_step={global_step}, "
            f"replay={restored}"
        )

    if start_episode > args.episodes:
        raise ValueError(
            f"Checkpoint is already at episode {start_episode - 1}; "
            f"--episodes must be at least {start_episode}."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = BASE_DIR / "saves" / f"carsim_sac_complete_{args.name}"
    log_dir = BASE_DIR / "runs" / f"complete_{args.name}_{run_stamp}"
    writer = SummaryWriter(log_dir=str(log_dir))
    env = CarSimGymWrapper(
        args.sim,
        target_speed=args.target_speed,
        path_file=args.path,
    )

    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Save directory:", save_dir)
    print("TensorBoard directory:", log_dir)
    print(
        f"Episodes {start_episode}..{args.episodes}, max_steps={args.max_steps}, "
        f"warmup_steps={args.warmup_steps}, learning_starts={args.learning_starts}"
    )

    last_completed_episode = start_episode - 1

    try:
        for episode in range(start_episode, args.episodes + 1):
            obs = env.reset()
            episode_reward = 0.0
            info = {}
            warmup_action = np.zeros(2, dtype=np.float32)
            episode_start = time.perf_counter()
            latest_losses = None

            for episode_step in range(1, args.max_steps + 1):
                global_step += 1
                if global_step <= args.warmup_steps:
                    if (episode_step - 1) % WARMUP_HOLD_STEPS == 0:
                        warmup_action = sample_warmup_action()
                    action = warmup_action.copy()
                else:
                    action = agent.select_action(obs, deterministic=False)

                next_obs, reward, done, info = env.step(action)
                truncated = episode_step >= args.max_steps and not done
                buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done or truncated,
                )
                obs = next_obs
                episode_reward += reward

                if (
                    global_step >= args.learning_starts
                    and len(buffer) >= args.batch_size
                ):
                    for _ in range(args.updates_per_step):
                        latest_losses = agent.update(
                            buffer.sample(args.batch_size)
                        )
                        for key, value in latest_losses.items():
                            writer.add_scalar(f"loss/{key}", value, global_step)

                if done:
                    break

            wall_time = time.perf_counter() - episode_start
            sim_time = float(env._env.t_current)
            off_track = bool(info.get("off_track", False))
            completed = bool(info.get("completed", False))
            last_completed_episode = episode

            writer.add_scalar("train/episode_reward", episode_reward, episode)
            writer.add_scalar("train/episode_length", episode_step, episode)
            writer.add_scalar("train/sim_time", sim_time, episode)
            writer.add_scalar("train/off_track", int(off_track), episode)
            writer.add_scalar("train/completed", int(completed), episode)
            writer.add_scalar("train/replay_size", len(buffer), episode)
            writer.add_scalar("train/final_speed", info.get("speed_kmh", 0.0), episode)
            writer.add_scalar("train/final_station", info.get("station", 0.0), episode)
            writer.add_scalar(
                "train/path_progress",
                info.get("path_progress", 0.0),
                episode,
            )
            writer.add_scalar(
                "train/final_lateral_error",
                info.get("lateral_error", 0.0),
                episode,
            )

            loss_text = ""
            if latest_losses:
                loss_text = (
                    f" q={latest_losses['q1_loss']:.3f}/"
                    f"{latest_losses['q2_loss']:.3f}"
                )
            print(
                f"Episode {episode:4d}/{args.episodes} | reward={episode_reward:8.2f} "
                f"len={episode_step:4d} sim={sim_time:6.2f}s wall={wall_time:6.2f}s "
                f"progress={100.0 * info.get('path_progress', 0.0):5.1f}% "
                f"off={int(off_track)} complete={int(completed)} buffer={len(buffer)}"
                f"{loss_text}"
            )

            if episode % args.eval_interval == 0:
                metrics = evaluate(
                    env,
                    agent,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                )
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{key}", value, episode)
                print(
                    f"  Eval | reward={metrics['mean_reward']:.2f} "
                    f"len={metrics['mean_length']:.1f} "
                    f"progress={100.0 * metrics['mean_path_progress']:.1f}% "
                    f"off={metrics['off_track_rate']:.2f} "
                    f"complete={metrics['completion_rate']:.2f}"
                )

                if metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = metrics["mean_reward"]
                    save_checkpoint(
                        save_dir / "best.pth",
                        agent,
                        buffer,
                        args,
                        episode,
                        global_step,
                        best_eval_reward,
                        replay_limit=0,
                    )
                    print("  Saved new best checkpoint.")

            if episode % args.checkpoint_interval == 0:
                save_checkpoint(
                    save_dir / "latest.pth",
                    agent,
                    buffer,
                    args,
                    episode,
                    global_step,
                    best_eval_reward,
                    replay_limit=args.checkpoint_replay_size,
                )

            writer.flush()

        save_checkpoint(
            save_dir / "final.pth",
            agent,
            buffer,
            args,
            last_completed_episode,
            global_step,
            best_eval_reward,
            replay_limit=args.checkpoint_replay_size,
        )
        print("Training completed. Final checkpoint:", save_dir / "final.pth")

    except KeyboardInterrupt:
        save_checkpoint(
            save_dir / "interrupted.pth",
            agent,
            buffer,
            args,
            last_completed_episode,
            global_step,
            best_eval_reward,
            replay_limit=args.checkpoint_replay_size,
        )
        print("Training interrupted safely; checkpoint saved.")
    except Exception:
        save_checkpoint(
            save_dir / "crash_recovery.pth",
            agent,
            buffer,
            args,
            last_completed_episode,
            global_step,
            best_eval_reward,
            replay_limit=args.checkpoint_replay_size,
        )
        print("Training failed; crash-recovery checkpoint saved.")
        raise
    finally:
        env.close()
        writer.close()
        print("CarSim environment and TensorBoard writer closed safely.")


def evaluate_only(args) -> None:
    if not args.model:
        raise ValueError("--mode eval requires --model CHECKPOINT.pth")

    device = select_device(args.device)
    checkpoint = load_checkpoint_file(args.model, device)
    agent = make_agent(args, device, checkpoint)
    load_agent_state(agent, checkpoint)
    env = CarSimGymWrapper(
        args.sim,
        target_speed=args.target_speed,
        path_file=args.path,
    )
    try:
        metrics = evaluate(env, agent, args.eval_episodes, args.max_steps)
        print("Evaluation:", metrics)
        if args.trajectory_output:
            rows, summary = record_deterministic_episode(
                env,
                agent,
                args.max_steps,
            )
            csv_path, svg_path = save_trajectory_report(
                args.trajectory_output,
                env,
                rows,
                summary,
            )
            print("Trajectory summary:", summary)
            print("Trajectory CSV:", csv_path)
            print("Trajectory SVG:", svg_path)
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete SAC training workflow for CarSim path tracking."
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--sim", default=DEFAULT_SIM_FILE)
    parser.add_argument("--scenario", choices=["uturn", "dlc"], default="uturn")
    parser.add_argument(
        "--path",
        default="",
        help="Override the reference path selected by --scenario",
    )
    parser.add_argument("--name", default="uturn_40kmh")
    parser.add_argument("--resume", default="", help="Resume a full training checkpoint")
    parser.add_argument("--model", default="", help="Checkpoint used by eval mode")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED_KMH)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--lr-alpha", type=float, default=5e-5)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-replay-size", type=int, default=50_000)
    parser.add_argument(
        "--trajectory-output",
        default="",
        help="Eval mode: output prefix for trajectory CSV and SVG",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.path = resolve_reference_path(arguments)
    if arguments.mode == "train":
        train(arguments)
    else:
        evaluate_only(arguments)
