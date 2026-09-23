"""Record a deterministic CarSim episode and create dependency-free reports."""

import csv
from pathlib import Path

import numpy as np


def row_from_state(time_s, state, reward=0.0, info=None):
    info = info or {}
    desired = info.get("desired_action", (0.0, 0.0))
    executed = info.get("executed_action", (0.0, 0.0))
    controls = info.get("applied_controls", (0.0, 0.0, 0.0))
    return {
        "time_s": float(time_s),
        "x_m": float(state["x"]),
        "y_m": float(state["y"]),
        "yaw_deg": float(state["yaw_deg"]),
        "station_m": float(state["station"]),
        "lateral_error_m": float(state["lateral_error"]),
        "heading_error_deg": float(state["heading_error_deg"]),
        "speed_kmh": float(state["speed"]),
        "reward": float(reward),
        "desired_lon": float(desired[0]),
        "desired_steer": float(desired[1]),
        "executed_lon": float(executed[0]),
        "executed_steer": float(executed[1]),
        "throttle": float(controls[0]),
        "brake_mpa": float(controls[1]),
        "steer_wheel_deg": float(controls[2]),
        "completed": int(bool(info.get("completed", False))),
        "off_track": int(bool(info.get("off_track", False))),
    }


def record_deterministic_episode(env, agent, max_steps: int):
    obs = env.reset()
    initial_state = env._extract_state(env._env.export_vars)
    rows = [row_from_state(env._env.t_current, initial_state)]
    total_reward = 0.0
    final_info = {}

    for _ in range(max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        state = {
            "x": info["x"],
            "y": info["y"],
            "yaw_deg": info["yaw_deg"],
            "station": info["station"],
            "lateral_error": info["lateral_error"],
            "heading_error_deg": info["heading_error_deg"],
            "speed": info["speed_kmh"],
        }
        rows.append(row_from_state(env._env.t_current, state, reward, info))
        total_reward += reward
        final_info = info
        if done:
            break

    summary = {
        "steps": len(rows) - 1,
        "total_reward": float(total_reward),
        "path_progress": float(final_info.get("path_progress", 0.0)),
        "completed": bool(final_info.get("completed", False)),
        "off_track": bool(final_info.get("off_track", False)),
    }
    return rows, summary


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(xs, ys, map_x, map_y):
    return " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(xs, ys))


def write_svg(path: Path, reference_path, rows, summary) -> None:
    width, height = 1200, 760
    left, right = 85, 35
    top1, bottom1 = 55, 390
    top2, bottom2 = 470, 700

    ref_x = np.asarray(reference_path.x, dtype=float)
    ref_y = np.asarray(reference_path.y, dtype=float)
    actual_x = np.asarray([row["x_m"] for row in rows], dtype=float)
    actual_y = np.asarray([row["y_m"] for row in rows], dtype=float)
    station = np.asarray([row["station_m"] for row in rows], dtype=float)
    lateral = np.asarray([row["lateral_error_m"] for row in rows], dtype=float)

    xmin = min(float(ref_x.min()), float(actual_x.min()))
    xmax = max(float(ref_x.max()), float(actual_x.max()), xmin + 1.0)
    ymin = min(float(ref_y.min()), float(actual_y.min()), -1.0)
    ymax = max(float(ref_y.max()), float(actual_y.max()), 4.5)
    ypad = max(0.5, 0.08 * (ymax - ymin))
    ymin -= ypad
    ymax += ypad

    def map_x(value):
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def map_y(value):
        return bottom1 - (value - ymin) / (ymax - ymin) * (bottom1 - top1)

    smax = max(float(reference_path.path_length), float(station.max()), 1.0)
    emax = max(1.0, float(np.max(np.abs(lateral))) * 1.15)

    def map_s(value):
        return left + value / smax * (width - left - right)

    def map_error(value):
        return (top2 + bottom2) / 2.0 - value / emax * (bottom2 - top2) / 2.0

    ref_points = _polyline(ref_x, ref_y, map_x, map_y)
    actual_points = _polyline(actual_x, actual_y, map_x, map_y)
    error_points = _polyline(station, lateral, map_s, map_error)
    status = "COMPLETED" if summary["completed"] else (
        "OFF TRACK" if summary["off_track"] else "TRUNCATED"
    )
    status_color = "#2ca02c" if summary["completed"] else "#d62728"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22">CarSim DLC Tracking Evaluation</text>
<text x="{left}" y="49" font-family="Arial" font-size="13">Blue: reference | Red: vehicle trajectory | Y axis magnified for readability</text>
<rect x="{left}" y="{top1}" width="{width-left-right}" height="{bottom1-top1}" fill="#fafafa" stroke="#444"/>
<line x1="{left}" y1="{map_y(0):.2f}" x2="{width-right}" y2="{map_y(0):.2f}" stroke="#bbb" stroke-dasharray="6,6"/>
<line x1="{left}" y1="{map_y(3.5):.2f}" x2="{width-right}" y2="{map_y(3.5):.2f}" stroke="#bbb" stroke-dasharray="6,6"/>
<polyline points="{ref_points}" fill="none" stroke="#1261a0" stroke-width="4" stroke-dasharray="10,5"/>
<polyline points="{actual_points}" fill="none" stroke="#d62728" stroke-width="3"/>
<circle cx="{map_x(actual_x[0]):.2f}" cy="{map_y(actual_y[0]):.2f}" r="6" fill="#2ca02c"/>
<circle cx="{map_x(actual_x[-1]):.2f}" cy="{map_y(actual_y[-1]):.2f}" r="6" fill="{status_color}"/>
<text x="{width/2}" y="425" text-anchor="middle" font-family="Arial">Global X / m</text>
<text x="22" y="{(top1+bottom1)/2}" text-anchor="middle" transform="rotate(-90 22 {(top1+bottom1)/2})" font-family="Arial">Global Y / m</text>
<text x="{left}" y="455" font-family="Arial" font-size="17">Lateral tracking error</text>
<rect x="{left}" y="{top2}" width="{width-left-right}" height="{bottom2-top2}" fill="#fafafa" stroke="#444"/>
<line x1="{left}" y1="{map_error(0):.2f}" x2="{width-right}" y2="{map_error(0):.2f}" stroke="#777"/>
<polyline points="{error_points}" fill="none" stroke="#9467bd" stroke-width="3"/>
<text x="{width/2}" y="738" text-anchor="middle" font-family="Arial">Station / m</text>
<text x="22" y="{(top2+bottom2)/2}" text-anchor="middle" transform="rotate(-90 22 {(top2+bottom2)/2})" font-family="Arial">Lateral error / m</text>
<text x="{width-right-360}" y="49" font-family="Arial" font-weight="bold" fill="{status_color}">{status}; progress={100*summary['path_progress']:.1f}%; reward={summary['total_reward']:.1f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_trajectory_report(output_prefix, env, rows, summary):
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    svg_path = prefix.with_suffix(".svg")
    write_csv(csv_path, rows)
    write_svg(svg_path, env._path, rows, summary)
    return csv_path, svg_path
