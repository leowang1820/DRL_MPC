"""Read-only, synthetic coast/steer example. Does not start CarSim or save files."""

import argparse
import json
import math

import numpy as np

from .model import STATE_NAMES, Controls, LoadMemory, Reference7DOF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=float, required=True, help="Explicit scenario friction coefficient")
    parser.add_argument("--speed-kmh", type=float, default=40.0)
    parser.add_argument("--front-steer-deg", type=float, default=0.0,
                        help="Actual road-wheel angle, NOT steering-wheel angle")
    parser.add_argument("--horizon", type=float, default=0.05)
    parser.add_argument("--substeps", type=int, default=100)
    args = parser.parse_args()
    if not math.isfinite(args.speed_kmh) or args.speed_kmh <= 0:
        parser.error("--speed-kmh must be finite and positive")
    model = Reference7DOF()
    vx = args.speed_kmh / 3.6
    initial = np.r_[vx, 0.0, 0.0, np.full(4, vx / model.vehicle.wheel_radius_m)]
    controls = Controls(np.full(2, math.radians(args.front_steer_deg)), np.zeros(4))
    result = model.predict(initial, controls, mu=args.mu, load_memory=LoadMemory(0.0, 0.0),
                           horizon_s=args.horizon, substeps=args.substeps)
    print("REFERENCE PORT ONLY: no CarSim comparison and no physical validation yet.")
    print("Synthetic initial state, zero net wheel torque, explicit zero initial load memory.")
    print(json.dumps({
        "mu": args.mu, "horizon_s": result.horizon_s, "substep_s": result.substep_s,
        "force_sample_time_s": result.force_sample_time_s,
        "initial_state": dict(zip(STATE_NAMES, initial.tolist())),
        "terminal_state": dict(zip(STATE_NAMES, result.state.tolist())),
        "last_force_Fx_N": result.last_evaluation.fx_tire_n.tolist(),
        "last_force_Fy_N": result.last_evaluation.fy_tire_n.tolist(),
        "last_force_Fz_N": result.last_evaluation.fz_n.tolist(),
        "quality_flags": list(result.flags),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
