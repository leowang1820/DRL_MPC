"""Collect one CarSim episode at solver-step resolution using the existing DLC controller.

This is raw signal capture, NOT residual-label generation or 7DOF validation.
Run `python -m dynamics7dof.collect --check-config-only` without loading the DLL.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .model import VehicleParameters
from .signals import EXPORT_NAMES, NATIVE_UNITS, SI_NAMES, decode_exports, inspect_configuration
from .timing import ObservedSolver, SolverClock, TIME_SOURCE


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIM = r"E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim"
META_COLUMNS = (
    "episode_id", "sample_index", "control_index", "phase",
    "wrapper_time_s", "api_time_argument_s", "solver_step_s", "return_code",
    "solver_done", "throttle_command", "brake_pressure_command_mpa",
    "steering_wheel_command_deg", "quality_flags",
    "solver_time_before_s", "solver_time_s", "solver_time_source",
)
PENDING_CHECKS = [
    "solver_state_force_time_alignment_unverified",
    "wheel_torque_sign_and_net_torque_definition_unverified",
    "carsim_acceleration_to_load_memory_mapping_unverified",
]


class CaptureWriter:
    """Stream raw and unit-converted rows; refuse to overwrite an existing run."""

    def __init__(self, output, manifest):
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest = manifest
        self.count = 0
        self.flagged_rows = 0
        self.previous_time = None
        self.min_dt = None
        self.max_dt = None
        self.first_si = None
        self.last_si = None
        self.si_min = None
        self.si_max = None
        self.clock_rows = 0
        self.first_solver_time = None
        self.last_solver_time = None
        self.raw_file = None
        self.si_file = None
        try:
            self.raw_file = (self.output / "raw_exports.csv").open("x", newline="", encoding="utf-8-sig")
            self.si_file = (self.output / "signals_si.csv").open("x", newline="", encoding="utf-8-sig")
            self.raw_writer = csv.DictWriter(self.raw_file, fieldnames=(*META_COLUMNS, *EXPORT_NAMES))
            self.si_writer = csv.DictWriter(self.si_file, fieldnames=(*META_COLUMNS, *SI_NAMES))
            self.raw_writer.writeheader()
            self.si_writer.writeheader()
            self._write_manifest("started")
        except BaseException:
            self.close_files()
            raise

    def _write_manifest(self, status, extra=None):
        data = {**self.manifest, "status": status, "rows": self.count,
                "flagged_rows": self.flagged_rows,
                "wrapper_dt_min_s": self.min_dt, "wrapper_dt_max_s": self.max_dt,
                "first_si": self.first_si, "last_si": self.last_si,
                "si_min": self.si_min, "si_max": self.si_max,
                "solver_clock_rows": self.clock_rows,
                "first_solver_time_s": self.first_solver_time,
                "last_solver_time_s": self.last_solver_time,
                "ready_for_residual_training": False,
                "pending_checks": PENDING_CHECKS, **(extra or {})}
        temporary = self.output / "metadata.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.output / "metadata.json")

    def append(self, exports, *, wrapper_time, api_time, dt, control_index, phase, action, info, done,
               solver_time=None, solver_time_before=None):
        si = decode_exports(exports)
        flags = []
        wrapper_time = float(wrapper_time)
        dt = float(dt)
        if not math.isfinite(wrapper_time) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid wrapper time or solver step")
        if api_time is not None and not math.isfinite(api_time):
            raise ValueError("Invalid API time argument")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected 3 finite applied command values")
        if phase == "initial":
            flags.append("initial_snapshot_not_transition")
        if self.count == 1:
            flags.append("first_integrate_call_timing_pending")
        if solver_time is None or (phase != "initial" and solver_time_before is None):
            flags.append("solver_clock_unavailable")
        for value in (solver_time, solver_time_before):
            if value is not None and not math.isfinite(value):
                raise ValueError("Non-finite solver time")
        if solver_time is not None and self.last_solver_time is not None and solver_time <= self.last_solver_time:
            flags.append("solver_clock_nonincreasing")
        if self.previous_time is not None:
            spacing = wrapper_time - self.previous_time
            if spacing <= 0 or not math.isclose(spacing, dt, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError(f"Wrapper sample interval mismatch: {spacing}, expected {dt}")
            self.min_dt = spacing if self.min_dt is None else min(self.min_dt, spacing)
            self.max_dt = spacing if self.max_dt is None else max(self.max_dt, spacing)
        if si["vx_m_s"] < 1.0:
            flags.append("low_speed_outside_initial_validation_domain")
        if info.get("return_code", 0) != 0 or info.get("error"):
            flags.append("solver_nonzero_or_error")
        if done:
            flags.append("solver_terminal_snapshot")
        common = {
            "episode_id": 1, "sample_index": self.count, "control_index": control_index,
            "phase": phase, "wrapper_time_s": wrapper_time,
            "api_time_argument_s": "" if api_time is None else float(api_time),
            "solver_step_s": dt, "return_code": int(info.get("return_code", 0)),
            "solver_done": int(done), "throttle_command": float(action[0]),
            "brake_pressure_command_mpa": float(action[1]),
            "steering_wheel_command_deg": float(action[2]), "quality_flags": "|".join(flags),
            "solver_time_before_s": "" if solver_time_before is None else solver_time_before,
            "solver_time_s": "" if solver_time is None else solver_time,
            "solver_time_source": "" if solver_time is None else TIME_SOURCE,
        }
        self.raw_writer.writerow({**common, **dict(zip(EXPORT_NAMES, map(float, exports)))})
        self.si_writer.writerow({**common, **si})
        self.count += 1
        self.flagged_rows += int(bool(flags))
        self.previous_time = wrapper_time
        if solver_time is not None:
            self.clock_rows += 1
            if self.first_solver_time is None:
                self.first_solver_time = solver_time
            self.last_solver_time = solver_time
        self.last_si = si
        if self.first_si is None:
            self.first_si = si.copy()
            self.si_min, self.si_max = si.copy(), si.copy()
        else:
            self.si_min = {k: min(self.si_min[k], v) for k, v in si.items()}
            self.si_max = {k: max(self.si_max[k], v) for k, v in si.items()}

    def flush(self):
        for file in (self.raw_file, self.si_file):
            if file is not None and not file.closed:
                file.flush()

    def close_files(self):
        for file in (self.raw_file, self.si_file):
            if file is not None and not file.closed:
                file.close()

    def finish(self, status, extra=None):
        try:
            self.flush()
            self._write_manifest(status, extra)
        finally:
            self.close_files()


class IntegrationRecorder:
    """Collector-only adapter; core CarSimEnv and control policy stay unchanged.

    Split a held control interval into calls with inner_steps=1 to observe each
    solver tick. The same inputs and existing integrate_io time convention are
    preserved. Nominal wrapper time is not asserted to be physical export time.
    """

    def __init__(self, base, sink, require_solver_time=False):
        self.base, self.sink = base, sink
        self.control_index = 0
        self.config = None
        self.require_solver_time = require_solver_time
        self.clock = SolverClock()
        self.observed_solver = None
        self.original_solver = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        if self.sink.count:
            raise RuntimeError("One capture directory is restricted to one episode")
        exports = self.base.reset()
        if self.base.n_import != 3 or self.base.n_export != 25:
            raise RuntimeError("Live solver counts differ from checked 3/25 contract")
        self.config = dict(self.base.config)
        self.original_solver = getattr(self.base, "solver", None)
        initial_time = self.clock.bind(getattr(self.original_solver, "dll_handle", None),
                                       float(self.base.t_current))
        self.sink.manifest["clock_probe"] = {"source": TIME_SOURCE, "binding": self.clock.reason,
                                            "all_export_timestamps_verified": False}
        if self.original_solver is not None:
            self.observed_solver = ObservedSolver(self.original_solver, self.clock)
            self.base.solver = self.observed_solver
        self.sink.append(exports, wrapper_time=self.base.t_current, api_time=None,
                         dt=self.base.t_step, control_index=0, phase="initial",
                         action=[0.0, 0.0, 0.0], info={}, done=False, solver_time=initial_time)
        if self.require_solver_time and initial_time is None:
            raise RuntimeError(f"Solver clock required but {self.clock.reason}")
        return exports

    def control_step(self, action, inner_steps):
        if not isinstance(inner_steps, int) or inner_steps < 1:
            raise ValueError("inner_steps must be a positive integer")
        self.control_index += 1
        for _ in range(inner_steps):
            api_time = float(self.base.t_current)
            exports, reward, done, info = self.base.control_step(action, inner_steps=1)
            before = self.observed_solver.before if self.observed_solver else None
            after = self.observed_solver.after if self.observed_solver else None
            self.sink.append(exports, wrapper_time=self.base.t_current, api_time=api_time,
                             dt=self.base.t_step, control_index=self.control_index,
                             phase="after_integrate", action=action, info=info, done=done,
                             solver_time_before=before, solver_time=after)
            if self.require_solver_time and (before is None or after is None):
                raise RuntimeError(f"Solver clock required but {self.clock.reason}")
            if done:
                break
        self.sink.flush()
        return exports, reward, done, info

    def close(self):
        self.clock.discard()
        try:
            self.base.close()
        finally:
            if self.original_solver is not None:
                self.base.solver = self.original_solver


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default=DEFAULT_SIM)
    parser.add_argument("--path", default=str(BASE_DIR / "paths" / "dlc_reference_path.par"))
    parser.add_argument("--check-config-only", action="store_true")
    parser.add_argument("--mu", type=float, help="Required for capture; scenario label, not a measured export")
    parser.add_argument("--seconds", type=float, default=2.0, help="Nominal simulation time, multiple of 0.05 s")
    parser.add_argument("--target-speed", type=float, default=40.0)
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--wheelbase", type=float, default=VehicleParameters().wheelbase_m)
    parser.add_argument("--steering-ratio", type=float, default=16.0)
    parser.add_argument("--output-dir", type=Path, help="New directory only; default creates a timestamped run")
    parser.add_argument("--require-solver-time", action="store_true",
                        help="Stop and retain partial capture if the read-only VS clock is unavailable")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contract = inspect_configuration(args.sim)
    print("Config check: PASS (3 Imports, 25 Exports, exact order)")
    print("Parameters:", contract["parameter_path"])
    print("OPT_IO_SYNC_FM:", contract["settings"].get("OPT_IO_SYNC_FM", "unspecified"))
    print("Configured MU_ROAD_CONSTANT:", contract["settings"].get("MU_ROAD_CONSTANT", "unspecified"))
    if args.check_config_only:
        print("Read-only check finished. No DLL loaded, no capture directory created.")
        return
    if args.mu is None or not math.isfinite(args.mu) or args.mu < 0:
        raise ValueError("Capture requires explicit finite --mu >= 0; it is not measured from these 25 exports")
    for key in ("seconds", "target_speed", "lookahead", "wheelbase", "steering_ratio"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    configured_mu = contract["settings"].get("MU_ROAD_CONSTANT")
    if isinstance(configured_mu, (int, float)) and not math.isclose(args.mu, configured_mu, abs_tol=1e-9):
        raise ValueError("--mu differs from generated MU_ROAD_CONSTANT; confirm the active Run before capture")
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Import only after the read-only preflight; these imports do not start training.
    from carsim_wrapper import CONTROL_DT_S, CarSimGymWrapper
    from run_dlc_baseline import pure_pursuit_action

    count = args.seconds / CONTROL_DT_S
    if round(count) < 1 or not math.isclose(count, round(count), abs_tol=1e-8):
        raise ValueError("--seconds must be a positive integer multiple of the 0.05 s control period")
    output = args.output_dir or (BASE_DIR / "artifacts" / "dynamics7dof" /
                                datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f"))
    sink = CaptureWriter(output, {
        "format": "carsim_25_signal_capture_v2", "contract": contract,
        "scenario_mu_declared": args.mu, "mu_source": "CLI/config label, not measured tire-road friction",
        "gravity_conversion_m_s2": 9.81, "reference_path": str(path),
        "control_period_s": CONTROL_DT_S,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "native_units": dict(zip(EXPORT_NAMES, NATIVE_UNITS)),
        "timing_note": "wrapper_time_s is nominal; solver_time_before_s/solver_time_s directly observe T around integrate_io before termination. T is not proof of all export timestamps; missing T remains blank, never shifted or reconstructed.",
        "initial_note": "initial commands are buffer placeholders, not measured actuator values",
        "controller": "existing pure pursuit function; wheelbase explicitly defaults to user 3.16 m; steering ratio 16 is not newly calibrated",
    })
    print("Capture directory:", sink.output)
    env, recorder = None, None
    status, final_info, extra = "failed", {}, {}
    try:
        # New capture only: preserve the exact reference/configuration for later audits.
        snapshots = {}
        for name, source, expected in (
            ("simfile_snapshot.sim", Path(contract["sim_path"]), contract["sim_sha256"]),
            ("parameters_snapshot.par", Path(contract["parameter_path"]), contract["parameter_sha256"]),
            ("reference_path_snapshot.par", path, None),
        ):
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if expected and digest != expected:
                raise RuntimeError("Configuration changed before snapshot")
            with (sink.output / name).open("xb") as file:
                file.write(data)
            snapshots[name] = {"source": str(source), "sha256": digest}
        sink.manifest["snapshots"] = snapshots
        env = CarSimGymWrapper(args.sim, target_speed=args.target_speed, path_file=str(path))
        recorder = IntegrationRecorder(env._env, sink, require_solver_time=args.require_solver_time)
        env._env = recorder
        env.reset()
        live_dt = float(recorder.t_step)
        if live_dt <= 0 or not math.isclose(CONTROL_DT_S/live_dt, round(CONTROL_DT_S/live_dt), abs_tol=1e-8):
            raise ValueError("Solver step must divide the 50 ms control period exactly")
        refreshed = inspect_configuration(args.sim)
        if any(refreshed[k] != contract[k] for k in ("sim_sha256", "parameter_sha256")):
            raise RuntimeError("CarSim configuration changed during initialization")
        print("Live configuration:", recorder.config)
        print("Solver clock:", recorder.clock.reason, "| initial T:", sink.first_solver_time)
        print("Initial SI state:", {k: sink.first_si[k] for k in
              ("vx_m_s", "vy_m_s", "yaw_rate_rad_s", "omega_fl_rad_s", "omega_fr_rad_s", "omega_rl_rad_s", "omega_rr_rad_s")})
        status = "captured_requested_duration"
        for step in range(1, int(round(count)) + 1):
            action = pure_pursuit_action(env, args.target_speed, args.lookahead,
                                         args.wheelbase, args.steering_ratio)
            _, _, done, final_info = env.step(action)
            if step % 20 == 0 or done:
                clock_text = "unavailable" if sink.last_solver_time is None else f"{sink.last_solver_time:.6f}s"
                print(f"capture t_nominal={recorder.t_current:.3f}s T_solver={clock_text} rows={sink.count} "
                      f"station={final_info['station']:.2f}m speed={final_info['speed_kmh']:.2f}km/h "
                      f"lat={final_info['lateral_error']:+.3f}m")
            if final_info.get("error"):
                raise RuntimeError(str(final_info["error"]))
            if done:
                status = "off_track" if final_info.get("off_track") else "episode_ended"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        print("Capture interrupted; keeping partial data.")
    except BaseException as exc:
        status = "failed"
        extra["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            extra.update({"live_configuration": recorder.config if recorder is not None else None,
                          "final_info": final_info,
                          "control_updates": recorder.control_index if recorder is not None else 0})
            sink.finish(status, extra)
            print("Capture status:", status, "| rows:", sink.count)
            print("Metadata:", sink.output / "metadata.json")
            print("Raw and SI data saved. Timing/torque/load-memory checks remain; NOT training labels.")


if __name__ == "__main__":
    main()
