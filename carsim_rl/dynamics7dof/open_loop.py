"""Open-loop sine / throttle / brake experiment and 7DOF comparison.

Default: read-only plan. --run explicitly starts CarSim (and writes its usual
LastRun results); --replay reads an existing capture without loading any DLL.
Current 3 Imports remain unchanged: throttle, brake pressure, steering wheel.
The requested nominal front angle is mapped by an approximate steering ratio;
the model replay ALWAYS uses measured actual left/right wheel angles and torques.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .collect import BASE_DIR, DEFAULT_SIM, CaptureWriter, IntegrationRecorder
from .comparison import write_comparison
from .signals import EXPORT_NAMES, NATIVE_UNITS, inspect_configuration


@dataclass(frozen=True)
class Excitation:
    front_amplitude_deg: float = 5
    frequency_hz: float = .3
    steer_start_s: float = 1.
    steer_cycles: float = 2.
    ramp_s: float = .5
    steering_ratio: float = 16.
    throttle: float = .12
    drive_until_s: float = 3.
    brake_mpa: float = .2
    brake_start_s: float = 5.
    brake_duration_s: float = 1.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Excitation settings must be finite')
        if not (0 <= self.front_amplitude_deg <= 5 and self.frequency_hz > 0
                and self.steer_cycles > 0 and self.steering_ratio > 0
                and 0 <= self.throttle <= 1 and 0 <= self.brake_mpa <= 10):
            raise ValueError('Require front amplitude 0..5 deg, positive frequency/cycles/ratio, throttle 0..1, brake 0..10 MPa')
        if min(self.steer_start_s, self.ramp_s, self.drive_until_s,
               self.brake_start_s, self.brake_duration_s) < 0:
            raise ValueError('Signal times must be nonnegative')
        if 2*self.ramp_s > self.steer_cycles/self.frequency_hz:
            raise ValueError('Steering duration must fit both smooth ramps')
        if self.throttle > 0 and self.brake_mpa > 0 and self.brake_duration_s > 0 and self.drive_until_s > self.brake_start_s:
            raise ValueError('Drive and brake intervals must not overlap')

    def front_angle(self, elapsed):
        t = elapsed-self.steer_start_s
        duration = self.steer_cycles/self.frequency_hz
        if t <= 0 or t >= duration:
            return 0.
        envelope = 1.
        if self.ramp_s > 0:
            edge = min(t, duration-t, self.ramp_s)/self.ramp_s
            envelope = .5*(1-math.cos(math.pi*edge))
        return self.front_amplitude_deg*envelope*math.sin(2*math.pi*self.frequency_hz*t)

    def command(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Elapsed time must be finite and nonnegative')
        throttle = self.throttle if elapsed < self.drive_until_s else 0.
        brake = self.brake_mpa if self.brake_start_s <= elapsed < self.brake_start_s+self.brake_duration_s else 0.
        return [throttle, brake, self.front_angle(elapsed)*self.steering_ratio]


def guard_reason(si, initial, args):
    speed = si['vx_m_s']*3.6
    beta = math.degrees(math.atan2(si['vy_m_s'], si['vx_m_s']))
    yaw_rate = math.degrees(si['yaw_rate_rad_s'])
    dx, dy = si['x_origin_m']-initial['x_origin_m'], si['y_origin_m']-initial['y_origin_m']
    lateral = -dx*math.sin(initial['yaw_rad'])+dy*math.cos(initial['yaw_rad'])
    checks = [(speed < args.min_speed_kmh, 'speed_below_min'),
              (speed > args.max_speed_kmh, 'speed_above_max'),
              (abs(beta) > args.max_beta_deg, 'sideslip_limit'),
              (abs(yaw_rate) > args.max_yaw_rate_deg_s, 'yaw_rate_limit'),
              (abs(lateral) > args.max_lateral_displacement_m, 'lateral_displacement_limit')]
    return next((reason for exceeded, reason in checks if exceeded), None)


def capture_open_loop(args, contract, signal, output, *, env_factory=None):
    """One episode only. No RL wrapper, tracking controller, or speed controller."""
    if env_factory is None:
        from carsim_env import CarSimEnv
        env_factory = CarSimEnv
    sink = CaptureWriter(output, {
        'format': 'carsim_25_signal_capture_v2', 'experiment_kind': 'open_loop_excitation',
        'contract': contract, 'scenario_mu_declared': args.mu,
        'gravity_conversion_m_s2': 9.81, 'native_units': dict(zip(EXPORT_NAMES, NATIVE_UNITS)),
        'control_period_s': contract['settings'].get('TSTEP', .001),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'excitation': asdict(signal),
        'controller': 'open-loop time schedule; no speed/path tracking; safety guards may terminate',
        'timing_note': 'Actual T observed; commands evaluated at pre-integration T and held for one solver call.',
        'input_note': 'Nominal front request maps to steering wheel by a ratio, not direct wheel-angle control.',
    })
    env, recorder, status, info, extra = None, None, 'failed', {}, {}
    learning, learning_active = None, False
    try:
        if getattr(args, 'learning_model', None):
            import torch
            from .learning.online import OnlineShadow
            torch.set_num_threads(1)
            learning = OnlineShadow(args.learning_model, adapt=args.learn_online,
                                    update_every=args.learning_update_every)
            learning_active = True
        def observe_learning():
            nonlocal learning_active
            if not learning_active:
                return
            from .model import Controls, STATE_NAMES
            si = sink.last_si
            controls = Controls([si['front_steer_fl_rad'], si['front_steer_fr_rad']],
                                [si[f'drive_{w}_nm']+si[f'brake_{w}_nm'] for w in ('fl', 'fr', 'rl', 'rr')])
            try:
                learning.observe(sink.last_solver_time, [si[k] for k in STATE_NAMES], controls, args.mu,
                                 valid=not recorder.done and info.get('return_code', 0) == 0)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                # The learner is only an observer. Failure disables it without
                # changing commands or breaking an otherwise valid raw capture.
                learning_active = False
                extra['learning_error'] = str(exc)
                print('Learning observer disabled:', exc, flush=True)
        snapshots = {}
        for name, source, expected in (
            ('simfile_snapshot.sim', contract['sim_path'], contract['sim_sha256']),
            ('parameters_snapshot.par', contract['parameter_path'], contract['parameter_sha256']),
        ):
            payload = Path(source).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected:
                raise RuntimeError('Configuration changed before capture')
            with (sink.output/name).open('xb') as f:
                f.write(payload)
            snapshots[name] = {'source': source, 'sha256': digest}
        sink.manifest['snapshots'] = snapshots
        env = env_factory(args.sim)
        recorder = IntegrationRecorder(env, sink, require_solver_time=True)
        recorder.reset()
        observe_learning()
        refreshed = inspect_configuration(args.sim)
        if any(refreshed[k] != contract[k] for k in ('sim_sha256', 'parameter_sha256')):
            raise RuntimeError('Configuration changed during initialization')
        dt = float(recorder.t_step)
        if not 0 < dt <= .01 or not math.isclose(args.seconds/dt, round(args.seconds/dt), abs_tol=1e-7):
            raise ValueError('Duration must be a multiple of live solver step (step <= 0.01 s)')
        sink.manifest['control_period_s'] = dt
        print('Live configuration:', recorder.config, flush=True)
        print('Initial speed:', sink.first_si['vx_m_s']*3.6, 'km/h | clock:', sink.first_solver_time, flush=True)
        initial_reason = guard_reason(sink.first_si, sink.first_si, args)
        if initial_reason:
            raise ValueError(f'Initial state outside experiment limits: {initial_reason}')
        status = 'captured_requested_duration'
        report_step = max(1, round(1/dt))
        for step in range(1, round(args.seconds/dt)+1):
            before = sink.last_solver_time
            elapsed = before-sink.first_solver_time
            action = signal.command(elapsed)
            _, _, done, info = recorder.control_step(action, 1)
            observe_learning()
            if sink.last_solver_time <= before:
                raise RuntimeError('Solver clock did not advance')
            if info.get('error'):
                raise RuntimeError(str(info['error']))
            reason = guard_reason(sink.last_si, sink.first_si, args)
            if step % report_step == 0 or done or reason:
                print(f"t={sink.last_solver_time-sink.first_solver_time:.4f}s "
                      f"vx={sink.last_si['vx_m_s']*3.6:.2f}km/h "
                      f"vy={sink.last_si['vy_m_s']:+.3f}m/s "
                      f"r={math.degrees(sink.last_si['yaw_rate_rad_s']):+.3f}deg/s", flush=True)
            if done or reason:
                status = 'episode_ended'
                if reason:
                    info = {**info, 'end_reason': 'safety_limit', 'safety_reason': reason}
                break
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Interrupted: preserving partial capture.', flush=True)
    except BaseException as exc:
        extra['error'] = f'{type(exc).__name__}: {exc}'
        status = 'failed'
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
            elif env is not None:
                env.close()
        except BaseException as exc:
            status = 'failed'
            extra['close_error'] = str(exc)
            raise
        finally:
            if learning is not None:
                try:
                    learning.save(sink.output.parent/'learning_shadow',
                                  {'live_CarSim_observer': True, 'capture': str(sink.output),
                                   'controls_unchanged_by_learning': True, 'observer_active_at_end': learning_active})
                except (ValueError, RuntimeError, OSError) as exc:
                    extra['learning_save_error'] = str(exc)
            extra.update(live_configuration=recorder.config if recorder else None,
                         final_info=info, control_updates=recorder.control_index if recorder else 0)
            sink.finish(status, extra)
            print('Capture:', sink.output, '| status:', status, '| reason:', info, flush=True)
    return status, info


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Actually run CarSim; updates CarSim LastRun results')
    mode.add_argument('--replay', type=Path, help='Existing capture; no CarSim DLL or simulation')
    mode.add_argument('--check-config-only', action='store_true', help='Read-only contract and schedule check')
    p.add_argument('--sim', default=DEFAULT_SIM)
    p.add_argument('--mu', type=float)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--front-amplitude-deg', type=float, default=.5)
    p.add_argument('--frequency-hz', type=float, default=.3)
    p.add_argument('--steer-start-s', type=float, default=1.)
    p.add_argument('--steer-cycles', type=float, default=2.)
    p.add_argument('--ramp-s', type=float, default=.5)
    p.add_argument('--steering-ratio', type=float, default=16.)
    p.add_argument('--throttle', type=float, default=.12)
    p.add_argument('--drive-until-s', type=float, default=3.)
    p.add_argument('--brake-mpa', type=float, default=.2)
    p.add_argument('--brake-start-s', type=float, default=5.)
    p.add_argument('--brake-duration-s', type=float, default=1.)
    p.add_argument('--model-substep-s', type=float, default=.0005)
    p.add_argument('--metric-warmup-s', type=float, default=.1)
    p.add_argument('--vehicle-json', type=Path, help='Optional complete VehicleParameters JSON; original defaults otherwise')
    p.add_argument('--allow-nominal-time', action='store_true', help='Legacy replay ONLY; time remains explicitly unverified')
    p.add_argument('--learning-model', type=Path, help='Optional trusted residual checkpoint; observe only, NEVER control')
    p.add_argument('--learn-online', action='store_true', help='Enable delayed-label candidate updates in the observer')
    p.add_argument('--learning-update-every', type=int, default=200)
    p.add_argument('--min-speed-kmh', type=float, default=5.)
    p.add_argument('--max-speed-kmh', type=float, default=80.)
    p.add_argument('--max-beta-deg', type=float, default=15.)
    p.add_argument('--max-yaw-rate-deg-s', type=float, default=45.)
    p.add_argument('--max-lateral-displacement-m', type=float, default=8.)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.learn_online and args.learning_model is None:
        p.error('--learn-online requires --learning-model')
    if args.learning_model and (args.replay or args.vehicle_json):
        p.error('Learning observer requires live default-parameter experiment; use learning shadow for replay')
    if args.learning_update_every < 20:
        p.error('--learning-update-every must be at least 20')
    # Fail before simulation or filesystem creation if comparison settings are bad.
    if not math.isfinite(args.model_substep_s) or not 0 < args.model_substep_s <= .001:
        p.error('--model-substep-s must be in (0, 0.001]')
    if not math.isfinite(args.metric_warmup_s) or args.metric_warmup_s < 0:
        p.error('--metric-warmup-s must be nonnegative')
    if args.vehicle_json:
        from .model import VehicleParameters
        values = json.loads(args.vehicle_json.read_text(encoding='utf-8-sig'))
        if set(values) != set(asdict(VehicleParameters())):
            p.error('--vehicle-json must specify all VehicleParameters fields')
        VehicleParameters(**values)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.replay:
        output = args.output_dir or BASE_DIR/'artifacts'/'dynamics7dof'/f'comparison_{stamp}'
        write_comparison(args.replay, output, substep_s=args.model_substep_s,
                         warmup_s=args.metric_warmup_s, allow_nominal=args.allow_nominal_time,
                         vehicle_json=args.vehicle_json)
        return 0
    if args.allow_nominal_time:
        p.error('--allow-nominal-time is for legacy replay only; live experiment requires actual T')
    signal = Excitation(**{k: getattr(args, k) for k in Excitation.__dataclass_fields__})
    for k in ('seconds', 'min_speed_kmh', 'max_speed_kmh', 'max_beta_deg',
              'max_yaw_rate_deg_s', 'max_lateral_displacement_m'):
        if not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0:
            p.error(f'--{k.replace("_", "-")} must be positive and finite')
    if args.min_speed_kmh < 3.6 or args.max_speed_kmh <= args.min_speed_kmh:
        p.error('Require max speed > min speed >= 3.6 km/h for forward reference model')
    if args.seconds <= args.metric_warmup_s:
        p.error('Duration must exceed metric warmup')
    contract = inspect_configuration(args.sim)
    configured_mu = contract['settings'].get('MU_ROAD_CONSTANT')
    if args.mu is not None and (not math.isfinite(args.mu) or args.mu < 0
                               or not isinstance(configured_mu, (float, int))
                               or not math.isclose(args.mu, configured_mu, abs_tol=1e-9)):
        p.error('--mu must match a finite constant MU_ROAD_CONSTANT in the generated configuration')
    print('Config check: PASS (3 Imports, 25 Exports) | configured mu:', configured_mu)
    print('Schedule:', json.dumps(asdict(signal), indent=2))
    print('Nominal front angle -> steering ratio -> steering-wheel command; model uses ACTUAL wheel angle/torque.')
    print('No path/speed controller. Confirm a flat wide road, initial speed, native units and no other running solver.')
    if not args.run:
        print('Read-only plan finished. Use --run --mu <configured value> to start a simulation.')
        return 0
    if args.mu is None:
        p.error('--run requires explicit --mu')
    output = (args.output_dir or BASE_DIR/'artifacts'/'dynamics7dof'/f'open_loop_{stamp}').resolve()
    if output.exists():
        p.error('Output must be a new directory; refusing to overwrite')
    print('Starting CarSim: its ordinary LastRun results WILL be updated.', flush=True)
    status, info = capture_open_loop(args, contract, signal, output/'capture')
    if status == 'interrupted':
        return 130
    write_comparison(output/'capture', output/'comparison', substep_s=args.model_substep_s,
                     warmup_s=args.metric_warmup_s, vehicle_json=args.vehicle_json)
    return 2 if info.get('safety_reason') else 0


if __name__ == '__main__':
    raise SystemExit(main())
