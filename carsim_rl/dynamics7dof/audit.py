"""Read-only capture audit. Never loads a CarSim DLL or generates training labels.

Example: python -m dynamics7dof.audit --capture <capture_directory>
Optional --report-dir creates a NEW separate directory with JSON/Markdown.
Exit 0: integrity checked (physical validation may still be pending).
Exit 2: unreadable/inconsistent data. No original files are changed.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .signals import EXPORT_NAMES, IMPORT_NAMES, NATIVE_UNITS, SI_NAMES, decode_exports
from .timing import TIME_SOURCE


COMMON = (
    "episode_id", "sample_index", "control_index", "phase", "wrapper_time_s",
    "api_time_argument_s", "solver_step_s", "return_code", "solver_done",
    "throttle_command", "brake_pressure_command_mpa", "steering_wheel_command_deg", "quality_flags",
)
CLOCK = ("solver_time_before_s", "solver_time_s", "solver_time_source")
PENDING = [
    "Per-export state/force timestamp alignment is not yet physically verified.",
    "Signed drive + brake is only a candidate net torque; wheel torque balance remains unverified.",
    "CarSim Ax/Ay are not automatically the reference model's LoadMemory.",
]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path, physical):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path.name}: duplicate column names")
        missing = set((*COMMON, *physical)) - set(fields)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")
        if any(k in fields for k in CLOCK) and not all(k in fields for k in CLOCK):
            raise ValueError(f"{path.name}: incomplete clock column set")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError(f"{path.name}: need at least 3 snapshots")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path.name}: ragged/truncated CSV row")
    return rows, fields


def _column(rows, name, *, allow_blank=False):
    values = []
    for i, row in enumerate(rows):
        text = row.get(name, "")
        if text == "" and allow_blank:
            values.append(np.nan)
            continue
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: invalid number at sample {i}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite value at sample {i}")
        values.append(value)
    return np.asarray(values)


def _stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "rms": None}
    return {"count": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "rms": float(np.sqrt(np.mean(a*a)))}


def _close(a, b):
    return np.allclose(a, b, rtol=1e-8, atol=1e-9)


def audit_capture(capture, *, warmup_s=0.01):
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("warmup_s must be finite and nonnegative")
    root = Path(capture).expanduser().resolve()
    report = {"format": "dynamics7dof_audit_v1", "capture": str(root),
              "integrity": "FAIL", "errors": [], "warnings": [],
              "ready_for_residual_training": False, "pending_physical_checks": PENDING.copy()}
    errors, warnings = report["errors"], report["warnings"]

    def check(condition, message):
        if not condition:
            errors.append(message)

    try:
        paths = [root / name for name in ("metadata.json", "raw_exports.csv", "signals_si.csv")]
        report["source_sha256"] = {p.name: _digest(p) for p in paths}
        meta = json.loads(paths[0].read_text(encoding="utf-8"))
        raw, raw_fields = _read_csv(paths[1], EXPORT_NAMES)
        rows, fields = _read_csv(paths[2], SI_NAMES)
        if len(raw) != len(rows):
            raise ValueError("Raw/SI row counts differ")
        n = len(rows)
        report.update(rows=n, capture_status=meta.get("status"), final_info=meta.get("final_info", {}))
        contract = meta.get("contract", {})
        check([s.upper() for s in contract.get("imports", [])] == list(IMPORT_NAMES),
              "Metadata import contract differs from 3-channel order")
        check([s.upper() for s in contract.get("exports", [])] == [s.upper() for s in EXPORT_NAMES],
              "Metadata export contract differs from 25-channel order")
        check(contract.get("native_units") == list(NATIVE_UNITS), "Metadata native units differ from expected contract")
        check(meta.get("rows") == n, "Metadata row count differs from CSV")
        check(meta.get("status") in ("captured_requested_duration", "episode_ended", "off_track", "interrupted"),
              "Capture is not finalized successfully (started/failed/unknown)")
        if meta.get("status") in ("off_track", "interrupted"):
            warnings.append("Partial/off-track episode: not a successful path completion")
        if meta.get("ready_for_residual_training") is not False:
            warnings.append("Source training-ready claim is not accepted by this audit")

        has_clock = all(k in fields for k in CLOCK)
        check(has_clock == all(k in raw_fields for k in CLOCK), "Raw/SI clock schemas differ")
        if meta.get("format") == "carsim_25_signal_capture_v2":
            check(has_clock, "v2 capture is missing clock columns")
        shared = (*COMMON, *(CLOCK if has_clock else ()))
        check(all(all(a.get(k) == b.get(k) for k in shared) for a, b in zip(raw, rows)),
              "Raw/SI sample metadata or timing differs")
        data = {key: _column(rows, key) for key in SI_NAMES}
        numeric = {key: _column(rows, key) for key in COMMON
                   if key not in ("phase", "quality_flags", "api_time_argument_s")}
        t = numeric["wrapper_time_s"]
        api = _column(rows, "api_time_argument_s", allow_blank=True)
        dt = numeric["solver_step_s"]
        check(np.array_equal(numeric["sample_index"], np.arange(n)), "Sample indices are not consecutive from zero")
        check(np.all(numeric["episode_id"] == 1), "Unexpected episode IDs; no cross-episode derivatives allowed")
        check(rows[0]["phase"] == "initial" and all(r["phase"] == "after_integrate" for r in rows[1:]),
              "Invalid initial/after_integrate phases")
        check(np.all(dt > 0) and _close(dt, dt[0]), "Invalid or changing configured solver step")
        check(np.all(np.diff(t) > 0) and _close(np.diff(t), dt[1:]), "Nominal time gaps or non-increasing times")
        requested = meta.get("arguments", {}).get("seconds")
        if meta.get("status") == "captured_requested_duration" and requested is not None:
            check(_close(t[-1]-t[0], float(requested)), "Claimed requested duration does not match recorded nominal duration")
        check(np.isnan(api[0]) and np.all(np.isfinite(api[1:])) and _close(api[1:], t[:-1]),
              "API time argument does not match prior nominal time")
        check(np.all(np.isin(numeric["solver_done"], (0, 1))), "Invalid solver_done values")
        check(np.all(numeric["return_code"] == np.floor(numeric["return_code"])), "Non-integer solver return codes")
        check(not np.any(numeric["solver_done"][:-1]), "Rows recorded after a solver terminal snapshot")
        if np.any(numeric["return_code"] != 0):
            warnings.append("Nonzero solver return codes: examine termination/error before using these samples")
        if (meta.get("final_info") or {}).get("error"):
            errors.append("Metadata reports a solver/controller error")

        conversion_error = 0.0
        gravity = float(meta.get("gravity_conversion_m_s2", 9.81))
        for i, row in enumerate(raw):
            decoded = decode_exports([float(row[k]) for k in EXPORT_NAMES], gravity)
            conversion_error = max(conversion_error, max(abs(decoded[k]-data[k][i]) for k in SI_NAMES))
        check(conversion_error <= 1e-9, f"Raw-to-SI conversion mismatch: {conversion_error}")
        report["max_unit_conversion_error"] = conversion_error
        for field, reducer in (("first_si", lambda x: x[0]), ("last_si", lambda x: x[-1]),
                               ("si_min", np.min), ("si_max", np.max)):
            recorded = meta.get(field, {})
            check(all(k in recorded and _close(float(recorded[k]), reducer(data[k])) for k in SI_NAMES),
                  f"Metadata {field} differs from actual CSV")

        ci = numeric["control_index"]
        check(ci[0] == 0 and ci[1] == 1 and np.all(ci == np.floor(ci))
              and np.all(np.isin(np.diff(ci[1:]), (0, 1))), "Invalid control block indices")
        blocks = len(np.unique(ci[1:]))
        check(meta.get("control_updates") == blocks, "Control update count differs from CSV")
        control_period = float(meta["control_period_s"])
        expected = control_period / dt[0] if dt[0] > 0 else 0
        check(expected >= 1 and math.isclose(expected, round(expected), abs_tol=1e-8),
              "Control period is not an integer multiple of solver step")
        held = ci[2:] == ci[1:-1]
        for key in ("throttle_command", "brake_pressure_command_mpa", "steering_wheel_command_deg"):
            check(np.all(np.diff(numeric[key][1:])[held] == 0), f"Command changed inside a held control block: {key}")
        counts = list(Counter(ci[1:]).values())
        check(all(v == round(expected) for v in counts[:-1]) and 0 < counts[-1] <= round(expected),
              "Missing/extra solver samples inside control blocks")
        check(np.all((numeric["throttle_command"] >= 0) & (numeric["throttle_command"] <= 1)),
              "Throttle command outside [0,1]")
        check(np.all(numeric["brake_pressure_command_mpa"] >= 0), "Negative brake pressure command")
        report["quality_flags"] = dict(Counter(flag for row in rows for flag in row["quality_flags"].split("|") if flag))
        check(meta.get("flagged_rows") == sum(bool(row["quality_flags"]) for row in rows),
              "Flagged row count differs from metadata")

        # No reconstruction: historical files keep their nominal time basis.
        time_base = t
        timing = {"diagnostic_time_basis": "wrapper_nominal_unverified", "clock_rows": 0,
                  "nominal_dt_s": _stats(np.diff(t)), "all_export_timestamps_verified": False}
        if has_clock:
            actual = _column(rows, "solver_time_s", allow_blank=True)
            before = _column(rows, "solver_time_before_s", allow_blank=True)
            known = np.isfinite(actual)
            timing["clock_rows"] = int(known.sum())
            check(all(rows[i]["solver_time_source"] == TIME_SOURCE for i in np.flatnonzero(known)),
                  "Unknown solver clock source")
            check(meta.get("solver_clock_rows") == int(known.sum()), "Solver clock row count differs from metadata")
            if np.any(known):
                check(_close(float(meta["first_solver_time_s"]), actual[known][0]) and
                      _close(float(meta["last_solver_time_s"]), actual[known][-1]),
                      "Metadata first/last solver time differs from CSV")
            if np.all(known) and np.all(np.isfinite(before[1:])):
                advancing = bool(np.all(np.diff(actual) > 0))
                continuous = _close(before[1:], actual[:-1])
                check(advancing, "Solver clock is not strictly increasing")
                check(continuous, "Solver time before integration differs from previous time after integration")
                check(np.all(actual[1:] > before[1:]), "Integration did not advance internal T")
                check(_close(np.diff(actual)[1:], dt[2:]), "Unexpected solver intervals after first integration")
                timing.update(first_interval_s=float(actual[1]-actual[0]),
                              subsequent_dt_s=_stats(np.diff(actual)[1:]),
                              wrapper_minus_solver_s=_stats(t[1:]-actual[1:]),
                              api_to_solver_after_s=_stats(actual[1:]-api[1:]))
                if advancing and continuous:
                    time_base = actual
                    timing["diagnostic_time_basis"] = "observed_solver_T_not_per_export_certification"
            else:
                warnings.append("Missing/partial solver T: nominal time used ONLY for provisional diagnostics")
        else:
            warnings.append("Legacy capture has no actual solver time; no 0.5 ms shift has been inferred or applied")
        report["timing"] = timing
        report["integration_settings"] = contract.get("settings", {})
        if contract.get("settings", {}).get("OPT_INT_METHOD") in (1, 2, 3, 4):
            warnings.append("AM/RK uses internal half steps; inspect OPT_IO_UPDATE. OPT_IO_SYNC_FM alone does not establish timing")

        report["torques"] = {"net_torque_mapping": "drive + brake is a candidate only; not applied to labels", "wheels": {}}
        brake_command = numeric["brake_pressure_command_mpa"]
        forward_braking = (brake_command > 1e-8) & (data["vx_m_s"] > 1)
        report["torques"]["forward_braking_rows"] = int(forward_braking.sum())
        report["torques"]["brake_pressure_mpa"] = _stats(brake_command)
        for w in ("fl", "fr", "rl", "rr"):
            brake = data[f"brake_{w}_nm"]
            mask = forward_braking & (data[f"omega_{w}_rad_s"] > 0)
            report["torques"]["wheels"][w] = {
                "drive_nm": _stats(data[f"drive_{w}_nm"]), "brake_nm": _stats(brake),
                "brake_nm_during_forward_braking": _stats(brake[mask]),
                "negative_brake_fraction": float(np.mean(brake[mask] < 0)) if np.any(mask) else None,
            }
        if not np.any(forward_braking):
            warnings.append("No active forward braking samples; brake sign cannot be established from this episode")

        # These are planar signal-consistency diagnostics, NOT 7DOF accuracy scores.
        if np.all(np.diff(time_base) > 0):
            valid = (time_base-time_base[0] >= warmup_s) & (data["vx_m_s"] >= 1)
            bad = (numeric["return_code"] != 0) | (numeric["solver_done"] != 0)
            bad[:2] = True
            bad[-1] = True
            for shift in (-1, 0, 1):
                valid &= ~np.roll(bad, shift)
            valid[:3] = False
            valid[-3:] = False
            ix = np.flatnonzero(valid)
            vx, vy, yaw = data["vx_m_s"], data["vy_m_s"], data["yaw_rate_rad_s"]
            derived = {"ax": np.gradient(vx, time_base)-yaw*vy,
                       "ay": np.gradient(vy, time_base)+yaw*vx}
            acceleration = {"warmup_excluded_s": warmup_s, "samples": int(ix.size),
                            "time_basis": timing["diagnostic_time_basis"],
                            "meaning": "planar finite-difference signal consistency; not model accuracy or LoadMemory validation"}
            for axis, calculated in derived.items():
                measured = data[f"{axis}_carsim_m_s2"]
                acceleration[axis] = {"difference_m_s2": _stats((calculated-measured)[ix]),
                                      "measured_m_s2": _stats(measured[ix])}
            report["acceleration"] = acceleration

        # New captures have an immutable path snapshot. Older ones use a clearly
        # marked present-day reference, never silently treating it as historical.
        snapshots = meta.get("snapshots", {})
        for name, description in snapshots.items():
            if Path(name).name != name:
                raise ValueError("Unsafe snapshot filename in metadata")
            check(_digest(root/name) == description["sha256"], f"Snapshot hash mismatch: {name}")
        snapshot = root / "reference_path_snapshot.par"
        path_file = snapshot if "reference_path_snapshot.par" in snapshots else Path(meta.get("reference_path", ""))
        if path_file.is_file():
            from frenet_path import FrenetPath
            reference = FrenetPath(path_file)
            lateral, heading = [], []
            for x, y, angle in zip(data["x_origin_m"], data["y_origin_m"], data["yaw_rad"]):
                projection = reference.project(x, y, np.degrees(angle))
                lateral.append(projection["lateral_error"])
                heading.append(projection["heading_error_deg"])
            report["tracking"] = {"reference": str(path_file), "reference_sha256": _digest(path_file),
                                  "historical_snapshot": path_file == snapshot,
                                  "lateral_error_m": _stats(lateral), "heading_error_deg": _stats(heading)}
            if path_file != snapshot:
                warnings.append("Tracking metrics use current path file; historical path identity is unverified for this legacy capture")
        else:
            warnings.append("Reference path unavailable: tracking metrics omitted, other checks retained")
        check(report["source_sha256"] == {p.name: _digest(p) for p in paths}, "Source files changed during audit")
    except (OSError, ValueError, KeyError, TypeError, csv.Error, RuntimeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["integrity"] = "PASS" if not errors else "FAIL"
    return report


def format_report(report):
    lines = ["# CarSim capture audit", "", f"Source: {report['capture']}",
             f"Data integrity: {report['integrity']}", f"Rows: {report.get('rows', 'unavailable')}",
             "Ready for residual training: NO", ""]
    timing = report.get("timing", {})
    lines.extend([f"Diagnostic time basis: {timing.get('diagnostic_time_basis', 'unavailable')}",
                  f"Actual solver clock rows: {timing.get('clock_rows', 0)}"])
    if "first_interval_s" in timing:
        lines.append(f"First solver interval: {timing['first_interval_s']:.9f} s")
        lines.append(f"Wrapper minus solver T: {timing['wrapper_minus_solver_s']}")
    tracking = report.get("tracking")
    if tracking:
        lat = tracking["lateral_error_m"]
        lines.append(f"Lateral error: RMS {lat['rms']:.6f} m, max abs {max(abs(lat['min']),abs(lat['max'])):.6f} m")
    torque = report.get("torques", {})
    lines.append(f"Forward braking rows: {torque.get('forward_braking_rows', 'unavailable')}")
    for w, result in torque.get("wheels", {}).items():
        lines.append(f"{w.upper()} brake moment: {result['brake_nm']['min']:.6f} .. {result['brake_nm']['max']:.6f} N*m; "
                     f"negative fraction during forward braking: {result['negative_brake_fraction']}")
    for axis in ("ax", "ay"):
        result = report.get("acceleration", {}).get(axis)
        if result:
            lines.append(f"{axis.upper()} planar consistency RMS difference: {result['difference_m_s2']['rms']} m/s^2 (NOT 7DOF model error)")
    for title, key in (("Errors", "errors"), ("Warnings", "warnings"), ("Pending physical checks", "pending_physical_checks")):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {message}" for message in report[key])
        if not report[key]:
            lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=0.01)
    parser.add_argument("--report-dir", type=Path, help="Optional NEW directory, outside the capture; never overwrite")
    args = parser.parse_args(argv)
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("--warmup-seconds must be finite and nonnegative")
    root = args.capture.expanduser().resolve()
    output = args.report_dir.expanduser().resolve() if args.report_dir else None
    if output is not None and (output == root or root in output.parents):
        parser.error("--report-dir must be outside the source capture directory")
    if output is not None and output.exists():
        parser.error("--report-dir already exists; choose a new directory")
    report = audit_capture(root, warmup_s=args.warmup_seconds)
    rendered = format_report(report)
    print(rendered)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "audit.json").open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        with (output / "audit.md").open("x", encoding="utf-8") as file:
            file.write(rendered)
        print("Report directory:", output)
    return 0 if report["integrity"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
