"""Fixed, checked CarSim 25-channel contract. No DLL calls or file writes."""

import hashlib
import locale
from pathlib import Path
import re

import numpy as np

from .model import STATE_NAMES, WHEEL_ORDER


IMPORT_NAMES = ("IMP_THROTTLE_ENGINE", "IMP_PCON_BK", "IMP_STEER_SW")
EXPORT_NAMES = (
    "Xo", "Yo", "Yaw", "AVz", "Lat_Veh", "Lat_Targ", "Vx", "Station", "Vy",
    "AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2", "Steer_L1", "Steer_R1", "Ax", "Ay",
    "My_Dr_L1", "My_Dr_R1", "My_Dr_L2", "My_Dr_R2",
    "My_Bk_L1", "My_Bk_R1", "My_Bk_L2", "My_Bk_R2",
)
NATIVE_UNITS = (
    "m", "m", "deg", "deg/s", "m", "m", "km/h", "m", "km/h",
    "rpm", "rpm", "rpm", "rpm", "deg", "deg", "g", "g",
    "N*m", "N*m", "N*m", "N*m", "N*m", "N*m", "N*m", "N*m",
)
SI_NAMES = (
    "x_origin_m", "y_origin_m", "yaw_rad", "lateral_to_carsim_path_m",
    "target_lateral_m", "station_at_origin_m", *STATE_NAMES,
    "front_steer_fl_rad", "front_steer_fr_rad", "ax_carsim_m_s2", "ay_carsim_m_s2",
    *(f"drive_{w.lower()}_nm" for w in WHEEL_ORDER),
    *(f"brake_{w.lower()}_nm" for w in WHEEL_ORDER),
)


def decode_exports(exports, gravity_m_s2=9.81):
    """Unit conversion only. Preserve signs; do NOT infer net torque or load memory."""
    x = np.asarray(exports, dtype=np.float64)
    if x.shape != (25,) or not np.all(np.isfinite(x)):
        raise ValueError("Expected exactly 25 finite exports in the verified order")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    state = np.r_[x[6]/3.6, x[8]/3.6, np.deg2rad(x[3]), x[9:13]*2*np.pi/60]
    values = np.r_[x[0:2], np.deg2rad(x[2]), x[4:6], x[7], state,
                   np.deg2rad(x[13:15]), x[15:17]*gravity_m_s2, x[17:25]]
    return dict(zip(SI_NAMES, map(float, values)))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_native_text(path):
    """CarSim 2022 generated files can use Windows ANSI, not UTF-8."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig", errors="strict"), "utf-8-sig"
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False)
        return data.decode(encoding, errors="strict"), encoding


def _parse_ports(value):
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) < 2 or parts[0] != len(parts)-1 or any(v < 1 for v in parts):
        raise ValueError(f"Invalid port declaration: {value}")
    return sum(parts[1:])


def inspect_configuration(sim_path):
    """Resolve generated INPUT and verify actual EXPORT names, not just counts.

    This narrow parser supports the current generated flat Run_all.par layout.
    Unknown macros/nested includes fail closed instead of guessing file paths.
    """
    sim_path = Path(sim_path).expanduser().resolve()
    sim_text, sim_encoding = _read_native_text(sim_path)
    macros, inputs, ports = {}, [], {}
    for raw in sim_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if line.upper().startswith("SET_MACRO "):
            match = re.match(r"SET_MACRO\s+\$\(([^)]+)\)\$\s+(.*)$", line, re.I)
            if not match:
                raise ValueError(f"Unsupported macro declaration: {line}")
            macros[match[1]] = match[2].strip()
        elif line.upper().startswith("INPUT "):
            inputs.append(line.split(None, 1)[1].strip())
        elif line.upper().startswith(("PORTS_IMP ", "PORTS_EXP ")):
            name, value = line.split(None, 1)
            ports[name.upper()] = _parse_ports(value)
    if len(inputs) != 1:
        raise ValueError("Expected one generated INPUT Run_all.par in simfile.sim")
    expression = inputs[0]
    for _ in range(20):
        previous = expression
        expression = re.sub(r"\$\(([^)]+)\)\$", lambda m: macros.get(m[1], m[0]), expression)
        if previous == expression:
            break
    if "$(" in expression:
        raise ValueError(f"Unresolved INPUT macro: {expression}")
    par_path = Path(expression.strip('"'))
    if not par_path.is_absolute():
        par_path = sim_path.parent / par_path
    par_path = par_path.resolve()
    par_text, par_encoding = _read_native_text(par_path)
    imports, exports = [], []
    settings = {}
    for raw in par_text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name = parts[0].upper()
        if name in ("INPUT", "INCLUDE"):
            raise ValueError("Nested parameter INPUT needs explicit contract resolution")
        if name == "IMPORT":
            if len(parts) < 3 or parts[2].upper() != "REPLACE":
                raise ValueError("Expected REPLACE mode for all three imports")
            imports.append(parts[1])
        elif name == "EXPORT":
            exports.append(parts[1])
        elif name in ("OPT_IO_SYNC_FM", "OPT_INT_METHOD", "MU_ROAD_CONSTANT", "TSTART", "TSTOP"):
            value = line[len(parts[0]):].strip().lstrip("=").strip().rstrip(";")
            try:
                settings[name] = float(value)
            except ValueError:
                settings[name] = value
    if tuple(n.upper() for n in imports) != IMPORT_NAMES:
        raise ValueError(f"Import order mismatch: {imports}")
    actual = tuple(n.upper() for n in exports)
    expected = tuple(n.upper() for n in EXPORT_NAMES)
    if actual != expected:
        differences = [f"#{i+1}: expected {name}, got {exports[i] if i<len(exports) else 'MISSING'}"
                       for i, name in enumerate(EXPORT_NAMES)
                       if i >= len(exports) or actual[i] != expected[i]]
        raise ValueError(f"Export order/count mismatch ({len(exports)}): " + "; ".join(differences))
    if ports != {"PORTS_IMP": 3, "PORTS_EXP": 25}:
        raise ValueError(f"simfile port count mismatch: {ports}")
    return {
        "sim_path": str(sim_path), "parameter_path": str(par_path),
        "sim_encoding": sim_encoding, "parameter_encoding": par_encoding,
        "sim_sha256": _sha256(sim_path), "parameter_sha256": _sha256(par_path),
        "imports": imports, "exports": exports, "native_units": list(NATIVE_UNITS),
        "settings": settings,
        "unit_assumption": "Native CarSim units checked against local channel metadata; not overridden at runtime",
    }
