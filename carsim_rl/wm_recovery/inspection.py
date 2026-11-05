"""Read-only inventory of a generated flat CarSim configuration.

The inventory is deliberately not a declaration that the vehicle is validated.
No DLL calls, database edits, or guessed target Run paths are made here.
"""

from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import re

from dynamics7dof.model import VehicleParameters
from dynamics7dof.signals import _parse_ports, _read_native_text

from .controls import DIRECT_IMPORT_NAMES


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_simfile(path):
    sim = Path(path).expanduser().resolve(strict=True)
    text, encoding = _read_native_text(sim)
    macros, inputs, ports = {}, [], {}
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        macro = re.fullmatch(r"SET_MACRO\s+\$\(([^)]+)\)\$\s+(.+)", line, re.I)
        if macro:
            macros[macro[1]] = macro[2]
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, value = parts[0].upper(), parts[1]
            if key == "INPUT":
                inputs.append(value)
            elif key in ("PORTS_IMP", "PORTS_EXP"):
                ports[key] = _parse_ports(value)
    if len(inputs) != 1:
        raise ValueError("Expected exactly one INPUT pointing to a generated flat parameter file")
    target = inputs[0]
    for _ in range(20):
        expanded = re.sub(r"\$\(([^)]+)\)\$", lambda m: macros.get(m[1], m[0]), target)
        if expanded == target:
            break
        target = expanded
    if "$(" in target:
        raise ValueError("Unresolved or recursive INPUT macro")
    par = Path(target.strip('"'))
    par = (par if par.is_absolute() else sim.parent / par).resolve(strict=True)
    par_text, par_encoding = _read_native_text(par)
    imports, exports, settings = [], [], {}
    for number, raw in enumerate(par_text.splitlines(), 1):
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key in ("INPUT", "INCLUDE", "PARSFILE") and len(parts) > 1:
            raise ValueError(f"Nested parameter file at line {number}; explicit resolution required")
        if key == "IMPORT":
            if len(parts) < 4:
                raise ValueError(f"Incomplete IMPORT at line {number}")
            imports.append({"name": parts[1].upper(), "mode": parts[2].upper(),
                            "initial": parts[3], "line": number})
        elif key == "EXPORT":
            if len(parts) != 2:
                raise ValueError(f"Unsupported EXPORT at line {number}")
            exports.append(parts[1])
        elif key in ("TSTEP", "TSTART", "TSTOP", "MU_ROAD_CONSTANT", "MU_ROAD_COMBINE",
                     "MU_ROAD_L_CONSTANT", "OPT_IO_UPDATE", "OPT_IO_SYNC_FM", "OPT_INT_METHOD"):
            settings[key] = " ".join(parts[1:]).strip("= ;")
    names = [item["name"] for item in imports]
    warnings = []
    counts_match = ports == {"PORTS_IMP": len(imports), "PORTS_EXP": len(exports)}
    if not counts_match:
        warnings.append("simfile port declarations do not match parameter-file channel counts")
    expected_channels = len(names) == 6 and set(names) == set(DIRECT_IMPORT_NAMES)
    if not expected_channels:
        warnings.append("direct four-wheel-torque / equal-front-steer channel set is not present")
    duplicates = {name: count for name, count in Counter(n.upper() for n in exports).items() if count > 1}
    if duplicates:
        warnings.append("duplicate exports: never convert this array to a name/value dict without checking")
    warnings.append("Import names and counts do not validate steering dynamics or torque balance")
    return {
        "format": "wm_recovery_configuration_inventory_v1",
        "sim_path": str(sim), "parameter_path": str(par),
        "sim_sha256": file_sha256(sim), "parameter_sha256": file_sha256(par),
        "encodings": {"sim": encoding, "parameters": par_encoding},
        "ports": ports, "imports": imports, "exports": exports,
        "duplicate_exports": duplicates, "settings": settings,
        "port_counts_match": counts_match, "direct_channel_names_match": expected_channels,
        "paper_reference_parameters": asdict(VehicleParameters()),
        "vehicle_parameter_alignment": "not_verified",
        "physical_interface_verified": False, "warnings": warnings,
    }
