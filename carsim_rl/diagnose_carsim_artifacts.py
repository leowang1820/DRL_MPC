# -*- coding: utf-8 -*-
"""Read-only CarSim artifact integrity diagnostic.

Run in the project root:
    python diagnose_carsim_artifacts.py

This tool NEVER writes, renames, or deletes user experiment files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


RAW_HASHES = {
    "train1": "d47be0cf16e116c28f5a3651daf65c545809fa6d3814a43ab814831c3e966f73",
    "train2": "59d38e1c7da16a95a1cf5ba2ed27757a8432bbe7078826fe1c52367282bfed4c",
    "validation": "c220a301d31bd74c7ccb471ad703e1756db645af9ba21b56f8e2389e3c2fe566",
    "test": "709279987b633920298cf6190e9e855851c923d9584ca056fd0ec7845f00fed2",
}

CANDIDATE_HASHES = {
    "train1": "d14b0046f79f92f7682e3c753828e6216b20f65a74c756b1ba95dcb78527dccf",
    "train2": "04e04dd1c567b2458d5652e244aa35c334033ffbd003a82804bbfe675641763a",
    "validation": "646268f5c688f30ab75ca45817bff5ee3cbbb8fae2bed7c7567342a6282e76d2",
    "test": "110e148f3589ed422e53ad1d2b91fa9f7966bb9672c54abe7928cc0fc903ef06",
}

KEYS = (
    "time_s",
    "state_si",
    "command_si",
    "nominal_7dof_input_si",
    "measured_steer_rad",
    "reported_external_torque_nm",
    "reported_accel_mps2",
    "import_native",
    "my_dr_native",
    "nominal_initial_load_memory_mps2",
)

WHEELS = ("L1", "R1", "L2", "R2")
EXPORT_NAMES = (
    "Xo", "Yo", "Vx", "Vy", "Yaw", "AVz", "Ax",
    *(f"AlphL_{w}" for w in WHEELS),
    *(f"Vx_WC_{w}" for w in WHEELS),
    *(f"AVy_{w}" for w in WHEELS),
    *(f"Fx_{w}" for w in WHEELS),
    *(f"Fy_{w}" for w in WHEELS),
    *(f"Fz_{w}" for w in WHEELS),
    *(f"KappL_{w}" for w in WHEELS),
    *(f"MuX_{w}" for w in WHEELS),
    "Steer_SW", "Beta", "Ay", "Steer_L1", "Steer_R1",
    *(f"My_Dr_{w}" for w in WHEELS),
    "Steer_SW",
    *(f"My_US_{w}" for w in WHEELS),
    *(f"MyRR_{w}" for w in WHEELS),
    *(f"My_WC_{w}" for w in WHEELS),
)
assert len(EXPORT_NAMES) == 61


def file_sha256(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fingerprint(arrays):
    """Exact physical-array digest used by carsim_candidate.py."""
    sha = hashlib.sha256()
    for key in KEYS:
        x = np.asarray(arrays[key], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"{key}: nonfinite values")
        sha.update(key.encode("ascii"))
        sha.update(str(x.shape).encode("ascii"))
        sha.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    return sha.hexdigest()


def cols(*names):
    out = []
    for name in names:
        if EXPORT_NAMES.count(name) != 1:
            raise ValueError(f"Expected a unique export: {name}")
        out.append(EXPORT_NAMES.index(name))
    return out


def canonical_from_raw(raw_arrays):
    """Recreate the physical subset in memory without modifying raw data."""
    t = np.asarray(raw_arrays["time_s"], dtype=np.float64)
    raw = np.asarray(raw_arrays["raw_exports"], dtype=np.float64)
    cmd = np.asarray(raw_arrays["command"], dtype=np.float64)
    imp = np.asarray(raw_arrays["import_values"], dtype=np.float64)

    n = len(t) - 1
    if n < 1 or raw.shape != (n + 1, 61):
        raise ValueError(f"raw exports shape mismatch: {raw.shape}")
    if cmd.shape != (n, 5) or imp.shape != (n, 6):
        raise ValueError("raw command/import shape mismatch")
    if not np.isfinite(t).all() or not np.allclose(
        np.diff(t), .01, rtol=0, atol=1e-9
    ):
        raise ValueError("raw time grid mismatch")

    steer_deg = np.rad2deg(cmd[:, 4])
    expected_imports = np.column_stack((cmd[:, :4], steer_deg, steer_deg))
    if not np.allclose(imp, expected_imports, rtol=0, atol=1e-8):
        raise ValueError("raw six-channel input consistency mismatch")

    state_columns = cols(
        "Vx", "Vy", "AVz",
        "AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2"
    )
    factors = np.array([
        1 / 3.6, 1 / 3.6, np.pi / 180,
        2 * np.pi / 60, 2 * np.pi / 60,
        2 * np.pi / 60, 2 * np.pi / 60
    ], dtype=np.float64)

    return {
        "time_s": t,
        "state_si": raw[:, state_columns] * factors,
        "command_si": cmd,
        "nominal_7dof_input_si": np.column_stack((
            cmd[:, 4], cmd[:, 4], cmd[:, :4]
        )),
        "measured_steer_rad": raw[:, cols(
            "Steer_L1", "Steer_R1"
        )] * (np.pi / 180),
        "reported_external_torque_nm": raw[:, cols(
            "My_US_L1", "My_US_R1", "My_US_L2", "My_US_R2"
        )].copy(),
        "reported_accel_mps2": raw[:, cols(
            "Ax", "Ay"
        )] * 9.80665,
        "import_native": imp,
        "my_dr_native": raw[:, cols(
            "My_Dr_L1", "My_Dr_R1", "My_Dr_L2", "My_Dr_R2"
        )].copy(),
        "nominal_initial_load_memory_mps2": np.zeros(
            2, dtype=np.float64
        ),
    }


def show_file(label, path):
    print(f"  {label}: {path}")
    if not path.is_file():
        print("    STATUS=MISSING")
        return False
    stat = path.stat()
    print(f"    SIZE={stat.st_size} MODIFIED={stat.st_mtime}")
    if not stat.st_size:
        print("    STATUS=EMPTY")
        return False
    return True


def check_json(label, path):
    if not show_file(label, path):
        return
    try:
        with path.open("r", encoding="utf-8") as file:
            obj = json.load(file)
        print(f"    JSON=OK FORMAT={obj.get('format')}")
    except Exception as error:
        print(f"    JSON=BAD ERROR={type(error).__name__}: {error}")


def load_npz(label, path):
    if not show_file(label, path):
        return None
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            failure = archive.testzip()
            if failure is not None:
                raise ValueError(f"CRC mismatch: {failure}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        print("    ZIP=OK")
        print("    ARRAYS=" + " ".join(
            f"{key}:{value.shape}" for key, value in arrays.items()
        ))
        return arrays
    except Exception as error:
        print(f"    ZIP=BAD ERROR={type(error).__name__}: {error}")
        return None


def run(root):
    pilot = root / "carsim_experimental_pilot_01"
    sources = {
        "train1": (
            root / "carsim_long_validation_01",
            root / "carsim_candidate_01",
        ),
        "train2": (
            pilot / "raw_train2",
            pilot / "candidate_train2",
        ),
        "validation": (
            pilot / "raw_validation",
            pilot / "candidate_validation",
        ),
        "test": (
            pilot / "raw_test",
            pilot / "candidate_test",
        ),
    }

    print("READ-ONLY DIAGNOSTIC: no file modifications")
    print(f"ROOT={root.resolve()}")

    for name, (raw_dir, candidate_dir) in sources.items():
        print(f"\n========== {name} ==========")
        raw_path = raw_dir / "raw_capture.npz"
        candidate_path = candidate_dir / "candidate.npz"

        raw = load_npz("RAW", raw_path)
        if raw is not None:
            actual_sha = file_sha256(raw_path)
            print(f"    RAW_SHA256={actual_sha}")
            print(
                "    RAW_BYTES_MATCH_HISTORY="
                + str(actual_sha == RAW_HASHES[name])
            )
            try:
                actual_canonical = fingerprint(canonical_from_raw(raw))
                print(
                    "    RAW_PHYSICAL_SUBSET_MATCH_HISTORY="
                    + str(actual_canonical == CANDIDATE_HASHES[name])
                )
                print("    RAW_DERIVED_CANONICAL=" + actual_canonical)
            except Exception as error:
                print(f"    RAW_SEMANTICS=BAD {type(error).__name__}: {error}")

        check_json("RAW_METADATA", raw_dir / "metadata.json")

        candidate = load_npz("CANDIDATE", candidate_path)
        if candidate is not None:
            try:
                actual_canonical = fingerprint(candidate)
                print(
                    "    CANDIDATE_MATCH_HISTORY="
                    + str(actual_canonical == CANDIDATE_HASHES[name])
                )
                print("    CANDIDATE_CANONICAL=" + actual_canonical)
            except Exception as error:
                print(f"    CANDIDATE_SEMANTICS=BAD {type(error).__name__}: {error}")

        check_json(
            "CANDIDATE_REPORT",
            candidate_dir / "candidate_report.json"
        )
        check_json(
            "EPISODE_MANIFEST",
            pilot / "workspace" / "episodes" / name
            / "experimental_episode.json"
        )

    print("\n========== MODEL FILE ==========")
    model = root / "carsim_experimental_model_01" / "world_model.pt"
    if show_file("WORLD_MODEL", model):
        print("    WORLD_MODEL_SHA256=" + file_sha256(model))

    print("\n========== DISK CHECK ==========")
    print("Check free disk space, backups, sync software and file modification times.")
    print("DO NOT overwrite files with failed hashes before investigating.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts/wm_recovery"),
        help="The directory containing CarSim experiment artifacts"
    )
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
