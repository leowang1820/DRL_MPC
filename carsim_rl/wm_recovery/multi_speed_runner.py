# -*- coding: utf-8 -*-
r"""批量检查和运行 40~180 km/h CarSim 配置，并建立统一 WM 训练集。

主命令：
  python -m wm_recovery.multi_speed_runner audit --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve"
  python -m wm_recovery.multi_speed_runner probe --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve"
  python -m wm_recovery.multi_speed_runner capture --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve" \
      --output artifacts/wm_recovery/multispeed_01 --speeds 40 60 80
  python -m wm_recovery.multi_speed_runner merge \
      --captures artifacts/wm_recovery/multispeed_01 \
      --output artifacts/wm_recovery/multispeed_workspace_01
  python -m wm_recovery.multi_speed_runner train \
      --workspace artifacts/wm_recovery/multispeed_workspace_01 \
      --output artifacts/wm_recovery/multispeed_model_01 --epochs 100

约束：
- audit 只读 SIM 和 PAR，无法从文件夹名确认真实车速。
- probe 实际初始化 CarSim，并验证 Export.Vx 初值（原生 km/h）。
- SIM 的 INPUT/FILEBASE 在不同车速之间必须唯一，禁止复用同一结果文件。
- capture 调用已有 scenario_runner.capture_all，18 个工况分别运行，不修改 DLL。
- merge 复用 carsim_experimental._candidate_facts 和原有 Manifest 契约，不复制原始轨迹。
- train 复用原来的训练/标定/测试代码和实验开关。
- 在 CarSim 实测确认前，不把任何速度、轮胎极限或制动接口称作已验证。
"""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import re

from .scenario_config import SCENARIOS, validate_scenarios

EXPECTED_SPEEDS = (40, 60, 80, 100, 120, 140, 160, 180)
DEFAULT_ROOT = (
    r"E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve"
)
SPEED_PATTERN = re.compile(r"^(\d{2,3})kmh$", re.IGNORECASE)
MACRO_PATTERN = re.compile(r"\$\(([^)]+)\)\$")
MACRO_LINE = re.compile(
    r"SET_MACRO\s+\$\(([^)]+)\)\$\s+(.+)", re.IGNORECASE
)
MANIFEST_VERSION = "wm_multispeed_speed_capture_v1"
AUDIT_VERSION = "wm_multispeed_sim_audit_v1"
REGISTRY_VERSION = "wm_recovery_experimental_registry_v1"


def sha256(path):
    """计算文件指纹，保证前后检查的是同一份配置。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_save_new(path, obj):
    """只写新文件，避免误覆盖历史实验。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def selected_speeds(values):
    """解析 --speeds 40 60 80 或 --speeds all。"""
    if values is None or values == ["all"]:
        return EXPECTED_SPEEDS
    if "all" in values:
        raise ValueError("--speeds all 不可与其他车速混用")
    nums = tuple(int(x) for x in values)
    if len(set(nums)) != len(nums):
        raise ValueError("不能重复选择同一车速")
    unexpected = set(nums) - set(EXPECTED_SPEEDS)
    if unexpected:
        raise ValueError(f"暂未定义这些速度目录：{sorted(unexpected)}")
    return tuple(sorted(nums))


def discover(root, speeds):
    """在固定目录中依次寻找 {speed}kmh/simfile.sim。"""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"多速度根目录不存在：{base}")
    found = {}
    for speed in speeds:
        path = base / f"{speed}kmh" / "simfile.sim"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {speed} km/h SIM 文件：{path}")
        found[speed] = path.resolve()
    return found


def read_simfile(path):
    """只读解析 SIM 宏，展开 INPUT 和 FILEBASE，不修改 CarSim 数据。"""
    source = Path(path).read_text(
        encoding="utf-8-sig", errors="replace"
    )
    macros, entries = {}, {}
    for raw in source.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = MACRO_LINE.fullmatch(line)
        if match:
            macros[match[1].upper()] = match[2].strip()
            continue
        pieces = line.split(None, 1)
        if len(pieces) == 2:
            entries[pieces[0].upper()] = pieces[1].strip()

    def expand(value):
        value = value or ""
        for _ in range(32):
            after = MACRO_PATTERN.sub(
                lambda m: macros.get(m[1].upper(), m[0]),
                value,
            )
            if after == value:
                break
            value = after
        if MACRO_PATTERN.search(value):
            raise ValueError(f"SIM 存在未解析或递归宏：{value}")
        return value.strip().strip('"')

    result = {
        "sim_path": str(Path(path).resolve()),
        "sim_sha256": sha256(path),
        "root_file_name": expand(macros.get("ROOT_FILE_NAME", "")),
    }
    for key, output in (
        ("INPUT", "input_par_path"),
        ("FILEBASE", "filebase"),
        ("ERDFILE", "erd_file"),
    ):
        result[output] = expand(entries.get(key, ""))

    result["ports_imp_declared"] = entries.get("PORTS_IMP")
    result["ports_exp_declared"] = entries.get("PORTS_EXP")
    result["ext_model_step_declared"] = entries.get("EXT_MODEL_STEP")
    par_string = result["input_par_path"]
    par = Path(par_string)
    if par_string and not par.is_absolute():
        par = Path(path).parent / par
    result["par_exists"] = bool(par_string) and par.is_file()
    result["parameter_sha256"] = (
        sha256(par) if result["par_exists"] else None
    )
    return result


def windows_key(path):
    """Windows 路径不区分大小写；按展开后的实际目标比较。"""
    if not path:
        return ""
    return ntpath.normcase(ntpath.normpath(path.strip().strip('"')))


def audit_records(sim_paths, strict_interface=True):
    """检查实际参数路径/输出路径是否互不冲突，并可严格校验接口。"""
    results = {}
    issues = []
    par_paths = defaultdict(list)
    output_paths = defaultdict(list)
    source_hashes = defaultdict(list)
    par_hashes = defaultdict(list)

    for speed, sim in sorted(sim_paths.items()):
        try:
            item = read_simfile(sim)
            item["expected_speed_kmh"] = int(speed)
            item["actual_initial_speed_kmh"] = None
            item["speed_measured_by_solver"] = False
            results[speed] = item
            if item["ports_imp_declared"] not in ("1,6", "6"):
                issues.append(f"{speed}kmh: PORTS_IMP 声明不是六通道")
            if item["ports_exp_declared"] not in ("1,61", "61"):
                issues.append(f"{speed}kmh: PORTS_EXP 声明不是 61 通道")
            if not item["par_exists"]:
                issues.append(
                    f"{speed}kmh: INPUT 指向的 Run_all.par 不存在："
                    f"{item['input_par_path']}"
                )
            for val, store in (
                (windows_key(item["input_par_path"]), par_paths),
                (windows_key(item["filebase"]), output_paths),
                (item["sim_sha256"], source_hashes),
                (item["parameter_sha256"], par_hashes),
            ):
                if val:
                    store[val].append(speed)

            if strict_interface and item["par_exists"]:
                from .inspection import inspect_simfile
                from .carsim_bridge import verify_inventory
                verify_inventory(inspect_simfile(sim))
                item["interface_check_passed"] = True
            else:
                item["interface_check_passed"] = None
        except Exception as exc:
            issues.append(f"{speed}kmh: 配置解析/接口校验失败：{exc}")

    for label, groups in (
        ("共用同一个 INPUT 参数文件", par_paths),
        ("共用同一个 FILEBASE 输出路径", output_paths),
        ("SIM 文件字节内容相同", source_hashes),
        ("Run_all.par 参数内容相同", par_hashes),
    ):
        for names in groups.values():
            if len(names) > 1:
                issues.append(f"{label}：{names} km/h")

    return {
        "format": AUDIT_VERSION,
        "status": "pass_static" if not issues else "blocked",
        "errors": issues,
        "entries": [results[x] for x in sorted(results)],
        "note": (
            "静态检查不能证明初始车速。必须再用 probe 读取 CarSim Solver 初始 Vx。"
        ),
    }


def show_audit(report):
    print("\n==== SIM 静态检查（不会启动 CarSim） ====")
    for entry in report["entries"]:
        speed = entry["expected_speed_kmh"]
        print(f"\n[{speed} km/h] {entry['sim_path']}")
        print("  INPUT :", entry["input_par_path"])
        print("  PARAM :", "存在" if entry["par_exists"] else "缺失")
        print("  SHA256:", entry["parameter_sha256"])
        print("  OUTPUT:", entry["filebase"])
        print("  I/O   :", entry["ports_imp_declared"],
              entry["ports_exp_declared"])
        print("  REAL V0: 尚未测量")
    if report["errors"]:
        print("\n静态检查未通过：")
        for message in report["errors"]:
            print("  -", message)
    else:
        print("\n静态检查通过，可以继续 probe 核对真实速度。")


def require_audit_good(report):
    if report["errors"]:
        raise RuntimeError(
            "SIM 配置检查未通过，禁止进行批量运行。"
            "请核对每个 Run 的 INPUT/FILEBASE 和接口配置。"
        )


def probe_initial_speeds(sim_paths, audit, tolerance_kmh=1.0,
                         bridge_factory=None):
    """每套 SIM 启动一次 Solver，只读取初始化后的 Vx。"""
    require_audit_good(audit)
    if bridge_factory is None:
        from .carsim_bridge import CarSimBridge
        bridge_factory = CarSimBridge
    if not math.isfinite(tolerance_kmh) or tolerance_kmh <= 0:
        raise ValueError("速度误差容许值必须为正")

    observations = {}
    for speed, path in sorted(sim_paths.items()):
        print(f"\n正在初始化 {speed}kmh 的 CarSim Solver...")
        with bridge_factory(str(path), prediction_dt_s=0.01) as bridge:
            actual = float(bridge.latest.data.Vx)
            if not math.isfinite(actual):
                raise ValueError(f"{speed}kmh: Solver 初始 Vx 非有限数")
            if abs(actual - speed) > tolerance_kmh:
                raise RuntimeError(
                    f"初始车速不匹配：文件夹={speed}kmh，"
                    f"CarSim 实际 Vx={actual:.3f}km/h；"
                    "请在对应 Run 内修改初速并重新生成 SIM/PAR"
                )
            observations[speed] = actual
            print(f"  标称 {speed:>3} km/h -> CarSim {actual:.3f} km/h：匹配")
    return observations


def _audit_and_probe(root, speeds, tolerance_kmh):
    sim_paths = discover(root, speeds)
    report = audit_records(sim_paths, strict_interface=True)
    show_audit(report)
    observations = probe_initial_speeds(
        sim_paths, report, tolerance_kmh=tolerance_kmh
    )
    return sim_paths, report, observations


def verify_completed_speed_folder(speed_dir, meta, expected_speed,
                                  tolerance_kmh=1.0):
    """续跑前核验旧轨迹不是空文件或丢失的占位文件。

    这里不删除任何内容；如果历史数据损坏，明确报错要求先归档。
    """
    import numpy as np
    registry_path = speed_dir / "workspace" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"缺少速度数据集注册表：{registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("episode_count") != len(validate_scenarios()):
        raise ValueError(f"{expected_speed}kmh 轨迹数量不正确")

    for row in registry["episodes"]:
        ep_file = Path(row["episode_path"]) / "experimental_episode.json"
        if not ep_file.is_file():
            raise FileNotFoundError(ep_file)
        manifest = json.loads(ep_file.read_text(encoding="utf-8"))
        candidate = Path(manifest["candidate_path"])
        file = candidate / "candidate.npz"
        report_file = candidate / "candidate_report.json"
        for required in (file, report_file):
            if not required.is_file() or required.stat().st_size == 0:
                raise RuntimeError(
                    f"历史候选文件不存在或为空：{required}；"
                    "请先归档不完整数据，不能直接续跑"
                )
        with np.load(file, allow_pickle=False) as archive:
            state = np.asarray(archive["state_si"], dtype=np.float64)
        initial_kmh = float(state[0, 0] * 3.6)
        if (
            not math.isfinite(initial_kmh)
            or abs(initial_kmh - expected_speed) > tolerance_kmh
        ):
            raise ValueError(
                f"{row['name']} 历史候选数据车速不匹配："
                f"{initial_kmh:.3f} km/h"
            )
        candidate_report = json.loads(
            report_file.read_text(encoding="utf-8")
        )
        if (
            candidate_report.get("capture_sim_sha256")
            != meta["sim_sha256"]
            or candidate_report.get("capture_parameter_sha256")
            != meta["parameter_sha256"]
        ):
            raise ValueError(
                f"{row['name']} 历史候选与当前 SIM/PAR 指纹不一致"
            )
        source = candidate_report.get("raw_source")
        if source:
            raw = Path(source)
            if not raw.is_file() or raw.stat().st_size == 0:
                raise RuntimeError(f"{row['name']} 原始记录已丢失：{raw}")


def capture_speeds(root, output, speeds, tolerance_kmh=1.0,
                   resume=False, allow_high_speed=False):
    """执行逐速度独立的场景采集；每个速度 18 条轨迹 + 2 次转角标定。"""
    if max(speeds) >= 120 and not allow_high_speed:
        raise ValueError(
            "选中了 120km/h 及以上极限操纵仿真，"
            "请先在低速试运行并显式加 --allow-high-speed"
        )
    sim_paths, audit, observations = _audit_and_probe(
        root, speeds, tolerance_kmh
    )

    # 所有文件/初速都确认无误以后，才创建新的采集目录。
    target = Path(output).expanduser().resolve()
    if target.exists() and not resume:
        raise FileExistsError(
            f"输出目录已存在：{target}；续跑请加 --resume，"
            "只允许跳过已经完整完成的车速"
        )
    target.mkdir(parents=True, exist_ok=True)

    from .scenario_runner import capture_all
    for speed, sim in sorted(sim_paths.items()):
        speed_dir = target / f"{speed}kmh"
        marker = speed_dir / "speed_capture_manifest.json"
        entry = next(
            item for item in audit["entries"]
            if item["expected_speed_kmh"] == speed
        )

        if speed_dir.exists():
            if not resume or not marker.is_file():
                raise FileExistsError(
                    f"{speed}kmh 存在不完整或未经核实的采集目录：{speed_dir}。"
                    "请先手动归档它，再用新目录运行；程序不会清空旧文件。"
                )
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if (
                saved.get("format") != MANIFEST_VERSION
                or saved.get("sim_sha256") != entry["sim_sha256"]
                or saved.get("parameter_sha256") != entry["parameter_sha256"]
                or not (speed_dir / "workspace" / "registry.json").is_file()
            ):
                raise ValueError(
                    f"{speed}kmh 旧数据与当前 SIM/PAR 不一致，拒绝跳过"
                )
            verify_completed_speed_folder(
                speed_dir, saved, speed,
                tolerance_kmh=tolerance_kmh
            )
            print(f"跳过已完成的 {speed}kmh，原始记录保持不变")
            continue

        print(f"\n========== 开始 {speed} km/h 的 18 条工况 ==========")
        capture_all(str(sim), speed_dir)

        calibration_file = speed_dir / "steer_calibration.json"
        coverage_file = speed_dir / "coverage_report.json"
        if not calibration_file.is_file() or not coverage_file.is_file():
            raise RuntimeError(f"{speed}kmh: 缺少标定或覆盖率报告")
        calibration = json.loads(
            calibration_file.read_text(encoding="utf-8")
        )
        for direction in ("positive", "negative"):
            measured = float(
                calibration[direction]["initial_speed_kmh"]
            )
            if abs(measured - speed) > tolerance_kmh:
                raise ValueError(
                    f"{speed}kmh: {direction} 标定时实际车速 {measured}km/h"
                    " 与预期不符"
                )

        coverage = json.loads(coverage_file.read_text(encoding="utf-8"))

        # 再次核验每条正式轨迹的初始速度和采集配置指纹。
        # 这样即使采集过程中有人切换 CarSim Run，也不能混入错误车速数据。
        import numpy as np
        registry_file = speed_dir / "workspace" / "registry.json"
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
        if registry.get("episode_count") != len(validate_scenarios()):
            raise ValueError(f"{speed}kmh: 采集完成但轨迹数量不符")
        for item in registry["episodes"]:
            ep_meta = json.loads(
                (Path(item["episode_path"]) / "experimental_episode.json")
                .read_text(encoding="utf-8")
            )
            candidate_path = Path(ep_meta["candidate_path"])
            with np.load(candidate_path / "candidate.npz",
                         allow_pickle=False) as archive:
                state = np.asarray(archive["state_si"], dtype=np.float64)
            first_speed = float(state[0, 0]) * 3.6
            if (
                not math.isfinite(first_speed)
                or abs(first_speed - speed) > tolerance_kmh
            ):
                raise ValueError(
                    f"{speed}kmh/{item['name']}: "
                    f"轨迹实际首帧={first_speed:.3f}km/h，"
                    "与目录标识不符"
                )
            candidate_report = json.loads(
                (candidate_path / "candidate_report.json")
                .read_text(encoding="utf-8")
            )
            if (
                candidate_report.get("capture_sim_sha256")
                != entry["sim_sha256"]
                or candidate_report.get("capture_parameter_sha256")
                != entry["parameter_sha256"]
            ):
                raise ValueError(
                    f"{speed}kmh/{item['name']}: "
                    "原始采集 SIM/PAR 指纹与配置审计不一致"
                )

        marker_data = {
            "format": MANIFEST_VERSION,
            "expected_speed_kmh": speed,
            "measured_initial_speed_kmh": observations[speed],
            "sim_path": str(sim),
            "sim_sha256": entry["sim_sha256"],
            "input_par_path": entry["input_par_path"],
            "parameter_sha256": entry["parameter_sha256"],
            "filebase": entry["filebase"],
            "scene_count": len(validate_scenarios()),
            "coverage_passed": bool(coverage.get("passed")),
            "workspace": str((speed_dir / "workspace").resolve()),
            "note": "实验性质的 CarSim 轨迹，未完成全部物理执行器核验",
        }
        json_save_new(marker, marker_data)
        print(f"{speed}kmh 采集结束；激励覆盖率：{coverage.get('passed')}")

    print("\n所有选择的速度均已完成或经核验跳过。")
    print("下一步运行 merge，建立多车速统一数据集。")
    return target


def merge_workspaces(captures_dir, output_dir, speeds,
                     validation_speed=None, test_speed=None):
    """不复制候选轨迹，只为多车速数据集建立统一 Episode 索引。

    默认：all-speeds（每个速度的 12 train/3 val/3 test）。
    可选：指定两个速度作为完全保留的验证和测试车速，用于外推评估。
    """
    from .carsim_experimental import _candidate_facts, MANIFEST_FORMAT
    from .offline import NOMINAL_MU

    selected = tuple(sorted(speeds))
    holdout = validation_speed is not None or test_speed is not None
    if holdout:
        if validation_speed is None or test_speed is None:
            raise ValueError("速度留出评估必须同时指定验证和测试车速")
        if validation_speed == test_speed:
            raise ValueError("验证与测试车速不能相同")
        if validation_speed not in selected or test_speed not in selected:
            raise ValueError("留出车速必须包含在 --speeds 中")
        if len(selected) < 4:
            raise ValueError("留出两个车速后，至少还需两个训练车速")

    capture_root = Path(captures_dir).expanduser().resolve(strict=True)
    dest = Path(output_dir).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"统一数据集目录已经存在：{dest}")

    # 先全部核对和去重，避免生成半成品 workspace。
    prepared = []
    fingerprint_seen = {}
    provenance = []
    coverage_summary = {}
    for speed in selected:
        speed_dir = capture_root / f"{speed}kmh"
        manifest_file = speed_dir / "speed_capture_manifest.json"
        registry_file = speed_dir / "workspace" / "registry.json"
        coverage_file = speed_dir / "coverage_report.json"
        for file in (manifest_file, registry_file, coverage_file):
            if not file.is_file():
                raise FileNotFoundError(
                    f"{speed}kmh 未完成采集或缺少记录：{file}"
                )
        speed_meta = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            speed_meta.get("format") != MANIFEST_VERSION
            or speed_meta.get("expected_speed_kmh") != speed
        ):
            raise ValueError(f"{speed}kmh 速度信息不合法")
        measured_speed = float(speed_meta["measured_initial_speed_kmh"])
        if abs(measured_speed - speed) > 1.0:
            raise ValueError(f"{speed}kmh 未通过初始车速核验")

        reg = json.loads(registry_file.read_text(encoding="utf-8"))
        if reg.get("format") != REGISTRY_VERSION:
            raise ValueError(f"{speed}kmh 不受支持的原始数据集格式")
        if reg.get("episode_count") != len(validate_scenarios()):
            raise ValueError(f"{speed}kmh 轨迹数量与场景配置不一致")

        coverage = json.loads(coverage_file.read_text(encoding="utf-8"))
        coverage_summary[f"{speed}kmh"] = bool(coverage.get("passed"))

        for row in reg["episodes"]:
            original_episode = Path(row["episode_path"]).resolve()
            meta = json.loads(
                (original_episode / "experimental_episode.json")
                .read_text(encoding="utf-8")
            )
            original_split = meta["split"]
            if holdout:
                if speed == validation_speed:
                    if original_split != "validation":
                        continue
                    split = "validation"
                elif speed == test_speed:
                    if original_split != "test":
                        continue
                    split = "test"
                else:
                    if original_split != "train":
                        continue
                    split = "train"
            else:
                split = original_split

            candidate = Path(meta["candidate_path"]).resolve(strict=True)
            facts = _candidate_facts(candidate)
            for field in ("candidate_sha256", "raw_source_sha256"):
                if facts[field] != meta.get(field):
                    raise ValueError(
                        f"原始清单指纹与物理数据不符：{candidate}"
                    )
                key = (field, facts[field])
                if key in fingerprint_seen:
                    raise ValueError(
                        f"两条跨车速轨迹复用了同一数据："
                        f"{fingerprint_seen[key]} 和 {speed}kmh/{row['name']}"
                    )
                fingerprint_seen[key] = f"{speed}kmh/{row['name']}"

            # 检查采集时的 SIM 指纹和本速度记录是否一致。
            candidate_meta = facts["candidate"].metadata
            if (
                candidate_meta.get("capture_sim_sha256")
                != speed_meta.get("sim_sha256")
                or candidate_meta.get("capture_parameter_sha256")
                != speed_meta.get("parameter_sha256")
            ):
                raise ValueError(
                    f"{speed}kmh/{row['name']} 的 SIM/PAR 配置指纹不一致"
                )

            new_name = f"v{speed}_{row['name']}"
            rewritten = dict(meta)
            rewritten.update({
                "format": MANIFEST_FORMAT,
                "trajectory_id": f"carsim_experimental_{new_name}",
                "condition_id": f"v{speed}_{meta['condition_id']}",
                "split": split,
                "actual_mu": NOMINAL_MU,
                "candidate_path": str(candidate),
                "candidate_sha256": facts["candidate_sha256"],
                "raw_source_sha256": facts["raw_source_sha256"],
                "assumption_acknowledged": True,
                "source_kind": "carsim_experimental",
                "ready_for_wm_training": False,
                "initial_speed_kmh": measured_speed,
                "expected_initial_speed_kmh": speed,
                "capture_sim_sha256": speed_meta["sim_sha256"],
                "capture_parameter_sha256": speed_meta["parameter_sha256"],
            })
            prepared.append((new_name, rewritten))
        provenance.append({
            "speed_kmh": speed,
            "sim_path": speed_meta["sim_path"],
            "sim_sha256": speed_meta["sim_sha256"],
            "par_sha256": speed_meta["parameter_sha256"],
            "coverage_passed": coverage_summary[f"{speed}kmh"],
        })

    counts = Counter(meta["split"] for _, meta in prepared)
    if counts["train"] < 2 or counts["validation"] < 1 or counts["test"] < 1:
        raise ValueError(f"数据分区不足：{dict(counts)}")
    if len({meta["condition_id"] for _, meta in prepared}) != len(prepared):
        raise ValueError("condition_id 发生重复")

    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, meta in prepared:
        episode_dir = dest / "episodes" / name
        episode_dir.mkdir(parents=True, exist_ok=False)
        json_save_new(episode_dir / "experimental_episode.json", meta)
        rows.append({
            "name": name,
            "split": meta["split"],
            "condition_id": meta["condition_id"],
            "episode_path": str(episode_dir.resolve()),
            "candidate_sha256": meta["candidate_sha256"],
            "raw_source_sha256": meta["raw_source_sha256"],
            "initial_speed_kmh": meta["initial_speed_kmh"],
        })

    registry = {
        "format": REGISTRY_VERSION,
        "status": "experimental_not_physically_validated",
        "explicit_opt_in_env": "WM_RECOVERY_EXPERIMENTAL_CARSIM",
        "no_cross_split_raw_duplicates": True,
        "nominal_mu": NOMINAL_MU,
        "episode_count": len(rows),
        "episodes": rows,
        "ready_for_wm_training": False,
        "speed_split_policy": (
            "speed_holdout" if holdout else "all_speeds_seen"
        ),
        "speed_provenance": provenance,
        "partition_counts": dict(counts),
        "note": (
            "同一速度可有不同转向工况；all_speeds_seen 测试并不等于"
            "未见过车速的泛化测试。严格物理核验尚未完成。"
        ),
    }
    json_save_new(dest / "registry.json", registry)

    # 借助本来就存在的 train_from_workspace 覆盖率门槛机制。
    from .scenario_runner import COVERAGE_FORMAT
    coverage = {
        "format": COVERAGE_FORMAT,
        "passed": all(coverage_summary.values()),
        "criteria_origin": "each_speed_project_defined_coverage_gate",
        "speed_coverage_passed": coverage_summary,
        "note": (
            "合并数据的 passed=True 仅表示各速度批次满足项目自定义的"
            "横向激励覆盖门槛，不是 ISO 认证或轮胎极限证明。"
        ),
    }
    json_save_new(dest / "multispeed_coverage_summary.json", coverage)
    # 由新 train 命令检查此文件，旧训练管线仍保持原有数据契约。
    print("\n多速度 WM 数据集创建完成：", dest)
    print("  数据条数：", len(rows))
    print("  Train/Validation/Test：", dict(counts))
    print("  每个车速激励覆盖率：", coverage_summary)
    return dest


def train_merged(workspace, output, epochs,
                 allow_underexcited=False):
    """检查跨速度覆盖率后复用现有 WM 训练/标定/测试流水线。"""
    workspace = Path(workspace).expanduser().resolve(strict=True)
    path = workspace / "multispeed_coverage_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少多速度覆盖率汇总报告：{path}"
        )
    coverage = json.loads(path.read_text(encoding="utf-8"))
    if not coverage.get("passed") and not allow_underexcited:
        raise RuntimeError(
            "至少一个车速未达到项目自定义的近极限激励覆盖率要求；"
            "请查看 multispeed_coverage_summary.json。"
            "如仅调试训练管线，显式指定 --allow-underexcited。"
        )
    from .scenario_runner import train_from_workspace
    return train_from_workspace(
        workspace, output, epochs,
        allow_underexcited=allow_underexcited,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="多初始车速 CarSim 配置核验/批量极限场景/WM 训练"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    common_root = argparse.ArgumentParser(add_help=False)
    common_root.add_argument("--root", default=DEFAULT_ROOT)
    common_root.add_argument(
        "--speeds", nargs="+", default=["all"],
        help="all 或 40 60 80 ... 180",
    )

    audit = sub.add_parser("audit", parents=[common_root],
                           help="只读检查八份 SIM/PAR 和输出路径")
    audit.add_argument("--output", type=Path, default=None)

    probe = sub.add_parser("probe", parents=[common_root],
                           help="真实初始化各 Solver，测量初始 Vx")
    probe.add_argument("--tolerance-kmh", type=float, default=1.0)
    probe.add_argument("--output", type=Path, default=None)

    capture = sub.add_parser("capture", parents=[common_root],
                             help="逐速度标定并采集 18 条极限场景")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--tolerance-kmh", type=float, default=1.0)
    capture.add_argument("--allow-high-speed", action="store_true")
    capture.add_argument("--resume", action="store_true")

    merge = sub.add_parser("merge", help="合并已完成的多车速 workspace")
    merge.add_argument("--captures", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--speeds", nargs="+", default=["all"])
    merge.add_argument("--validation-speed", type=int)
    merge.add_argument("--test-speed", type=int)

    train = sub.add_parser("train", help="训练、标定和独立测试")
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--allow-underexcited", action="store_true")

    args = parser.parse_args(argv)
    if args.action in {"audit", "probe", "capture", "merge"}:
        speeds = selected_speeds(args.speeds)

    if args.action == "audit":
        report = audit_records(
            discover(args.root, speeds), strict_interface=True
        )
        show_audit(report)
        if args.output:
            json_save_new(args.output, report)
        require_audit_good(report)

    elif args.action == "probe":
        paths, report, actual = _audit_and_probe(
            args.root, speeds, args.tolerance_kmh
        )
        if args.output:
            json_save_new(args.output, {
                "format": "wm_multispeed_initial_speed_probe_v1",
                "expected_to_actual_speed_kmh": actual,
                "sim_sha256_by_speed": {
                    str(speed): sha256(path)
                    for speed, path in paths.items()
                },
            })

    elif args.action == "capture":
        capture_speeds(
            args.root, args.output, speeds,
            tolerance_kmh=args.tolerance_kmh,
            resume=args.resume,
            allow_high_speed=args.allow_high_speed,
        )

    elif args.action == "merge":
        merge_workspaces(
            args.captures, args.output, speeds,
            validation_speed=args.validation_speed,
            test_speed=args.test_speed,
        )

    elif args.action == "train":
        train_merged(
            args.workspace, args.output, args.epochs,
            allow_underexcited=args.allow_underexcited,
        )


if __name__ == "__main__":
    main()
