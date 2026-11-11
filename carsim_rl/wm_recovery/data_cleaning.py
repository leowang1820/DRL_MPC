# -*- coding: utf-8 -*-
"""CarSim 多车速 World Model：采集完成后的离线质量清洗。

用法：
    python -m wm_recovery.data_cleaning audit \
        --workspace artifacts/wm_recovery/workspace_40_60_01

    python -m wm_recovery.data_cleaning clean \
        --workspace artifacts/wm_recovery/workspace_40_60_01 \
        --output artifacts/wm_recovery/workspace_40_60_clean_01

输入：multi_speed_runner merge 生成的 workspace。
输出：新的 workspace，含原样分区、清洗记录和前缀裁剪的候选数据。
不会触碰原始 raw_capture.npz、candidate.npz 或原始 workspace。

清洗规则：
1. 过滤 Vx < min_vx_kmh（默认 0，排除反向车身纵向速度）。
2. 过滤 |atan2(Vy,Vx)| > beta_limit_deg（默认 15°）。
3. 实际左右前轮角度异常速率或加速度：默认分别为 400deg/s
   和 20000deg/s²，属于待校准的「启发式质量阈值」，不是标准规定。
4. 实际四轮外加转矩速率阈值默认关闭，因为此通道通常回显我们
   有意设计的尖锐指令，直接把高变化率判成噪声容易误删激励。

重要：由于 7DOF 的 LoadMemory 沿轨迹连续传播，默认只保留
第一处违规之前的连续前缀，不拼接不连续区间，也不重置记忆伪造训练段。
裁剪后的测试数据不能证明模型可在被剔除的失稳状态中工作。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FORMAT = "wm_recovery_postcapture_quality_clean_v1"
REGISTRY_FORMAT = "wm_recovery_experimental_registry_v1"
DT_S = 0.01
G = 9.80665


@dataclass(frozen=True)
class QualityPolicy:
    min_vx_kmh: float = 0.0
    beta_limit_deg: float = 15.0
    max_steer_rate_deg_s: float = 400.0
    max_steer_accel_deg_s2: float = 20000.0
    max_external_torque_rate_nm_s: float | None = None
    guard_s: float = 0.05
    min_keep_s: float = 1.20

    def validate(self):
        for key in (
            "min_vx_kmh", "beta_limit_deg", "max_steer_rate_deg_s",
            "max_steer_accel_deg_s2", "guard_s", "min_keep_s",
        ):
            val = float(getattr(self, key))
            if not math.isfinite(val):
                raise ValueError(f"{key} 必须为有限数字")
        if self.beta_limit_deg <= 0 or self.beta_limit_deg >= 90:
            raise ValueError("beta_limit_deg 必须在 (0,90) 范围")
        if self.max_steer_rate_deg_s <= 0 or self.max_steer_accel_deg_s2 <= 0:
            raise ValueError("执行器速率和加速度阈值必须大于 0")
        if self.min_keep_s < 1.01 or self.guard_s < 0:
            raise ValueError("至少保留 1.01s 且 guard_s 不能为负")
        if self.max_external_torque_rate_nm_s is not None:
            if not (
                math.isfinite(self.max_external_torque_rate_nm_s)
                and self.max_external_torque_rate_nm_s > 0
            ):
                raise ValueError("转矩速率阈值必须为正的有限数字")


def policy_as_dict(policy):
    return dict(vars(policy))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_json_new(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def _array_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key], dtype=np.float64).copy()
                for key in z.files}


def analyze_candidate_arrays(arrays, policy):
    """返回有问题的区间 mask 和安全前缀长度，绝不在原数组中删点。"""
    policy.validate()
    state = arrays["state_si"]
    steer = arrays["measured_steer_rad"]
    torque = arrays["reported_external_torque_nm"]
    time_s = arrays["time_s"]
    command = arrays["command_si"]
    n = len(command)
    if (
        state.shape != (n + 1, 7)
        or steer.shape != (n + 1, 2)
        or torque.shape != (n + 1, 4)
        or len(time_s) != n + 1
    ):
        raise ValueError("候选数组形状与 CarSim 的状态/区间对齐约定不一致")
    dt = np.diff(time_s)
    if not np.allclose(dt, DT_S, rtol=0, atol=1e-8):
        raise ValueError("清洗仅支持完整的 0.01 秒 CarSim 时间栅格")
    if any(
        not np.all(np.isfinite(array)) for array in arrays.values()
    ):
        raise ValueError("原始候选含非有限数；禁止静默过滤此异常")

    vx_m_s = state[:, 0]
    vy_m_s = state[:, 1]
    beta_deg = np.degrees(np.arctan2(vy_m_s, vx_m_s))
    bad_vx = (vx_m_s * 3.6 < policy.min_vx_kmh)
    bad_beta = (np.abs(beta_deg) > policy.beta_limit_deg)
    bad_state = bad_vx | bad_beta

    # measured_steer_rad 是 CarSim 左右前轮各自输出，不是下发的共同指令。
    steer_deg = np.rad2deg(steer)
    steer_rate = np.diff(steer_deg, axis=0) / dt[:, None]
    bad_steer_rate = np.any(
        np.abs(steer_rate) > policy.max_steer_rate_deg_s,
        axis=1,
    )

    # 两次相邻转角速度之差 / dt；不能把正常 SWD/step 误当作噪声。
    steer_accel = np.diff(steer_rate, axis=0) / DT_S
    bad_steer_accel = np.zeros(n, dtype=bool)
    if len(steer_accel):
        bad_steer_accel[1:] = np.any(
            np.abs(steer_accel) > policy.max_steer_accel_deg_s2,
            axis=1,
        )

    # 外加转矩通常近似等于 Import 的回显；先统计，不强制判异常。
    torque_rate = np.diff(torque, axis=0) / dt[:, None]
    bad_torque_rate = np.zeros(n, dtype=bool)
    if policy.max_external_torque_rate_nm_s is not None:
        bad_torque_rate = np.any(
            np.abs(torque_rate) > policy.max_external_torque_rate_nm_s,
            axis=1,
        )

    reasons = {
        "vx_too_low": bad_vx[:-1] | bad_vx[1:],
        "beta_too_large": bad_beta[:-1] | bad_beta[1:],
        "steer_rate_too_large": bad_steer_rate,
        "steer_accel_too_large": bad_steer_accel,
        "torque_rate_too_large": bad_torque_rate,
    }
    bad_interval = np.zeros(n, dtype=bool)
    for mask in reasons.values():
        bad_interval |= mask

    first_bad = (
        int(np.flatnonzero(bad_interval)[0])
        if np.any(bad_interval) else n
    )
    guard_steps = int(math.ceil(policy.guard_s / DT_S - 1e-9))
    # 完全正常时不扣保护余量；只有真的出现违规时才提前退出。
    keep = n if first_bad == n else max(0, first_bad - guard_steps)
    min_keep = int(math.ceil(policy.min_keep_s / DT_S - 1e-9))

    return {
        "original_intervals": n,
        "first_bad_interval": first_bad if first_bad < n else None,
        "first_bad_time_s": float(time_s[first_bad]) if first_bad < n else None,
        "keep_intervals": int(keep),
        "keep_duration_s": float(keep * DT_S),
        "excluded_intervals": int(n - keep),
        "kept": bool(keep >= min_keep),
        "max_abs_beta_deg": float(np.max(np.abs(beta_deg))),
        "min_vx_kmh": float(np.min(vx_m_s) * 3.6),
        "max_abs_steer_rate_deg_s": float(np.max(np.abs(steer_rate))),
        "max_abs_steer_accel_deg_s2": (
            float(np.max(np.abs(steer_accel))) if len(steer_accel) else 0.0
        ),
        "max_abs_external_torque_rate_nm_s": float(
            np.max(np.abs(torque_rate))
        ),
        "reason_counts": {
            key: int(mask.sum()) for key, mask in reasons.items()
        },
        "first_bad_reasons": [
            key for key, mask in reasons.items()
            if first_bad < n and mask[first_bad]
        ],
    }


def _trim_arrays(arrays, keep):
    state_fields = {
        "time_s", "state_si", "measured_steer_rad",
        "reported_external_torque_nm", "reported_accel_mps2",
        "my_dr_native",
    }
    interval_fields = {"command_si", "nominal_7dof_input_si", "import_native"}
    other_fields = {"nominal_initial_load_memory_mps2"}
    if set(arrays) != state_fields | interval_fields | other_fields:
        raise ValueError("候选数组字段列表与已验证格式不一致")
    result = {}
    for name, a in arrays.items():
        if name in state_fields:
            result[name] = a[:keep + 1].copy()
        elif name in interval_fields:
            result[name] = a[:keep].copy()
        else:
            result[name] = a.copy()
    return result


def _candidate_hash(arrays):
    from .carsim_candidate import _fingerprint
    return _fingerprint(arrays)


def _candidate_report_after_trim(original, arrays, keep, source_sha, policy):
    report = dict(original)
    report["samples"] = keep + 1
    report["intervals"] = keep
    report["canonical_physical_arrays_sha256"] = _candidate_hash(arrays)
    diff = arrays["measured_steer_rad"][:, 0] - arrays["measured_steer_rad"][:, 1]
    report["max_measured_steer_difference_rad"] = float(
        np.max(np.abs(diff))
    )
    report["max_measured_steer_difference_deg"] = float(
        np.rad2deg(np.max(np.abs(diff)))
    )
    torque_error = float(np.max(np.abs(
        arrays["reported_external_torque_nm"][1:]
        - arrays["import_native"][:, :4]
    )))
    report["max_torque_echo_error_nm"] = torque_error
    report["torque_echo_matches"] = bool(torque_error <= 1e-7)
    # 严格沿用 CandidateEpisode.load() 的候选数据版本契约：
    # 状态必须保持 candidate_created_not_training_ready，
    # 不能为了标记已清洗而改变它（否则校验器会拒绝读取）。
    if report.get("status") != "candidate_created_not_training_ready":
        raise ValueError(
            "原始 candidate_report 的状态与严格 V1 契约不一致"
        )
    report["ready_for_wm_training"] = False
    report["quality_cleaned"] = True
    report["quality_parent_candidate_sha256"] = source_sha
    report["quality_policy"] = policy_as_dict(policy)
    report["quality_keep_intervals"] = keep
    report["note"] = (
        "沿原始 CarSim 连续轨迹保留前缀，未拼接或重采样；"
        "该候选仍属于实验性质，不能声称通过物理接口验证。"
    )
    return report


def _split_speed(name):
    import re
    match = re.match(r"^v(\d+)_(.+)$", str(name))
    if not match:
        raise ValueError(
            f"场景名称缺少 v40_ / v60_ 这样的多车速前缀：{name}"
        )
    return int(match.group(1)), match.group(2)


def _coverage_summary(quality_rows, source_registry, output_dir):
    """基于清洗后保留的真实 Ay 重新计算覆盖率，而非沿用未清洗结果。"""
    from .scenario_runner import summarize_coverage

    # 读取采集当时的 coverage_report，而不是查询以后可能已改过的 SCENARIOS。
    # 防止在清洗旧数据时，把修改后的新场景定义误认为原始实验条件。
    per_speed = defaultdict(list)
    all_reports = []
    for row in quality_rows:
        if not row["kept"]:
            continue
        speed, scene_name = _split_speed(row["name"])
        source_candidate = Path(row["source_candidate_path"])
        source_coverage_file = (
            source_candidate.parent / "coverage_report.json"
        )
        if not source_coverage_file.is_file():
            raise FileNotFoundError(
                f"缺少采集时保存的覆盖率报告：{source_coverage_file}"
            )
        source_coverage = json.loads(
            source_coverage_file.read_text(encoding="utf-8")
        )
        source_matches = [
            item for item in source_coverage.get("scenarios", [])
            if item.get("scenario") == scene_name
        ]
        if len(source_matches) != 1:
            raise ValueError(
                f"原始 coverage_report 不能唯一找到 {scene_name} 的采集记录"
            )
        capture_row = source_matches[0]
        if capture_row.get("split") != row["split"]:
            raise ValueError(
                f"{scene_name} 的原始训练分区与当前 workspace 不符"
            )

        cand_path = Path(row["clean_candidate_path"])
        data = _array_dict(cand_path / "candidate.npz")
        ay_g = data["reported_accel_mps2"][1:, 1] / G
        init_ay_g = data["reported_accel_mps2"][0, 1] / G
        ay_abs = np.abs(ay_g - init_ay_g)
        near_count = int((ay_abs >= (0.75 * 0.8)).sum())
        strong_count = int((ay_abs >= (0.90 * 0.8)).sum())
        summary = {
            "scenario": row["name"],
            "split": row["split"],
            "sign": int(capture_row["sign"]),
            "peak_abs_ay_g": float(np.max(ay_abs)),
            "duration_ay_ge_0p75_mu_s": near_count * DT_S,
            "duration_ay_ge_0p90_mu_s": strong_count * DT_S,
        }
        per_speed[speed].append(summary)
        all_reports.append(summary)

    if not all_reports:
        raise ValueError("清洗后无任何有效轨迹")
    gate = summarize_coverage(all_reports)
    speed_summaries = {}
    speed_coverage_passed = {}

    # 保留原 multi_speed_runner 的 speed_coverage_passed 结构；
    # all_speeds_seen 时，每个车速都必须独立通过同一组覆盖率指标，
    # 防止某一车速的高激励掩盖另一车速覆盖不足。
    for speed, records in per_speed.items():
        speed_gate = summarize_coverage(records)
        speed_key = f"{speed}kmh"
        speed_coverage_passed[speed_key] = bool(speed_gate["passed"])
        speed_summaries[speed_key] = {
            "retained_episodes": len(records),
            "retained_splits": dict(Counter(r["split"] for r in records)),
            "coverage_passed": bool(speed_gate["passed"]),
            "coverage_checks": speed_gate["checks"],
        }

    # 对留出完整速度作为 validation/test 的实验，单一速度不包含
    # 所有 split，不能强制套用“单速度 12/3/3”的门槛；
    # 这时只报告分速度覆盖率，但总体 passed 以合并数据的 gate 为准。
    split_policy = source_registry.get("speed_split_policy", "all_speeds_seen")
    require_per_speed = (split_policy == "all_speeds_seen")
    expected_speeds = {
        f"{int(item['speed_kmh'])}kmh"
        for item in source_registry.get("speed_provenance", [])
    }
    missing_speeds = sorted(expected_speeds - set(speed_coverage_passed))
    for speed_key in missing_speeds:
        speed_coverage_passed[speed_key] = False
        speed_summaries[speed_key] = {
            "retained_episodes": 0,
            "retained_splits": {},
            "coverage_passed": False,
            "coverage_checks": {},
        }
    passed = bool(gate["passed"]) and (
        all(speed_coverage_passed.values()) if require_per_speed else True
    )
    result = {
        "format": gate["format"],
        "passed": passed,
        "checks": gate["checks"],
        "criteria_origin": "postcapture_cleaned_project_defined_coverage_gate",
        "speed_split_policy": split_policy,
        "speed_coverage_passed": speed_coverage_passed,
        "source": "postcapture_quality_cleaned_candidate_prefixes",
        "quality_policy": "详见 quality_report.json",
        "speed_retained": speed_summaries,
        "warning": (
            "passed 表示清洗后符合项目自定义的横向加速度代理指标，"
            "不等于轮胎极限、有效控制可恢复性或物理接口验证。"
        ),
    }
    _save_json_new(output_dir / "multispeed_coverage_summary.json", result)
    return result


def audit_or_clean(workspace, output, policy, write=False):
    """audit：只打印审计；clean：创建新的可用于 WM train 的 workspace。"""
    from .carsim_experimental import _candidate_facts

    policy.validate()
    workspace = Path(workspace).expanduser().resolve(strict=True)
    regpath = workspace / "registry.json"
    registry = json.loads(regpath.read_text(encoding="utf-8"))
    if registry.get("format") != REGISTRY_FORMAT:
        raise ValueError("输入必须是已合并的实验性多车速 workspace")
    dest = Path(output).expanduser().resolve() if write else None
    if write and dest.exists():
        raise FileExistsError(f"输出目录已存在：{dest}；拒绝覆盖")
    original_items = []
    for row in registry["episodes"]:
        name = row["name"]
        ep_path = Path(row["episode_path"])
        episode_meta = json.loads(
            (ep_path / "experimental_episode.json").read_text(encoding="utf-8")
        )
        cand_path = Path(episode_meta["candidate_path"]).resolve(strict=True)
        facts = _candidate_facts(cand_path)
        if (
            episode_meta["candidate_sha256"] != facts["candidate_sha256"]
            or episode_meta["raw_source_sha256"] != facts["raw_source_sha256"]
        ):
            raise ValueError(f"{name} 的候选来源指纹不匹配")
        arrays = _array_dict(cand_path / "candidate.npz")
        result = analyze_candidate_arrays(arrays, policy)
        original_items.append({
            "name": name,
            "split": episode_meta["split"],
            "episode_meta": episode_meta,
            "candidate_path": str(cand_path),
            "candidate_meta": facts["candidate"].metadata,
            "source_candidate_sha256": facts["candidate_sha256"],
            "raw_source_sha256": facts["raw_source_sha256"],
            "quality": result,
        })
        print(
            f"{name:34} | {result['keep_duration_s']:>5.2f}s / "
            f"{result['original_intervals']*DT_S:>5.2f}s"
            f" | {'保留' if result['kept'] else '剔除'}"
            f" | 首次违规={result['first_bad_time_s']}"
            f" | 原因={','.join(result['first_bad_reasons']) or '-'}"
        )

    accepted = [it for it in original_items if it["quality"]["kept"]]
    rejected = [it for it in original_items if not it["quality"]["kept"]]
    splits = Counter(it["split"] for it in accepted)
    summary = {
        "format": FORMAT,
        "policy": policy_as_dict(policy),
        "source_workspace": str(workspace),
        "input_episodes": len(original_items),
        "retained_episodes": len(accepted),
        "rejected_episodes": len(rejected),
        "retained_splits": dict(splits),
        "total_raw_intervals": sum(
            it["quality"]["original_intervals"] for it in original_items
        ),
        "total_kept_intervals": sum(
            it["quality"]["keep_intervals"] for it in accepted
        ),
        "episodes": [
            {
                "name": it["name"], "split": it["split"],
                "source_candidate_sha256": it["source_candidate_sha256"],
                "source_raw_sha256": it["raw_source_sha256"],
                **it["quality"],
            }
            for it in original_items
        ],
        "note": (
            "只读源文件。清洗时仅生成独立的连续前缀候选，不重排、不插值。"
            "由于截断后近极限时间可能减少，覆盖率必须重新计算。"
        ),
    }

    print(
        f"\n结果：保留 {len(accepted)}/{len(original_items)} 条轨迹，"
        f"有效区间 {summary['total_kept_intervals']}/"
        f"{summary['total_raw_intervals']}。"
    )
    if not write:
        return summary

    if splits["train"] < 2 or splits["validation"] < 1 or splits["test"] < 1:
        raise ValueError(
            "清洗后分区不足（至少 2 train、1 validation、1 test）；"
            "请调整数据采集工况而非绕过检查"
        )
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    source_hashes = set()
    clean_hashes = set()
    quality_rows = []

    for item in accepted:
        name = item["name"]
        quality = item["quality"]
        old_path = Path(item["candidate_path"])
        old_meta = item["episode_meta"]
        cutoff = quality["keep_intervals"]

        if cutoff == quality["original_intervals"]:
            # 无违规：直接引用原候选，不复制、不改动。
            new_cand_path = old_path
            new_hash = item["source_candidate_sha256"]
        else:
            # 有违规：创建新候选，所有状态与命令裁剪到同一个连续时间前缀。
            new_cand_path = dest / "clean_candidates" / name
            new_cand_path.mkdir(parents=True, exist_ok=False)
            arrays = _array_dict(old_path / "candidate.npz")
            clean_arrays = _trim_arrays(arrays, cutoff)
            new_report = _candidate_report_after_trim(
                item["candidate_meta"], clean_arrays, cutoff,
                item["source_candidate_sha256"], policy
            )
            if new_report["torque_echo_matches"] is not True:
                raise RuntimeError(f"{name} 裁剪后转矩回显不一致")
            np.savez_compressed(
                new_cand_path / "candidate.npz", **clean_arrays
            )
            _save_json_new(
                new_cand_path / "candidate_report.json", new_report
            )
            # 用原有严格格式读取器二次核验，禁止只靠写文件宣称成功。
            new_facts = _candidate_facts(new_cand_path)
            new_hash = new_facts["candidate_sha256"]

        if (new_hash in clean_hashes
                or item["raw_source_sha256"] in source_hashes):
            raise ValueError(
                f"{name} 清洗后与其他训练/验证/测试数据存在来源重复"
            )
        clean_hashes.add(new_hash)
        source_hashes.add(item["raw_source_sha256"])

        ep_out = dest / "episodes" / name
        ep_out.mkdir(parents=True, exist_ok=False)
        new_manifest = dict(old_meta)
        new_manifest.update({
            "candidate_path": str(new_cand_path),
            "candidate_sha256": new_hash,
            "raw_source_sha256": item["raw_source_sha256"],
            "quality_cleaned": True,
            "quality_keep_intervals": cutoff,
            "quality_original_candidate_sha256":
                item["source_candidate_sha256"],
            "ready_for_wm_training": False,
            "note": (
                "实验性清洗前缀，仅限用户明确许可的 WM 实验；"
                "未核验实际净转矩/制动执行器"
            ),
        })
        _save_json_new(
            ep_out / "experimental_episode.json", new_manifest
        )
        row = dict(
            name=name,
            split=old_meta["split"],
            condition_id=old_meta["condition_id"],
            episode_path=str(ep_out),
            candidate_sha256=new_hash,
            raw_source_sha256=item["raw_source_sha256"],
        )
        rows.append(row)
        quality_rows.append({
            "name": name,
            "split": old_meta["split"],
            "kept": True,
            "source_candidate_path": item["candidate_path"],
            "clean_candidate_path": str(new_cand_path),
        })

    reg_out = dict(registry)
    reg_out.update({
        "episodes": rows,
        "episode_count": len(rows),
        "partition_counts": dict(splits),
        "quality_cleaned": True,
        "quality_policy": policy_as_dict(policy),
        "quality_source_workspace": str(workspace),
        "ready_for_wm_training": False,
        # 原 registry 的状态是现有训练路径的契约，保持原值不变。
        "status": registry.get("status", "experimental_not_physically_validated"),
    })
    _save_json_new(dest / "registry.json", reg_out)
    _save_json_new(dest / "quality_report.json", summary)
    coverage = _coverage_summary(quality_rows, registry, dest)
    print("\n清洗后的 Workspace：", dest)
    print("清洗后横向激励覆盖率：", coverage["passed"])
    print("注意：模型只在保留的时间区间内训练/评价。")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CarSim 多车速 WM：采集后数据质量审计/清洗"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for action in ("audit", "clean"):
        p = sub.add_parser(action)
        p.add_argument("--workspace", required=True, type=Path)
        if action == "clean":
            p.add_argument("--output", required=True, type=Path)
        p.add_argument("--min-vx-kmh", type=float, default=0.0)
        p.add_argument("--beta-limit-deg", type=float, default=15.0)
        p.add_argument("--max-steer-rate-deg-s", type=float, default=400.0)
        p.add_argument("--max-steer-accel-deg-s2", type=float, default=20000.0)
        p.add_argument("--max-torque-rate-nm-s", type=float, default=None)
        p.add_argument("--guard-s", type=float, default=0.05)
        p.add_argument("--min-keep-s", type=float, default=1.20)
    args = parser.parse_args(argv)
    policy = QualityPolicy(
        min_vx_kmh=args.min_vx_kmh,
        beta_limit_deg=args.beta_limit_deg,
        max_steer_rate_deg_s=args.max_steer_rate_deg_s,
        max_steer_accel_deg_s2=args.max_steer_accel_deg_s2,
        max_external_torque_rate_nm_s=args.max_torque_rate_nm_s,
        guard_s=args.guard_s,
        min_keep_s=args.min_keep_s,
    )
    audit_or_clean(
        args.workspace,
        args.output if args.mode == "clean" else None,
        policy,
        write=(args.mode == "clean"),
    )


if __name__ == "__main__":
    main()
