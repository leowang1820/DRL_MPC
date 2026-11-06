"""Causal windows and trajectory bootstrap; no cross-trajectory histories."""

from dataclasses import dataclass
import os

import numpy as np

from dynamics7dof.model import Reference7DOF

from .offline import Episode, NOMINAL_MU, audit_partition, controls_at


FEATURE_FLOOR = np.array([.1, .02, .01, .2, .2, .2, .2, .001, .001, 10, 10, 10, 10])
STATE_FLOOR = FEATURE_FLOOR[:7]
RESIDUAL_FLOOR = np.array([.001, .001, .0001, .02, .02, .02, .02])


def feature_vector(state, applied, history, memory):
    return np.concatenate((state, applied, history.reshape(-1), memory), axis=-1)


@dataclass
class PreparedEpisode:
    episode: Episode
    memory: np.ndarray
    nominal_next: np.ndarray
    valid: np.ndarray
    starts: np.ndarray

    def history_at(self, index, steps):
        return np.concatenate((self.episode.state[index - steps:index], self.episode.model_input[index - steps:index]), axis=1)


def prepare_episode(episode, config):
    if not np.allclose(np.diff(episode.time), config.prediction_dt_s, rtol=0, atol=1e-8):
        raise ValueError("Trajectory grid differs from prediction_dt_s; resampling requires an explicit input-timing policy")
    model = Reference7DOF()
    memory = episode.initial_memory
    memories, nominal, valid = [], [], []
    for i in range(len(episode.time) - 1):
        memories.append([memory.ax_m_s2, memory.ay_m_s2])
        p = model.predict(episode.state[i], controls_at(episode, i), mu=NOMINAL_MU,
                          load_memory=memory, horizon_s=config.prediction_dt_s, substeps=config.substeps)
        nominal.append(p.state)
        valid.append(not p.flags)
        memory = p.load_memory
    valid = np.asarray(valid, dtype=bool)
    # Once nominal load memory is invalid, subsequent anchors are also unverified.
    valid = np.logical_and.accumulate(valid)
    starts = np.array([i for i in range(config.history_steps, len(valid) - config.rollout_steps + 1)
                       if valid[i - config.history_steps:i + config.rollout_steps].all()], dtype=int)
    if not len(starts):
        raise ValueError(f"No valid causal windows in trajectory {episode.metadata['trajectory_id']}")
    return PreparedEpisode(episode, np.asarray(memories), np.asarray(nominal), valid, starts)


def load_partition(paths, config, split, *, allow_synthetic=False, excluded_ids=(), excluded_hashes=(), excluded_conditions=()):
    episodes = [Episode.load(path) for path in paths]
    if not episodes:
        raise ValueError("At least one trajectory is required")
    audit_partition(episodes)

    # 单一分区不得混用严格 V1 与实验性命令近似轨迹。
    experimental = [
        bool(ep.metadata.get("experimental_training_only")) for ep in episodes
    ]
    if any(experimental):
        if not all(experimental):
            raise ValueError("不能在一个分区里混用严格 V1 和实验性 CarSim 轨迹")
        if os.getenv("WM_RECOVERY_EXPERIMENTAL_CARSIM") != "1":
            raise ValueError("实验性 CarSim 数据必须显式开启环境开关")

    for episode in episodes:
        meta = episode.metadata
        if meta["split"] != split:
            raise ValueError(f"Expected {split} trajectory, got {meta['split']}")
        if (meta["trajectory_id"] in excluded_ids or episode.source_sha256 in excluded_hashes
                or meta["condition_id"] in excluded_conditions):
            raise ValueError("Evaluation/calibration trajectory overlaps a previously used partition")
        if meta["source_kind"] == "synthetic":
            if not allow_synthetic:
                raise ValueError("Synthetic data requires --allow-synthetic and is never CarSim evidence")
        elif meta["source_kind"] == "carsim_experimental":
            # 用户明确允许先忽略净转矩/制动核验，仅运行开发性质的实验。
            # 这条分支绝不宣称已经完成严格物理核验。
            if not meta.get("experimental_training_only"):
                raise ValueError("必须明确声明为实验性 CarSim 训练")
            if meta.get("ready_for_wm_training") is not False:
                raise ValueError("实验性数据不得冒充严格物理验证通过")
            if meta.get("parameter_alignment") != "user_confirmed_not_rechecked":
                raise ValueError("缺少用户确认的车辆参数一致性标记")
            if meta.get("input_semantics") != "nominal_equal_steer_command":
                raise ValueError("缺少明确的名义同角输入语义")
        elif meta["parameter_alignment"] != "verified":
            raise ValueError("CarSim/paper parameter alignment is unverified; complete it before nominal training")
    return [prepare_episode(episode, config) for episode in episodes]


def fit_scaler(prepared, config):
    features, residuals, states = [], [], []
    for data in prepared:
        for i in data.starts:
            features.append(feature_vector(data.episode.state[i], data.episode.model_input[i],
                                           data.history_at(i, config.history_steps), data.memory[i]))
            residuals.append(data.episode.state[i + 1] - data.nominal_next[i])
            states.append(data.episode.state[i + 1:i + config.rollout_steps + 1])
    features, residuals, states = np.asarray(features), np.asarray(residuals), np.concatenate(states)
    floor = np.r_[FEATURE_FLOOR, np.tile(FEATURE_FLOOR, config.history_steps), [.1, .1]]
    return {"feature_mean": features.mean(0), "feature_scale": np.maximum(features.std(0), floor),
            "residual_scale": np.maximum(residuals.std(0), RESIDUAL_FLOOR),
            "state_scale": np.maximum(states.std(0), STATE_FLOOR)}


def trajectory_bootstrap(prepared, member_count, seed):
    """Sample entire trajectories with replacement, retaining all their windows."""
    if len(prepared) < 2:
        raise ValueError("Trajectory-level bootstrap needs at least two distinct training trajectories")
    draws, pools = [], []
    for k in range(member_count):
        rng = np.random.default_rng(seed + 104729 * (k + 1))
        sampled = rng.integers(0, len(prepared), size=len(prepared))
        draws.append(sampled.tolist())
        pools.append(np.array([(int(ep), int(i)) for ep in sampled for i in prepared[ep].starts], dtype=int))
    return draws, pools


def provenance(prepared):
    return [{"trajectory_id": p.episode.metadata["trajectory_id"], "condition_id": p.episode.metadata["condition_id"],
            "sha256": p.episode.source_sha256, "hash_basis": "canonical_physical_arrays",
            "source_kind": p.episode.metadata["source_kind"],
             "split": p.episode.metadata["split"], "windows": len(p.starts)} for p in prepared]


# ============================================================
# CarSim 候选轨迹：仅执行名义 7DOF 诊断，不产生训练窗口/缩放器
# ============================================================


@dataclass
class CandidateNominalAudit:
    """候选数据的只读名义模型预测记录，不能作为训练 PreparedEpisode。"""

    source_sha256: str
    memory_at_start: np.ndarray
    nominal_next: np.ndarray
    valid: np.ndarray
    report: dict


def audit_carsim_candidate(candidate, *, max_substep_s=0.0005, model=None):
    """使用共同转向指令做名义 7DOF 一步预测，检查数据衔接。

    与 prepare_episode() 的区别：
    1. CandidateEpisode.nominal_input 不叫 applied，无法误入正式训练入口；
    2. 始终保留 measured_steer、CarSim Ax/Ay，二者不覆盖模型输入和记忆；
    3. 模型记忆由 7DOF 自己生成并在微步之间传递；
    4. 返回的误差仅是诊断，不能据此宣布训练数据已验证。
    """

    from dynamics7dof.model import Controls, STATE_NAMES
    from .offline import CandidateEpisode

    if not isinstance(candidate, CandidateEpisode):
        raise TypeError(
            "本函数仅接受 CandidateEpisode.load() 创建的候选轨迹"
        )
    if candidate.metadata.get("ready_for_wm_training") is not False:
        raise ValueError("候选轨迹不能宣称已经批准正式训练")
    if candidate.metadata.get("torque_echo_matches") is not True:
        raise ValueError("转矩回显不匹配，不可进行名义控制诊断")
    if not np.isfinite(max_substep_s) or max_substep_s <= 0:
        raise ValueError("max_substep_s 必须为正的有限数值")

    model = Reference7DOF() if model is None else model
    n = len(candidate.time) - 1
    memories = np.full((n, 2), np.nan, dtype=np.float64)
    predicted = np.full((n, 7), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    memory = candidate.initial_memory
    first_failure = None

    for i in range(n):
        memories[i] = [memory.ax_m_s2, memory.ay_m_s2]

        # 从本区间五维指令生成的名义同角输入读取，
        # 不使用“实测左右转角”或未来状态作为输入。
        u = candidate.nominal_input[i]
        controls = Controls(u[:2], u[2:])
        dt = float(candidate.time[i + 1] - candidate.time[i])
        steps = max(1, int(np.ceil(dt / max_substep_s - 1e-9)))

        try:
            step = model.predict(
                candidate.state[i],
                controls,
                mu=NOMINAL_MU,
                load_memory=memory,
                horizon_s=dt,
                substeps=steps,
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            first_failure = {"interval": i, "reason": str(error)}
            break

        if step.flags:
            first_failure = {
                "interval": i,
                "reason": "model_domain_flags",
                "flags": list(step.flags),
            }
            break

        # 记忆只来自原始 7DOF 的力学运算，绝不覆盖成 CarSim Ax/Ay。
        predicted[i] = step.state
        memory = step.load_memory
        valid[i] = True

    residual = predicted[valid] - candidate.state[1:][valid]
    scores = None
    if residual.size:
        scores = {
            "rmse": dict(zip(
                STATE_NAMES,
                np.sqrt(np.mean(residual ** 2, axis=0)).astype(float).tolist(),
            )),
            "mae": dict(zip(
                STATE_NAMES,
                np.mean(np.abs(residual), axis=0).astype(float).tolist(),
            )),
        }

    report = {
        "format": "wm_recovery_carsim_candidate_nominal_audit_v1",
        "status": "diagnostic_only",
        "candidate_sha256": candidate.source_sha256,
        "intervals": n,
        "valid_intervals": int(valid.sum()),
        "coverage": float(valid.mean()),
        "max_substep_s": float(max_substep_s),
        "nominal_mu": NOMINAL_MU,
        "one_step_scores_si": scores,
        "first_failure": first_failure,
        "nominal_input": "common_command_steer_plus_external_torque",
        "measured_steer_preserved": True,
        "carsim_ax_ay_used_in_load_memory": False,
        "reference_force_memory_carried": True,
        "other_wheel_actuation_verified_zero":
            candidate.metadata.get("other_wheel_actuation_verified_zero", False),
        "friction_brake_actuation_verified_zero":
            candidate.metadata.get("friction_brake_actuation_verified_zero", False),
        "ready_for_wm_training": False,
        "warning": (
            "结果仅用于检查候选数据与 7DOF 预测接口；"
            "尚不能认定外加转矩为轮端总转矩，也不能作为论文正式预测精度"
        ),
    }
    return CandidateNominalAudit(
        source_sha256=candidate.source_sha256,
        memory_at_start=memories,
        nominal_next=predicted,
        valid=valid,
        report=report,
    )
