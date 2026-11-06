"""CarSim 六输入、六十一输出的原始数据采集桥接。

本模块只负责：
1. 检查生成的 CarSim 配置及通道顺序。
2. 把五维算法控制指令转换为六维 CarSim Import。
3. 复用现有 CarSimEnv 进行仿真步进。
4. 记录带有实际仿真时间的六十一维原始 Export。
5. 将原始数据保存为未完成物理验证的记录。

本模块不改变 CarSim 配置、不转换未知单位、不生成正式训练轨迹。
导入本模块不会加载或启动 CarSim Solver。
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .controls import Command, COMMAND_NAMES, DIRECT_IMPORT_NAMES
from .carsim_export_schema import (
    EXPORT_COUNT,
    EXPORT_NAMES,
    EXPORT_SCHEMA_VERSION,
    to_named_exports,
    validate_export_order,
)
from .inspection import inspect_simfile


# 四个轮端外加转矩使用 Add，两个前轮直接转角使用 Replace
EXPECTED_IMPORT_MODES = {
    "IMP_MYUSM_L1": "ADD",
    "IMP_MYUSM_R1": "ADD",
    "IMP_MYUSM_L2": "ADD",
    "IMP_MYUSM_R2": "ADD",
    "IMP_STEER_L1": "REPLACE",
    "IMP_STEER_R1": "REPLACE",
}

# 原始记录格式与正式 WM 训练轨迹格式严格分开
RAW_FORMAT = "wm_recovery_carsim_raw_capture_v1"


def verify_inventory(inventory):
    """逐项核对 CarSim 生成配置的输入、输出和模式。

    此检查仅证明静态配置符合约定，不能证明物理执行器已验证。
    """

    imports = inventory["imports"]
    exports = inventory["exports"]
    ports = inventory["ports"]

    actual_names = tuple(item["name"].upper() for item in imports)
    expected_names = tuple(name.upper() for name in DIRECT_IMPORT_NAMES)

    # 输入通道顺序也冻结，避免错误的控制动作被发送到别的位置
    if actual_names != expected_names:
        raise ValueError(
            "CarSim Import 名称或顺序不符合固定的六通道接口："
            f"实际 {actual_names}"
        )

    for item in imports:
        name = item["name"].upper()
        mode = item["mode"].upper()
        if mode != EXPECTED_IMPORT_MODES[name]:
            raise ValueError(
                f"Import {name} 的模式为 {mode}，"
                f"预期 {EXPECTED_IMPORT_MODES[name]}"
            )

    if ports != {"PORTS_IMP": 6, "PORTS_EXP": EXPORT_COUNT}:
        raise ValueError(
            f"CarSim 端口声明错误：{ports}"
        )

    # 按位置逐项比较，允许两个 Steer_SW，但不能调整位置
    validate_export_order(exports)

    return True


@dataclass(frozen=True)
class CarSimSnapshot:
    """保存一个仿真时刻的时间戳和六十一维原始输出。"""

    time_s: float
    raw_exports: np.ndarray

    @property
    def data(self):
        """支持 snapshot.data.Vx、snapshot.data.Fx_L1 等按名称访问。"""
        return to_named_exports(self.raw_exports)


def _make_snapshot(time_s, raw_exports):
    """复制一个独立的输出快照，避免底层缓冲区被后续步进覆盖。"""

    timestamp = float(time_s)
    values = np.asarray(raw_exports, dtype=np.float64)

    if not math.isfinite(timestamp):
        raise ValueError("CarSim 仿真时间不是有限数值")

    if values.shape != (EXPORT_COUNT,):
        raise ValueError(
            f"CarSim 每个时刻必须输出 {EXPORT_COUNT} 个数值，"
            f"实际形状为 {values.shape}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("CarSim Export 中存在非有限数值")

    return CarSimSnapshot(timestamp, values.copy())


class CarSimBridge:
    """对已有 CarSimEnv 增加 WM 专用的严格接口及原始记录功能。

    env_factory 和 inspection_fn 可在单元测试中注入模拟对象。
    正式使用时不需要指定它们。
    """

    def __init__(
        self,
        sim_path,
        prediction_dt_s=0.01,
        *,
        env_factory=None,
        inspection_fn=None,
    ):
        self.sim_path = Path(sim_path).expanduser()
        self.prediction_dt_s = float(prediction_dt_s)

        if (
            not math.isfinite(self.prediction_dt_s)
            or self.prediction_dt_s <= 0
        ):
            raise ValueError("prediction_dt_s 必须为正的有限数值")

        self._env_factory = env_factory
        self._inspection_fn = (
            inspect_simfile if inspection_fn is None else inspection_fn
        )

        self._env = None
        self._inventory = None
        self._started = False
        self._ended = False
        self._inner_steps = None
        self._solver_t_step_s = None
        self._latest = None

        self._times = []
        self._raw_exports = []
        self._commands = []
        self._imports = []

    def reset(self):
        """先做只读配置检查，再初始化 CarSim 并采集初始状态。

        正式运行会加载 DLL；单元测试应注入模拟 env_factory。
        """

        if self._started:
            raise RuntimeError("本记录已启动；请为新轨迹创建新的 CarSimBridge")

        # 任何 DLL 操作之前，先核对磁盘上生成的配置
        inventory = self._inspection_fn(self.sim_path)
        verify_inventory(inventory)

        env = None

        try:
            # 延迟导入：仅执行 reset() 时才访问现有 CarSim 通信层
            if self._env_factory is None:
                from carsim_env import CarSimEnv
                factory = CarSimEnv
            else:
                factory = self._env_factory

            env = factory(str(self.sim_path))
            initial_exports = env.reset()

            config = env.config
            if not isinstance(config, dict):
                raise ValueError("CarSimEnv 未提供有效的配置字典")

            if int(config.get("n_import", -1)) != 6:
                raise ValueError("Solver 报告的 Import 数量不是 6")

            if int(config.get("n_export", -1)) != EXPORT_COUNT:
                raise ValueError("Solver 报告的 Export 数量不是 61")

            t_step = float(config.get("t_step", float("nan")))
            if not math.isfinite(t_step) or t_step <= 0:
                raise ValueError("Solver 的积分时间步长无效")

            # 只允许预测周期恰好对应整数个 Solver 积分步
            ratio = self.prediction_dt_s / t_step
            inner_steps = int(round(ratio))

            if (
                inner_steps < 1
                or not math.isclose(
                    inner_steps * t_step,
                    self.prediction_dt_s,
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                )
            ):
                raise ValueError(
                    f"预测周期 {self.prediction_dt_s} s "
                    f"不是 Solver 步长 {t_step} s 的整数倍"
                )

            snapshot = _make_snapshot(
                env.t_current,
                initial_exports,
            )

        except Exception:
            # 失败时关闭可能已经初始化的 Solver
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            raise

        self._env = env
        self._inventory = inventory
        self._inner_steps = inner_steps
        self._solver_t_step_s = t_step
        self._started = True
        self._ended = bool(getattr(env, "done", False))
        self._latest = snapshot

        self._times = [snapshot.time_s]
        self._raw_exports = [snapshot.raw_exports.copy()]
        self._commands = []
        self._imports = []

        return snapshot

    def step(self, command):
        """执行一个预测周期，记录该区间的控制和结束时刻状态。

        command 为 Command 对象或五维数组：
        [T_FL, T_FR, T_RL, T_RR, delta_f_rad]

        记录中的 command[i] 对应时间区间
        [time_s[i], time_s[i + 1])。
        """

        if not self._started or self._env is None:
            raise RuntimeError("请先调用 reset()")

        if self._ended:
            raise RuntimeError("仿真已结束，不能继续步进")

        if not isinstance(command, Command):
            command = Command.from_vector(command)

        command_values = command.as_vector()

        # 映射时严格检查全部六个 Import，禁止补零或截断
        import_values = command.to_imports(DIRECT_IMPORT_NAMES)

        if (
            import_values.shape != (6,)
            or not np.all(np.isfinite(import_values))
        ):
            raise ValueError("控制映射未产生完整有效的六维 Import")

        previous_time = self._latest.time_s

        try:
            exports, _, done, info = self._env.control_step(
                import_values,
                inner_steps=self._inner_steps,
            )
        except Exception:
            self.close()
            raise

        # 非零返回码和 Solver 错误一律阻止静默生成训练数据
        info = info if isinstance(info, dict) else {}
        if info.get("error") or info.get("return_code", 0) != 0:
            self.close()
            raise RuntimeError(f"CarSim Solver 步进失败：{info}")

        try:
            snapshot = _make_snapshot(
                self._env.t_current,
                exports,
            )

            # 使用 Solver 实际时间，而不是自己假定时间已经前进
            actual_dt = snapshot.time_s - previous_time
            tolerance = max(1e-9, self.prediction_dt_s * 1e-7)

            if not math.isclose(
                actual_dt,
                self.prediction_dt_s,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    f"实际前进 {actual_dt} s，"
                    f"不等于预期 {self.prediction_dt_s} s"
                )
        except Exception:
            self.close()
            raise

        self._times.append(snapshot.time_s)
        self._raw_exports.append(snapshot.raw_exports.copy())
        self._commands.append(command_values.copy())
        self._imports.append(import_values.copy())

        self._latest = snapshot
        self._ended = bool(done)
        return snapshot

    @property
    def latest(self):
        """取得最近一个仿真时刻的输出快照。"""
        if self._latest is None:
            raise RuntimeError("还没有读取任何 CarSim 状态")
        return self._latest

    @property
    def inner_steps(self):
        """每个 WM 预测周期对应的 CarSim 积分步数。"""
        return self._inner_steps

    def close(self):
        """安全结束当前 CarSim 仿真，不丢弃已采集的原始数据。"""
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._ended = True

    def save_raw(self, output_dir):
        """保存未经物理验证的 CarSim 原始记录，不覆盖已有目录。

        保存 N+1 帧状态和 N 组区间控制。
        此文件不能直接冒充 offline.Episode 所需的正式训练数据。
        """

        if not self._started:
            raise RuntimeError("尚未开始记录")

        if not self._commands:
            raise ValueError("尚未采集任何完整控制区间")

        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)

        times = np.asarray(self._times, dtype=np.float64)
        exports = np.stack(self._raw_exports)
        commands = np.stack(self._commands)
        imports = np.stack(self._imports)

        np.savez_compressed(
            root / "raw_capture.npz",
            time_s=times,
            raw_exports=exports,
            command=commands,
            import_values=imports,
        )

        metadata = {
            "format": RAW_FORMAT,
            "source_kind": (
                "carsim_env" if self._env_factory is None else "mock_or_injected"
            ),
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "sim_path": str(self._inventory.get("sim_path", self.sim_path)),
            "parameter_path": self._inventory.get("parameter_path"),
            "sim_sha256": self._inventory.get("sim_sha256"),
            "parameter_sha256": self._inventory.get("parameter_sha256"),
            "import_names": list(DIRECT_IMPORT_NAMES),
            "export_names": list(EXPORT_NAMES),
            "command_names": list(COMMAND_NAMES),
            "samples": int(len(times)),
            "control_intervals": int(len(commands)),
            "prediction_dt_s": self.prediction_dt_s,
            "solver_t_step_s": self._solver_t_step_s,
            "solver_steps_per_interval": self._inner_steps,
            "recorded_time_source": "CarSimEnv.t_current",
            "state_time_alignment": "raw_exports[i] 位于 time_s[i]",
            "command_time_alignment": (
                "command[i] 在 time_s[i] 至 time_s[i+1] 区间发送"
            ),
            "native_units_not_converted": True,
            "si_units_verified": False,
            "physical_interface_verified": False,
            "actual_actuation_verified": False,
            "parameter_alignment": "not_verified",
            "ready_for_wm_training": False,
            "notes": (
                "原始数据仅用于采集核验；"
                "需单独核实实际输入、单位、附着、车辆参数和初始载荷记忆"
            ),
        }

        (root / "metadata.json").write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return {
            "status": "raw_capture_saved_not_validated",
            "output": str(root),
            "samples": len(times),
            "control_intervals": len(commands),
        }

    def __enter__(self):
        """支持 with 语法，自动初始化。"""
        self.reset()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """离开 with 代码块时自动关闭 Solver。"""
        self.close()
        return False
