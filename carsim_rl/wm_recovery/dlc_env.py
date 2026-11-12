# -*- coding: utf-8 -*-
"""兼容现有 carsim_env.CarSimEnv 的零 Import（内置驾驶员）适配器。

公开的 CarSimEnv.reset() 常见实现会检查 n_import <= 0 并拒绝运行。
这对外部控制场景合理，但 CarSim 的内置预瞄驾驶员并不需要外部
控制通道。因此，此处只覆盖 reset() 中的缓冲区创建逻辑。

底层仍然使用你项目已经安装的 CarSimEnv 和 VS Solver，不拷贝
或修改第三方源码。具体 DLL/SDK 的零 Import 行为需在本机检验。
"""
from __future__ import annotations

import ctypes
import math
import numpy as np


def make_preview_env(sim_path):
    """为一个零 Import 的 CarSim DLC SIM 创建环境实例。"""
    from carsim_env import CarSimEnv

    class _CarSimPreviewEnv(CarSimEnv):
        def reset(self):
            # 关闭上一条未完成的轨迹，防止旧 solver 状态残留。
            if self.initialized and not self.done:
                self.close()

            self.config = self.solver.read_configuration(self.sim_path)
            if not isinstance(self.config, dict):
                raise RuntimeError("无法读取 CarSim SIM 配置")

            self.n_import = int(self.config["n_import"])
            self.n_export = int(self.config["n_export"])
            self.t_start = float(self.config["t_start"])
            self.t_stop = float(self.config["t_stop"])
            self.t_step = float(self.config["t_step"])

            if self.n_import != 0:
                raise ValueError(
                    "DLC 内置预瞄驾驶员应使用零个外部 Import，"
                    f"实际报告 {self.n_import}"
                )
            if self.n_export < 1:
                raise ValueError("DLC 没有配置可读取的 Export")
            if not all(math.isfinite(v) for v in
                       (self.t_start, self.t_stop, self.t_step)):
                raise ValueError("CarSim 仿真时间参数不合法")
            if self.t_step <= 0:
                raise ValueError("CarSim t_step 必须为正")

            # 保留一个安全的实存 double 作为底层 C 指针的占位。
            # **CarSim 真正的 n_import 仍为 0，不给车辆写入任何动作**。
            # 避免 ctypes 对长度 0 数组的地址处理差异。
            self._import_c_array = (ctypes.c_double * 1)()
            self._export_c_array = (
                ctypes.c_double * self.n_export
            )()
            self.import_np = np.empty((0,), dtype=np.float64)
            self.export_np = np.ctypeslib.as_array(self._export_c_array)
            self.import_vars = []
            self.export_vars = []

            self.t_current = self.t_start
            self.solver.copy_export_vars_into(
                self._export_c_array, self.n_export, return_list=False
            )
            self.export_vars = self.export_np.tolist()
            self.initialized = True
            self.done = False
            return tuple(self.export_vars)

    return _CarSimPreviewEnv(str(sim_path))
