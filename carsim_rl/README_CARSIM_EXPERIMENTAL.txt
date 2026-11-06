# CarSim 实验性数据接入补丁

本补丁以你已上传的 offline_updated.py / dataset_updated.py 为基底。
严格的 `wm_recovery_trajectory_v1` 逻辑保持原样；新增独立的
`experimental_episode.json` 与环境变量门禁。

## 包含文件

- wm_recovery/offline.py：兼容旧版 V1，增加实验清单加载与 model_input 属性
- wm_recovery/dataset.py：名义输入历史窗口和实验性分区显式检查
- wm_recovery/carsim_experimental.py：真实三条新轨迹采集、候选转换、四分区清单
- wm_recovery/tests/test_carsim_experimental.py：六项不启动 CarSim 的契约测试

## 注意

1. 替换两个现有文件前必须备份。
2. `carsim_candidate.py`、`carsim_si.py`、`carsim_bridge.py` 已由前期建立，本补丁不重复包含。
3. 训练指令需要显式执行 `$env:WM_RECOVERY_EXPERIMENTAL_CARSIM='1'`。
4. 实验输入是同角转向指令及四轮外加转矩指令，不代表实测两个车轮转角严格相等。
5. 旧版严格数据和实验性数据禁止混合在同一个训练分区。
6. 这里的验证和测试均属于同一 CarSim 配置下不同激励的开发试验，
   不能据此声称跨车辆、跨附着、跨道路的外推泛化。
