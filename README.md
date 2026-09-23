# DRL MPC 实验工程

正式 Python 工程位于 `carsim_rl/`，请在 PyCharm 中打开该目录或将其设为工作目录。

- 日常入口和命令：见 `carsim_rl/README.md`。
- CarSim DLC 配置：见 `carsim_rl/DLC_GUIDE.md`。
- 完整训练和续训：见 `carsim_rl/TRAINING_GUIDE.md`。
- Word 操作手册：保留在 `carsim_rl/docs/`。

2026-09-23 已清理调试脚本、旧训练入口及通用示例。历史代码备份位于本项目上一级的
`_cleanup_backups/`；本项目中的 `.idea/` 与嵌套 `DRL_MPC/.git/` 未改动。

后续接入 7DOF 残差学习时，以当前 DLL 环境和控制包装为基础扩展。
