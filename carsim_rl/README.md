# CarSim SAC 正式工程

本目录是当前保留的训练和部署版本。解释器使用 Conda 环境 `drl_mpc_carsim`；
PyCharm 的工作目录设置为本目录。

## 日常运行入口

| 文件 | 用途 |
| --- | --- |
| `train_carsim_sac_complete.py` | 正式 SAC 训练、恢复训练、模型评估 |
| `export_sac_actor.py` | 从完整检查点导出确定性 Actor |
| `run_carsim_sac_actor.py` | 加载 Actor 控制 CarSim，保存轨迹 |
| `run_dlc_baseline.py` | DLC 纯跟踪基线及场景验证 |
| `generate_dlc_path.py` | 重新生成 DLC 参考路径 |

## 运行所需的核心模块

| 文件 | 用途 |
| --- | --- |
| `carsim_wrapper.py` | 7 维观测、奖励、2 维动作到 3 个 Import 的映射 |
| `carsim_env.py` | CarSim reset、步进与关闭 |
| `vs_solver.py` | 底层 Solver DLL API |
| `SACModule.py` | SAC 网络、经验回放、参数更新 |
| `frenet_path.py` | 全局位置转路径坐标和跟踪误差 |
| `trajectory_visualization.py` | 轨迹 CSV 与 SVG 报告 |

核心模块由入口脚本调用，不需要单独点击运行。`carsim_wrapper.py` 从原
`train_carsim_sac_v3.py` 中提取；计算逻辑、网络结构和 state_dict 字段保持不变，
旧版训练入口已移出工程，已有正式检查点和导出模型可以继续使用。

## 常用命令

先激活环境并进入本目录，确认 CarSim 当前 Run 和生成的 simfile.sim 一致。
配置变更后重新生成仿真文件；不得同时运行共享同一 CarSim 输出文件的多个仿真进程。

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
python run_dlc_baseline.py
```

新训练请用新实验名，避免覆盖此前保存结果：

```powershell
python train_carsim_sac_complete.py --mode train --scenario dlc --name dlc_40_new --episodes 300 --max-steps 450 --target-speed 40 --device cuda
```

使用已保存的 DLC 策略进行评估、导出和控制：

```powershell
python train_carsim_sac_complete.py --mode eval --scenario dlc --model ".\saves\carsim_sac_complete_dlc_40_seed42\best.pth" --max-steps 450 --eval-episodes 1 --device cuda
python export_sac_actor.py --help
python run_carsim_sac_actor.py --device cuda
```

`export_sac_actor.py` 默认从 `saves/carsim_sac_complete_dlc_40_seed42/best.pth`
导出到 `exports/dlc_40_best_actor.pt`，需要更新模型时运行
`python export_sac_actor.py`。导出会更新同名文件；如需另存，使用 `--output`。
部署时同时保留 Actor 的 `.pt` 和同名 `.json` 文件。

TensorBoard：

```powershell
python -m tensorboard.main --logdir ".\runs" --port 6006
```

## 数据与文档

- `paths/`：参考轨迹。
- `saves/`：训练检查点，包含历史实验。
- `exports/`：部署 Actor 和元数据。
- `runs/`：TensorBoard 事件。
- `artifacts/`、`Results/`：轨迹报告与 CarSim 结果。
- `docs/`：已交付的 Word 操作手册。

以上数据本次均保留，包括早期训练结果。`requirements-lock.txt` 和
`environment.yml` 继续使用原版本。

## 2026 年 9 月清理说明

主工程已移除 `smoke_test_*.py`、`test_*.py`、`diagnose_carsim_api.py`、
旧 `simple_controller.py`、旧 `train_carsim_sac_v3.py`、根目录示例
`main.py/init.py`、Word 生成脚本及排版预览缓存。原文件均可从项目上一级
`_cleanup_backups/` 的本次备份目录恢复，备份含清单和文件哈希。

SACModule 中的 Pendulum 演示训练、底层环境及 Frenet 模块中的调试 main 已移除。
网络计算、SAC 更新、环境步进和奖励逻辑未修改。

旧 Word 手册作为历史交付保留。其中提到的 smoke 脚本不再使用，场景验证改用
`run_dlc_baseline.py`；环境包装模块名改为 `carsim_wrapper.py`。
如果 PyCharm 的旧 Run Configuration 仍指向已移除文件，请改选本页列出的入口。
