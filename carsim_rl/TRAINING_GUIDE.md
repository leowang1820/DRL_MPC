# CarSim SAC 完整训练指南

## 1. 训练前检查

在 `carsim_rl` 目录、`drl_mpc_carsim` Conda 环境中操作。
旧版 smoke/test/diagnose 调试脚本已移出工程。DLC 配置验证使用
`python run_dlc_baseline.py`，详细步骤见 `DLC_GUIDE.md`。
下面的训练命令保留默认 Uturn 场景；切换到 DLC 时，必须同时设置
`--scenario dlc --max-steps 450` 并选用对应 CarSim Run。

检查 CarSim 配置：

- CarSim: `n_import=3`, `n_export=8`, `t_stop=65`, `t_step=0.001`
- 初始车速：`40 km/h`
- 观测维度：7
- Solver 返回码：0

运行 Python 前关闭 MATLAB/Simulink 和其他仍在占用 CarSim Solver 的 Python Console。

## 2. 第一阶段：20 轮可恢复训练

```powershell
python train_carsim_sac_complete.py --mode train --name uturn_40_v1 --episodes 20 --device cuda --eval-interval 5 --checkpoint-interval 5
```

本阶段用于检查奖励趋势、离轨率、完成率和损失稳定性。

## 3. 第二阶段：恢复并训练到 300 轮

```powershell
python train_carsim_sac_complete.py --mode train --name uturn_40_v1 --resume ".\saves\carsim_sac_complete_uturn_40_v1\latest.pth" --episodes 300 --device cuda
```

`--episodes 300` 表示训练到总计第 300 轮，不是再追加 300 轮。

## 4. 评估最优模型

```powershell
python train_carsim_sac_complete.py --mode eval --model ".\saves\carsim_sac_complete_uturn_40_v1\best.pth" --eval-episodes 10 --device cuda
```

## 5. TensorBoard

```powershell
tensorboard --logdir ".\runs"
```

浏览器打开终端显示的本地地址，重点观察：

- `train/episode_reward`
- `train/episode_length`
- `train/off_track`
- `train/completed`
- `train/final_lateral_error`
- `eval/mean_reward`
- `eval/off_track_rate`
- `eval/completion_rate`
- `loss/q1_loss`, `loss/q2_loss`, `loss/policy_loss`

## 6. 检查点文件

每个训练目录会包含：

- `latest.pth`：定期检查点，包含最近的回放数据
- `best.pth`：当前评估奖励最高的模型
- `final.pth`：全部训练正常结束时生成
- `interrupted.pth`：用 `Ctrl+C` 中断时自动生成
- `crash_recovery.pth`：发生异常时尽可能保存

检查点保存策略、回放数据数量和评估周期均可通过命令行参数调整。
