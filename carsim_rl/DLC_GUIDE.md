# CarSim + SAC 双移线（DLC）操作说明

## 1. 本项目生成的轨迹

这是一条用于控制器开发和强化学习训练的平滑 DLC 轨迹，不作为 ISO 3888
认证场地声明。

- 0–30 m：原车道直行；
- 30–60 m：使用五次平滑曲线向左横移 3.5 m；
- 60–85 m：在相邻车道保持；
- 85–115 m：使用五次平滑曲线回到原车道；
- 115–160 m：出口直行；
- 采样间隔：0.1 m；
- 实际弧长：约 160.58 m；
- 最大曲率：约 0.02215 1/m。

相关文件位于 `paths` 文件夹：

- `dlc_reference_path.par`：CarSim Parsfile，同时也是 Python 的参考路径；
- `dlc_reference_table.txt`：三列 X/Y/S，可直接复制到 CarSim 表格；
- `dlc_reference_path.csv`：便于检查和后处理；
- `dlc_reference_preview.svg`：轨迹预览。

如需重新生成：

```powershell
python .\generate_dlc_path.py
```

## 2. 在 CarSim 2022.1 中建立 DLC 数据集

1. 在 Run Control 页面用 `Ctrl+N` 复制当前已经能够 Python 联仿的 Run，命名为
   `Python SAC - DLC 40 kmh`。不要直接修改原来的 Uturn Run。
2. 进入 `Path: X-Y Coordinates` 页面，复制一个非闭合路径数据集，命名为
   `DLC Smooth 40 kmh`。
3. 打开 `paths\dlc_reference_table.txt`，复制全部内容。
4. 清空新路径页面的 `X, Y, S` 表格，然后从第一格粘贴。确认起点为
   `(0, 0, 0)`，终点约为 `(160, 0, 160.58)`，并保持 `Closed/Loop` 关闭。
5. 如果你的 CarSim 页面提供 Library Tool/Parsfile 导入，也可以直接导入
   `paths\dlc_reference_path.par`，效果与粘贴表格相同。

CarSim 官方把 `Path: X-Y Coordinates` 作为创建路径的标准方式。不同数据库模板的
黄色链接位置可能不同，因此以页面标题为准。

## 3. 配置道路和联仿接口

1. 物理道路选择平直、平坦路面，建议可用宽度至少 10–12 m，保证横移 3.5 m 后
   车辆仍在路面上。
2. 如果 Driver/Path 页面有独立的 target/reference path 选项，选择
   `DLC Smooth 40 kmh`，但方向盘、油门和制动仍必须保持外部 Import 控制。
3. 保持三个 Import 的顺序不变：
   `IMP_THROTTLE_ENGINE, IMP_PCON_BK, IMP_STEER_SW`。
4. 保持八个 Export 的顺序不变：
   `XO, YO, YAW, AVZ, Lat_Veh, Lat_Targ, VX, Station`。
   Python 真正使用的是前四项和 `VX`，横向误差由 Python 按 DLC 路径重新计算。
5. Procedure 建议：初始车速 40 km/h，停止时间 20 s 以上；现有 65 s 也可以，
   Python 到达 160.58 m 后会提前成功终止。
6. 在复制后的 Run 中执行 `Send to Simulink`，让当前 Run 重新写出
   `CarSim2022.1_Data\simfile.sim`。即使不启动 Simulink，也必须执行这一步。

不要把 DLC 路径直接当成弯曲道路中心线，除非你就是想显示一条弯曲道路。典型双移线
应在宽阔的直道路面上运动；参考线可由 CarSim target path 或 Python SVG 显示。

## 4. 先跑无需训练的基线

关闭可能占用 Solver DLL 的 MATLAB/Simulink 运行，然后执行：

```powershell
python .\run_dlc_baseline.py
```

它使用纯跟踪控制器验证坐标、符号和路径配置，并生成：

- `artifacts\dlc_baseline.csv`
- `artifacts\dlc_baseline.svg`

成功标准：起点误差接近 0、进度接近 100%、`completed=True`、
`off_track=False`。如果只在转向段方向相反，先不要训练，检查 CarSim 的
`IMP_STEER_SW` 符号。

## 5. DLC 强化学习训练

只有基线验证后再开始新训练。Uturn 模型不能作为 DLC 正式结果，建议从新随机种子训练：

```powershell
python .\train_carsim_sac_complete.py `
  --mode train `
  --scenario dlc `
  --name dlc_40_seed42 `
  --episodes 300 `
  --target-speed 40 `
  --device cuda `
  --seed 42 `
  --max-steps 450 `
  --eval-interval 10 `
  --eval-episodes 3 `
  --checkpoint-interval 10
```

TensorBoard：

```powershell
python -m tensorboard.main --logdir ".\runs" --port 6006
```

重点观察 `eval/completion_rate`、`eval/off_track_rate`、
`eval/mean_path_progress`、`train/final_lateral_error`。

## 6. 评估并画出参考轨迹与实际轨迹

```powershell
python .\train_carsim_sac_complete.py `
  --mode eval `
  --scenario dlc `
  --model ".\saves\carsim_sac_complete_dlc_40_seed42\best.pth" `
  --device cuda `
  --eval-episodes 3 `
  --max-steps 450 `
  --trajectory-output ".\artifacts\dlc_sac_best"
```

结果为 `artifacts\dlc_sac_best.csv` 和 `artifacts\dlc_sac_best.svg`。

## 7. 在 CarSim 中观看车辆动画

Python 基线或评估正常结束后，回到刚才复制的 Run：

1. 确认该 Run 的输出 ERD/动画输出处于启用状态；
2. 打开 Video/VS Visualizer，加载该 Run 最新生成的结果；
3. 使用俯视相机观察车辆先左移 3.5 m、保持、再返回；
4. 若 CarSim 动画中没有绘制蓝色参考线，以 Python 生成的 SVG 作为参考线与实际
   轨迹叠加结果；CarSim 动画主要用于检查车辆姿态、轮胎和道路位置。

注意：CarSim 的 GUI 仅负责配置和结果动画；当前 Python DLL 联仿不是实时同步播放。

## 8. 提取 Actor 并部署回 CarSim

完整训练检查点包含 Actor、两个 Critic、目标网络、优化器和训练状态。部署时只需 Actor：

```powershell
python .\export_sac_actor.py `
  --checkpoint ".\saves\carsim_sac_complete_dlc_40_seed42\best.pth" `
  --output ".\exports\dlc_40_best_actor.pt"
```

导出的 `.pt` 是确定性 TorchScript Actor，配套 `.json` 记录观测顺序、动作含义、
目标车速、源检查点和 SHA-256。使用它控制 CarSim：

```powershell
python .\run_carsim_sac_actor.py `
  --actor ".\exports\dlc_40_best_actor.pt" `
  --device cuda `
  --max-steps 450 `
  --output ".\artifacts\dlc_actor_control"
```

数据流为：CarSim Export → 7 维归一化观测 → Actor → 2 维动作 → 变化率限制和
油门/制动/方向盘映射 → CarSim Import。不要直接把 Actor 的两个输出当作 CarSim 的
三个 Import；必须保留包装层的映射和 50 ms 控制周期。
