# 单条 CarSim DLC 轨迹：纯 NMPC vs 残差 NMPC（可直接运行的实验）

## 目标与边界

使用 **已有** `wm_recovery.nmpc_reference.py`（与 MATLAB continuousDynamics 对齐）、`nmpc_residual_model.py`（8→16→16→3 MLP）和 `nmpc_rollout.py`（连续 RK4 自由滚动）。

输入是 CarSim 2022.1 预瞄 Driver DLC 的实际采集文件：

`artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz`

已知文件包含 `1001×61`、0~10 秒、0.01 秒间隔，DLC 实际起速约 **90 km/h**，不要因为旧的文件夹名称把它误当成 40 km/h。

**本次不是跨场景泛化实验。** 同一条轨迹的：
- Train：`[0,5)` 秒，500 个区间；
- Validation：`[5,10)` 秒，500 个区间。
- 验证的真实状态不会参与训练梯度、归一化或 checkpoint 选择；只用于最后评估；
- 但在 5 秒预测里 Validation 只有 **1 个完整独立时间起点**，不能进行严格统计推断；
- 3 秒有多个重叠窗口，彼此相关，不等于独立仿真实验。

**重要物理限制：** 当前可用的未来输入来自 CarSim 实际测得的前轮转角和四轮轮胎力。这些轮胎力取决于 CarSim 的真实未来车辆状态。因此本实验属于 **已知未来 CarSim 等效响应的 oracle replay / 诊断性上界对照**，而不等于仅使用未来可知转向/轮端转矩指令的实用 MPC 开环预测。两种预测器共享同一条外生输入轨迹，因此诊断性比较仍是对称的。

## 文件说明

此 ZIP 只增加三个生产文件和一个测试文件，**不覆盖已有 NMPC 模型**：

```text
wm_recovery/
  dlc_si.py                       # 61 Export -> SI 车辆状态
  dlc_actuation.py                # DLC NMPC 8 维输入构造
  dlc_single_scene_experiment.py  # 训练+3/5s RK4 对比+报告+绘图
  tests/
    test_dlc_single_scene.py      # 合成数据 Mock 测试，不需要 CarSim DLL
README_DLC_SINGLE_SCENE_NMPC.md
```

### 前轮等效转角

`delta = (Steer_L1 + Steer_R1)/2`，单位从 deg 转为 rad；这是简化单轨等效转角，不代表任意几何条件下严格等效。

### Mz

默认 `--mz-mode zero`：假设当前 DLC 的内置 Driver / 车辆模型没有独立外加 TV 力矩。**需要在 CarSim 车辆控制配置核实**。
`--mz-mode force-diff-diagnostic` 通过轮距与左右轮实测纵向力差估算，仅供消融诊断；这个估计不等于经验证的 TV 执行器力矩，可能混入转向/轮胎侧向力投影项。

### Fx 与 lamda

默认：
- `Fx_all = Fx_L1 + Fx_R1 + Fx_L2 + Fx_R2`；
- `--lambda-mode fixed-half` 表示设定 `lamda = 0.5` 常量；
- `--lambda-mode ratio-strict` 才尝试 `lamda=Fx_front/Fx_all`，对总纵向力近零且前后轴互相抵消的情况**直接报错**，不自行裁剪或填补。

默认固定 lambda 只是此场景的便捷简化假设，**不是对真实轴力分配的确认**。如果真实车辆纵向力分配远离 0.5，这会显著改变名义 NMPC 的受力模型与学习目标。

`--mu-hat 0.8` 也需要在 CarSim Procedure 中确认。`MuX_L1/R1/L2/R2` 不直接代替道路名义 `mu_hat`。

## 安装

在 Windows PowerShell：

```powershell
conda activate drl_mpc_carsim
Set-Location 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl'

Expand-Archive `
  -Path "$HOME\Downloads\wm_dlc_single_scene_nmpc_v1.zip" `
  -DestinationPath . -Force
```

需要现有项目中已经有：
`wm_recovery/nmpc_reference.py`、
`wm_recovery/nmpc_residual_model.py`、
`wm_recovery/nmpc_rollout.py`，
并能 import `numpy`, `torch`。绘图额外需要 `matplotlib`，若没有会自动跳过 PNG。

### 先运行离线单元测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_single_scene -v
```

我们运行过 **7/7 项合成数据测试**，以及完整 10 秒合成 DLC 的 3/5 秒滚动烟雾测试。它们验证接口与训练/评估流程，不等于实测 DLC 已取得改善。

### 运行你的真实 DLC 90km/h 轨迹

```powershell
python -m wm_recovery.dlc_single_scene_experiment `
  --data "artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz" `
  --output "artifacts/wm_recovery/dlc_single_scene_90_01" `
  --mu-hat 0.8 `
  --train-end 5.0 `
  --epochs 250 `
  --batch-size 128 `
  --lr 0.001 `
  --horizons "300,500" `
  --anchor-stride 50 `
  --mz-mode zero `
  --lambda-mode fixed-half
```

**注意：每次运行都必须使用一个新的 `--output` 目录。** 如目录已存在可改成 `dlc_single_scene_90_02`，避免覆盖旧实验。

控制台会打印：

```text
Train 500 intervals [0,5)s, Validation 500 intervals [5,10]s
epoch ... | Train residual normalized MSE=...
[train      H=300] windows=5 | pure=... | residual=... | improvement=...%
[train      H=500] windows=1 | pure=... | residual=... | improvement=...%
[validation H=300] windows=5 | pure=... | residual=... | improvement=...%
[validation H=500] windows=1 | pure=... | residual=... | improvement=...%
```

`improvement_pct = 100*(pure - residual)/pure`；正数代表改善，负数代表退化。

如果残差长时域劣于纯 NMPC，程序**仍然保存所有真实误差，不会强制宣布训练成功**。

## 输出文件

```text
artifacts/wm_recovery/dlc_single_scene_90_01/
  final.pt                    # 固定 epoch 的 8→16→16→3 MLP，兼容原模型加载函数
  dlc_residual_dataset.npz    # 1001×8 SI 输入、1000×3 残差、原始时间戳
  history.json                # Train 单步训练曲线（数值）
  normalization.json          # 仅 Train 拟合的归一化统计量
  report.json                 # 训练段/时间隔离验证段的 3s、5s 指标
  validation_3s_rollout.npz
  validation_5s_rollout.npz
  validation_3s_vy.png
  validation_3s_yaw_rate.png
  validation_3s_vx.png
  validation_5s_vy.png
  validation_5s_yaw_rate.png
  validation_5s_vx.png
```

### 打印关键指标

```powershell
$j = Get-Content `
  "artifacts/wm_recovery/dlc_single_scene_90_01/report.json" `
  -Raw | ConvertFrom-Json

"=== Validation 3s ==="
$j.rollouts.validation."300_steps" | Select-Object `
  nominal_normalized_path_rmse, residual_normalized_path_rmse, `
  normalized_path_improvement_pct, nominal_terminal_rmse_si, `
  residual_terminal_rmse_si

"=== Validation 5s ==="
$j.rollouts.validation."500_steps" | Select-Object `
  nominal_normalized_path_rmse, residual_normalized_path_rmse, `
  normalized_path_improvement_pct, nominal_terminal_rmse_si, `
  residual_terminal_rmse_si
```

## 标签 / 训练 / 滚动公式

状态：
`q = [vy, r, vx]`（单位 `[m/s, rad/s, m/s]`）

八维输入：
`z = [vy, r, vx, delta_f, Mz_tv, Fx_all, lamda, mu_hat]`

对每个 10ms 区间：

`q_nominal_next = RK4_NMPC(q_CarSim[k], external_CarSim[k], dt=0.01)`

`residual_target[k] = (q_CarSim[k+1] - q_nominal_next)/0.01`

MLP 输入 z 输出 3 维 SI 残差加速度。
训练仅用前 500 个区间，使用 Adam、固定 epoch、不访问验证标签选择 checkpoint。

在完整 3s/5s 预测中：

`q_nominal[k+1] = RK4(f_NMPC, q_nominal[k], exogenous[k])`

`q_residual[k+1] = RK4(f_NMPC + MLP, q_residual[k], exogenous[k])`

两条预测从**同一 CarSim 初始真实状态**出发，之后不回灌未来真实 q；
两个预测器使用相同的未来**实测**等效外部序列。

误差：
- 每个状态的路径 RMSE（SI）；
- 每个状态的终点 RMSE（SI）；
- 由训练段 q 标准差归一化的整段 RMSE；
- 相对纯 NMPC 的百分比改善。

## 风险与后续

1. 此 10s 场景可能包含最高 17.3 度质心侧偏与 ~0.77g 横向加速度；验证段是否覆盖高动态，应检查状态时间曲线。
2. 仅 1 个 5s 验证起点，不应写成统计显著的改善。对论文结论需要独立 DLC Run 与更多窗口。
3. 记录的实际轮胎 Fx 不属于可从当前状态、未来驱动指令直接得到的外部输入；当前实验**不可**直接用于声称部署时的闭环 5s 精度。
4. 两个转向角直接平均、`Mz=0`、`lamda=0.5`、轮胎局部 Fx 均需要物理模型审计；对显著不合理的残差或滚动退化不要靠网络调参掩盖。
5. 请将真实 `report.json` 的 Train/Validation 3s/5s 指标及轨迹图反馈给我，再决定修正动力学映射还是改训练方式。
