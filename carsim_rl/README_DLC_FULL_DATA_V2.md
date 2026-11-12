# DLC 单场景全数据训练与同数据评价补丁 v2

## 本次修改

根据要求：**不再划分 0–5s 训练、5–10s 验证**。

给现有 `wm_recovery.dlc_single_scene_experiment` 增加开关 `--full-data`：

- 对单条 10 秒 DLC 轨迹，使用所有 1000 个连续 10ms 动力学区间作为 MLP 训练样本。
- 同样使用全 1000 个区间进行 pure NMPC / residual NMPC 的滚动预测对照。
- 3 秒（300 step）以 50 step 间隔取 **15** 个固定窗口；起点 0.0, 0.5, ..., 7.0 秒。
- 5 秒（500 step）以 50 step 间隔取 **11** 个固定窗口；起点 0.0, 0.5, ..., 5.0 秒。
- 两种预测器每个窗口使用相同的真实初始状态，同一组 CarSim 未来实测转向与纵向轮胎力。
- 窗口内部不回灌未来真实车辆状态，因此是自由滚动轨迹对比。
- 网络结构、名义 NMPC 参数、训练优化器和损失函数都保持不变。

**这叫同数据训练/自拟合回放 (in-sample reconstruction)，不是独立验证，更不能直接宣称泛化能力。**
当前未来 Fx 和实际前轮转角来自真实 CarSim 轨迹，所以仍是 oracle-actuation 离线重放诊断，而不是可部署的闭环 NMPC。

## 文件

该 ZIP 只覆盖两个文件，不修改已完成的 CarSim 数据采集：

```text
wm_recovery/dlc_single_scene_experiment.py
wm_recovery/tests/test_dlc_single_scene.py
README_DLC_FULL_DATA_V2.md
```

保留 `wm_recovery/dlc_si.py`、`wm_recovery/dlc_actuation.py`、`wm_recovery/nmpc_reference.py`、
`wm_recovery/nmpc_residual_model.py`、`wm_recovery/nmpc_rollout.py` 不动。

## 安装（PowerShell）

```powershell
conda activate drl_mpc_carsim

Set-Location 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl'

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item ".\wm_recovery\dlc_single_scene_experiment.py" `
    ".\wm_recovery\dlc_single_scene_experiment.py.$stamp.bak"

Expand-Archive `
    -Path "$HOME\Downloads\wm_dlc_full_data_fit_v2.zip" `
    -DestinationPath . -Force
```

## 单元测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_single_scene -v
```

本补丁在无 CarSim DLL 的合成数据上完成 **9 项单元测试全部通过**，另完成 10 秒、1001 帧合成场景的全量模式冒烟测试，确认 3 秒共 15 窗口、5 秒共 11 窗口；测试不等于真实 DLC 的残差预测已有改善。

## 真实 DLC 完整训练 + 同数据评价命令

不要再传 `--train-end 5.0`；直接添加 **`--full-data`**：

```powershell
python -m wm_recovery.dlc_single_scene_experiment `
    --data "artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz" `
    --output "artifacts/wm_recovery/dlc_single_scene_90_full_01" `
    --full-data `
    --mu-hat 0.8 `
    --epochs 250 `
    --batch-size 128 `
    --lr 0.001 `
    --horizons "300,500" `
    --anchor-stride 50 `
    --mz-mode zero `
    --lambda-mode fixed-half
```

输出目录需要是新的目录。示例中 `mu_hat=0.8`、`Mz=0`、`lambda=0.5` 仍是尚待核对的物理假设；新补丁没有改变等效动力学输入的物理定义。

预期打印：

```text
数据：1001 帧、61 个 Export，模式=同数据全量训练与回放
训练：1000 区间 [0.00,10.00)s；评价：1000 区间 [0.00,10.00]s
注意：本次 Train = Evaluation，为完整 DLC 自拟合；此评价不具备独立泛化意义。
...
[all_data   H=300 (3.0s)] windows=15 | pure=... | residual=... | improvement=...%
[all_data   H=500 (5.0s)] windows=11 | pure=... | residual=... | improvement=...%
```

报告 `report.json` 仍有 `rollouts.train` 和 `rollouts.validation`，**但在 `--full-data` 下两者是同一批数据的同一组指标**，不会重复计算，也不能作为训练/泛化差异解读。

## 核对结果（PowerShell）

```powershell
$r = Get-Content `
    "artifacts/wm_recovery/dlc_single_scene_90_full_01/report.json" `
    -Raw | ConvertFrom-Json

"Experiment: $($r.experiment_type)"
"Train intervals: $($r.train_intervals)"
"Evaluation intervals: $($r.validation_intervals)"
"Same data: $($r.train_and_validation_same_intervals)"

"== Same-data 3s =="
$r.rollouts.validation."300_steps" |
    Select-Object n_windows, nominal_normalized_path_rmse, `
      residual_normalized_path_rmse, normalized_path_improvement_pct, `
      nominal_terminal_rmse_si, residual_terminal_rmse_si

"== Same-data 5s =="
$r.rollouts.validation."500_steps" |
    Select-Object n_windows, nominal_normalized_path_rmse, `
      residual_normalized_path_rmse, normalized_path_improvement_pct, `
      nominal_terminal_rmse_si, residual_terminal_rmse_si
```

会生成：

```text
artifacts/wm_recovery/dlc_single_scene_90_full_01/
  final.pt
  report.json
  history.json
  normalization.json
  dlc_residual_dataset.npz
  validation_3s_rollout.npz
  validation_5s_rollout.npz
  validation_3s_vy.png
  validation_3s_yaw_rate.png
  validation_3s_vx.png
  validation_5s_vy.png
  validation_5s_yaw_rate.png
  validation_5s_vx.png
```

**注意画图所显示的轨迹**：上述 PNG 是每个时域的**首个代表窗口**，即 3s 用 0~3s、5s 用 0~5s。报告中的 RMSE 是全场景所有 15/11 窗口合并统计，不仅仅是图中的一个窗口。如果需要画 5~10s 的最后窗口，可单独导出该起点的滚动预测。

## 输出元信息

- `full_data: true`
- `experiment_type: in_sample_full_data_reconstruction`
- `train_intervals: 1000`
- `validation_intervals: 1000`
- `train_and_validation_same_intervals: true`
- `train_validation_no_interval_overlap: false`
- `independent_validation: false`
- `normalization_includes_evaluation_data: true`
- `training.used_validation_labels_for_training_or_checkpoint_selection: true`
- `checkpoint.training_mode: one_step_fixed_epochs_all_data_in_sample`

这些标志用于避免未来将同数据训练结果误写为独立验证效果。

## 如何解释结果

该实验能回答：
> 现有的三状态名义 NMPC + 8→16→16→3 残差网络，能否**重构已经看过的完整 90 km/h DLC**，以及 3/5s 自由滚动误差有没有明显降低？

它不能回答：
> 对未见的另一条 DLC、其他速度/摩擦工况，是否仍然泛化？

如果同数据训练后 3/5 秒仍然很差，优先核对物理执行器映射、横摆力矩平衡、轮胎动态状态与残差 RK4 注入方式；不要把 Epoch 继续调高作为唯一措施。
