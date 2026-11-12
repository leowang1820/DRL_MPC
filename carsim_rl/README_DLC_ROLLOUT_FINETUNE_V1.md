# DLC 多步自由滚动微调 v1 — 已有 MLP 的下一阶段实验

## 为什么现在做它？

你在同一个正向+负向 DLC 的 5.5~10.5 秒窗口得到了：

| 真实状态重置间隔 | pure NMPC | NMPC + MLP | 改善率 |
|---|---:|---:|---:|
| 0.1s | .05644 | .00764 | +86.46% |
| 0.5s | .22110 | .01492 | +93.25% |
| 1.0s | .44884 | .03453 | +92.31% |
| 2.0s | 1.39801 | .73233 | +47.62% |
| 5.0s | 8.37160 | 10.30265 | -23.07% |

这支持**长时域自由滚动误差传播**的重要性，但不单独证明误差来自 MLP、轮胎物理模型、外生输入还是离散化方式。

这次先**不更换 8→16→16→3 MLP**、不修改原 `nmpc_reference.py` 或 CarSim 数据采集。
直接从你当前最好的 **全时域训练**模型 `final.pt` 出发做微调：

- 单步预训练标签维持原定义；
- 新增 0.5s、1s、1.5s、2s 的**可微连续 RK4 自由滚动损失**；
- 每个训练窗口只有窗口起点取自 CarSim，之后用模型自身预测状态推进；
- 未来的 CarSim 真值只参与训练 loss，**不用于下一步预测状态**；
- 新模型与原模型在 **相同完整轨迹、相同 3s/5s 预测起点**上评价；
- 特别报告 **5.5–10.5s** 5 秒自由滚动窗口，比较微调前后；
- 如果微调后退化，`report.json` 如实记录，不隐瞒，不自动强行宣布成功。

物理边界：未来 CarSim 实测的前轮转角和轮胎纵向力仍被视作已知外生序列（offline oracle actuation），
`Mz=0`、`lambda=0.5`、`mu_hat=0.8` 仍是当前实验的简化假设。训练和评价使用同一轨迹，
属于**同场景自拟合/回放**，不是独立未见 DLC 的泛化评估。

## 文件

ZIP 中包含：

```text
wm_recovery/
  dlc_rollout_finetune.py
  tests/
    test_dlc_rollout_finetune.py
optional_dependency/
  nmpc_reference_torch.py
README_DLC_ROLLOUT_FINETUNE_V1.md
```

其中 `optional_dependency/nmpc_reference_torch.py` 是此前使用过的可微 Torch 物理模型实现，
**不会自动覆盖你项目的同名模块**。如果本地已经存在 `wm_recovery/nmpc_reference_torch.py`，
只需确保它是与当前 NumPy NMPC 一致的版本（程序启动时会进行一致性检查）。

原有 `dlc_si.py`、`dlc_actuation.py`、`nmpc_residual_model.py`、
`dlc_all_windows_viewer.py`、`dlc_single_scene_experiment.py`
保持不变。

## 1. 安装（Windows PowerShell）

```powershell
conda activate drl_mpc_carsim
Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

Expand-Archive `
  -Path "$HOME\Downloads\wm_dlc_multistep_finetune_v1.zip" `
  -DestinationPath . -Force

# 只有在原工程缺少可微 Torch NMPC 模块时才安装支持文件
if (-not (Test-Path ".\wm_recovery\nmpc_reference_torch.py")) {
    Copy-Item ".\optional_dependency\nmpc_reference_torch.py" `
              ".\wm_recovery\nmpc_reference_torch.py"
}
```

需已安装 `numpy`、`torch`；若想弹出 Figure，额外要求本机的
`matplotlib` + `tkinter`（TkAgg 后端）。

## 2. 运行离线单元测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_rollout_finetune -v
```

这套程序在合成数据上已验证：基本单位与 Torch/NumPy 名义 RK4 对齐、
反向传播不截断历史梯度、保存的 checkpoint 可用原模型加载器读取、
从原模型再训练、相同窗口评估和实际报告数值。
**没有用户本地实际 CarSim 文件，不能事先报告真实微调改善比例。**

## 3. 直接微调目前的正负 DLC 模型（不重新做单步 250 Epoch）

```powershell
python -m wm_recovery.dlc_rollout_finetune `
  --data "artifacts/wm_recovery/dlc_pos_neg_90_01/run_001_CarSim2022_1_Data/dlc_exports.npz" `
  --checkpoint "artifacts/wm_recovery/dlc_pos_neg_90_train_01/final.pt" `
  --output "artifacts/wm_recovery/dlc_pos_neg_90_multistep_01" `
  --mu-hat 0.8 `
  --stages "50:30,100:30,150:20,200:20" `
  --batch-size 4 `
  --single-batch-size 128 `
  --lr 0.0001 `
  --single-weight 0.2 `
  --terminal-weight 0.1 `
  --horizons "300,500" `
  --anchor-stride 50 `
  --focus-start 5.5 `
  --focus-horizon 500 `
  --mz-mode zero `
  --lambda-mode fixed-half
```

**这里每组 `50:30` 是 50 个 10ms RK4 步 / 30 次 Adam 更新。**
`--batch-size 4` 是每次训练同时计算 4 条滚动窗口，
`--single-batch-size 128` 用 128 个真实状态样本维持原单步残差拟合。

默认保持旧 `final.pt` 不动，训练新输出为 `fine_tuned.pt`。
输出目录必须不存在，禁止覆盖旧数据。

如果你希望结束后自动打开所有 3s/5s Figure，在命令最后增加：

```powershell
--show-figures --backend TkAgg
```

不要和之前的 `--show-all-windows` 混淆：这里是新程序的参数 `--show-figures`。

## 4. 运行后看什么？

终端会打印：

```text
阶段1：0.5s自由滚动微调...
阶段2：1.0s自由滚动微调...
阶段3：1.5s自由滚动微调...
阶段4：2.0s自由滚动微调...

=== 同一窗口微调前后 ===
3s: Pure=...; MLP before=...; MLP after=...; change=...%
5s: Pure=...; MLP before=...; MLP after=...; change=...%
重点 5.50~10.50s: MLP before=...; after=...; change=...%
```

`change` 是 **相对于原 MLP 的改变百分比**，不是相对纯 NMPC 的改善。
正值表示新 MLP 比旧 MLP 好，负值表示微调后变差。

参考你已经看到的难点窗口，正常情况下**微调前**的原模型重点窗口应接近：
- Pure NMPC = 8.37160
- NMPC + 原 MLP = 10.30265

如果微调前数值明显不同，先检查训练权重、数据 NPZ、
输入物理假设和当前项目的动力学模型版本。

`report.json` 中尤其关注：

```text
all_windows_before
all_windows_after
focus_window_before
focus_window_after
```

重点窗口的 `unreset_path_rmse_at_horizons` 包含：
`10_steps`、`50_steps`、`100_steps`、`150_steps`、`200_steps`、
`300_steps`、`400_steps`、`500_steps` 等：
它们**都是从 5.50 秒只初始化一次、不重置状态**而得到的预测误差，
与之前“每 0.1s 重置再拼接”的结果不同，可以更明确查看长时域误差开始加速的时刻。

查看重点窗口每个预测长度的数值：

```powershell
$r = Get-Content `
  "artifacts/wm_recovery/dlc_pos_neg_90_multistep_01/report.json" `
  -Raw | ConvertFrom-Json

"=== Fine-tuning before ==="
$r.focus_window_before.unreset_path_rmse_at_horizons | Format-List

"=== Fine-tuning after ==="
$r.focus_window_after.unreset_path_rmse_at_horizons | Format-List
```

打印整体改善：

```powershell
foreach ($h in @("300_steps","500_steps")) {
  $before = $r.all_windows_before.$h
  $after = $r.all_windows_after.$h
  "$h : before=$($before.mlp_normalized_path_rmse) ; after=$($after.mlp_normalized_path_rmse)"
}
```

注：所有评价都使用相同样本窗口；5 秒窗口彼此重叠，不是统计独立的试验。

## 5. 如果训练完成后想再看所有 Figure

原来的全窗口浏览器可直接加载新的 `fine_tuned.pt`：

```powershell
python -m wm_recovery.dlc_all_windows_viewer `
  --data "artifacts/wm_recovery/dlc_pos_neg_90_01/run_001_CarSim2022_1_Data/dlc_exports.npz" `
  --checkpoint "artifacts/wm_recovery/dlc_pos_neg_90_multistep_01/fine_tuned.pt" `
  --horizons "300,500" `
  --anchor-stride 50 `
  --mu-hat 0.8 `
  --mz-mode zero `
  --lambda-mode fixed-half `
  --backend TkAgg
```

## 6. 如何解释结果

- 3s 和 5s 都显著降低：说明**训练目标与状态误差传播**确实是主要可改善因素之一；
  下一步可公平比较 GRU/LSTM，看历史记忆是否能在此基础上继续改善。
- 3s 改善但 5s 不改善：可能需要更长时域训练，但也应优先检查物理输入、
  侧向力模型、扰动状态的适用范围，不建议盲目提高反向传播步数。
- 反而变差：不要掩盖；先降低学习率、阶段更新次数，或加大单步保持权重，
  核查 fine-tune 的路径与终点损失是否被个别难样本主导。
- 若少量难窗口仍持续发散，而所有重置短窗都准确，
  可以进一步实验物理力模型修正、残差约束和历史记忆网络。

**务必注意：这里没有对预测过程中的 `vx` 或 `r` 进行非法裁剪来制造改善。**
如果物理输入导致速度预测为负，报告会忠实保留该结果，
下一步需针对模型适用域和车辆力学进行诊断。

本阶段最有价值的目标不是让训练 Loss 更低，而是：
让难点 5.5~10.5s 的**不重置 5s 预测 RMSE** 相对于原 MLP 真实下降。
