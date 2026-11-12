# CarSim DLC A/B/C/D 数据量与动力学覆盖消融（完整代码 v1）

## 目标

在 **同一条 90 km/h、10 s、1001 帧、61 通道 DLC 数据** 上，直接比较
“500 帧集中在前半段”“500 帧覆盖全程”“750 帧覆盖全程”“全部 1000 个区间”的单步残差学习差异。

四组使用完全一致的：
- 名义单轨 NMPC（既有 `nmpc_reference.py`；不修改物理方程）；
- MLP 网络结构 `8→16→16→3`；
- 每个随机种子下的相同初始网络权重；
- Adam 固定优化更新次数，而非各组固定 Epoch；
- 批大小、学习率、权重衰减和梯度裁剪；
- 完整 DLC 上相同 3 s / 5 s 预测起点；
- 名义 NMPC 基线（一次计算、四组共享）；
- **同一套完整 DLC 的状态评价尺度**，避免不同训练组归一化不同造成 RMSE 不可比。

各个模型的输入、残差标签的**训练归一化只使用各自选中的训练样本**，
不使用其他时间点来训练该模型。公共 RMSE 尺度仅用于结果比较，不影响权重拟合。

## 实验设计

| 组 | 训练间隔数 | 从 DLC 中选择的位置 | 实验意义 |
|---|---:|---|---|
| A | 500 | [0,5) s 的前 500 个连续间隔 | 只覆盖前半段动力学 |
| B | 500 | [0,10) s 全时间轴均匀分布 | 与 A 相同样本量，比较覆盖范围 |
| C | 750 | 包含全部 B，再从其余时间点均匀添加 250 个 | 保持全程覆盖，增加采样密度 |
| D | 1000 | 整条 DLC 的全部可用 10ms 区间 | 全量上限 |

B ⊂ C ⊂ D（严格保证其索引嵌套），方便归因样本密度提升。
各组每个随机种子均采用**相同 2000 次 Adam 更新**，
对应旧全量训练 D 的约 250 epoch × 8 batch/epoch。
A/B/C 采样采用固定种子、有放回 minibatch，避免 Epoch 计数本身不公平。

固定窗口：`--anchor-stride 50`，所以 10 s 轨迹上
- 3 秒窗口：15 个，起点为 0.0、0.5、...、7.0 秒；
- 5 秒窗口：11 个，起点为 0.0、0.5、...、5.0 秒。

**评价含已见数据：不是独立验证。** 本实验仅比较同轨迹上的
数据覆盖 vs 样本量贡献，不用于宣称对新 CarSim 场景的泛化性能。

此外，现有 DLC 等效输入仍采用 CarSim **未来实测**前轮角与轮胎纵向力。
这是离线 `oracle` 外生信号回放，不能当作真实未来可知的驱动/制动指令。
其中 `Mz=0`、`lambda=0.5` 和 `mu_hat=0.8` 均需核实车辆设置和模型定义。

## 包内文件

```text
wm_recovery/
  dlc_data_ablation.py
  tests/
    test_dlc_data_ablation.py
README_DLC_ABCD_ABLATION.md
```

这份 ZIP **只新增文件**，不覆盖现有代码。依赖你之前已经安装并测试通过的：
- `wm_recovery/dlc_si.py`
- `wm_recovery/dlc_actuation.py`
- `wm_recovery/nmpc_reference.py`
- `wm_recovery/nmpc_residual_model.py`
- `wm_recovery/nmpc_rollout.py`

依赖 NumPy、PyTorch；安装 Matplotlib 会自动生成 3 张图，不安装也可保存所有数值结果。

## 一、安装

在 Conda PowerShell：

```powershell
conda activate drl_mpc_carsim

Set-Location 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl'

Expand-Archive `
    -Path "$HOME\Downloads\wm_dlc_abcd_ablation_v1.zip" `
    -DestinationPath . -Force
```

## 二、运行测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_data_ablation -v
```

已通过 9 项无 CarSim DLL 的测试：包含 A/B/C/D 抽样、嵌套性、固定锚点、共享误差尺度、
批量 RK4 与旧逐窗口 RK4 相符、随机种子复现、四组端到端生成模型和指标。
并通过合成 10 秒、1001 帧的完整 3 s / 5 s 冒烟测试，确认 15/11 窗口及绘图。

这些测试 **没有使用真实 `dlc_exports.npz`**，不代表真实残差补偿已经成功。

## 三、真实 DLC 正式运行

当前原始数据（已验收）：

`artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz`

完整命令：

```powershell
python -m wm_recovery.dlc_data_ablation `
    --data "artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz" `
    --output "artifacts/wm_recovery/dlc_abcd_90_01" `
    --mu-hat 0.8 `
    --updates 2000 `
    --batch-size 128 `
    --lr 0.001 `
    --seeds "42" `
    --horizons "300,500" `
    --anchor-stride 50 `
    --mz-mode zero `
    --lambda-mode fixed-half
```

模型训练期间结果可能类似：

```text
共同基线 H=300 (3s) | 15 个窗口 | Pure NMPC=...
共同基线 H=500 (5s) | 11 个窗口 | Pure NMPC=...
[A seed=42 H=3s] n=500 windows=15 pure=... residual=... gain=...%
[B seed=42 H=3s] n=500 windows=15 pure=... residual=... gain=...%
[C seed=42 H=3s] n=750 windows=15 pure=... residual=... gain=...%
[D seed=42 H=3s] n=1000 windows=15 pure=... residual=... gain=...%
...
```

名义 NMPC 基线预计应与你上一次全数据实验相近：
- 3 s: `2.3464930`
- 5 s: `3.2588092`

若有显著偏差，需要先核对是否使用同一个 DLC NPZ、10ms 时间网格与模型版本。
新程序的共享尺度来自全部 1000 个区间左端状态，与旧 D 组尺度定义相同。

如果想考察训练随机性，请跑一个全新的目录、多随机种子：

```powershell
python -m wm_recovery.dlc_data_ablation `
    --data "artifacts/wm_recovery/dlc_preview_40_02/run_001_CarSim2022_1_Data/dlc_exports.npz" `
    --output "artifacts/wm_recovery/dlc_abcd_90_3seeds_01" `
    --mu-hat 0.8 `
    --updates 2000 `
    --batch-size 128 `
    --lr 0.001 `
    --seeds "42,43,44" `
    --horizons "300,500" `
    --anchor-stride 50 `
    --mz-mode zero `
    --lambda-mode fixed-half
```

不同 `--output` 目录互不覆盖。建议先单种子检验，再多种子做最终研究记录。

## 四、生成的文件

```text
artifacts/wm_recovery/dlc_abcd_90_01/
  experiment_config.json          # 参数、每组覆盖率、物理假设
  training_indices.npz            # A/B/C/D 训练索引 + 两时域全部窗口起点
  summary.json                    # 每组所有 seed 的路径和终点误差
  summary_table.csv               # 3s/5s 汇总表，可由 Excel 打开
  metrics_by_seed.csv             # 每组每个随机种子的评价
  per_window_metrics.csv          # 每个预测起点的实际提升/退化

  ablation_rmse_summary.png       # 四组 vs Pure NMPC 的 RMSE 柱状图
  mean_error_curve_300steps.png   # 全 3s 窗口平均预测误差随时间
  mean_error_curve_500steps.png   # 全 5s 窗口平均预测误差随时间

  models/
    A_seed_42.pt
    B_seed_42.pt
    C_seed_42.pt
    D_seed_42.pt
    A_seed_42_history.json
    ... 

  rollouts/
    A_seed_42_300steps.npz
    A_seed_42_500steps.npz
    ...
    D_seed_42_500steps.npz
```

Rollout NPZ 保存 `(H+1, N_windows, 3)`：
- `truth_si`：CarSim 真实状态；
- `pure_nmpc_si`：纯 NMPC 开环积分；
- `residual_nmpc_si`：带 MLP 的开环积分；
- `time_s`：每个窗口的绝对时间戳；
- `anchors`：每个窗口原始区间起点。

报告的改善率为：

`100 * (Pure_RMSE - Residual_RMSE) / Pure_RMSE`

正为改善，负为退化。所有组共用 `Pure_RMSE`。

## 五、查看结果（PowerShell）

```powershell
Import-Csv `
    "artifacts/wm_recovery/dlc_abcd_90_01/summary_table.csv" `
    -Encoding UTF8 |
    Format-Table -AutoSize
```

另外可以看 5s 后半段出发的窗口误差：
（5s 窗口后半段只可能从 5.0s 起步）

```powershell
Import-Csv `
    "artifacts/wm_recovery/dlc_abcd_90_01/per_window_metrics.csv" `
    -Encoding UTF8 |
    Where-Object {
        $_.horizon_steps -eq "500" -and
        [double]$_.anchor_time_s -ge 5.0
    } |
    Select-Object arm,seed,anchor_time_s, `
        nominal_normalized_path_rmse, `
        residual_normalized_path_rmse,improvement_pct |
    Format-Table -AutoSize
```

## 六、结果如何回答“数量还是覆盖”

- **A vs B**：都是 500 个样本、2000 次更新。若 B 显著优于 A，更支持**覆盖动态区域**而非简单样本数量是关键。
- **B vs C vs D**：全部覆盖 [0,10) 秒且 C 包含 B。若误差持续下降，说明更密的样本有额外价值。
- **B 接近 D**：表示主要收益可能来自覆盖完整过程，而不是增加大量相邻点。
- **所有组 3s 很好但 5s 仍大幅偏离**：提示还有长时域积分、模型输入定义或缺失动态记忆问题。

注意本次仍是单条已知 DLC，且 B/C/D 用了整条轨迹的部分或全部标签，所以对固定窗口
不能视为严格的样本外测试。建议多随机种子与多条真实 DLC 轨迹后再做论文结论。

**运行后请提供 `summary_table.csv` 与 `per_window_metrics.csv`，我们能用配对窗口方法拆分“样本量效果”和“动力学覆盖效果”。**
