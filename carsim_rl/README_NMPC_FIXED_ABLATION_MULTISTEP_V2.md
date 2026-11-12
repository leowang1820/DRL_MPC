# NMPC 残差学习 v2：固定窗口消融 + 可微物理 + 多步训练

## 0. 范围与约定

本版本基于已生成的数据集：

```text
artifacts/wm_recovery/nmpc_residual_dataset_40_60_01/
  train.npz
  validation.npz
  test.npz
  dataset_manifest.json
```

以及已有的 100 epochs 上限实验最优模型：

```text
artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt
```

该旧实验实际在 epoch 42 早停、最佳 epoch 27。

**不需要重新采集 CarSim、不需要重新清洗数据、不修改 MATLAB/FORCESPRO、不重新训练原 7DOF World Model。**

这次只新增/修改：

```text
wm_recovery/
  nmpc_rollout_evaluate.py          # 覆盖更新：保留旧 standard，增加 --analysis ablation
  nmpc_rollout_ablation.py          # 新增：固定同窗口残差强度和积分形式消融
  nmpc_reference_torch.py           # 新增：可微 NMPC 名义动力学和 RK4
  nmpc_residual_multistep_train.py  # 新增：从旧 best.pt 继续多步训练
  tests/
    test_nmpc_multistep_ablation.py # 新增：7 项回归测试
```

先备份旧 `nmpc_rollout_evaluate.py`。其他旧文件保留：`nmpc_reference.py`、`nmpc_rollout.py`、`nmpc_residual_model.py`、`nmpc_residual_train.py`、`nmpc_residual_dataset.py`、`nmpc_actuation.py`。

运行依赖 NumPy + PyTorch，使用原来已成功运行的 Conda 环境即可；**不调用 DLL、CarSim、MATLAB 或 FORCESPRO**。

## 一、科学意义：两个积分模型不相同

已知未来 CarSim 等效执行器量：

```text
a_k = [delta_f,k, Mz_tv,k, Fx_all,k, lamda_k, mu_hat,k]
```

各方法均使用相同的当前预测状态 `q=[vy,r,vx]`、同一起点、同一 `a_k`，未来 CarSim 车辆真值**只计算损失和误差**。

纯物理：

```math
q_{k+1}=RK4(f_nom, q_k, a_k, dt_k)
```

**连续残差**（最终仍更接近你希望集成进 NMPC 微分方程的形式）：

```math
q_{k+1}=RK4(f_nom+gamma*Delta_f_theta, q_k, a_k, dt_k)
```

**离散残差**（用于检查你的单步等效残差标签和滚动实现是否匹配）：

```math
q_{k+1}=RK4(f_nom,q_k,a_k,dt_k)+dt_k*gamma*Delta_f_theta([q_k,a_k])
```

`gamma` 可以是一个统一标量，或逐通道 `[gamma_vy,gamma_r,gamma_vx]`。

特别说明：离散残差积分类似于你的标签构造方式，但不是原 MATLAB continuousDynamics 的微分方程形式。**不能直接以离散形式的误差更小为理由，声称其可无修改插入连续时间 NMPC。**

## 二、安装（PowerShell）

将 `wm_nmpc_fixed_ablation_multistep_v2.zip` 下载至 Windows Downloads 目录。

```powershell
conda activate drl_mpc_carsim

Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

# 只会覆盖旧评价入口；其他都是新文件
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item .\wm_recovery\nmpc_rollout_evaluate.py `
    ".\wm_recovery\nmpc_rollout_evaluate.py.$stamp.bak"

Expand-Archive `
    -Path "$HOME\Downloads\wm_nmpc_fixed_ablation_multistep_v2.zip" `
    -DestinationPath . -Force
```

注意：以上命令假设 ZIP 已放在 `$HOME\Downloads`。如实际下载到其他目录，替换 `-Path`。

## 三、必须先跑新/旧回归测试

```powershell
python -m unittest wm_recovery.tests.test_nmpc_multistep_ablation -v

python -m unittest wm_recovery.tests.test_nmpc_residual_pipeline -v
```

预期新测试 7 项、旧测试 6 项，共 13 项通过（隔离合成数据已验证）。
测试覆盖以下内容：

- NumPy vs PyTorch 名义轮胎力和三通道导数的双精度一致性；
- 同一个 10ms RK4 参考模型在两套代码中的数值一致性；
- 真正沿 5 个时间步回传梯度到网络参数；
- 离散与连续残差公式在同输入下不同，且 gamma=0 完全退化为原名义模型；
- 固定窗口所有变体相同、不会跨 `segment_id` 和数据缺口；
- 将未来 `z_si[:,0:3]` 改成任意伪造数字，预测状态仍不变；
- 从原最佳模型保存的新 `best.pt` 能被原 `load_model_checkpoint()` 和评价器加载；
- 未经确认不使用 Test 参与训练/验证、不会覆盖已有输出。

测试使用**合成数据**，真实 40/60 km/h 数据还需运行下述消融/微调命令。

## 四、第一步：固定 5s 预测窗口做消融（建议先跑）

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --analysis ablation `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --split validation `
    --fixed-horizon 500 `
    --anchor-stride 50 `
    --gamma-grid "0,0.25,0.5,0.75,1" `
    --output artifacts/wm_recovery/nmpc_fixed5s_ablation_100ep_01
```

预计固定 **15 个 5s** Validation 窗口，来自五条连续长度足够的 Validation 轨迹。这些窗口会重叠，**不是 15 条独立试验轨迹**。

`gamma` 的含义：
- `0`：纯 NMPC 名义动力学；
- `0.25 / 0.5 / 0.75`：部分残差补偿，探查是否过度补偿；
- `1`：现有残差模型全量补偿。

每个值均用 `continuous`、`discrete` 两种积分实现。因此默认共有 10 个被评估变体。它们会使用完全相同的 15 个起点；不再出现 3s/5s 指标使用不同数量窗口的误导。

### 第二个固定 3s 组：不要遗漏 Brake-turn

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --analysis ablation `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --split validation `
    --fixed-horizon 300 `
    --anchor-stride 50 `
    --gamma-grid "0,0.25,0.5,0.75,1" `
    --channel-ablation `
    --output artifacts/wm_recovery/nmpc_fixed3s_ablation_100ep_01
```

预计固定 36 个 3s Validation 窗口，**包括清洗截断后仍有约 3.14 秒的 `v60_validation_brake_turn`**。

`--channel-ablation` 再补充 6 种连续 RK4 残差方案：分别完全关闭一个通道、分别将一个通道减半，用于判断 SWD/Brake-turn 是哪个残差通道导致的长时域退化。

输出目录各包含：

```text
ablation_report.json
example_fixed.npz
```

`ablation_report.json`：
- `fixed_cohort_sha256`：固定预测窗口指纹，用于核对换模型后使用完全相同的起点。
- `windows`：每个真实窗口的 scene、segment、起始时间、原始 interval_index。
- `horizons`：同一批固定窗口在 0.1/0.5/1/2/3/4/5s 的指标（不超过 `fixed_horizon`）。
- `by_scene`、`by_speed`：逐场景/初速比较所有变体。
- `ranked_by_final_path_normalized_rmse`：按固定全长的**整段** Train 尺度归一化 RMSE 排序，不只按终点误差排序。
- 每个变体：`endpoint_rmse_q_si`、`endpoint_max_abs_q_si`、`endpoint_p90_abs_q_si`、`path_rmse_q_si`、`worst_window_endpoint_normalized_norm`、相对纯物理的改善率等。

`example_fixed.npz`：
- `q_true_si`: (H+1,3)
- `q_variants_si`: (变体数,H+1,3)
- `variant_names`: 按上方变体数排列
- `known_external_si`: (H,5)
- `t_from_start_s`: (H+1,)

固定组的实际数量以真实运行输出为准，不假定新模型已经改善这些指标。

## 五、第二步：多步训练（先短后长）

**第一轮只跑 0.2 秒，确认可微训练收敛且能被旧模型加载器识别。**

```powershell
python -m wm_recovery.nmpc_residual_multistep_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --pretrained artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --output artifacts/wm_recovery/nmpc_multistep_20_01 `
    --horizon 20 `
    --epochs 5 `
    --batch-size 8 `
    --steps-per-epoch 10 `
    --train-stride 10 `
    --validation-stride 50 `
    --lr 0.0001 `
    --single-weight 0.25 `
    --rollout-weight 1.0
```

中间不重新加载 CarSim 的真实车辆状态，已知的未来执行器序列使用 `z_si[:,3:]`。每个 RK4 阶段使用网络自己预测的 `[vy,r,vx]`。训练目标：

```math
L_train = 1.0 * L_rollout + 0.25 * L_single + 0.0001 * L_reg
```

- `L_rollout`：Train-only 状态尺度归一化后 20×10ms 区间的全部状态误差。
- `L_single`：仍保留原 10ms 等效加速度残差监督，防止网络失去单步物理意义。
- `L_reg`：轻量幅值正则（网络输出相对于 Train 残差尺度）。

**训练选模指标：固定 Validation 20 步自由滚动归一化 MSE，而非旧版单步 Validation Loss。**

保存 `best.pt` 前会计算微调前的 Validation 滚动指标，并在 Epoch 0 保留原预训练权重的拷贝。如果所有微调轮次都没改善，最终 `best.pt` 就是 **Epoch 0 原模型**，不会因微调强制覆盖更好的基线。

### 第二轮：0.5 秒微调

确定第 1 轮 Validation 没有异常，再逐级延长：

```powershell
python -m wm_recovery.nmpc_residual_multistep_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --pretrained artifacts/wm_recovery/nmpc_multistep_20_01/best.pt `
    --output artifacts/wm_recovery/nmpc_multistep_50_01 `
    --horizon 50 `
    --epochs 8 `
    --batch-size 8 `
    --steps-per-epoch 10 `
    --lr 0.00005
```

接下来如有改善，可尝试：
- `--horizon 100`：1 秒；
- `--horizon 200`：2 秒；
- 最终仍通过 **固定 5 秒 Validation** 验证。

不建议直接用 500 步反向传播作为第一轮，因为 500×RK4 将显著增加计算图、梯度累积及训练代价。

### 多步微调输出

```text
nmpc_multistep_20_01/
  best.pt
  history.json
  normalization.json
  training_report.json
```

模型架构仍是 **8→16→16→3 + Tanh**，且训练归一化 buffer 没变，旧版 `nmpc_rollout_evaluate.py` 的 `--analysis standard` 仍能使用这些新 `best.pt`。

微调状态用 float64 计算，以减少轮胎摩擦圆平滑公式的数值差异；保存为与已有 MLP 加载器兼容的 float32 权重。

## 六、第三步：用同一批固定窗口比较微调前后模型

从新的 0.5 秒 `best.pt` 再做一次固定 5s 消融：

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --analysis ablation `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_multistep_50_01/best.pt `
    --split validation `
    --fixed-horizon 500 `
    --anchor-stride 50 `
    --gamma-grid "0,0.25,0.5,0.75,1" `
    --output artifacts/wm_recovery/nmpc_fixed5s_ablation_multistep50_01
```

同样应该再做 3 秒 Validation Brake-turn 组。

**比较两个消融报告中的 `fixed_cohort_sha256`；只有相等才是同一批原始预测窗口。**

看 5 个核心指标：
1. 完整 5 秒 `path_rmse_q_si` 和 `endpoint_rmse_q_si`；
2. 5 秒各通道的 `endpoint_max_abs_q_si`（最坏窗口）；
3. 同一 5 秒窗口组在 1、2、3、4 秒处的中途误差；
4. 单独 3 秒组中的 `v60_validation_brake_turn`；
5. `continuous_g1`（希望的连续微分方程）相对 `discrete_g1` 的差距。

建议不要自动使用验证集上的“最佳 gamma”或仅凭综合分数立即部署，而是记录各通道的真实物理误差与不利场景，最后再选择对照模型。

## 七、几个值得注意的边界

- “未来 CarSim 等效执行器参数已知”是你确认的**条件预测假设**。报告不能延伸成未知未来执行器的因果预测结论。
- `Mz_tv` 仍来自 CarSim 左右轮胎力响应的等效估计，不是已验证的 NMPC 横摆力矩执行器量。
- `neutral lamda=0.5` 是人为定义的物理建模约定，不是轮胎分配比例实测值。
- 训练数据包含单一 `mu_hat=0.8` 维度，不能外推多附着系数泛化能力。
- 5 秒同窗口组仅含 Validation 中能支持 5 秒的轨迹；困难的短 Brake-turn 要用 3 秒组补充分析。
- `test.npz` 没有参与多步微调或选模。在确定 Validation 最终方案之前，先不要运行 Test 消融。
- **本版本仅完成模型结构、实验工具和合成测试；没有真实 40/60 km/h Windows 数据的运行结果，因此无法承诺 5 秒误差已经改善。**
