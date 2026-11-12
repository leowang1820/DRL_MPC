# NMPC 残差网络 v3：多时域联合损失 + 终点约束 + 场景均衡采样

## 1. 本次修改什么？

**只需要覆盖一个生产模块：**

```text
wm_recovery/nmpc_residual_multistep_train.py
```

新增测试：

```text
wm_recovery/tests/test_nmpc_multihorizon_v3.py
```

保留现有其他程序，不改动：

- `nmpc_reference.py`、`nmpc_reference_torch.py`
- `nmpc_residual_model.py`、`nmpc_residual_dataset.py`
- `nmpc_rollout.py`、`nmpc_rollout_evaluate.py`、`nmpc_rollout_ablation.py`
- 现有 CarSim 40/60kmh Workspace、`.npz`、`best.pt`、MATLAB/FORCESPRO

兼容旧命令 `--horizon 50`，但**新功能需要显式传入新参数**。

## 2. 三项核心算法修改

### (A) 多预测时域联合滚动误差

```math
L =
lambda_roll * sum_j(w_j [L_path(H_j) + beta*L_terminal(H_j)])
+ lambda_single*L_single
+ lambda_reg*L_reg
```

例如 `H=[50,100,200]`，分别对应 0.5、1.0、2.0 秒，每个时域路径 MSE 都**先除以自身预测步数并使用 Train-only 的状态标准差归一化**，然后按归一化的 `w_j` 加权，避免 2s 时域因区间数量更多而天然压过 0.5s。

`--horizon-strategy joint`：每个优化步骤真正计算 H=50、100、200 的全部加权损失，一次反向传播。

`--horizon-strategy mixed`：每个优化步骤按照 `w_j` 随机选一个 H，单步期望目标仍是上面的联合目标，计算量更低。在少量训练步时会存在随机采样波动；无论训练使用何种策略，**每个 Epoch 的 Validation 都完整、确定地评价所有 H**。

### (B) 单独约束终点误差

```math
L_terminal(H) = mean_over_windows(mean_over_3_channels(
    ((q_pred[H]-q_CarSim[H])/q_scale_train)^2
))
```

因此 `--terminal-weight 0.3` 不等于替换整段误差，而是在整段平均误差上额外强调终点状态。

单步残差监督仍保留：`--single-weight 0.25`。这不是预测训练时把 CarSim 真值状态注入模型：真实状态仅在 t0 初始化和每步计算误差。

### (C) 按 Train 场景均衡抽样

`--scene-balance` 支持：

- `none`：与旧版类似的候选窗口均匀抽样。
- `scene`：每条 Train 轨迹出现概率接近相同。
- `family`：按速度与工况类型均衡，例如 `v40:brake_turn`、`v60:swd`、`v40:circle`，左右版本合并为同一类。对于同类工况，仍从 Train 的真实窗口中取数据，不合成新数据。

训练批次按 group 洗牌轮转，**一个轮转周期内各组出现次数尽量相同**。辅助单步损失也用同样的 Train 分组采样。对于不能容纳某个 H 的短轨迹，它不会被该 H 的候选窗口集合包含；报告会记录各 H 的可用场景数量。

Validation 不进行随机抽样。每个 H 的起点固定、场景统计完整，可以用 `--validation-scene-weight 0.5` 在窗口平均与场景宏平均之间做 50/50 组合用于 Early Stopping。即使使用此参数，Test 仍完全不参与调参或选模。

## 3. 在 Windows PowerShell 安装

把 `wm_nmpc_multihorizon_finetune_v3.zip` 放到 Downloads 文件夹：

```powershell
conda activate drl_mpc_carsim
Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item .\wm_recovery\nmpc_residual_multistep_train.py `
    ".\wm_recovery\nmpc_residual_multistep_train.py.$stamp.bak"

Expand-Archive `
    -Path "$HOME\Downloads\wm_nmpc_multihorizon_finetune_v3.zip" `
    -DestinationPath . -Force
```

不需要再次执行 `nmpc_residual_dataset build`，不覆盖任何现有 `best.pt`。

## 4. 单元测试

```powershell
python -m unittest wm_recovery.tests.test_nmpc_multihorizon_v3 -v
```

隔离合成数据环境实际执行 **7 项测试全部通过**：
1. 时域配置/权重合法性；
2. 场景名称与类别识别；
3. 按速度×工况轮转采样均衡；
4. 路径及终点损失都可对残差网络反向传播；
5. 禁止跨原始区间或时间戳断点；
6. 200 步（2 秒）可微 RK4 反向传播；
7. 从真实接口格式的合成 Train/Validation 创建预训练检查点，执行联合、随机混合和旧版单时域微调；验证只使用 Train/Validation、模型权重可由旧加载器读取、输出目录拒绝覆盖。

此前的 `wm_recovery.tests.test_nmpc_multistep_ablation` 和 `test_nmpc_residual_pipeline` 也可以继续运行。

## 5. 第一轮：建议先运行“轻量混合时域”调试

你的旧基准是：

```text
artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt
```

而 50 步微调的完整 5 秒整体改善很小，因此**建议直接从原 100ep 基准开始一组新的独立多时域实验**，避免把不确定的 50 步微调收益继续叠加。

```powershell
python -m wm_recovery.nmpc_residual_multistep_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --pretrained artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --output artifacts/wm_recovery/nmpc_multihorizon_mixed_smoke_01 `
    --horizons "50,100,200" `
    --horizon-weights "0.4,0.35,0.25" `
    --horizon-strategy mixed `
    --terminal-weight 0.3 `
    --scene-balance family `
    --validation-scene-weight 0.5 `
    --epochs 2 `
    --steps-per-epoch 5 `
    --batch-size 4 `
    --lr 0.00005
```

这次是**检查训练管线**，不是收敛实验。输出：

```text
nmpc_multihorizon_mixed_smoke_01/
  best.pt
  normalization.json
  history.json
  training_report.json
```

重要字段：
- `initial_validation.validation_joint_normalized_mse`：原预训练模型在新联合指标下的基准；
- `best_validation_joint_normalized_mse`：联合 Validation 最佳；
- `best_epoch`：若为 0，说明模型未改善，保留原预训练权重；
- `initial_validation.per_horizon` / `history[...].per_horizon`：每个时域的路径、终点误差、最坏场景；
- `training_sampling_by_horizon`：候选窗口与实际训练抽样类别统计；
- `test_not_used` 应为 `true`。

**切勿拿新联合 MSE 直接与旧单时域 0.0174852 或旧单步 Validation MSE 0.3171 做数值比较，它们的优化目标和标度不同。**

## 6. 第二轮：正式使用联合优化版本（更耗费 CPU）

调试通过后，使用 `joint`，每个优化更新都反向传播三组窗口：

```powershell
python -m wm_recovery.nmpc_residual_multistep_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --pretrained artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --output artifacts/wm_recovery/nmpc_multihorizon_joint_01 `
    --horizons "50,100,200" `
    --horizon-weights "0.4,0.35,0.25" `
    --horizon-strategy joint `
    --terminal-weight 0.3 `
    --scene-balance family `
    --validation-scene-weight 0.5 `
    --epochs 8 `
    --steps-per-epoch 8 `
    --batch-size 4 `
    --lr 0.00003 `
    --patience 6
```

训练时保留原来的归一化参数，并始终使用已知未来 CarSim 等效外部量 `[delta_f, Mz_tv, Fx_all, lamda, mu_hat]`。每个 RK4 子阶段重新计算模型自身状态 `q=[vy,r,vx]`；真实未来状态只作监督。

`joint` 比 `mixed` 占用更多计算和内存，如果 CPU 计算压力明显，可先在 `mixed` 模式下增加 `steps-per-epoch` 并用相同的 Validation 指标对照。**不保证某种方式在真实 5s 上必然胜出。**

## 7. 第三步：保持旧固定 5 秒评价基准不变

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --analysis ablation `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_multihorizon_joint_01/best.pt `
    --split validation `
    --fixed-horizon 500 `
    --anchor-stride 50 `
    --gamma-grid "0,0.25,0.5,0.75,1" `
    --output artifacts/wm_recovery/nmpc_fixed5s_multihorizon_joint_01
```

与原始 `nmpc_fixed5s_ablation_100ep_01`、`nmpc_fixed3s_ablation_100ep_01` 比较：
- 确认 `fixed_cohort_sha256` 相同；
- 查看 `continuous_g1` 的完整 5s 整段归一化 RMSE 是否低于 **0.1827922**；
- 查看固定 3s 归一化 RMSE 是否低于 **0.2275847**；
- 进一步检查 `v60_validation_brake_turn`、`v40_validation_swd_left` 的场景级误差；
- 关注 3s/5s 最坏窗口绝对误差是否变小，而非只追求总体平均。

若希望同时检查 Brake-turn，另做 `--fixed-horizon 300 --channel-ablation` 的 Validation 评价（**输出目录必须另取新名字**）。

## 8. 当前研究边界

- 5 秒是你的 NMPC 完整预测时域，但目前多时域训练只用到 2 秒；5 秒依然是严格独立的最终 Validation 验收指标。
- 当前 8 维网络输入中 `mu_hat` 没有跨附着变化，轮胎内部未观测状态也没有显式记忆。
- 将已知未来 CarSim 等效执行器量当作外部输入是你确定的条件预测设定，不代表完整闭环 NMPC 已验证。
- 全部新旧模型沿用 8→16→16→3，**没有对原 MATLAB/FORCESPRO 控制器做改动**。
- 当前仅在合成数据上测试了代码正确性；不能声称新算法已经提升真实 CarSim 的预测精度。
