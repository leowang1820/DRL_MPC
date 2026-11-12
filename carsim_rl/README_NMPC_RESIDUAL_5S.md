# NMPC Residual MLP + 5 秒已知执行器条件滚动评价（Python v1）

本补丁新增四个模块和一个回归测试文件，不覆盖你已验证的数据清洗、61 通道 Export、`nmpc_actuation.py`、`nmpc_reference.py` 或 `nmpc_residual_dataset.py`。

## 一、模块与数据契约

- `wm_recovery/nmpc_residual_model.py`：轻量 MLP（8→16→16→3），内部输入和输出归一化；外部 `forward` 始终接收和返回 SI 物理量。
- `wm_recovery/nmpc_residual_train.py`：数据契约检查、train-only 归一化、Adam 训练、基于验证集早停与最佳权重保存。
- `wm_recovery/nmpc_rollout.py`：名义 / 残差三状态的自由 RK4 积分，已知未来等效执行器轨迹 ZOH。
- `wm_recovery/nmpc_rollout_evaluate.py`：0.1、0.5、1、2、3、4、5 秒的同起点滚动误差、逐车速/逐场景评价、保存 5 秒代表轨迹。
- `wm_recovery/tests/test_nmpc_residual_pipeline.py`：6 项离线单元测试，含合成数据端到端模型训练+5s 滚动。

项目需要已经安装：

```text
wm_recovery/nmpc_reference.py
wm_recovery/nmpc_residual_dataset.py
wm_recovery/nmpc_actuation.py
```

数据目录（根据你已经成功的 build）：

```text
artifacts/wm_recovery/nmpc_residual_dataset_40_60_01/
  train.npz
  validation.npz
  test.npz
  dataset_manifest.json
```

运行环境：Python、NumPy、PyTorch，`torch.load(weights_only=True)` 需要支持此参数的 PyTorch 版本。无需 CarSim DLL、无需 FORCESPRO、无需 MATLAB。

## 二、确认科学问题

你要求验证的是 **已知未来实际等效执行器序列条件下，车辆速度/横摆动力学的独立自由滚动**。

八维网络输入（SI）：

```text
z(t)=[vy(t), r(t), vx(t), delta_f(t), Mz_tv(t), Fx_all(t), lamda(t), mu_hat(t)]
```

用于滚动的 `delta_f, Mz_tv, Fx_all, lamda, mu_hat` 按照已经记录在 CarSim 数据集里的未来真实等效外部量提供。

名义动力学：

```math
q=[vy, r, vx]
dq_nom/dt=f_NMPC(q,a(t))
```

残差动力学：

```math
dq_res/dt=f_NMPC(q,a(t))+Delta_f_theta([q,a(t)])
```

其中 `a(t)=[delta_f,Mz_tv,Fx_all,lamda,mu_hat]`。

**每个 RK4 阶段都用模型自己预测的 q；不会把未来的 CarSim vy/r/vx 注入求导、积分或残差网络。**

用户 NMPC 时域是 `N=50, Ts=0.1s, 总计5s`；本离线数据使用 `dt=0.01s`，因此 5s=500 个原始 CarSim 采样区间。相当于 50 个 0.1s NMPC 阶段、每阶段可见 10 个 10ms 外部样本。此处不修改求解器的积分设置。

这是一种**条件预测**；不评价“仅通过四轮转矩命令和转向命令、无未来实测力时”所能做到的完整执行器自主预测，也不构成闭环 NMPC 物理验证。

## 三、安装

下载 `wm_nmpc_residual_training_rollout_v1.zip` 到 Windows 的 Downloads 目录。

PowerShell 项目根目录执行：

```powershell
conda activate drl_mpc_carsim

Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

Expand-Archive `
    -Path "$HOME\Downloads\wm_nmpc_residual_training_rollout_v1.zip" `
    -DestinationPath . -Force
```

该 ZIP 新增四个文件和一个测试文件，不覆盖你的参考模型或已完成的数据集。

## 四、回归测试

```powershell
python -m unittest wm_recovery.tests.test_nmpc_residual_pipeline -v
```

预期：

```text
Ran 6 tests
OK
```

本次已在隔离环境用 PyTorch+NumPy 和合成数据运行，六项全部通过；合成已知残差成功验证了 5s 的纯物理与残差积分差异、Train-only 归一化、时间对齐、样本分段、模型加载与校验失败拒绝。**没有在你的真实 Windows CarSim 数据上运行模型训练与 5s 评价，需按后面的命令验证。**

## 五、第一轮：只训练 10 epochs 验证管线

```powershell
python -m wm_recovery.nmpc_residual_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --output artifacts/wm_recovery/nmpc_residual_mlp_10ep_01 `
    --epochs 10 `
    --batch-size 256 `
    --lr 0.001
```

输出：

```text
nmpc_residual_mlp_10ep_01/
  best.pt                   # 验证集表现最佳的残差网络
  normalization.json        # Train-only 输入/输出归一化
  history.json              # 训练 Loss 与验证 Loss
  training_report.json      # 数据签名、最佳 epoch 和物理局限声明
```

这里只访问 `train.npz` 和 `validation.npz`，不会用 `test.npz` 参与归一化、早停或模型选择。训练损失是按训练集残差通道标准差归一化后的 MSE。训练用 `sample_weight`，验证 Loss 不受训练样本权重影响。当前数据集默认 neutral 权重 1.0。

**数据归一化只从 Train 计算**；`mu_hat` 恒定为 0.8 时自动使用安全的 scale=1，不会除零。

若输出目录已存在，程序拒绝覆盖，请改用 `_02` 名称。

## 六、第一轮：在 Validation 做 5s 滚动评价

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_residual_mlp_10ep_01/best.pt `
    --split validation `
    --anchor-stride 50 `
    --max-horizon 500 `
    --output artifacts/wm_recovery/nmpc_rollout_val_10ep_01
```

输出：

```text
nmpc_rollout_val_10ep_01/
  rollout_report.json
  example_5s.npz
```

`rollout_report.json` 内容：
- `horizons["0.10s"]` ... `horizons["5.00s"]`
- 每个时域的名义模型和残差模型：**endpoint_rmse_q_si**（终点 RMSE）、**endpoint_mae_q_si**、**path_rmse_q_si**（从起点到该时域所有中间步的整段 RMSE）
- 同一指标的相对改善百分比（正值=残差更好，负值=残差更差）
- `by_speed`：v40、v60 初速分类
- `by_scene`：逐真实场景对照
- `num_anchors`：选用多少起点
- `notes`：明确当前仅为已知未来执行器条件预测

`example_5s.npz`：
- `scene`：代表性 5 秒场景名称
- `t_from_start_s`：501 个时间戳（0～5s）
- `q_true_si`、`q_nominal_si`、`q_residual_si`：形状都为 (501,3)
- `known_external_si`：形状 (500,5)，仅用于预测中的外部执行器序列

统计时不会跨任何 `segment_id` 边界；将真实 q 仅用于预测起点初始条件和预测后指标，绝不会用于中间预测状态纠偏。

默认 `--anchor-stride 50`：每个连续段每隔 50×10ms=0.5s 启动一个新滚动预测窗口。**窗口可以重叠，报告不是独立样本置信区间。**

根据你上传的 `dataset_manifest.json` 计算，5s 的窗口在此配置下预计：
- Train：69（对应 23 条 5s 足够长的轨迹）
- Validation：15（5 条轨迹）
- Test：6（仅 2 条轨迹）

所以 5s 的 Test 评价在场景覆盖方面还比较窄；必须同时报告逐场景结果。其他已经清洗截断或 `cancellation` 拆段的轨迹仍用于更短的预测时域。

## 七、正式训练：100 epochs 上限，按 Validation 早停

10ep 试验确认数值稳定后：

```powershell
python -m wm_recovery.nmpc_residual_train `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --output artifacts/wm_recovery/nmpc_residual_mlp_100ep_01 `
    --epochs 100 `
    --patience 15 `
    --batch-size 256 `
    --lr 0.001
```

这是一个**从头训练**的独立实验，不会默认接续 10ep 模型。

接着使用最佳模型进行 Validation 完整滚动评价：

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --split validation `
    --anchor-stride 50 `
    --max-horizon 500 `
    --output artifacts/wm_recovery/nmpc_rollout_val_100ep_01
```

**确定训练策略并在 Validation 上选定模型后，才进行最终 Test**：

```powershell
python -m wm_recovery.nmpc_rollout_evaluate `
    --dataset artifacts/wm_recovery/nmpc_residual_dataset_40_60_01 `
    --checkpoint artifacts/wm_recovery/nmpc_residual_mlp_100ep_01/best.pt `
    --split test `
    --anchor-stride 50 `
    --max-horizon 500 `
    --output artifacts/wm_recovery/nmpc_rollout_test_100ep_01
```

不要因为 Test 结果不佳就反复用 Test 调整模型，否则它不再是独立测试。

## 八、阅读输出文件

例如，用 Python 查看 5 秒终点与整段的三通道 RMSE：

```python
import json
from pathlib import Path

path = Path(
    "artifacts/wm_recovery/"
    "nmpc_rollout_val_100ep_01/rollout_report.json"
)
report = json.loads(path.read_text(encoding="utf-8"))
five = report["horizons"]["5.00s"]

print("5 秒窗口数:", five["windows"])
print("名义模型终点 RMSE:", five["nominal"]["endpoint_rmse_q_si"])
print("残差模型终点 RMSE:", five["residual"]["endpoint_rmse_q_si"])
print("名义模型整段 RMSE:", five["nominal"]["path_rmse_q_si"])
print("残差模型整段 RMSE:", five["residual"]["path_rmse_q_si"])
print("终点改善百分比:", five["endpoint_rmse_improvement_pct"])
```

通道固定为 `[vy (m/s), r (rad/s), vx (m/s)]`，不同单位不应未经归一化混成一个三维欧氏误差。

## 九、注意的物理和训练边界

1. 单步标签是 10ms 等效加速度残差，不是瞬时真值导数；预训练最优不保证 5s 最优。
2. 当前 `Mz_tv` 基于 CarSim 轮胎力估算。满足用户要求的“已知未来等效外部量”条件，但还没有验证与 NMPC 可执行附加横摆力矩的完全等价。
3. 当前 `neutral=0.5` 是建模约定，非直接测得。报告应同时审阅带外/中性工况与场景误差。
4. 当前 2 个初始车速、名义 `mu_hat=0.8`，不能证明多附着/任意车速泛化。
5. 如果出现长时域数值发散，评价程序会抛出异常，而不是把发散窗口悄悄移除后报告虚假的改善。
6. 不能把本条件开环结果当作完整闭环 NMPC 安全性和求解性能验证。

完成本版本后，**下一步才考虑引入多步滚动损失微调，或测试分布外附着/初速**。若本次单步 MLP 在 5s 时域不如名义模型，先看 `by_scene` 及 `example_5s.npz` 诊断，而不是马上增加网络规模。
