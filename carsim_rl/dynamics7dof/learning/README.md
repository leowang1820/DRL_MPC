# 第一层：7DOF 动力学残差学习与在线观察

本包是可运行的实验框架，不是经过全工况验收的控制器。它保留原 7DOF、轮胎拟合和 CarSim/SAC 控制输入；不创建 Risk 标签，不接管车辆。

2026-09-28 更新：新增左右镜像结构约束、按工况平衡损失、分时段 RMSE/MAE 和开发对照标记。
新训练默认使用镜像结构，旧检查点仍保持原来的预测行为，不会在加载时偷偷变成新模型。
v2 使用方法与适用条件见 [README_V2.md](README_V2.md)；以下记录同时保留 v1 的实验历史。

进一步的文献依据、多步仿真误差训练与限制见 [RESEARCH_BASIS_V3.md](RESEARCH_BASIS_V3.md)。
v3 检查点可以冻结旁路预测，但禁止直接用旧导数目标在线更新；是否改善应看完整回放报告。

## 1. 当前已经完成什么

- 原始采集 → 有实际时钟、带延迟的残差监督数据集。
- 5 个独立初始化、独立 bootstrap 的残差网络。
- 原 7DOF ＋网络均值补偿，以及归一化网络分歧。
- 完整回合隔离的训练/验证；训练内部还留出小块历史样本保护在线更新。
- CarSim / 原模型 / 补偿模型三条连续预测曲线，全部只初始化一次。
- 顺序数据回放和可选真实 CarSim 在线观察入口；候选网络验证后才替换观察器里的模型。
- 模型/数据合同、原始文件哈希、版本、回退记录和不覆盖旧结果。

2026-09-28 已运行：105 项测试、两组真实采集的数据准备、GPU 上 80 轮训练、两个回合的连续预测及顺序在线更新回放。真实 CarSim 在线观察入口通过模拟接口测试，本次没有替用户启动新的真实求解任务。

## 2. 模型和数据含义

```text
z = [vx, vy, r, omega_FL, omega_FR, omega_RL, omega_RR,
     delta_FL, delta_FR, T_FL, T_FR, T_RL, T_RR, mu,
     memory_ax, memory_ay]                       共 16 维

y = dx_CarSim/dt - f_7DOF(z)                    共 7 维
f_augmented(z) = f_7DOF(z) + mean_k network_k(z)
```

状态、输入、输出使用 SI。T 是实际轮端 `My_Dr + My_Bk`，保留负制动力矩；delta 是实际前轮角，而不是方向盘角或转向指令。输出单位依次为 m/s²、m/s²、rad/s²、四个 rad/s²。

原模型有载荷记忆，所以准确写法是 `f(x,u,mu,memory)`，不是七维状态自身闭合的 `f(x,u)`。记忆从零初始化一次，之后按名义物理模型的轮胎合力连续传递；不能把网络补偿的总加速度直接当作轮胎力写回记忆。

### 导数标签不“偷看未来”做预测

使用中心点前后各 5 个样本，共 11 点的局部二次多项式估计中心时刻状态导数。1 ms 采样时，标签延迟 5 ms 才能到达：

```text
t 时刻：已知 x/u/mu/载荷记忆 → 预测并保存
t+5 ms：11 点窗口完整 → 得到 t 时刻的导数标签 → 评价旧预测 → 才允许更新
```

未来采样只参与延迟到达的标签，不进入 t 时刻的特征。排除初始化、结束/错误邻域、低速、模型保护以及窗口内过大的转角/扭矩突变。每 5 个采样点保留一条标签，约 200 Hz，不把 8000 行相邻数据当作 8000 个独立实验。

这些仍是实验性监督标签：滤波窗、实际输入时间戳、遗漏物理项会影响它们。不是精确无噪声的真实导数，也不是 Risk 标签。

## 3. 文件分工

| 文件 | 功能 |
|---|---|
| `data.py` | 16 维特征、载荷记忆、延迟导数标签、完整回合划分及数据保存 |
| `symmetry.py` | 左右车轮交换及横向量变号；镜像结构的物理假设 |
| `ensemble.py` | 5 个 MLP、标准化、bootstrap 训练、检查点、分歧与回退 |
| `dynamics.py` | 可调用的 `Learned7DOF` 增强导数和积分器 |
| `online.py` | 有界缓存、历史回放、延迟标签、候选更新和验证门槛 |
| `workflow.py` | 三条连续预测曲线，以及按时间顺序回放在线更新 |
| `__main__.py` | 命令行入口，使用 `python -m dynamics7dof.learning ...` |

不要右键直接运行包内文件。PyCharm 工作目录仍是 `E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl`，解释器仍是 `drl_mpc_carsim`。

## 4. 先看已经生成的结果

根目录：`artifacts/dynamics7dof/residual_framework_v1/`。

- `dataset/`：本次实验数据集与来源清单。
- `model/ensemble.pt`：初始离线训练网络，5 个成员，隐藏层 64×64。
- `model/training_report.json`：训练历史、内部保护集和独立验证指标。
- `evaluation_training_episode/comparison.html`：训练回合的三条曲线。
- `evaluation_held_out_episode/comparison.html`：未参与离线训练的另一个回合。
- `online_replay_guarded/online_report.json`：在线顺序回放，含每次候选接受/拒绝。
- `online_replay_guarded/shadow_ensemble.pt`：在线回放后的实验网络，不能继续把该回放回合称为独立验证数据。
- `RUN_REPORT.md`：本次实际结果和限制。

`online_replay_held_out/` 保留早期门槛版本的开发结果；最终在线回放以 `online_replay_guarded/` 为准。

## 5. 如何重跑离线流程

以下路径变量供 PowerShell 使用。为了不覆盖已完成结果，示例使用新的输出名称：

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
$residualTrainCapture = ".\artifacts\dynamics7dof\open_loop_20260927_233218_823895\capture"
$residualValCapture = ".\artifacts\dynamics7dof\open_loop_20260927_223412_368297\capture"
$residualWork = ".\artifacts\dynamics7dof\residual_my_run_01"
```

### A. 数据集

```powershell
python -m dynamics7dof.learning build --train $residualTrainCapture --validation $residualValCapture --output "$residualWork\dataset"
```

多个训练回合可以重复写 `--train 路径`，验证回合同理。必须有不同的训练和验证回合；拒绝相同路径、同一物理录制的副本，以及大量近乎相同的跨集合标签窗口。只有实际 solver T 完整的数据才允许进入学习，不使用旧版名义时间作静默替代。

### B. 训练

```powershell
python -m dynamics7dof.learning train --dataset "$residualWork\dataset" --output "$residualWork\model" --epochs 100 --device cuda --symmetry none --loss-weighting global
```

没有可用 GPU 时显式改成 `--device cpu`。不会自动安装任何包。
上面显式保留 v1 的无镜像、全局损失结构；v2 改用 `--symmetry mirror --loss-weighting episode-rms`。

输入均值/尺度、输出尺度只从拟合样本计算，不包含验证集或历史保护集。离线 bootstrap 按 50 ms 小块采样，以降低相邻样本重复带来的过度自信；它不替代独立工况验证。

训练内部每隔一组时间块保留 0.5 s 作为历史保护样本，并在拟合边界留 20 ms 间隔。这不是“独立验证集”，不能用其改善宣称泛化。

### C. 补偿前后连续预测

```powershell
python -m dynamics7dof.learning evaluate --checkpoint "$residualWork\model\ensemble.pt" --capture $residualValCapture --output "$residualWork\validation_plot"
```

打开 `validation_plot/comparison.html`。先看覆盖率、是否停止，再看逐状态 RMSE 和相位/幅值。训练回合报告会明确标记 `TRAINING_EPISODE`。

已被在线更新使用的回合标记为 `ADAPTATION_EPISODE_NOT_HELD_OUT`，不能因原来列在 validation 中就继续声称其独立。

## 6. 在线学习策略及实际调用

### 先用记录的数据按时间回放

```powershell
python -m dynamics7dof.learning shadow --checkpoint "$residualWork\model\ensemble.pt" --capture $residualValCapture --adapt --output "$residualWork\online_replay"
```

不带 `--adapt` 就只记录预测，不更新网络。这条命令不运行 CarSim。它拒绝把初始模型已经训练/适应过的回合伪装成新的在线检验。

### 然后才在真实 CarSim 仿真中打开观察器

当前已经生成的初始网络可这样加载。先仅预测观察，不更新：

```powershell
python -m dynamics7dof.open_loop --run --mu 0.5 --front-amplitude-deg 3 --frequency-hz 0.6 --throttle 0 --brake-mpa 0 --learning-model ".\artifacts\dynamics7dof\residual_framework_v1\model\ensemble.pt"
```

需要试验在线更新时，再添加 `--learn-online`。它仍然只改变旁路观察器里的网络，不影响方向盘、油门、制动命令：

```powershell
python -m dynamics7dof.open_loop --run --mu 0.5 --front-amplitude-deg 3 --frequency-hz 0.6 --throttle 0 --brake-mpa 0 --learning-model ".\artifacts\dynamics7dof\residual_framework_v1\model\ensemble.pt" --learn-online
```

新实验目录的 `learning_shadow/` 保存观察器指标与版本模型。车辆由原来的开环输入驱动；没有 Residual-MPC、SAC 更新或学习控制接管。

同一次运行还会自动生成 `learning_comparison_initial/comparison.html`，显示 CarSim、原 7DOF、
固定初始网络补偿的三条连续曲线。即使启用在线更新，这张图仍使用仿真开始前的检查点；
不能把回合结束后学到的权重重放到回合开头，并称其为在线预测。真正按时间更新的表现见 learning_shadow 日志。

### 候选网络如何更新

1. 最新缓存最多 4096 条干净标签，历史回放最多保留 2048 条。
2. 默认累计每 200 条新标签才尝试一次，约对应 1 s 有效采样。
3. 最近数据前 80% 用于拟合，后 20% 作为此次更新的时间后置检查，边界再次剔除延迟窗口。
4. 训练批次一半来自近期数据、一半来自历史回放；只训练候选副本，3 轮、小学习率。
5. 输入/输出标准化保持冻结，不通过重算尺度把旧网络含义改变；输入覆盖边界只根据候选拟合数据扩展。
6. 候选必须改善近期后置检查、保持历史保护集总体误差，并满足逐通道退化上限，才能替换观察器里的当前模型；否则丢弃候选。
7. 日志先保存更新前的预测，再计算到达的标签误差。不会用刚训练完的网络重新评价过去预测冒充在线效果。

这些门槛只检查已经看到的数据，不能保证下一阶段不退化。本次在线回放就出现了横向导数总体误差上升，故不建议直接把在线补偿接入控制。

预测日志只保留最后 10000 条，事件保留最后 2000 条，累计误差统计覆盖所有已完成标签。在线更新会增加墙钟耗时，目前没有硬实时保证。

## 7. 不确定性与回退

每个成员预测一个七维残差。成员均值用于补偿，成员标准差按训练得到的各输出尺度归一化，再合成为无量纲 `U_f`。它是认知不确定性的**代理**，不包含完整观测噪声建模。

- 输入超出训练覆盖边界：回到原 7DOF。
- 网络分歧超过阈值：回到原 7DOF。
- 补偿超过训练残差经验幅值：限制补偿并记录。
- 物理模型触发保护：停止精度对比或回退，不能把限幅结果当正常样本。

阈值目前来自训练分歧的经验分位数，不是经过校准的置信水平。低分歧不能保证准确；经验幅值限幅也不等于能量、轮胎摩擦圆或闭环稳定性证明。

高不确定状态可以为后续动力学学习提供数据，**本阶段绝不使用它直接生成新的稳定风险标签**。Risk Network、受控恢复时间、DRL-MPC 的接入属于后续独立阶段。

## 8. 下一步真正需要补什么数据

现有训练回合主要是 5°、0.5 Hz、无主动驱动/制动的转向滑行，不能覆盖驱动/制动和不同附着。建议先完成一个独立的 3°、0.6 Hz 滑行回合：先不学习，只采集，然后用固定初始网络 evaluate，看同类但未见工况能否改善。

随后将已有大转角滑行与驱动/制动回合共同用于训练，另留出新回合做验证。再逐步覆盖不同速度、频率、附着及联合驱动制动。不要把随机打散相邻采样行当独立验证，不要为了提高某张图的效果反复调整并继续称那张图为“测试集”。

## 9. 参考与本实现的区别

- [Deep Ensembles 原论文](https://arxiv.org/abs/1612.01474)：独立网络及预测分歧思想。本实现是确定性残差回归 ensemble，没有复刻论文的完整概率似然模型。
- [PETS 原论文](https://arxiv.org/abs/1805.12114)：动力学 ensemble 与模型不确定性思路。本实现没有 PETS 的完整概率动力学、轨迹采样或 MPC 优化器。
- [PyTorch 模型保存/加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：保存参数与必要状态；本包加载使用 `weights_only=True`，仍只应打开可信检查点。
