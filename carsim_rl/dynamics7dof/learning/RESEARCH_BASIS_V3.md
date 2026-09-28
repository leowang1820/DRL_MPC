# v3：有文献依据的 7DOF 残差辨识

核对日期：2026-09-28。论文均核对出版方/作者公开版本，而非仅看二手介绍。
这里区分文献方法、项目适配和未获验证的假设，不把工程实现冠以论文的理论保证。

## 1. 采用哪些论文，具体采用什么

| 文献 | 已核对出处 | 本项目采用的部分 | 不声称复现的部分 |
|---|---|---|---|
| Donati、Mammarella、Dabbene、Novara、Lagoa，2025，*Combining off-white and sparse black models in multi-step physics-based systems identification* | Automatica 179, 112409；[DOI](https://doi.org/10.1016/j.automatica.2025.112409)、[作者机构全文](https://iris.polito.it/retrieve/b71ae976-e5d5-43d6-bb53-7716a07a0c03/1-s2.0-S0005109825003036-main.pdf) | §2.1 的物理模型加残差、递推多步误差最小化 | 不联合辨识车辆参数，不实现论文稀疏基函数字典，不引用其稀疏恢复或参数误差界 |
| Beintema、Schoukens、Tóth，2023，*Deep subspace encoders for nonlinear system identification* | Automatica 156, 111210；[DOI](https://doi.org/10.1016/j.automatica.2023.111210)、[作者全文](https://arxiv.org/pdf/2210.14816)、[实验代码](https://github.com/GerbenBeintema/encoder-automatica-experiments) | §3.1 的短序列递推误差训练，§5.4 的 k-step 误差诊断 | CarSim 已提供状态，所以不实现 SUBNET 编码器或创新噪声模型；不套用其一致性证明 |
| Ribeiro、Tiels、Umenberger、Schön、Aguirre，2020，*On the smoothness of nonlinear system identification* | Automatica 121, 109158；[DOI](https://doi.org/10.1016/j.automatica.2020.109158)、[作者全文](https://arxiv.org/pdf/1905.00820) | 将长序列优化难度作为截断训练的理由 | 没有优化每段初值和连接约束，因此本实现不是论文的严格 multiple shooting |
| Chrosniak、Ning、Behl，2024，*Deep Dynamics: Vehicle Dynamics Modeling With a Physics-Constrained Neural Network for Autonomous Racing* | IEEE Robotics and Automation Letters 9(6), 5292–5297；[DOI](https://doi.org/10.1109/LRA.2024.3388847)、[作者代码](https://github.com/linklab-uva/deep-dynamics) | 作为保留动力学结构、区分物理参数与自由预测量的车辆应用参照 | 该文是单轨系数学习；没有把其网络或系数范围直接移植到本项目四轮 7DOF |

以上是方法借鉴，不是整篇论文复现。原仓库代码没有被复制进工程，也没有安装额外依赖。

## 2. 本项目实际优化的问题

原训练最小化实测状态处的导数残差误差。新训练直接最小化一段预测状态与 CarSim 状态的差：

\[
\xi=(x,m),\quad x=[v_x,v_y,r,\omega_{FL},\omega_{FR},\omega_{RL},\omega_{RR}],
\quad m=[a_{x,\mathrm{force}},a_{y,\mathrm{force}}].
\]

\[
\hat x_{j+1}=\hat x_j+h\big[f_{7D}(\hat x_j,u_j,\mu,\hat m_j)
                  +\Delta f_{\theta_k}(\hat x_j,u_j,\mu,\hat m_j)\big].
\]

每个 1 ms 采样间隔使用两个 0.5 ms Euler 子步，并在每个子步更新原模型的力驱动载荷记忆。
不是一次跨越 100 ms 的大步 Euler；记录的实际输入也不是每个窗口只取第一点。

\[
L=\frac{1}{KBH}\sum_{k=1}^{K}\sum_{b=1}^{B}\sum_{j=1}^{H}
\left\|D_{e(b)}^{-1}\big(\hat x^{(k,b)}_j-x^{\mathrm{CarSim},b}_j\big)\right\|_2^2.
\]

这里的 D 是只从 FIT 状态计算的逐回合状态尺度，有原工程量纲下限；不是根据对照集调出的权重。
初值取窗口起点的实测状态，记忆取因果历史递推值；窗口内部不重置到真值，也不从未来真值更新记忆。
网络参数的梯度通过所有预测状态、轮胎力与载荷记忆递推反向传播。车辆和轮胎参数保持冻结。

这种目标用状态序列作为监督，不需要对状态差分来产生导数标签。之前的导数拟合仍可用作预训练。

## 3. 从论文到工程：明确的适配与限制

1. **窗口长度不是论文给定常数。** 本次对照 100 ms 和 400 ms，初始网络、学习率和更新次数相同。
   前者是短窗起点，后者用于检查更长记忆；并未证明 400 ms 覆盖车辆全部主要时间尺度。
   文献提示结合系统时间尺度与 k-step 曲线选择窗口，而非默认短窗成绩就能代表长时预测。
2. **每个成员独立递推。** 用独立时间块 bootstrap 训练 5 个成员；实际部署则用残差均值。
   成员曲线的均值一般不等于均值模型的曲线，最终还必须用原 NumPy 增强模型做全程回放。
3. **门控差异明示。** 训练保留输入范围门控和残差限幅，但不在不同成员各自的轨迹之间计算分歧门控。
   最终评估仍保留原来的完整分歧/OOD 门控。因此训练损失不能冒充实际观察器性能。
4. **原公式没有被平滑改写。** 保留魔术公式、组合滑移、阻尼、载荷记忆及限幅。
   在零滑移等非光滑点使用有限子梯度；有限差分梯度校验在远离这些拐点处执行。
   不能直接套用要求全局连续可微的理论结果。
5. **窗口不跨训练/内部保护边界。** 端点目标也必须处于同一分区，不跨回合，不跨时间缺口。
   内部保护窗口用于选择检查点，不是独立测试集。重叠窗口也不是新增独立实验。
6. **训练数据仍只有两个回合。** 3°回合已用于开发，不能再作为独立泛化证明。
   多附着、多速度、联合大转向和制动、不同初始状态的泛化尚需新数据。
7. **镜像约束是本项目结构先验。** 不是上述论文逐字给出的 7DOF 方法，仍仅适用平路、均匀附着、
   左右对称车辆。它消除对称直行伪横向运动，但不证明真实车辆或整个模型稳定。
8. **网络分歧不是校准置信度。** 本轮不改风险网络，不生成恢复时间标签，也不声称闭环安全。

## 4. 文件与使用方式

- `torch_physics.py`：原物理方程的可微实现，批量 float64 运算；不替换原 `model.py`。
- `rollout.py`：因果窗口准备、分区、物理递推、短序列训练、内部保护集选点和 k-step 诊断。
- `tests/test_rollout_learning.py`：数值/梯度一致性、无教师强制、分区隔离和旧接口兼容测试。
- 结果位于 `artifacts/dynamics7dof/residual_framework_v3/`；本轮结果与候选是否通过见 `RUN_REPORT.md`。

在 carsim_rl 工作目录、drl_mpc_carsim 解释器中执行；输出需改为尚不存在的目录：

```powershell
python -m dynamics7dof.learning.rollout --checkpoint ".\artifacts\dynamics7dof\residual_framework_v2\model_balanced\ensemble.pt" --output ".\artifacts\dynamics7dof\my_rollout_candidate" --horizon-steps 400 --updates 40 --batch-size 8 --lr 0.0001 --device cpu
```

这里只做离线训练，不加载 CarSim DLL。使用 CPU 是本轮小网络/float64 参考运行选择，
不表示必须卸载 GPU PyTorch；`--device cuda` 也可用，但实际硬件时间与误差需另行记录。

检查当前模型的 k-step 误差（也不会启动 CarSim）：

```powershell
python -m dynamics7dof.learning.rollout --checkpoint ".\artifacts\dynamics7dof\residual_framework_v2\model_balanced\ensemble.pt" --output ".\artifacts\dynamics7dof\my_horizon_profile" --profile-only --horizon-steps 400
```

`horizon_profile.json` 统计的是同一起点上各成员的端点误差，不是最终门控均值模型的连续回放。
最终连续回放继续使用 `python -m dynamics7dof.learning evaluate ...`。

## 5. 暂不恢复导数式在线更新

多步训练的检查点允许 `open_loop --learning-model ...` 做冻结权重旁路预测。
但它明确禁止旧的 `--learn-online` 导数式更新，防止刚按序列训练好的模型又被另一目标覆盖。
以后若做真正在线适应，应先用已完整到达的序列窗口、历史序列回放与多步验证实现同目标更新；
不能因为有一个在线开关就声称实现了论文式在线系统辨识。
