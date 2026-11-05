# 第一阶段：分布式驱动车辆的 World Model 离线框架

本模块与旧 SAC/残差学习流程独立，复用原论文的 `dynamics7dof.model` 和经过数值对照的可微物理实现。**2026-11-04 按用户最新要求，仅编写算法框架，不修改任何 CarSim 参数，也不启动 CarSim 仿真。**

## 先读哪里

| 文档 | 用途 |
|---|---|
| 本 README | 模块功能、模型定义、默认设置、运行入口 |
| [CARSIM_ALIGNMENT_GUIDE.md](CARSIM_ALIGNMENT_GUIDE.md) | 从激活环境到真实 CarSim 数据训练的逐步操作；区分现在能执行的步骤与待接入部分 |
| [DATA_FORMAT.md](DATA_FORMAT.md) | 每条轨迹的状态、输入、时间和元数据格式 |
| [CARSIM_STATUS_20261104.md](CARSIM_STATUS_20261104.md) | 已检查的目标车辆配置、参数差异与未验证项 |
| [RUN_REPORT_20261104.md](RUN_REPORT_20261104.md) | 已执行测试和合成示例结果的适用范围 |

**当前可以直接运行：配置只读检查、合成示例、已整理轨迹的检查/训练/标定/评价。新 6 通道方案的 CarSim 实时采集与数据转换程序尚未接入。** `controls.py` 是动作映射，`inspect` 是静态检查；两者都不会自动改造车辆或生成真实训练数据。下一段代码工作是按核验后的目标接口接通采集与时间对齐。

## 已确认的研究设定

| 项目 | 当前约定 |
|---|---|
| 名义动力学 | 原论文 7DOF、轮胎拟合系数和载荷记忆保持原样 |
| 质量、横摆惯量、轮惯量 | 2026 kg、4095 kg·m²、2 kg·m² |
| 轴距、前后质心距 | 3.16 m、1.265 m、1.895 m |
| 轮距、质心高度、有效滚动半径 | 前后均 1.605 m、0.59 m、0.353 m |
| 训练附着 | 7DOF 与 CarSim 均 μ=0.8；其他 μ 仅作独立未知条件测试 |
| 算法动作 | `[T_FL, T_FR, T_RL, T_RR, delta_f]`，N·m、rad |
| 转矩语义 | `IMP_MYUSM_*` 的有符号轮端外加转矩，不包含摩擦制动；负值不被转换成制动压力 |
| 转向语义 | 实际左右前轮角相等，后续 CarSim 配置需处理被动转向效应 |
| 时间与历史 | 10 ms 预测周期，前 10 个周期的状态–实际输入历史，共 100 ms；均可配置 |
| 网络 | 五个独立初始化的残差 MLP；完整轨迹 Bootstrap |
| 网络真实附着输入 | 无；`actual_mu` 只用于数据集标记和结果解释 |

原物理模型使用 `LoadMemory(ax, ay)` 保存上一微步力产生的加速度，不能直接省略。新框架显式保存、推进该记忆，并将它作为网络条件信息；不能把这套模型声称为仅由七个状态决定的无记忆映射。历史输入为过去的七维状态与六维实际输入，当前输入另行拼接，没有未来观测。

```
x = [vx, vy, r, omega_FL, omega_FR, omega_RL, omega_RR]
u_actual = [delta_FL, delta_FR, T_FL, T_FR, T_RL, T_RR]
H_t = [(x_(t-H),u_(t-H)), ..., (x_(t-1),u_(t-1))]

xhat_(t+1)^k = F_7DOF(x_t, u_t; mu_nom=0.8, memory_t)
               + MLP_k(x_t, u_t, H_t, memory_t)
```

残差单位为**下一时刻状态增量**，不是状态导数。旧学习包的导数残差检查点不能直接加载。模型文件保存物理代码指纹、原参数、归一化统计、历史定义和训练轨迹来源；物理模型变化后拒绝静默加载旧模型。

## 当前已完成

| 文件 | 负责什么 | 何时使用 |
|---|---|---|
| `__init__.py` | 包入口与动作接口导出 | 其他代码导入本模块时 |
| `__main__.py` | 解析 `python -m wm_recovery ...` 命令，调度各流程 | 日常运行入口，不直接执行包内单个文件 |
| `config.py` | 预测周期、历史长度、网络宽度、积分子步、随机种子和原物理模型指纹 | 修改实验配置；不修改 CarSim |
| `controls.py` | 校验 5 维动作，按真实通道顺序展开为四轮转矩＋左右轮角，rad→deg | 后续采集桥接；通道映射本身不证明实际轮角同角 |
| `inspection.py` | 只读解析 `.sim` 与平铺参数文件，检查通道数量、名称、重复输出和附着设置 | CarSim 生成配置后核查 |
| `offline.py` | 加载完整轨迹、检查数据约定与重复记录、原 7DOF 单步/连续/固定时域对照 | `check-data`、`evaluate-nominal` |
| `dataset.py` | 构造因果历史与载荷记忆，按完整轨迹 Bootstrap，仅用训练数据计算尺度 | 训练前的数据准备 |
| `world_model.py` | 五个独立残差 MLP、物理＋残差预测、Ensemble 方差、模型保存加载 | 训练与推理核心 |
| `training.py` | 单步＋多步滚动损失，独立优化五个成员并记录训练来源 | `train`；默认训练滚动 5 步，即 50 ms |
| `confidence.py` | 方差/预测残差的归一化、窗口统计、阈值标定、持续超限与恢复判断 | 可信度评价；历史不足时不标为可信 |
| `evaluation.py` | 独立单步、1 s、连续预测评价，以及三种可信度监测对比 | `calibrate`、`evaluate`；滚动预测不注入未来真值 |
| `report.py` | 将状态和可信度曲线、指标写入 HTML/SVG | `evaluate` 自动生成报告 |
| `synthetic.py` | 生成明确标记为合成的轨迹，并串联全流程 | `demo`，不加载 CarSim DLL |
| `tests/test_foundation.py` | 输入语义、配置解析、轨迹格式与隔离等测试 | 检查基础接口 |
| `tests/test_world_model.py` | 物理等价性、历史因果性、训练与模型存取、可信度监测测试 | 检查学习和评价实现 |

本轮未实现原 SE、稳定律 π_s、Risk Network、NMPC 接管及在线增量更新。它们属于后续阶段，且原 SE/π_s 实现仍需用户提供或确认，不能自行另写一个控制器代替。

## 运行环境与入口

推荐解释器：`D:\anaconda3\envs\drl_mpc_carsim\python.exe`。工作目录必须为：

```
E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
```

先激活环境并进入工作目录：

```powershell
conda activate drl_mpc_carsim
Set-Location 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl'
python -c "import sys; print(sys.executable)"
```

确认解释器为上述环境中的 Python，再执行：

```powershell
python -m wm_recovery --help
python -m unittest discover -s wm_recovery/tests -v
python -m wm_recovery inspect --sim "E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim"
```

### 先检查全流程能否运行

```powershell
python -m wm_recovery demo --output artifacts/wm_recovery/my_synthetic_demo --epochs 1
```

会生成五条合成轨迹：两条训练、一条验证、一条名义测试、一条独立低附着测试；随后训练五个网络、标定阈值并输出报告。默认只训练 1 轮，用途是验证流程，不是形成合格模型。所有模型和报告标记 `synthetic_demo_only=true`。

查看：

```
my_synthetic_demo/test_evaluation/report.html
my_synthetic_demo/unknown_mu_evaluation/report.html
```

### 后续接入真实轨迹

真实数据先按 [DATA_FORMAT.md](DATA_FORMAT.md) 整理，再执行：

```powershell
python -m wm_recovery check-data data/train_a data/train_b data/validation data/test
python -m wm_recovery train --episodes data/train_a data/train_b --output artifacts/wm_recovery/model_01 --epochs 20
python -m wm_recovery calibrate --checkpoint artifacts/wm_recovery/model_01/world_model.pt --episodes data/validation --output artifacts/wm_recovery/confidence_01.json
python -m wm_recovery evaluate --checkpoint artifacts/wm_recovery/model_01/world_model.pt --calibration artifacts/wm_recovery/confidence_01.json --episode data/test --output artifacts/wm_recovery/test_01
```

`data/...` 是待采集的示意路径，目前并不存在。新训练入口要求至少两条不同训练轨迹；正式不确定性实验需要更丰富的轨迹和工况，两条只是接口最低要求。

默认 CPU、float64、隐藏宽度 64、Adam 学习率 0.001、批量 32、20 轮。多步损失权重 1、归一化尺度下限、梯度裁剪 10 均是可检查的首版工程设置，尚未通过真实车辆实验标定。CLI 可覆盖主要训练参数，不使用测试集自动选模型。在线速度也尚未测定。

可信度默认使用 100 ms 窗口、连续 3 个完整窗口超限触发、连续 10 个完整窗口恢复，阈值取独立名义验证统计的 99% 分位数。它们是可配置的初值，不是安全保证，也不是恢复失败概率。异常检测时刻按下一观测实际到达时间计算。

### 只运行原 7DOF 对照

```powershell
python -m wm_recovery evaluate-nominal --episode data/test --max-substep-s 0.0005 --output artifacts/wm_recovery/nominal_01
```

输出目录与模型文件不允许覆盖。低速、载荷限幅或其它原模型保护触发时报告无效覆盖率，不把有限片段成绩冒充全程成绩。

## CarSim 现状和下一步

逐步执行入口见 [CarSim 对齐与操作指南](CARSIM_ALIGNMENT_GUIDE.md)。建议先完成环境检查、合成示例和只读检查，再一起核对车辆配置并实现新采集桥接。

只读检查结果见 [CARSIM_STATUS_20261104.md](CARSIM_STATUS_20261104.md)。目前目标 CarSim 还没有满足已确认的 μ=0.8、实际前轮同角及论文参数一致要求，因此旧采集数据没有直接转为本框架的正式训练集。

下一次共同核对车辆时，应先冻结目标 Run，核对整车质量/惯量与轮胎＋轮毂旋转惯量，处理同角转向，检查四个外加转矩的符号和内部动力/制动残留，再确定 10 ms 区间内实际输入的采样方式。原状态方程、载荷记忆及轮胎拟合都已保留，之后的参数微调可以明确记录与比较。

评价回放会使用已记录的后续实际执行器输入，这是离线模型精度评价；不能据此假定在线控制器提前知道未来执行器响应。若后续引入执行器动态，应另行明确命令→实际输入模型。
