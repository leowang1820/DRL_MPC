# wm_recovery：CarSim 对齐与接下来怎么操作

本指南对应 2026-11-04 的代码。按照当前约定，算法保留论文参数；CarSim 的值后面共同微调。本次只补充说明、只读核对配置，没有修改 CarSim 或运行仿真。

工作顺序是：**环境与离线示例 → 目标配置核查 → 车辆/信号对齐 → 新采集接口 → 原 7DOF 基线 → 正式采集与 WM 训练 → 独立评价**。

## 1. 现在就能做：激活环境、确认工作目录

在 PowerShell 终端逐行执行：

```powershell
conda activate drl_mpc_carsim
Set-Location 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl'
python -c "import sys; print(sys.executable)"
python -m wm_recovery --help
```

解释器应是 `D:\anaconda3\envs\drl_mpc_carsim\python.exe`。帮助应列出 `inspect`、`check-data`、`evaluate-nominal`、`train`、`calibrate`、`evaluate`、`demo`。

当前没有 `collect` 或 `run-carsim` 子命令。看到这些离线命令可用，不等于已经接通 CarSim。

## 2. 现在就能做：看已有报告，或自己跑合成示例

已有报告可直接打开，不必重新训练：

```powershell
Invoke-Item 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery\framework_v1_20261104\test_evaluation\report.html'
```

想亲自确认环境可运行，执行：

```powershell
python -m wm_recovery demo --output 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery\my_demo_01' --epochs 1
```

目录必须是新目录；若已存在，将末尾改为 `my_demo_02`。示例依次生成合成轨迹、训练五个成员、标定可信度阈值、生成两份独立测试报告。完成后打开：

```powershell
Invoke-Item 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery\my_demo_01\test_evaluation\report.html'
Invoke-Item 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery\my_demo_01\unknown_mu_evaluation\report.html'
```

这一步的验收是流程无报错、模型与报告生成、曲线可打开。合成示例只有 1 轮训练，不用于决定真实车辆预测精度或检测阈值。

## 3. 现在就能做：只读检查目标 CarSim 配置

确认 CarSim 当前选中的 Run 是目标车辆对应的 Run。读取已经生成的配置：

```powershell
python -m wm_recovery inspect --sim 'E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim'
```

也可将检查结果保存到新的 JSON 文件：

```powershell
python -m wm_recovery inspect --sim 'E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim' --output 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery\carsim_inventory_01.json'
```

文件必须不存在；父目录若尚不存在，先创建。检查读取的是 `.sim` 指向的已生成参数文件，不能反映尚未重新生成的界面修改。

| 检查字段 | 如何理解 |
|---|---|
| `sim_path` / `parameter_path` | 确认所查的确是目标 Run，而非另一个实验生成的入口 |
| `imports` / `exports` | 逐项看通道名称、顺序和输入模式，不能只看数量 |
| `port_counts_match` | 声明数量与列出的通道数量一致，不代表物理定义正确 |
| `direct_channel_names_match` | 是否包含当前代码需要的四转矩＋左右轮角六个通道，仅检查名称集合 |
| `duplicate_exports` | 检查重复输出，以免转成名称字典时丢失列 |
| `settings.MU_ROAD_CONSTANT` | 生成配置中的常值路面附着设置；正式使用还要核对路面组合/空间变化配置 |
| `paper_reference_parameters` | Python 论文参数目标，不是从 CarSim 读出的整车等效值 |
| `vehicle_parameter_alignment` / `physical_interface_verified` | 当前静态检查器始终返回未验证；不能靠修改输入名称使它自动认定物理验证通过 |

此次重新读取仍指向 `Python_RL_CSRvx_DLC_frenet`，5 输入/61 输出，路面常值 μ=0.5，第五输入为 `IMP_STEER_SW`。这解释了为什么目前不能直接开始 μ=0.8 的参数一致训练。具体差异见 [配置记录](CARSIM_STATUS_20261104.md)。

## 4. 后续共同调整：让车辆物理接口与论文模型一致

这一阶段先确定独立实验 Run 与其引用的数据集，再逐项微调。复制 Run 时还要核对它是否仍引用旧 Run 共用的车辆、转向、轮胎、路面数据集；修改共用数据集会影响其它实验。不要只编辑生成的 `Run_all.par` 并把它当作永久配置。

### 4.1 四轮转矩与前轮角

当前 Python 接口期望以下通道。表中的 1～6 是建议排列；代码可按已核对的实际顺序映射。

| 算法动作 | CarSim 输入 | 接口单位 | 所需核验 |
|---|---|---|---|
| T_FL | `IMP_MYUSM_L1` | N·m | 左前轮外加转矩 |
| T_FR | `IMP_MYUSM_R1` | N·m | 右前轮外加转矩 |
| T_RL | `IMP_MYUSM_L2` | N·m | 左后轮外加转矩 |
| T_RR | `IMP_MYUSM_R2` | N·m | 右后轮外加转矩 |
| δ_f | `IMP_STEER_L1` | deg | 左前轮主动转向输入 |
| 同一个 δ_f | `IMP_STEER_R1` | deg | 右前轮主动转向输入 |

算法仍为 **5 维动作**；映射到 CarSim 时，公共转角复制到左右两个通道，所以边界为 **6 个输入**。网络内部采用 rad，`Command.to_imports()` 转成 deg。不能把当前第五个方向盘角通道直接当作前轮角，也不能只改端口数量而不改模型。

本机 CarSim 2022.1 输入通道表将 `IMP_MYUSM_*` 定义为簧下部分施加到车轮的外力矩；它不是完整电机模型。用户要求的负值保持为反向外加转矩，不转换为摩擦制动。内部传动、摩擦制动是否还有贡献，要通过输出记录核验。

本机通道表也明确说明，`IMP_STEER_L1/R1` 描述转向系统产生的转角，不包括悬架跳动/侧倾引起的被动转向。因此**左右指令相同，并不能保证实际左右轮角相同**。需要共同核对外部转向模型选项、toe/柔顺性等贡献，最后检查 `Steer_L1` 与 `Steer_R1` 实际曲线。

目前数据加载器要求实际左右角差在 `1e-10 rad` 以内。这是首版代码的严格同角检查，不是已标定的 CarSim 数值容差。若以后存在仅由数值求解造成的微小差异，需一起确定容差并修改检查，不能先把左右实际角取平均来掩盖真实差异。

### 4.2 参数对齐目标

| 参数含义 | 论文/代码目标 | 对齐时注意 |
|---|---|---|
| 整车总质量 | 2026 kg | 不是仅簧载质量 `M_SU` |
| 整车绕质心横摆惯量 | 4095 kg·m² | 不是仅簧载横摆惯量 `IZZ_SU` |
| 单轮旋转惯量 | 2 kg·m² | 核对轮胎＋轮毂组合值，不能只看 `IT` |
| 质心到前/后轴距离 | 1.265 / 1.895 m | 核对整车质心及原点定义 |
| 轴距 | 3.16 m | 与前后轴位置一致 |
| 前/后轮距 | 1.605 / 1.605 m | 同时核对定义与单位 |
| 整车质心高度 | 0.59 m | 不是仅簧载部分质心高度 |
| 轮胎有效滚动半径 | 0.353 m | 不是轮胎外形半径 |
| 名义路面附着 | 0.8 | 首版训练与名义验证一致 |

分项质量、质心、惯量如何调整并不唯一，按你要求留待共同微调。后续以目标配置的新求解回显核对总量，不把历史 `LastRun_echo.par` 当作当前配置已验证的证据。

轮胎拟合也需要单独核对：相同附着、相同轮径，不意味着 CarSim 轮胎力特性与原 7DOF 拟合曲线相同。原 7DOF 省略的自由度、胎力瞬态和其它模型差异可形成 WM 要学习的残差；先记录和解释这些差异，不要求 CarSim 与 7DOF 逐点完全重合。质量/惯量匹配本身不能证明轮胎模型匹配。

## 5. 下一段需要编写的代码：新接口采集与时间对齐

这部分尚未交付。计划复用底层 DLL 封装和已有求解器时钟读取方法，新增针对已核验通道的采集桥接、开环激励及轨迹转换。旧 `dynamics7dof` 采集器按 3 输入/25 输出固定，不能直接用于当前目标车辆。

采集桥接需要同时保存以下内容：

| 数据 | 候选 CarSim 来源 | 写入算法前的处理 |
|---|---|---|
| vx、vy | `Vx`、`Vy` | 核对速度参考点、坐标方向、当前输出单位，统一为 m/s |
| r | `AVz` | 核对正方向和单位，统一为 rad/s |
| 四轮 ω | `AVy_L1/R1/L2/R2` | 核对轮序、正方向和单位，统一为 rad/s |
| 两前轮实际角 | `Steer_L1`、`Steer_R1` | 核对是否为所需实际轮角，统一为 rad，检查同角 |
| 实际外加轮端转矩 | 核验后的外加转矩信号或求解器变量 | 不直接用 `My_Dr_*`、`My_US_*` 或 `My_WC_*` 猜测替代；必要时新增明确含义的诊断输出 |
| 算法命令 | 采集程序发出的五维动作 | 与实际输入分别存储 |
| 真实时刻 | 求解器时钟与原始导出时序 | 检查当前积分选项下的前后帧对齐，不仅依赖 Python 循环计数 |
| 诊断信号 | 胎力、垂向载荷、驱动/制动相关转矩、加速度等 | 检查轮端平衡、内部残留、载荷记忆及单位 |

若原生输出为 km/h、deg、deg/s、rpm，换算分别为 `/3.6`、`×π/180`、`×π/180`、`×2π/60`；这些规则只在确认该通道实际单位后使用。原始输出和换算结果均保留，避免二次换算。

默认时间设置：CarSim 当前配置为 1 ms 求解步长；WM 为 10 ms 预测周期；7DOF 内部为 0.5 ms 积分子步。三者作用不同。

```
t_i 的状态 x_i + 在 [t_i, t_i+10ms) 内施加的实际输入
                         ↓
                 t_i+10ms 的状态 x_(i+1)
```

v1 数据约定要求每个预测区间的实际输入可以按保持值处理。实际转角/转矩若在这 10 ms 内变化，就应记录子步输入并扩展物理积分/数据接口，或共同选择更小预测周期。不能简单每 10 行取 1 行，更不能拿区间末端输入解释前一段状态变化。

初始载荷记忆需要有明确的初始化/预热方法，记录到元数据；不能直接把传感器 Ax/Ay 当作原模型的力导出记忆，也不能每步清零。

## 6. 接通后先验证物理链路，再做大批量采集

| 试验 | 需要看到的证据 |
|---|---|
| 前进行驶、零外加转矩、零转角 | 记录基线；区分空气阻力/滚阻与内部驱动或制动残留，不把速度自然下降一概当作接口错误 |
| 四个车轮依次施加小幅正/负转矩 | 轮序与施加位置一致，正负作用符合约定，记录实际执行与车轮响应 |
| 四轮相同转矩 | 纵向响应合理，同时检查非预期横摆来源 |
| 左右差动转矩 | 横摆响应方向与坐标约定一致 |
| 正/负公共前轮角 | 实际两前轮同角，数值、方向和时序正确 |
| 转向＋转矩联合输入 | 采集链路、实际输入记录与状态时序仍一致 |

试验速度、转矩/转角幅值、持续时间和判定容差需要在目标车辆上一起确定，当前指南不把任意数值写成验收标准。

通过后，将录制整理为 [DATA_FORMAT.md](DATA_FORMAT.md) 的轨迹。先比较原 7DOF 单步和连续预测；发现数量级、符号或时间偏差时，先修复数据链路，不用残差网络掩盖错误。记录正常模型差异及覆盖率，再进入 WM 训练。

## 7. 数据准备好以后：检查、训练、标定、评价

以下命令只有在采集与转换完成后可用。`data\wm_recovery\...` 是约定示例目录，当前没有自动生成真实数据；参数未核验时也不能把元数据改成 `verified` 来跳过检查。

### 7.1 按完整轨迹与工况组划分

| 分区 | 工况 | 用途 |
|---|---|---|
| `train` | μ=0.8，多条不同速度/操纵轨迹 | 网络梯度训练和完整轨迹 Bootstrap |
| `validation` | μ=0.8，独立工况组 | 可信度阈值标定 |
| `test` | μ=0.8，独立工况组 | 名义泛化误差和误报评价 |
| `unknown_mu_test` | 非训练附着值，仍使用名义 μ=0.8 预测 | 预测失效与监测表现；不输入真实 μ |

至少两条不同训练轨迹只是 Bootstrap 最低要求，正式实验需要更丰富的激励。用于调参的测试结果应改作开发数据，另留未见工况作最终测试。恢复试验需等原 π_s/SE 接入后补充。

### 7.2 检查数据并运行原 7DOF 对照

```powershell
$wmData = 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\data\wm_recovery'
$wmResults = 'E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\wm_recovery'
python -m wm_recovery check-data "$wmData\train_a" "$wmData\train_b" "$wmData\validation" "$wmData\test" "$wmData\unknown_mu_test"
python -m wm_recovery evaluate-nominal --episode "$wmData\test" --max-substep-s 0.0005 --output "$wmResults\nominal_01"
```

`check-data` 只检查数据约定与分区，不自动验证车辆参数。原 7DOF 对照入口生成 JSON/NPZ；HTML 曲线报告由后面的 WM `evaluate` 入口生成。

### 7.3 训练五成员 World Model

```powershell
python -m wm_recovery train --episodes "$wmData\train_a" "$wmData\train_b" --output "$wmResults\model_01" --prediction-dt-s 0.01 --history-steps 10 --physics-substep-s 0.0005 --rollout-steps 5 --epochs 20
```

默认训练滚动长度为 5 步，即 50 ms，**不等于已用 1 s 损失训练**；后面会独立检查 1 s 预测。训练长度、轮数与损失权重后续根据独立开发数据调整。

### 7.4 用独立名义数据标定阈值

```powershell
python -m wm_recovery calibrate --checkpoint "$wmResults\model_01\world_model.pt" --episodes "$wmData\validation" --output "$wmResults\confidence_01.json"
```

阈值仅从名义验证数据获得，不能为了让未知工况容易触发而拿它的测试结果反复标定。默认窗口/分位数是首版实验设置，不是已验证的控制边界。

### 7.5 分别评价名义与未知附着

```powershell
python -m wm_recovery evaluate --checkpoint "$wmResults\model_01\world_model.pt" --calibration "$wmResults\confidence_01.json" --episode "$wmData\test" --output "$wmResults\test_01" --horizon-s 1
python -m wm_recovery evaluate --checkpoint "$wmResults\model_01\world_model.pt" --calibration "$wmResults\confidence_01.json" --episode "$wmData\unknown_mu_test" --output "$wmResults\unknown_mu_01" --horizon-s 1
Invoke-Item "$wmResults\test_01\report.html"
Invoke-Item "$wmResults\unknown_mu_01\report.html"
```

先看有效覆盖率、单步/1 s/连续预测，再比较原 7DOF、WM 和保持当前状态基准。随后检查“误差变大时方差/残差是否变大”，对比仅方差、仅残差和联合三种监测的误报与延迟。预测残差只在下一观测到达后可用。

本版未知附着测试支持每条轨迹一个常值 μ；行驶中附着突变还需要扩展工况时间序列/事件记录。即使检测或精度改善，也不能直接当作 1 s 受控恢复风险已准确；那需要后续的 π_s、SE、Risk 标签及独立恢复试验。

## 本轮接下来最具体的工作

现在可先执行第 1～3 步。共同检查 `Python_RL_CSRvx_DLC_frenet` 的转向、输入输出和参数定义之后，下一次代码工作是第 5 步：**新增符合四轮外加转矩＋实际前轮同角约定的采集桥接，先得到一条可核验的小幅开环轨迹。** 这条轨迹通过时序/单位/物理输入检查，再开始批量数据采集。
