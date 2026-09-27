# 第三阶段：一次批量验证时间、轮端扭矩和载荷记忆

本工具使用已有文件离线计算，不运行 CarSim DLL，不改 SAC，不覆盖原始采集，不改用户给定的 7DOF 参数和轮胎拟合。它已对 2026-09-27 的三组数据完成实测对比。结果不是新的监督训练标签，也不是控制接管验收。

## 1. 直接查看已完成的报告

在 PyCharm 项目树打开：

```text
artifacts/dynamics7dof/physics_verification_162739/report_v2/physics_check.md
```

同目录 `physics_check.json` 保存完整数值、状态顺序、文件哈希、参数组及适用范围。
`report_v1` 是开发过程中的首轮数值结果，最终可读报告以 `report_v2` 为准。

本次三个输入目录：

- `capture_20260927_162739_297136`：实际 T 时钟、40→30 km/h 制动前 2 s。
- `capture_20260927_155420_045453`：目标 40 km/h 的完整 DLC，13.6 s；无实际时钟，仅诊断相对间隔。
- `capture_20260927_160434_220127`：目标 30 km/h 的完整 DLC，17.7 s；无实际时钟，仅诊断相对间隔。

最新 2 s 与旧减速回合前缀重复，不能分别当作训练集和验证集来声称泛化。

## 2. 在 PyCharm 重新运行

解释器仍用 `D:\anaconda3\envs\drl_mpc_carsim\python.exe`；Terminal 切到：

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
python -m unittest discover -s dynamics7dof/tests -v
```

下面命令可作为一整行复制；输出目录必须尚不存在（重跑请把 `report_manual_01` 改成新名字）：

```powershell
python -m dynamics7dof.physics_check --capture ".\artifacts\dynamics7dof\capture_20260927_162739_297136" --capture ".\artifacts\dynamics7dof\capture_20260927_155420_045453" --capture ".\artifacts\dynamics7dof\capture_20260927_160434_220127" --evidence-capture ".\artifacts\dynamics7dof\capture_20260927_162739_297136" --evidence-dir ".\artifacts\dynamics7dof\physics_verification_162739\solver_evidence" --output-dir ".\artifacts\dynamics7dof\physics_verification_162739\report_manual_01"
```

Run Configuration 也可使用 **Module name = dynamics7dof.physics_check**，将上面的 `--capture ...` 等部分填入 Parameters，Working directory 同上。不要直接运行包内的 `.py` 文件。

已保存的 `solver_evidence` 足够重跑以上命令，不需要 CarSim 软件运行，也不需要再次 Send to Simulink。

## 3. 独立证据从哪里来

这次从对应 Run 的 `LastRun.vs/.vsb`、`LastRun_all.par`、`LastRun_echo.par`、`LastRun_log.txt` 制作独立副本，然后在副本目录调用本机官方 `ERDConverter.exe`。原结果未改动，副本清单及哈希保存在 `source_manifest.json`。

官方转换器语法：`ERDConverter.exe -b <结果.vs> -a <结果_all.par> -c`；详见本机 `CarSim2022.1_Prog/Help/Manuals/erd_converter.pdf` 第 11–12 页。新一轮转换需要 CarSim Browser 和相应许可可用；转换只是读取已有结果，不启动车辆仿真。

代码中 `convert_results(vs_path, output, converter)` 供新工况建立副本后转换。它只接受新输出目录，保留来源清单并核对原结果未变化。不要直接对 CarSim 的活动结果目录执行转换、覆盖或手工编辑。

**不能把另一次运行的 LastRun 与旧 capture 配对。** 当前验证器检查参数哈希，并用 Vx、Xo、Steer_L1 的逐时刻一致性检查配对；即使有同一模型名称也不够。新工况的结果应在下一次仿真覆盖 LastRun 之前归档。

## 4. 验证的物理约定

### 时间

当前 AM-2 配置观察到第一段实际间隔 0.5 ms，后续 1 ms；实际 T 比 Python 名义时间少 0.5 ms。用实际 T 对齐 CarSim 保存的运动信号，吻合明显优于再次平移半步。正式构造转移时排除初始化和结束邻域，不能使用固定名义步长替代观测到的首间隔。

这不是“所有 CarSim 配置都固定少半步”的规则。积分方法、I/O 更新或输出配置改变后需要重新核对。旧数据没有 T，报告不会编造其绝对时间。

### 有符号执行器合扭矩

```text
T_act = My_Dr + My_Bk
简化轮速平衡：Ispin * domega/dt ≈ T_act - R * Fx
```

当前前进制动时 My_Bk 为负。使用 CarSim 保存的真实 Fx 和轮胎＋轮毂旋转惯量独立检查，`相加` 明显优于 `相减` 和 `只用驱动扭矩`。残余不平衡约前轮 11 N·m、后轮 6 N·m，并不为零：滚阻等详细作用未全部纳入。不能据此声称完整轮端力矩已完全闭合，也不能把这个值再次扣去 R*Fx 后作为 7DOF 输入。

### 载荷记忆

`LoadMemory` 的定义是上一模型微步得到的 `Sum_Fx/m, Sum_Fy/m`，不是随便选两个传感加速度。每回合显式初始化一次，预测后携带返回的记忆。报告同时试验：

1. `carried`：连续传递模型记忆，保持参考模型语义。
2. `sensor_lag`：用上一采样行 CarSim Ax/Ay 代入，仅作敏感性对比。
3. `zero_each_step`：每次预测重置零，仅作反例/消融对比。

当前缓和工况下不同方式的部分车身误差接近，不能因此认定它们物理等价，更不能因为某项误差碰巧较小就默认清零。未保存实测 Fz/Fy，因此不声称已经验证真实四轮载荷；报告给出的 Fz 差异仅是模型内敏感性。

### 参数组与预测精度

原模型保留 `m=2026, Iz=4095, Iwheel=2`。CarSim 输出显示初始总车 `m=2126.4, Iz=5067.232381`，每轮旋转总惯量 `ISPIN=3.23`；其中 `IT=2` 只是轮胎部分，另有轮毂 `IW=1.23`。`4095` 在 CarSim 对应未加载簧载车身，不是本次整车的横摆惯量。

验证器单独增加 `initial_total_vehicle_proxy` 对照组，同时使用整车初始质心位置/高度；不是自动校准，不会替换 `model.py`。它有些指标变好，有些变差，反映其余拟合和简化结构仍有失配。

报告既有 1 ms 一步误差、瞬时导数误差，也有 50 ms 回放和“保持当前状态不变”基准，另比较 0.5/0.25 ms 积分子步。50 ms 回放读取实际未来轮端输入，仅用于离线排查模型，不是假装控制器在线知道未来输入。不能因为一步误差绝对值很小就认定模型已足够准确。

## 5. 后续直接进入残差原型

不需要继续反复采 2 s 调试片段。下一阶段可以一次实现离线残差数据集、Ensemble 训练、留出回合评估和只观察不接管的在线更新。

建议保留原始参考组为首个基准（已有明确参数版本），同时保留整车对照组做消融。输入约定为七维状态、两个实际前轮角、四个执行器合扭矩、mu、两个载荷记忆；均一附着时共 16 维。分别标准化状态导数输出后才合成 Ensemble disagreement，不能直接相加不同量纲方差。

后续正式训练数据应来自带完整实际时钟的多个独立回合，并按回合/工况划分训练验证；先检查差分和滤波的时刻对齐。改进必须在留出工况上相对原模型成立，Ensemble 分歧需与真实误差校准。当前误差报告不自动转为训练标签，不开启 Risk 标签刷新，也不接管车辆控制。
