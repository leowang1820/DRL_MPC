# 第二阶段：采集 25 通道 CarSim 数据

更新：下文保留采集阶段的操作与待核对记录。2026-09-27 已完成的时间/扭矩批量验证，
请看 [PHYSICS_CHECK_GUIDE.md](PHYSICS_CHECK_GUIDE.md)；不需要再重复末尾的 2 秒验证步骤。
采集器仍保留原始信号，不把一次工况的验证结论自动推广到所有后续配置。

本阶段只采集原始物理量和 SI 单位换算值。**不训练 SAC/Ensemble、不生成正式残差标签，
也不把 7DOF 预测用于车辆控制。** 控制复用 run_dlc_baseline.py 的 pure_pursuit_action。
代码未修改 carsim_env.py、carsim_wrapper.py、SACModule.py 或原有训练/部署入口。

## 1. 当前已核对的接口

| Export 序号 | 名称（严格按顺序） | 原单位 |
|---|---|---|
| 1–8 | Xo, Yo, Yaw, AVz, Lat_Veh, Lat_Targ, Vx, Station | m,m,deg,deg/s,m,m,km/h,m |
| 9 | Vy | km/h |
| 10–13 | AVy_L1, AVy_R1, AVy_L2, AVy_R2 | rpm |
| 14–15 | Steer_L1, Steer_R1 | deg |
| 16–17 | Ax, Ay | g |
| 18–21 | My_Dr_L1, My_Dr_R1, My_Dr_L2, My_Dr_R2 | N·m |
| 22–25 | My_Bk_L1, My_Bk_R1, My_Bk_L2, My_Bk_R2 | N·m |

Imports 仍是 IMP_THROTTLE_ENGINE、IMP_PCON_BK、IMP_STEER_SW，三项均为 Replace。
单靠 PORTS_EXP=25 不足以证明映射正确，因此运行前会解析 simfile.sim 中的 INPUT，
检查真正生成的 Run_all.par 的全部通道顺序，并保存两份配置文件的 SHA-256。
支持 UTF-8 及本机 Windows ANSI 编码；不修改 CarSim 文件编码。
若更换了导出单位设置，即使变量名相同也应重新核对。本采集器按上表原单位解码。

## 2. 第一次先采 2 秒

先停止 MATLAB/Simulink 和其他 Python 的 CarSim 仿真，不同时占用同一 Run 的输出文件。
确认 CarSim 当前工况已经 Send to Simulink 生成，初速 40 km/h，宽阔平坦道路，mu=0.5。
当前生成文件中 TSTOP=40 s，OPT_IO_SYNC_FM=0，不由采集器修改。

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
python -m dynamics7dof.collect --check-config-only
python -m dynamics7dof.collect --mu 0.5 --seconds 2
```

第一条模块命令是只读配置检查，不加载 DLL，也不创建采集目录。
第二条命令**实际运行 CarSim**，车辆由已有纯跟踪控制逻辑驾驶。
50 ms 更新控制，每个求解器积分步采样一次；t_step=1 ms 时，完整 2 秒正常采集预期为
2001 行数据（1 行初始化 + 2000 行积分输出，不含 CSV 表头），40 次控制更新。
若求解器提前结束、离轨或发生异常，实际行数会更少，结束原因写入 metadata.json。

初始 SI 读数应大致满足：vx≈11.11 m/s、vy/r≈0、四轮转速约 31.5 rad/s。
轮胎有效半径及初始状态可能造成小差异，这些不是硬阈值。符号或数量级明显不对先停下来核对。
不要因为没有抛出异常就认定输出对齐或模型精度通过。

这个采集入口默认轴距 3.16 m（用户参数），不是旧基线入口的 2.8 m；转向比 16 与
前视距离 10 m 沿用基线设置，转向比还没有重新标定。SAC 的控制器和参数完全不变。
`--mu` 是实验标签，会与配置中的 MU_ROAD_CONSTANT 核对，但不是实时测得的四轮附着。

## 3. 输出文件

默认目录：

```text
E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl\artifacts\dynamics7dof\capture_时间戳\
```

- raw_exports.csv：25 个原单位输出，加采样序号、控制序号、时间、施加的命令和标记。
- signals_si.csv：SI 单位状态/转角/加速度/扭矩，保留原始符号；不是残差标签。
- metadata.json：配置哈希、参数、原单位、实际步长、样本数量、首尾值/范围、终止原因和待核对项。
- v2 新采集另存 simfile_snapshot.sim、parameters_snapshot.par、reference_path_snapshot.par，
  仅作为本次配置/参考路径证据，不是可直接运行的便携 CarSim 工程。后续修改路径不影响旧快照。

默认每次建立新目录，避免覆盖。`--output-dir` 可以指定新目录；目录已存在会拒绝运行。
文件按控制周期刷新；Ctrl+C 会保留部分数据并写 interrupted 状态，异常会写 failed。
强制终止进程/断电不保证最终元数据完成，不能把 started 状态的数据当作完整采集。

## 4. 时间与扭矩：刻意不作未验证推断

- wrapper_time_s：现有 Python 环境推进后的名义计时。
- api_time_argument_s：本次实际传给 vs_integrate_io 的时间参数；初始化行留空。
- 两者通常相差一个 t_step。采集器保持已有步进调用方式，不偷偷修改全工程的时序。
- 对 Euler/AB-2，OPT_IO_SYNC_FM 会影响运动学与力输出的时序；具体见官方
  https://www.carsim.com/downloads/pdf/Simulink_ABS_Example.pdf 中的同步说明。
  当前实际 OPT_INT_METHOD=2 是 AM-2，包含内部半步；本机 LastRun_echo.par 显示
  TSTEP=0.001、T_DT=0.0005、OPT_IO_UPDATE=0。因此不能仅凭 OPT_IO_SYNC_FM=0
  判断全部输出时序，也不要为消除半步差异而自行切换积分法。
- v2 新增 solver_time_before_s、solver_time_s、solver_time_source，使用
  vs_get_var_ptr("T") 只读观察积分前后的内部时钟。观察发生在终止求解器之前，关闭后释放指针。
  不修改原控制输入、API 时间参数、积分频率和 25 个 Exports。不需要再配置一个 Time Export。
- 初始化绑定失败/接口缺失时，实际时间留空并标记 solver_clock_unavailable；不以名义时间顶替。
  使用 --require-solver-time 可要求拿不到实际时间就停止，保留失败采集目录供排查。
  T 被读取不等于每个导出量时间已独立验证；三项 pending_checks 仍保留，训练就绪仍为 false。
  初始化、首次积分调用和结束行不能未经核对直接连成训练标签。
- 每个控制周期保持同一油门/制动/方向盘命令，但实际车轮转角和轮端力矩仍可能在周期内变化，
  因此逐积分步保存实际输出，而不是只保存 50 ms 终点。
- My_Dr/My_Bk 各自保存，不擅自推断刹车为正/负或相减。净扭矩的符号、滚阻/传动惯量作用
  还需实测片段核对。
- Ax/Ay 只从 g 乘 9.81 换成 m/s²；不直接当 dvx/dt/dvy/dt，也不直接替代模型的 LoadMemory。
- metadata 中 ready_for_residual_training 固定为 false，这不是错误，而是当前阶段的边界。

## 5. 下一次验证

先检查 2 秒的控制台和 metadata.json。无接口/单位问题后，再运行：

```powershell
python -m dynamics7dof.collect --mu 0.5 --seconds 20
```

路径终点可能在 20 秒前达到，属于正常结束。固定 DLC 上的这一段数据只用于初步对齐，
不代表已经覆盖动力学模型的工作范围。下一阶段先确认时序/扭矩/加速度定义，再进行影子一步
预测与误差报告；原轮胎联合滑移和附加记忆问题仍需独立处理，不能靠增加训练量绕过。

## 6. 离线测试

```powershell
python -m unittest discover -s dynamics7dof/tests -v
```

新增测试验证完整通道顺序、单位、Windows 参数文件编码、模拟求解器的采样与输入保持、
提前结束、失败时保留数据和关闭资源。模拟测试不调用真实 CarSim，也不验证车辆动力学精度。

## 7. 离线检查已有数据（不运行 CarSim）

在 carsim_rl 目录的终端执行，例如检查本次减速数据：

```powershell
python -m dynamics7dof.audit --capture ".\artifacts\dynamics7dof\capture_20260927_160434_220127"
```

默认只在终端显示结果，不写文件、不加载 DLL，也不改采集目录。
需要保存 JSON/Markdown 报告时，额外指定一个不存在、位于采集目录外的目录：

```powershell
python -m dynamics7dof.audit --capture ".\artifacts\dynamics7dof\capture_20260927_160434_220127" --report-dir ".\artifacts\dynamics7dof\my_braking_audit_01"
```

输出目录已存在会拒绝覆盖。报告保存原始三文件的 SHA-256，便于复核。
检查内容包括行号/时间连续性、raw/SI 对应和单位换算、元数据首尾及范围一致性、
控制保持、制动期间四轮扭矩符号、路径误差、平面加速度一致性。
旧数据没有实际时间列时，明确显示 wrapper_nominal_unverified，绝不推断或补写半步时间。
旧数据的路径没有快照，重算跟踪误差会提示使用的是当前路径，不能证明历史路径未变。

平面一致性使用 Ax≈dvx/dt-r*vy、Ay≈dvy/dt+r*vx。中心差分排除初始化/首次积分及其邻点、
结束边界、求解器异常/终止及邻点、低速区域，默认另排除前 0.01 s，可用 --warmup-seconds 调整。
这只是信号诊断，不是完整三维恒等式，不是 7DOF 模型预测误差，也不等于 LoadMemory 定义验证。
全量实际时钟连续且递增时使用它进行诊断，否则明确标注名义时间。

Data integrity: PASS 只表示上述数据一致性检查通过；Ready for residual training: NO
是刻意保留的物理验证边界，不是程序失败。完整性错误返回退出码 2；仅待核对项/警告返回 0。
减速样本支持保留制动扭矩负号，但 drive+brake 仍是候选净扭矩映射，未生成标签。

## 8. 验证新时钟采集（由用户手动运行 2 秒）

本次代码修改只完成离线测试；真实 DLL 的时间读取必须用这一步验证。
保持 CarSim 配置不变，不需要重新 Send to Simulink，也不需要增加 Exports：

```powershell
python -m dynamics7dof.collect --mu 0.5 --seconds 2 --target-speed 30 --require-solver-time
```

查看 Solver clock: available 和逐秒 T_solver；记录实际结果，不硬编码期望的 0.5 ms 偏移。
将新的 Capture directory 传给第 7 节 audit 命令。若绑定失败，停止操作并发送报错与
metadata.json，不要删除 --require-solver-time 后把缺失时间当作验证成功。

本机官方依据：Help/Manuals/vs_commands.pdf 第 14 页（只读 T）；
Help/Memos/VS_Commands_API.pdf 第 26 页（变量指针访问）；
Help/Manuals/system_parameters.pdf 第 17–18、27 页（积分法、T_DT、I/O 时序）。
这些文件位于 CarSim2022.1_Prog 目录。尚未接入 Ensemble、Risk 或残差标签生成。
