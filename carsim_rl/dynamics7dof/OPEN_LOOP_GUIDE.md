# 正弦转向、驱动/制动：CarSim 与 7DOF 曲线对比

入口：`python -m dynamics7dof.open_loop`。不接入 SAC、Ensemble 或轨迹跟踪器，不改原 7DOF 参数/轮胎拟合。

## 1. 本实验比较什么

当前 CarSim 的三个输入是油门、制动压力、方向盘角，不能直接传入两个前轮角和四轮扭矩。因此采用：

```text
名义前轮正弦角 × 近似转向比 → 方向盘指令 ┐
油门脉冲、制动压力脉冲 ─────────────────┼→ CarSim → 真实状态曲线
                                       └→ 实际前轮角、轮端扭矩
同一个初始七维状态 + 实际轮端输入 ─────────→ 7DOF 连续积分 → 对比曲线
```

7DOF 使用实际 `Steer_L1/Steer_R1`，以及各轮 `My_Dr + My_Bk`（保留制动负号），不把方向盘角当成前轮角，也不随意把油门、压力换算为扭矩。

这是**实测实际输入条件下的动力学验证**，不验证转向/动力总成执行器本身，也不声称控制器可以提前知道未来实际输入。若要求前轮严格等于指定正弦、直接指定四轮力矩，需要另外配置 CarSim 直接轮端输入；本工具不修改当前 3/25 接口。

主曲线只在起点使用一次 CarSim 状态，之后连续积分，不每步重置到真实状态。初始 LoadMemory 显式为零，此后连续传递。实际输入在相邻采样点间按左端值保持；输入突变处仍有离散采样误差。

另计算一步预测作为独立诊断：它每步使用真实状态，但不画成主连续预测曲线。不要混淆二者。

## 2. 在 CarSim 中准备

1. 建议复制一个单独 Run，使用平坦、足够宽且足够长的道路，关闭不需要的路面扰动；约 40 km/h 的初速即可。
2. 初次使用 mu=0.5，仿真结束时间不少于 8 s。时长只是上限，CarSim 自身事件也可能提前结束。
3. 保留之前的 3 Imports / 25 Exports，顺序、Replace 模式和原生单位不变。若更改了 Run/参数，重新 Send to Simulink 生成文件；不需要运行 Simulink。
4. 关闭其他正在运行的 CarSim/Python/Simulink 求解任务。仅打开 CarSim Browser 不等于正在运行求解器。

真实仿真仍会更新 CarSim 的常规 LastRun 结果，重要旧结果请先归档。Python 采集和报告使用新时间戳目录，不覆盖已有实验。

本实验没有轨迹控制或恒速控制，不能期望车辆自动跟随 DLC 或始终保持 40 km/h。安全阈值只会终止仿真，不会接管转向。

## 3. PyCharm 中运行

解释器：`D:\anaconda3\envs\drl_mpc_carsim\python.exe`。
工作目录：`E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl`。

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
python -m dynamics7dof.open_loop --check-config-only --mu 0.5
```

这一步不加载 DLL，不创建目录。无参数运行也仅显示只读计划。确认配置后运行一次完整实验：

```powershell
python -m dynamics7dof.open_loop --run --mu 0.5
```

默认实验如下，所有时刻均相对实验起点：

| 项目 | 默认值 |
|---|---|
| 名义时长 | 8 s |
| 名义前轮角振幅 | 0.5° |
| 正弦频率 | 0.3 Hz，共 2 个周期 |
| 转向起始 | 1 s，首尾各 0.5 s 平滑包络 |
| 近似转向比 | 16，名义方向盘峰值约 8° |
| 驱动 | 0–3 s，油门 0.12 |
| 制动 | 5–6 s，压力 0.2 MPa |
| 其他时刻 | 油门、制动为零；并非固定车速 |
| 模型积分子步 | 不超过 0.5 ms，按观测时间间隔整分 |

显式指定参数的等价命令：

```powershell
python -m dynamics7dof.open_loop --run --mu 0.5 --seconds 8 --front-amplitude-deg 0.5 --frequency-hz 0.3 --throttle 0.12 --drive-until-s 3 --brake-mpa 0.2 --brake-start-s 5 --brake-duration-s 1
```

两个便于分离误差来源的变体：

```powershell
# 只做正弦转向，油门/制动置零；车速会自然变化
python -m dynamics7dof.open_loop --run --mu 0.5 --throttle 0 --brake-mpa 0

# 不转向，只做驱动和制动
python -m dynamics7dof.open_loop --run --mu 0.5 --front-amplitude-deg 0
```

修改振幅/频率后仍从温和工况开始，不直接用大转角验证极限区域。默认终止阈值为纵向速度 5–80 km/h、|beta|≤15°、|r|≤45°/s，以及相对初始朝向的侧向位移≤8 m。终止原因保存在 metadata；触发安全终止不是实验成功完成。

Run Configuration 可选 **Module name = dynamics7dof.open_loop**，Parameters 填 `--run --mu 0.5`，Working directory 同上。不要直接运行包内 `.py` 文件。

## 4. 输出图在哪里

控制台会打印绝对路径，默认结构为：

```text
artifacts/dynamics7dof/open_loop_时间戳/
  capture/             原始 25 通道、SI 换算、实际时钟、配置快照和输入计划
  comparison/
    comparison.html    用浏览器打开的完整图表报告
    states.svg         vx、vy、r、beta、四轮角速度的对比曲线
    errors.svg         连续预测减 CarSim 的误差曲线
    inputs.svg         指令与实际前轮角/轮端扭矩
    metrics.json       连续预测、一步预测、保持当前值基准及覆盖率
    comparison.csv     所有状态逐点对比，SI 单位
```

双击 `comparison.html` 用 Edge/Chrome 打开即可。图形为可缩放 SVG，不依赖 Matplotlib，不需要安装新库或启动服务器。可在浏览器放大查看。

CarSim 曲线为蓝色，7DOF 连续积分为橙色。图上横摆角速度用 deg/s、侧偏角用 deg；CSV 和七维状态指标保持 SI（r 为 rad/s）。

## 5. 如何判断准确程度

先看 `inputs.svg`：实际前轮角是否确实产生正负激励？油门与制动时段是否正确？四轮扭矩是否具有正确符号？名义前轮角经过转向系统后不一定严格等于请求值。

再看状态曲线：

- vy、r 的方向是否相同，正负峰值、相位和过零时刻是否接近。
- vx 是否在驱动/制动阶段呈现一致变化。
- 长时间是否出现持续偏移。即使一步误差很小，也可能积累成明显连续预测误差。
- 四轮轮速是否合理；若仅轮速误差明显，要检查轮胎纵向模型、滚阻和轮毂＋轮胎转动惯量。

指标给出 RMSE、MAE、最大绝对误差和偏差，不以单一“准确率百分比”判断。参考信号几乎为零时，归一化误差留空，避免用接近零的分母制造误导数字。

默认排除前 0.1 s 和求解器终止行。模型发生保护限幅、进入低速域或积分异常时，连续曲线停止，报告保留剩余空白及覆盖率；不能把短暂有效片段当成全程通过。

原参数 `m=2026, Iz=4095, Iwheel=2` 没有被自动替换。上一阶段发现的整车参数定义差异仍会影响结果。可用 `--vehicle-json` 指定包含全部 VehicleParameters 字段的独立参数文件进行对照，不修改 model.py。

## 6. 不重跑 CarSim，也可重新画图

对新采集的 capture 目录：

```powershell
python -m dynamics7dof.open_loop --replay "这里填写完整capture目录"
```

可用 `--model-substep-s 0.00025` 单独检查积分步长敏感性；重画结果总是保存到新目录，不改源数据。

旧数据缺少实际 T 时默认拒绝。仅用于历史诊断时可显式加 `--allow-nominal-time`，所有图都会标出 `NOMINAL_TIME_UNVERIFIED`。部分实际时间缺失仍拒绝，不静默混用时间。

已使用旧 DLC 数据完成一次离线预览，报告在 `artifacts/dynamics7dof/open_loop_preview_dlc40_20260927_v1/comparison.html`。该预览不是新正弦实验，使用旧数据名义时间；不能把它当作新实验已成功运行的证据。

## 7. 验证范围

2026-09-27：84 项回归测试通过，涵盖激励计划、真实时间选择、异常关闭、持续传递记忆、不重置连续预测、终止样本处理、指标、SVG 格式和文件不覆盖。当前 CarSim 3/25 配置只读检查通过；旧 DLC 离线重放连续预测覆盖率 100%。

本次没有自动启动新的真实 CarSim 正弦实验；需要用户执行带 `--run` 的命令。SVG 已经过格式测试，因工具审批服务额度限制尚未完成最终的图片渲染目检。
