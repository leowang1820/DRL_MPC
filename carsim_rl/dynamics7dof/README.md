# 7DOF 第一阶段：MATLAB 物理模型的独立 Python 对照版

状态：**reference-only，尚未通过 CarSim 精度验证，不用于控制接管或自动生成正式残差标签。**

本模块不加载 CarSim DLL、不导入 SAC、不修改已有模型、参考路径和训练日志。
它保留用户 MATLAB 物理部分的行为，先让逐项比较有据可查；不是已经修正/标定完成的新模型。

## 1. 文件与运行

- `model.py`：参数、原拟合轮胎力、瞬时状态导数、带显式载荷记忆的 Euler 预测。
- `__main__.py`：合成初值的一步预测示例，不启动 CarSim、不写出文件。
- `tests/test_reference.py`：正式保留的回归/物理结构测试，不是整车精度证明。
- 本说明：参数映射、已知限制以及下一步需要的 CarSim 信号。

在 PyCharm Terminal 中：

```powershell
conda activate drl_mpc_carsim
cd E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl
python -m unittest discover -s dynamics7dof/tests -v
python -m dynamics7dof --mu 0.5
python -m dynamics7dof --mu 0.5 --front-steer-deg 1
```

PyCharm Run Configuration 使用 **Module name = dynamics7dof**，参数为 `--mu 0.5`；
工作目录为 `E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl`，解释器为
`D:\anaconda3\envs\drl_mpc_carsim\python.exe`。不要直接运行包内的 `__main__.py`。

最后一个示例的 1 度是**实际前轮转角**，不是方向盘角。示例采用自由滚动初始轮速、
四轮零净扭矩和显式零初始载荷记忆，不是从 CarSim 采集的数据。
示例没有空气阻力/滚阻；零转向、零扭矩、自由滚动时匀速是这个原模型的预期行为。

`miu` 没有默认值。2026-09-27 检查当前生成的 Run_all.par 时，其中路面数据集为
`Constant: 0.5`、`MU_ROAD_CONSTANT 0.5`。这是一次配置快照，不是自动同步机制，
也不能用轮胎参数中的 `MU_REF_X/Y=1` 代替路面附着。更换工况后必须重新确认；
分离附着或变附着时应输入四轮对应值并按时间更新。测试中的 mu=0 或 1 仅是合成测试。

## 2. 用户参数的映射（SI）

| 用户参数 | Python 字段 | 使用方式 |
|---|---|---|
| m=2026 | mass_kg | 车身平面方程与原版准静态载荷的总质量 |
| Lf=1.265, Lr=1.895 | lf_m, lr_m | 质心到前/后轴距离；轴距 3.16 m |
| Bf=Br=1.605 | track_front_m, track_rear_m | 前/后轮距 |
| hg=0.59 | cg_height_m | 按用户给定值保留；与总质量的等效质心定义后续核对 |
| Iz=4095 | yaw_inertia_kg_m2 | 横摆转动惯量 |
| Iz_wheel=2 | wheel_inertia_kg_m2 | 每个车轮转动惯量 |
| R_wheel=0.353 | wheel_radius_m | 轮胎有效半径 |
| g=9.81 | gravity_m_s2 | 重力加速度 |
| camber=1.9554 | TireFit.camber_fit | 原轮胎曲线拟合常数，数值原样使用，不擅自转换单位或置零 |

`mb=1820`、`mw=25` 为用户提供的分项质量记录；当前 7DOF 没有独立垂向/悬架自由度，
因此不再把它们加到 m 上。`mb+4*mw=1920` 不等于总质量 2026，不擅自补算或替换。
用户已确认使用 `Iz_wheel=2`，`Ir=1.1` 不参与当前计算。`cf=1.0649e5`、`cr=7.3746e4` 未叠加到 Magic Formula
轮胎力中；若另建线性模型，需先确认其是单轮还是单轴刚度以及单位。

## 3. 状态与输入合同

Python 状态顺序固定：

```text
x = [vx, vy, r, omega_FL, omega_FR, omega_RL, omega_RR]
单位：m/s, m/s, rad/s, rad/s, rad/s, rad/s, rad/s
车体坐标：x 向前、y 向左、逆时针横摆为正
```

MATLAB 原入口的前三项是 `vy_cur, r_cur, vx_cur`，移植调用时必须重排。
各轮顺序始终为 FL、FR、RL、RR。输入为：

```text
Controls.front_steer_rad = [delta_FL, delta_FR]
Controls.wheel_net_torque_nm = [T_FL, T_FR, T_RL, T_RR]
LoadMemory = [上一微步的 Sum_Fx/m, 上一微步的 Sum_Fy/m]
```

T 是传入轮速方程的有符号轮端净扭矩，不是油门比例、制动压力或轮胎接地力。
轮胎反力矩 `R*Fx` 已在模型中扣除，不应重复扣除。制动力矩的符号必须与轮速约定匹配。
CarSim 的轮速单位/方向、转角测量位置、力与速度参考点需逐项确认。

显式接口：

```python
evaluation = model.evaluate(x, controls, mu=mu, load_memory=memory)
prediction = model.predict(x, controls, mu=mu, load_memory=memory,
                           horizon_s=0.05, substeps=100)
x_next = prediction.state
memory_next = prediction.load_memory
```

不能把零载荷记忆当作每次实车/CarSim 样本的默认值。连续预测要传回 memory_next。
模型目前是带额外记忆的映射，不能标为仅由七维 x/u 决定的闭合 f_7D(x,u)。
Frenet 的 e_phi/ey/s、曲率多项式 p、始终为零的 sr_out 未移入物理模块；
跟踪误差仍可由已有 frenet_path.py 独立计算。

## 4. 有意保留、尚需验证的问题

1. 原拟合纯滑移参数以及 camber_fit 全部保留。还需确认拟合覆盖的载荷、附着、侧偏角和滑移率范围。
2. 原联合滑移方向权重保留；它会在小联合滑移下削弱力，是否也经过联合滑移数据拟合待确认。
3. 原经验侧向阻尼保留。LoadMemory 的 ay 不包含该项，不是完整的状态导数。
4. 原法向力 100 N 下限、vx>=0.1、omega>=-1 保护保留，并输出 quality_flags。
   触发时不能将限幅样本不加区分地用于残差监督。没有 flags 也不代表物理精度已验证。
5. 原显式 Euler/100 个子步保留，输入与 mu 在预测区间内不变；50 ms 对应 0.5 ms 子步，
   100 ms 对应 1 ms 子步。小步长不保证稳定，后续需要实际工况下的步长收敛检查。
6. 返回力对应最后更新前的状态，保存在 last_force_state；force_sample_time_s 明确为
   horizon-substep。不能与 terminal state 混当同一时刻。若要终点力，需明确用终点状态
   和终点 load_memory 再 evaluate，而不是给原输出换标签。
7. derivative 才是 [dvx/dt,dvy/dt,dr/dt,domega/dt]。tire_ax/ay 不能直接充当 dvx/dt/dvy/dt。

向量形状/有限值检查、异常时显式失败、不同前后轮距支持是实现层改进；在用户给定的
相同前后轮距和正常有限输入下，保持原公式。轮胎拟合若产生负峰值会明确拒绝，
不继续以未知拟合外推结果积分。本次不宣称与 MATLAB 可执行程序逐位一致；尚缺 MATLAB
生成的黄金输入输出样例，当前测试是数值公式复算、结构不变量与 Python 回归检查。

2026-09-27 已使用现有 `drl_mpc_carsim` 解释器执行 18 项测试，全部通过；同时完成了
mu=0.5、40 km/h 初速、1 度实际前轮转角、50 ms 时域的合成预测，未触发保护标记。
这些结果只支持移植和接口检查，不证明 MATLAB 跨语言一致性或 CarSim 预测精度。

## 5. 下一步：补 CarSim 信号，不改原 SAC 观测

当前已生成配置仍是 3 Imports / 8 Exports。保持前 8 个 Export 名称与顺序不变，
新增量追加在后面，不替换现有 VX/AVZ 的位置。SAC 仍接收原来的七维跟踪观测。

追加信号需先在 CarSim 变量选择器中核对“名称、描述、单位、参考点/坐标系”，
不要依赖未经核实的变量缩写。第一批所需物理量：

| 物理量 | 作用/核对点 |
|---|---|
| 质心处车体 vx、vy | 原 VX 是否满足该定义要确认；速度统一转 m/s |
| 车身横摆角速度 r | 原 AVZ 的坐标和符号要确认；deg/s 转 rad/s |
| 四轮角速度 | FL/FR/RL/RR；确认 rpm、deg/s 或 rad/s 后转换 |
| 前左、前右实际轮转角 | 不是方向盘指令；统一 rad |
| 四轮实际驱动/制动扭矩 | 按符号和物理作用位置组合为轮端净扭矩 |
| 与模型定义对应的 ax、ay | 校验惯性加速度与车体速度导数区别，以及重力/阻尼贡献 |
| 四轮 Fz、Fx、Fy（建议同时采集） | 分开诊断载荷、纯轮胎拟合和整车模型误差；确认轮胎/车体坐标系 |
| 四轮实际路面附着信息 | 当前原模型每轮一个 mu；纵/横附着不同或路面变化时须重新约定 |

这些信号未核对前，不把油门乘一个任意系数当轮端扭矩，也不把方向盘角直接传入 delta。
后续数据采集器需记录每条状态转移的时间、episode、实际控制区间及异常标志；
先做影子预测，不改变 CarSim 控制器。完成对齐和精度验证后，再设计残差标签与 Ensemble。
