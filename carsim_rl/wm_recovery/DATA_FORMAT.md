# 离线轨迹约定 v1

每个目录只容纳一条完整轨迹：`metadata.json` 与 `trajectory.npz`。所有动力学计算使用 SI。旧 3 输入/25 输出采集文件不能直接复用，因为输入语义、时间以及车辆匹配状态不同。

## 数组

| 名称 | 形状 | 含义 |
|---|---|---|
| `time_s` | N | 真实观测时间，严格递增；学习入口要求间隔等于 `prediction_dt_s` |
| `state` | N×7 | vx、vy、r、ω_FL、ω_FR、ω_RL、ω_RR；m/s、rad/s |
| `applied_input` | (N−1)×6 | 实际 δ_FL、δ_FR、四个实际外加转矩；rad、N·m |
| `command` | (N−1)×5 | 四轮转矩命令、公共前轮转角命令；N·m、rad |
| `initial_load_memory_m_s2` | 2 | 轨迹起点显式名义载荷记忆 ax、ay；不能默认为传感器 Ax/Ay |

第 i 行输入用于 `[time_s[i], time_s[i+1])`，监督标签为 `state[i+1]`。状态历史不包含当前或未来时刻。初始历史不足的样本不进行学习。

**当前 v1 要求该区间内实际输入按保持值处理，并经采集验证。** 不能将区间末端轮角/转矩当作整个区间输入；若 10 ms 内存在明显执行器变化，应保留子步输入并扩展积分接口，或明确选用更小预测周期，不静默降采样。CarSim 1 ms 求解步长不等于 World Model 10 ms 周期。

## 元数据示例

```json
{
  "format": "wm_recovery_trajectory_v1",
  "trajectory_id": "unique_capture_id",
  "condition_id": "mu08_speed40_steer_sine_group_a",
  "source_kind": "carsim",
  "split": "train",
  "nominal_mu": 0.8,
  "actual_mu": 0.8,
  "state_names": ["vx_m_s", "vy_m_s", "yaw_rate_rad_s", "omega_fl_rad_s", "omega_fr_rad_s", "omega_rl_rad_s", "omega_rr_rad_s"],
  "applied_input_names": ["actual_front_steer_fl_rad", "actual_front_steer_fr_rad", "actual_external_torque_fl_nm", "actual_external_torque_fr_nm", "actual_external_torque_rl_nm", "actual_external_torque_rr_nm"],
  "command_names": ["external_torque_fl_nm", "external_torque_fr_nm", "external_torque_rl_nm", "external_torque_rr_nm", "front_steer_rad"],
  "input_alignment": "held_over_interval_t_i_to_t_i_plus_1",
  "actuation_definition": "external_wheel_torque_without_friction_brake",
  "other_wheel_actuation_verified_zero": true,
  "parameter_alignment": "verified",
  "initial_memory_source": "填写实际载荷记忆初始化和预热方法"
}
```

示例中的 `verified` / `true` 是完成真实核验后才能填写的声明，不是转换脚本可自动补出的默认值。当前 CarSim 参数尚未对齐，应标记 `parameter_alignment="unverified"`；可以做数据检查/名义诊断，但不能当作满足“参数一致离线训练”要求的数据。

`split` 只允许 `train`、`validation`、`test`、`unknown_mu_test`。前三者要求实际 μ=0.8。其他附着仅用于 `unknown_mu_test`；`actual_mu` 不进入网络或名义物理预测。

初版一次轨迹只标记一个常值实际 μ。突变附着数据应扩展为明确的工况事件/时间序列约定后再接入，不能仅靠单个 `actual_mu` 表示。`mismatch_start_s` 可记录未知工况起点，以计算检测延迟；未知常值附着从起始时刻生效时可填起始时间。

训练、验证、测试按完整轨迹和工况组划分，不随机拆相邻样本。每个 `condition_id` 只能属于一个分区。模型记录训练来源，标定阈值和独立评价时检查来源重叠。哈希基于物理数组内容，重新压缩不会绕过检查。

外加轮端转矩不能直接替换为 `My_Dr_*`，也不能未经定义核验就把 `My_US_*` 或 `My_WC_*` 当作网络输入。后续需通过 CarSim 通道说明与单轮激励，确认实际转矩来源，并验证其它驱动/制动贡献为零。四轮次序固定 FL、FR、RL、RR，坐标和单位应与原 7DOF 一致。
