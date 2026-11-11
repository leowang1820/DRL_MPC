# NMPC 等效执行器/物理量映射 — CarSim 61 Export

本补丁只增加一个独立模块 `wm_recovery/nmpc_actuation.py`，
不修改 CarSim、World Model、数据清洗模块和 MATLAB。

## 1. 八维输入

    z[k] = [vy, yaw_rate, vx, delta_f,
            mz_tv_estimated, fx_all, lamda, mu_hat]

全部采用 SI 物理单位。

- vy, vx：CarSim km/h 转 m/s。
- AVz：CarSim deg/s 转 rad/s。
- delta_f：左右实际 Steer_L1/R1 的平均值，deg 转 rad。
- Fx_all：四轮 tire-local 纵向 Fx 之和，单位 N。
- lamda：前轴 tire-local Fx / 四轮 Fx 之和。
- Mz_tv：四轮车身纵向力差乘前、后轮距一半，单位 N*m。
- mu_hat：默认名义 0.8，不用 MuX_* 假装物理估计。

## 2. 模型假设

1. Fx/Fy 为轮胎局部坐标系中的 N；正 X 向前、正 Y 向左。
2. 后轮无转向；正偏航逆时针；左右为车身左右。
3. 前轮采用：
       Fx_body = Fx*cos(delta) - Fy*sin(delta)
   后轮假定：
       Fx_body = Fx
4. 正方向：
       Mz_tv = (Bf/2)*(Fx_body_FR-Fx_body_FL)
             + (Br/2)*(Fx_body_RR-Fx_body_RL)
5. 前/后轮距默认均为 1.605m，待 CarSim 车辆配置核对。
6. Mz_tv 是*轮距纵向力差产生的等效附加偏航力矩*，
   不是全部偏航力矩或已验证的 NMPC 实际可执行力矩。
7. lambda 不裁剪：不限制 [0.4,0.6]，超过此区间仅作统计；
   当 |Fx_all| <= 50N 时视为不可辨识，返回 NaN 和 valid_mask=False。
8. 这里的全部映射是用户指定的工作假设，不宣称经过独立物理验证。

## 3. 安装

在 PowerShell 项目根目录：

    conda activate drl_mpc_carsim
    Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"
    Expand-Archive `
      -Path "$HOME\Downloads\wm_nmpc_actuation_v1.zip" `
      -DestinationPath . -Force

安装后新增：

    wm_recovery/nmpc_actuation.py
    wm_recovery/tests/test_nmpc_actuation.py

**不覆盖已有 `carsim_export_schema.py`**，自动使用你原来的固定 61 通道定义。

## 4. 单元测试

    python -m unittest wm_recovery.tests.test_nmpc_actuation -v

已在不调用 CarSim DLL 的隔离环境中测试 9 项：单位、公式、
角平均、轮距符号、横向力投影、lambda 无界、零分母、输入数组、
原始 SHA256 及连续前缀读取，全部通过。

## 5. 审计已清洗 Workspace（无需再次采集 CarSim）

    python -m wm_recovery.nmpc_actuation audit `
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02

可以另存报告：

    python -m wm_recovery.nmpc_actuation audit `
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
      --output artifacts/wm_recovery/nmpc_actuation_audit_01.json

如果希望将分母门槛由 50N 改为 10N：

    python -m wm_recovery.nmpc_actuation audit `
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
      --min-abs-fx-all-n 10

建议先采用默认 50N 审计，查看多少样本的 lambda 不可辨识，
以及多少有效 lambda 超出了你预期的 0.4~0.6 区间。

## 6. 在下游 Python 中使用

    from wm_recovery.nmpc_actuation import load_from_candidate

    time_s, result = load_from_candidate(
        r"...\candidate_train_circle_left"
    )

    z_k = result.z_si[:-1]              # [N,8]，状态时刻 t_k
    valid = result.valid_mask[:-1]      # [N]，是否具备全部有限输入
    z_train = z_k[valid]                # 示例：只看可用输入

注意：真正监督训练还必须以同一 `valid` 掩码筛选 t_{k+1} 的标签，
并不能跨过清洗过的数据时间段拼接未来状态。

可以直接用原生 CarSim 输出数组：

    from wm_recovery.nmpc_actuation import derive_nmpc_actuation
    result = derive_nmpc_actuation(raw_exports)
    z = result.z_si

`z_si` 总是 `[T,8]`，即使传入单个 `[61]` 样本也返回 `[1,8]`。

## 7. 输出的特殊状态

- `lamda_observable_mask`：总纵向力绝对值大于门槛；
- `lamda_in_expected_band_mask`：0.4<=lamda<=0.6，仅诊断；
- `valid_mask`：八维特征是否全部有限；
- `delta_left_right_diff_deg`：左右实际轮角差；
- `fx_front_n`、`fx_rear_n`、`mz_tv_estimated_nm`：物理审计量。

下一步：在得到真实数据的审计结果后，建立
`nmpc_reference.py` 和 `nmpc_residual_dataset.py`，
避免在力输入不可辨识时硬凑网络标签。
