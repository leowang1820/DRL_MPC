# CarSim → NMPC 等效输入转换 v2：三状态 λ 分类

## 一、修改范围

本补丁只覆盖两个文件：

- `wm_recovery/nmpc_actuation.py`：三状态分类、有效性掩码、连续有效区间、审计统计。
- `wm_recovery/tests/test_nmpc_actuation.py`：13 项不启动 CarSim 的单元测试。

**不修改** CarSim Export 61 通道、原始轨迹、清洗 Workspace、`offline.py`、`world_model.py` 或 MATLAB。

## 二、三状态判定

定义：

- `Fx_front = Fx_L1 + Fx_R1`
- `Fx_rear = Fx_L2 + Fx_R2`
- `Fx_all = Fx_front + Fx_rear`

默认阈值：

- `min_abs_fx_all_n=50.0`：总纵向力是否足够大，可直接求比例。
- `max_abs_neutral_axle_force_n=50.0`：低总力时，各轴是否也接近零。
- `neutral_lamda=0.5`：低力工况的显式中性约定值，不是实测值。

分类按优先级进行：

1. **ratio（真实力比值可计算）**：`abs(Fx_all)>50`，`lamda=Fx_front/Fx_all`，**不设上、下界**。
2. **neutral（近零纵向力）**：`abs(Fx_all)<=50` 且 `abs(Fx_front)<=50` 且 `abs(Fx_rear)<=50`。`lamda=0.5`，标记为约定值。
3. **cancellation（前后轴力明显抵消）**：不满足上面两种情况。`lamda=NaN`，训练输入无效。

注意：50 N 是启发式判定阈值，不是车辆法规标准。若修改阈值，应在实验配置中记录。`expected_lamda_low/high` 只用于统计，不能控制计算结果。

## 三、结果数组

`derive_nmpc_actuation(raw_exports)` 返回 `NMPCActuationResult`。

保留既有输出：

- `z_si`：`(N+1,8)`，顺序为 `[vy,r,vx,delta_f,Mz_tv,Fx_all,lamda,mu_hat]`。
- `valid_mask`：八维值全部有限；此版本同时包含 `ratio` 和 `neutral`。
- `lamda_observable_mask`：**只标记**由真实轴力比值直接计算的 `ratio`。
- `lamda_in_expected_band_mask`：预期区间的诊断掩码，不参与裁剪。

新增：

- `lamda_neutral_mask`
- `lamda_cancellation_mask`
- `lamda_regime`（逐状态时刻的 `ratio` / `neutral` / `cancellation` 标签）

示例：

```python
from wm_recovery.nmpc_actuation import derive_nmpc_actuation

result = derive_nmpc_actuation(raw_exports)

z_k = result.z_si[:-1]                   # N 个控制区间的状态时刻输入
valid = result.valid_mask[:-1]          # ratio + neutral
ratio = result.lamda_observable_mask[:-1]
neutral = result.lamda_neutral_mask[:-1]
cancellation = result.lamda_cancellation_mask[:-1]
```

**训练时**应保留 ratio/neutral 来源标记：例如给 neutral 样本较小权重或单独评价。不要把 neutral 当作 CarSim 实测分配系数，也不要拼接被 cancellation 切断的非连续时间窗口。

## 四、安装到项目

在 Windows PowerShell / PyCharm Terminal：

```powershell
conda activate drl_mpc_carsim
Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

Copy-Item .\wm_recovery\nmpc_actuation.py `
    ".\wm_recovery\nmpc_actuation.py.$stamp.bak"

Expand-Archive `
    -Path "$HOME\Downloads\wm_nmpc_actuation_three_regimes_v2.zip" `
    -DestinationPath . -Force
```

回归测试：

```powershell
python -m unittest wm_recovery.tests.test_nmpc_actuation -v
```

已在隔离测试环境验证 13 项测试通过，覆盖单位、横摆力矩公式、三类判定、边界、无限幅、连续有效段、Workspace 分组以及严格原始候选 SHA256/时序契约。尚未在你本机的真实 36 条 CarSim 轨迹上运行本版本。

## 五、真实数据审计

使用已完成清洗的 Workspace，无须重新采集：

```powershell
python -m wm_recovery.nmpc_actuation audit `
    --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
    --min-abs-fx-all-n 50 `
    --max-abs-neutral-axle-force-n 50 `
    --neutral-lamda 0.5 `
    --output artifacts/wm_recovery/nmpc_actuation_audit_02.json
```

输出逐场景显示：

- `直接比例`：由有效总纵向力比值计算的控制区间数
- `中性约定`：用 0.5 模型约定补充的控制区间数
- `抵消不可用`：总力接近零但轴力明显抵消的控制区间数
- `有效`：ratio+neutral 的控制区间数
- `最长连续`：不跨越 NaN/轨迹边界的最长连续有效时长

审计 JSON 中还提供：

- `regime_totals`：完整数据集三类比例和数量
- `by_speed`：分初始车速统计
- `by_split`：分 train/validation/test 统计
- `episodes[*].longest_contiguous_valid_run`：每轨迹最长 ratio+neutral 连续区间
- `episodes[*].longest_contiguous_ratio_run`：每轨迹**仅 ratio** 的最长连续区间
- `longest_contiguous_valid_run`：全局最长单轨迹区间，绝不跨轨迹拼接

控制区间取 `z_si[:-1]`，所以每条有 600 个控制区间的轨迹仍按 600 个而不是 601 个状态时刻计数。真实有效率需要运行后才能知道。

## 六、重要边界

本模块只实现用户指定的等效物理映射假设；尚未独立核验 CarSim Fx/Fy 的准确坐标系以及 Mz_tv 与 NMPC 实际执行器响应的等价性。中性补值会提高八维数值输入覆盖率，但不会自动证明 8→3 残差网络已具备因果可学习性或长时域预测精度。
