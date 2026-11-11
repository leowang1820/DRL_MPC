# NMPC 物理残差训练数据构造 v1（Python 专用）

本版本实现两个独立模块：

1. `wm_recovery/nmpc_reference.py` — 复现现有 MATLAB `continuousDynamics()` 中 [vy_dot, r_dot, vx_dot] 三个车辆动力学导数，以及 10 ms 单步零阶保持（ZOH）RK4。
2. `wm_recovery/nmpc_residual_dataset.py` — 从 `workspace_40_60_clean_02` 加载原始 CarSim 61 通道的清洗后前缀、严格核对 SHA256、构造 8 维输入与 3 维区间等效残差标签；保持 train/validation/test 分离与无效区间分段。

**本版本不运行 CarSim，不修改已清洗数据，不修改原 7DOF World Model，不修改 MATLAB/FORCESPRO，也不包含神经网络训练器。**

## 一、文件放置

解压 ZIP 至项目根目录，新增：

```text
wm_recovery/
  nmpc_reference.py
  nmpc_residual_dataset.py
  tests/
    test_nmpc_reference.py
    test_nmpc_residual_dataset.py
```

依赖你已经安装并通过审计的现有文件：

- `wm_recovery/nmpc_actuation.py`（三状态 lambda v2）
- `wm_recovery/carsim_export_schema.py`
- `wm_recovery/offline.py`
- `wm_recovery/carsim_candidate.py`
- 原清洗结果 `artifacts/wm_recovery/workspace_40_60_clean_02`

## 二、输入与输出

网络输入顺序固定：

```python
z = [vy, r, vx, delta_f, Mz_tv_estimated, Fx_all, lamda, mu_hat]
```

物理量单位：m/s、rad/s、m/s、rad、N·m、N、无量纲、无量纲。

名义 NMPC 动力学计算：

```math
\dot v_y^{nom}=-v_xr+\Sigma F_y^{nom}/m
\dot r^{nom}=\Sigma M_z^{nom}/I_z
\dot v_x^{nom}=\Sigma F_x^{nom}/m+v_yr
```

名义模型精确复现用户提供的 MATLAB 轴荷转移、轴荷平滑、摩擦圆降额、载荷相关刚度、tanh 饱和、前轮坐标变换和牛顿–欧拉公式。物理常数：

- `lf=1.265 m`、`lr=1.895 m`、`L=3.16 m`
- `m=2053.8 kg`、`Iz=4095 kg·m²`、`h_cg=0.59 m`
- `g=9.81 m/s²`
- `C1=8735.0825`、`C2=59.6128`
- 轴荷平滑常数 `10000`、摩擦圆裕量平滑常数 `1e6`

**标签**（时间单位 s）：

```math
q_k=[v_{y,k}, r_k, v_{x,k}]
q_{k+1}^{nom}=RK4(q_k,z_k,dt_k)
y_k=(q_{k+1}^{CarSim}-q_{k+1}^{nom})/dt_k
```

标签三个分量分别为：横向加速度等效残差 [m/s²]、横摆角加速度等效残差 [rad/s²]、纵向加速度等效残差 [m/s²]。

说明：
- `z_k` 只能来自 `t_k` 的真实观测值，不能使用 `t_{k+1}` 轮胎力。
- `q_{k+1}^{CarSim}` 只作训练监督真值。
- RK4 内的 `delta_f、Mz、Fx_all、lambda、mu` 在该 10ms 区间内零阶保持。它不积分 NMPC 中的状态变化率，因此是**区间等效**而非严格瞬时连续时间残差。
- `Mz_tv_estimated` 来自同一时刻的 CarSim 轮胎力差，是等效执行器估计，仍未经完整控制量对应验证。
- 当 lambda 为 `neutral` 时 0.5 属于模型约定，保留分类标签，不把它当作实测值。
- `cancellation` 样本不会进入网络训练，连续片段不得跨过它们。

## 三、安装（Windows PowerShell）

```powershell
conda activate drl_mpc_carsim

Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

Expand-Archive `
    -Path "$HOME\Downloads\wm_nmpc_reference_dataset_v1.zip" `
    -DestinationPath . `
    -Force
```

无需覆盖原来的 `nmpc_actuation.py`、`offline.py` 或 `data_cleaning.py`。

## 四、先做不调用 CarSim 的单元测试

```powershell
python -m unittest wm_recovery.tests.test_nmpc_reference -v

python -m unittest wm_recovery.tests.test_nmpc_residual_dataset -v
```

在隔离环境中结合现有 `nmpc_actuation` 的回归测试运行了 **27 项单元测试，全部通过**。
包括：
- 从源 MATLAB 逐公式独立计算标量结果并与 Python 参考模型比较
- 零输入/纯纵向力/附加横摆力矩的正确性
- 批量 RK4 与逐行 RK4 一致
- 已知合成残差标签的数值复现
- 三个分区的候选文件与来源 SHA256
- `cancellation` 导致连续区间拆分
- 测试集不能参与训练样本加权
- 未清洗 Workspace 被拒绝
- 校验失败时不创建输出目录

测试使用**模拟数据**；真实 40/60kmh 数据要运行下面的 `audit` 来确认。

## 五、用你已经清洗过的数据只读审计

```powershell
python -m wm_recovery.nmpc_residual_dataset audit `
    --workspace artifacts/wm_recovery/workspace_40_60_clean_02
```

此操作不创建新数据集，只验证全部候选来源和名义 RK4，并报告：
- 每个场景的有效样本数、直接 `ratio`、约定 `neutral`、连续段数
- train、validation、test 三个分区的样本数
- 可用 0.1s、0.5s、1.0s 的连续训练窗口数量
- 名义模型单步误差与三通道残差分布统计

在真实来源与清洗阈值没有变化、并且无新的模型域错误时，预期：
- 总区间 20,542
- cancellation 导致不可用 7
- 数据集可用区间约 20,535
- train/validation/test 分别 14,140 / 3,314 / 3,081

上述数量是基于你此前审计日志的**预期**，不是在你的真实 Windows CarSim 文件上重新跑出的结果。

## 六、创建真实训练数据文件

```powershell
python -m wm_recovery.nmpc_residual_dataset build `
    --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
    --output artifacts/wm_recovery/nmpc_residual_dataset_40_60_01
```

输出：

```text
nmpc_residual_dataset_40_60_01/
  train.npz
  validation.npz
  test.npz
  dataset_manifest.json
```

如果曾创建过 `_01` 输出目录，请改用新的 `_02`，程序不会覆盖任何已有输出。

### 可选参数

```powershell
python -m wm_recovery.nmpc_residual_dataset build `
    --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
    --output artifacts/wm_recovery/nmpc_residual_dataset_40_60_weighted_01 `
    --min-segment-intervals 10 `
    --neutral-train-weight 0.5 `
    --rk4-substeps 2
```

- `--min-segment-intervals 10`：仅保留长度至少 0.1 秒的连续有效段，默认 1，默认不丢失额外有效样本。
- `--neutral-train-weight 0.5`：仅训练分区的中性约定区间权重减半，验证与测试权重仍全部为 1；默认 1，不人为降权。
- `--rk4-substeps 2`：在每个 CarSim 10ms 区间内用两次 RK4 子步，提高积分精度；默认 1。
- CarSim 对应映射阈值继续使用已审计的 `50 N / 50 N / neutral=0.5`，如需改变可以用 `--min-abs-fx-all-n`、`--max-abs-neutral-axle-force-n`、`--neutral-lamda`，更改后必须记录新版本。

## 七、训练文件字段

每个 split `.npz` 中包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `z_si` | (K, 8) | 八维物理输入 |
| `residual_y_si` | (K, 3) | 监督标签 |
| `q_k_si` | (K, 3) | 当前真实三维状态 |
| `q_true_next_si` | (K, 3) | 下一状态 CarSim 真值 |
| `q_nominal_next_si` | (K, 3) | 名义 RK4 预测 |
| `nominal_accel_si` | (K, 3) | 名义连续时间动力学导数 |
| `time_k_s` | (K,) | 原轨迹时间 |
| `dt_s` | (K,) | 实际时间间隔 |
| `lamda_regime_code` | (K,) | 0=ratio，1=neutral |
| `sample_weight` | (K,) | 训练权重 |
| `episode_index` | (K,) | 原始轨迹编号 |
| `interval_index` | (K,) | 原始控制区间索引 |
| `segment_id` | (K,) | 连续段唯一编号 |
| `segment_step` | (K,) | 连续段内从 0 开始的步数 |

训练时：

```python
import numpy as np

with np.load(
    "artifacts/wm_recovery/nmpc_residual_dataset_40_60_01/train.npz",
    allow_pickle=False,
) as data:
    z_train = data["z_si"]
    y_train = data["residual_y_si"]
    weights = data["sample_weight"]
    segment_id = data["segment_id"]
```

如果以后做多步训练：

```python
from wm_recovery.nmpc_residual_dataset import iter_contiguous_windows

windows_100 = iter_contiguous_windows(segment_id, horizon_intervals=100)
```

这样每一个窗口都是同一条轨迹中一个连续有效区间的 100 个真实间隔，**不会把无效 λ 之间的数据当作相邻时间步**。

## 八、研究边界

这两个模块解决了“NMPC 名义模型基准”和“八维输入到三通道等效残差标签”的程序问题，但：

1. 在正式训练和 MPC 使用前，仍需验证 61 Export 中轮胎 Fx/Fy 的坐标系、车辆参数与真实 CarSim Run 的一致性。
2. 需要检查 `dataset_manifest.json` 的三通道 `residual_abs_p99_y_si`，确认不存在极端残差标签或时间对齐异常。
3. 对 `neutral` 数据要单独给出统计；该约定值不是实测力分配。
4. 目前只有名义 `mu_hat=0.8`，没有跨附着系数泛化证据。
5. 这版**没有训练 PyTorch 网络**；下一步才是 `nmpc_residual_model.py` 和 `nmpc_residual_train.py`。
