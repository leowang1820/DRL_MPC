# CarSim 2022.1 内置预瞄驾驶员 DLC：Python 自动采集说明

## 目的

使用 **CarSim 内置 Driver Model** 跟踪 DLC，不由 Python 控制方向盘、轮端转矩或油门。
Python 通过现有 `carsim_env` / `vs_solver.py` 每次创建一个 CarSim 运行，并以 10 ms 为采样间隔记录当前 CarSim 配置中的全部 `EXPORT` 数组值。

**请不要把之前需要 6 个 Import（四轮转矩 Add + 两个前轮转角 Replace）的 `CarSimBridge` 直接用于这个新 DLC 场景。**

重要术语区别：

1. **Export 通道**：CarSim DLL 在 `vs_integrate_io` 中交给 Python 的实时数组。本代码只保存**配置好的全部 Export**。
2. **Output 结果变量**：CarSim 写入 VS/ERD、CSV 等结果文件的变量。可在 GUI 选择“全部可用输出”或指定子集；它们不自动变成 Export 通道。
3. **内部状态**：CarSim 数学模型内部积分状态、控制器内部状态不等同于 Output/Export。并非任意内部变量都可直接读取。

如果想尽可能保存所有物理量：
- GUI 中针对本次车辆/驾驶员模型导出可用输出变量清单；
- 选中需要的输出写入完整 VS/ERD 结果文件（根据数据量决定是否选所有可用输出）；
- Python 实时 Export 选择你需要逐采样访问的变量；
- 完整原始 VS/ERD 文件也要归档，不要仅保存整理后的 8 维状态。

## CarSim GUI 配置步骤

1. 在 CarSim 2022.1 中复制一份 Run Control，例如“DLC_Preview_40”。
2. Procedure 选择 DLC 道路/目标路径；Driver Controls 选择 **内置预瞄式闭环路径跟踪**，并配置目标车速。
3. 禁用旧方案所有外部 Import，包括两个 `IMP_STEER_L1 / IMP_STEER_R1 REPLACE` 和四个 `IMP_MYUSM` 外加转矩通道。新的 DLL 接口应报告 `n_import == 0`。
4. 保留或增加 `EXPORT` 通道：车身速度/横摆率、四轮速度、转角、轮胎力、驾驶员控制量、轨迹误差、横摆角等。变量的实际 Keyword 与单位从 CarSim “View / Output variables (text/Excel)” 查，不要凭猜测写。
5. 将 CarSim 结果输出时间间隔也设为 0.01 s，必要时另外选择完整 Output 结果变量。
6. 在 GUI 中先按“Run Math Model”成功运行一遍，确认 DLC 轨迹及驾驶员转向有效。这一步同时生成 `simfile.sim` 和 Flat All Parsfile。
7. 根据生成的 SIM/PAR 检查 `INPUT` 指向的文件、`EXPORT` 顺序以及 `TSTEP`、`TSTOP`。备份每个 Run Control 的**实际配置依赖**，保证不同工况不会共用并覆盖旧的 All Parsfile。

> 在 GUI 显示的是“内置预瞄驾驶员”并不表示 Python 自动确认了该控制方式；代码只能校验零个外部 Import，还应检查驾驶员角度输出及路径跟踪结果。

## 安装

将 `wm_carsim_preview_dlc_capture_v1.zip` 下载至 Windows Downloads，在项目根目录：

```powershell
conda activate drl_mpc_carsim
Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

Expand-Archive `
    -Path "$HOME\Downloads\wm_carsim_preview_dlc_capture_v1.zip" `
    -DestinationPath . -Force
```

文件结构：

```text
wm_recovery/
    dlc_capture.py
    dlc_env.py
    tests/
        test_dlc_capture.py
```

本代码沿用你现有项目中的 `wm_recovery.inspection.inspect_simfile()`、`carsim_env.CarSimEnv` 和 `vs_solver.py`。

**特别说明：** 目前公开版本的 `CarSimEnv.reset()` 对 `n_import <= 0` 会报错，因此新 `dlc_env.py` 实现一个零 Import 适配器，不修改你的第三方 `carsim_env.py`。它在 ctypes 底层保留一个未使用的占位 double 指针，但 CarSim 声明的外部 Import 数量仍然为 0，不会写入任何控制命令。其真实 DLL 兼容性仍需你的 CarSim 2022.1 本机验证。

## 先跑单元测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_capture -v
```

离线 Mock 下 4 项测试通过：
- 连续时刻完整采集 Export；
- 原始名称重复时不合并通道；
- 外部方向盘 `REPLACE` 仍存在时直接阻止运行；
- 零 Import 适配器初始化正确；
- 运行结束后释放资源、拒绝覆盖旧输出。

不需 DLL 的单元测试不是 CarSim 2022.1 真机运行保证。

## 先采集一个 DLC 场景

> **下面 SIM 路径仅为结构示例。请换成实际 GUI 生成的 `simfile.sim` 绝对路径。**

```powershell
python -m wm_recovery.dlc_capture `
    --sim "E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim" `
    --output artifacts/wm_recovery/dlc_preview_40_01 `
    --sample-dt 0.01 `
    --max-time 30
```

程序会先检查 SIM/PAR，要求没有外部 Import；接着加载 CarSim DLL，调用 `reset()`，随后每 10 ms 读取一次全部 Export。

如果 Solver 原始 `t_step=0.001s`，采集时将自动通过 10 个内部积分步得到一个采样点；若 `t_step=0.01s` 则每一个内部步都采集。输出时间戳来自 `env.t_current`，不是 Python 人工增加的时间。

## 批量运行多个已生成的 DLC 配置

```powershell
python -m wm_recovery.dlc_capture `
    --sim "E:\CarSimRuns\DLC_40\simfile.sim" `
    --sim "E:\CarSimRuns\DLC_60\simfile.sim" `
    --sim "E:\CarSimRuns\DLC_80\simfile.sim" `
    --output artifacts/wm_recovery/dlc_preview_multi_01 `
    --sample-dt 0.01 `
    --max-time 30
```

这三条 `.sim` 必须是**真正对应不同 Run Control 的完整配置入口**，不应都指向同一个被 GUI 覆写的 `Run_all.par`。不要只复制 `simfile.sim` 而忘记它的配置依赖。

每一条运行结果：

```text
artifacts/wm_recovery/dlc_preview_multi_01/
    run_001_DLC_40/
        dlc_exports.npz
        metadata.json
    run_002_DLC_60/
        dlc_exports.npz
        metadata.json
    run_003_DLC_80/
        dlc_exports.npz
        metadata.json
    batch_manifest.json
```

如果用单个 SIM 重复运行且不改变任何配置，得到的轨迹通常只是同一实验的重复，不一定增加动力学激励覆盖。

## 在 Python 中读取采集结果

```python
import numpy as np

path = "artifacts/wm_recovery/dlc_preview_40_01/run_001_CarSim2022_1_Data/dlc_exports.npz"
with np.load(path, allow_pickle=False) as archive:
    t = archive["time_s"]          # (N,)
    Y = archive["raw_exports"]     # (N, M)
    names = archive["export_names"].tolist()

print("采样数量:", Y.shape[0])
print("通道数量:", Y.shape[1])
print("前五个 Export 名称:", names[:5])

# 注意：重复 Export 名称需按位置索引，不要直接转成 dict。
for idx, name in enumerate(names):
    print(idx, name, Y[:, idx].min(), Y[:, idx].max())
```

**示例中的目录名会随 SIM 的父目录名称变化**，请按实际 `run_001_*` 文件夹选择结果。

## World Model / NMPC 必采物理量

为了将来构造你的 8 维输入和 3 维加速度残差，建议至少完整保存：

- 时间戳
- 车身纵向、横向速度 \(v_x,v_y\)
- 横摆角速度 \(r\)
- 车身纵向与横向加速度
- 四个车轮转速
- 左右前轮实际转角（区别于方向盘角度）
- 实际方向盘控制指令/内部 Driver 输出
- 各轮纵向、横向轮胎力，以及轮胎载荷
- 各轮滑移率、侧偏角（如当前模型支持输出）
- 车身/路径位姿、航向角、横向偏差、道路曲率
- 实际发动机驱动 / 制动力或可获得的轮端驱制动转矩
- 道路附着系数/路面状态（如当前模型支持）

实际关键字和单位请从当前车辆的 CarSim 2022.1 输出清单中逐项确认。
例如 \(v_x\) 的原生单位可能为 km/h，而 \(r\) 可能使用 deg/s；本代码只保留原生值，不能直接当作 SI 状态喂给 NMPC。

对于 DLC，**不要把方向盘角乘/除固定转向比 17 来替代实际前轮转角**。应该直接读取两侧前轮实际转角并确认算法需要的前轴等效转角定义。

## 如果零 Import DLL 接口仍然不能运行

使用官方 `VS_Solver_Wrapper_CLI.exe` 运行 **Self-Contained Solver**，让 CarSim 输出完整的 VS/ERD 或 CSV 结果，然后在 Python 中后处理。

CarSim 官方确认支持 VS Solver Wrapper 命令行以及 VS API，但你本机 2022.1 的 CLI 精确选项需要在命令行运行：

```powershell
& "实际安装目录\VS_Solver_Wrapper_CLI.exe" -help
```

这里不预设其 `simfile` 参数的具体开关。可以在帮助输出确认后用 Python `subprocess.run([...])` 顺序运行多个 SIM，记录返回码和标准输出，再读取对应的 CSV/VS 结果文件。

这一途径对“批量采集全部 Output 变量”通常更省心；实时 Export 方式适合“每一步立刻在 Python 得到状态”的需要。

## 物理边界

内置预瞄 Driver 完成的 DLC 是“CarSim 内部驾驶员给出实际控制、车辆真实响应”的闭环轨迹。
下一阶段构造 NMPC 残差样本时，需要继续验证 `delta_front`、`Fx_all`、`Mz_tv_estimated` 与 CarSim 轮胎受力的一致性。新场景不应直接与旧六通道外部力矩控制数据混为同一个已验证的执行器接口。

本代码已通过 Mock 单元测试，尚未在你的 CarSim 2022.1 许可证和本机 DLL 上实际执行。
