# CarSim 内置预瞄 DLC：时间戳异常诊断与终止短步分离修复（v2）

## 当前错误

原 `dlc_capture.py` 在 DLL 完成仿真以后强制检查：

```python
np.allclose(np.diff(times), 0.01, rtol=0.0, atol=1e-7)
```

只要任何相邻时间戳间隔偏离 10ms 就会报：
`ValueError: 采样间隔与设定值不一致。`

仅凭旧错误文本，**无法知道偏差发生在中途还是最后**。
这种报错不是 `PORTS_IMP`/`PORTS_EXP` 的配置解析问题，也不能直接证明内置预瞄控制失效。

## v2 修改

仅覆盖一个生产文件：

```text
wm_recovery/dlc_capture.py
```

新增/更新：

```text
wm_recovery/tests/test_dlc_capture.py
```

保留原来的 `dlc_env.py`、`inspection.py`、`carsim_env.py` 不变。

新增的行为：

1. 严格报告真实的采样间隔最小值/中位数/最大值，以及第一批异常时间戳与位置。
2. 如果**仅最后一个区间短于 10ms**、前面所有间隔都正确，且 CarSim 明确报告 done：
   - 正常保留完整采样网格到 `dlc_exports.npz`；
   - 最后的非等间隔帧保存至 `terminal_partial_step.npz`；
   - `metadata.json` 记录实际 Solver 终止时刻、尾步长度和诊断信息；
   - 不把尾步误当作 10ms 的监督样本。
3. 如果中途发生 9ms、15ms、20ms 等异常或有多个异常，不会自动删除、修正、插值或重采样：
   - 保留全部原始采集为 `INVALID_TIMING_do_not_train.npz`；
   - 保存 `INVALID_TIMING_details.txt`，包含出错时刻、统计和 Solver 返回信息；
   - 仍会抛出异常，提示**不能直接用于 NMPC 10ms 标签**。
4. 如果你想严格禁止末尾短步，可加 `--terminal-partial-policy reject`。
5. 保留 `--time-tolerance 1e-7` 默认精度；不能设置为比采样间隔 0.1% 更大的容差。

这些措施避免在精度异常时把采样间隔错误的结果直接混入数据集。

## 安装（PowerShell）

将 `wm_carsim_dlc_timing_fix_v2.zip` 放到下载目录：

```powershell
conda activate drl_mpc_carsim

Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

# 备份旧代码
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item .\wm_recovery\dlc_capture.py `
  ".\wm_recovery\dlc_capture.py.$stamp.bak"

Expand-Archive `
  -Path "$HOME\Downloads\wm_carsim_dlc_timing_fix_v2.zip" `
  -DestinationPath . -Force
```

## 运行回归测试

```powershell
python -m unittest wm_recovery.tests.test_dlc_capture -v
```

已用 FakeCarSimEnv 运行 **7 项离线测试全部通过**，包括：
- 10ms 等时序通道采集、重复 Export 名称保留；
- 外部方向盘 Import 拒绝；
- Solver Export 维度不匹配拒绝；
- `n_import=0` 环境适配；
- 最后 7ms 短步被独立保存；
- 中途丢失采样被隔离，并且明确报错；
- 明确诊断错误时间间隔。

注意：这些测试不能替代 CarSim 2022.1 DLL 的真实集成测试。

## 重新执行 CarSim DLC

**使用新的 `--output` 目录名。** 原程序的 `main()` 在采集前就创建了根目录；
第一次失败后旧的 `dlc_preview_40_01` 可能仍然存在，重复使用会触发“输出目录已存在”。

```powershell
python -m wm_recovery.dlc_capture `
  --sim "E:\modelcodeE\carsim2022\CarSim2022.1_Data\simfile.sim" `
  --output artifacts/wm_recovery/dlc_preview_40_02 `
  --sample-dt 0.01 `
  --max-time 30
```

### 如果是最后一个不满 10ms 的时间步

将打印类似：

```text
采样检查：N 帧，期望 dt=0.01s，实际中位 dt=0.01s，最小=0.005s，最大=0.01s
检测到终止短步：前一时刻=8.230000000s，终止时刻=8.235000000s，末尾 dt=0.005000000s；
将短步单独保存，主 NPZ 只包含等间隔数据。
```

结果：

```text
dlc_exports.npz                  # 严格等间隔 10ms
terminal_partial_step.npz        # 最后一帧短步，不参与 10ms 标签
metadata.json                    # 记录原始终止时间、采样诊断
```

### 如果是中间时间出现不规则间隔

将**继续抛异常**，但这次会打印类似：

```text
实际 dt 范围=[0.005, 0.02]s，中位数=0.01s，异常数量=...
第一个异常时刻=[{"i": ..., "t0": ..., "t1": ..., "dt": ...}]
最后十个时间戳=[...]
原始时序已隔离保存：...INVALID_TIMING_do_not_train.npz
```

如果这些异常大量发生在每一个区间，需要核对 `env.control_step` 的实际步进、
`env.t_current` 的更新时间点，以及 `.sim` 中的 `TSTEP`/`TSTART`/`TSTOP`。

**不要直接将 `np.diff(times)` 替换成固定的 `0.01`，也不要把所有异常时间戳强制改成均匀时间轴。**

## 研究上的重要边界

本补丁仅修复 Python 采样网格的安全校验和结果留存，不改变 CarSim 内置预瞄驾驶员，不自动修改车辆物理参数、Import/Export，也不会增加新的 CarSim Output 变量。

`dlc_exports.npz` 仍然只包含配置过的全部 Export；若需要全部可用 Output，请使用 CarSim 独立求解器的结果文件功能。

如果 v2 仍报错，请提供：
- 新报错里的 `实际 dt 范围`、`中位数`、`异常数量`、`第一个异常时刻`；
- `INVALID_TIMING_details.txt`；
- `.sim` 中的 `TSTEP`、`TSTART`、`TSTOP`；
- 你目前 `CarSimEnv.control_step` 和 `t_current` 更新片段。

根据这些信息就能判断是不是求解器积分推进与 Python 读取时间不一致，而不是继续猜测。
