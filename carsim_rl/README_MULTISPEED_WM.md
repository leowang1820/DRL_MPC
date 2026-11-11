# CarSim 多车速场景采集与 World Model 联合训练（中文指南）

## 这次新增什么？

你已有的根目录：

    E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve\
        40kmh\simfile.sim
        60kmh\simfile.sim
        80kmh\simfile.sim
        100kmh\simfile.sim
        120kmh\simfile.sim
        140kmh\simfile.sim
        160kmh\simfile.sim
        180kmh\simfile.sim

新版将这些文件按各自的 CarSim Run 读取，逐速度生成数据，
再合并到统一的 WM 训练/验证/测试 Workspace。

本补丁包含：
- `wm_recovery/multi_speed_runner.py`：audit/probe/capture/merge/train 主入口
- `wm_recovery/scenario_runner.py`：原 18 场景版本，初速允许上限由 110 提到 181 km/h
- `wm_recovery/scenario_config.py`：保留原 18 个极限场景的定义
- `wm_recovery/tests/test_multi_speed_runner.py`：10 项无 DLL 单元测试
- `wm_recovery/tests/test_scenario_refactor.py`：兼容 18 场景的老测试

原有 `carsim_bridge.py` / `carsim_env.py` / `vs_solver.py` /
`carsim_candidate.py` / `offline.py` / `dataset.py` / `training.py` /
`world_model.py` / `dynamics7dof` 均继续使用，不覆盖、不删除。

## 第 0 步：非常重要的 SIM 事实

上传的样例 `simfile.sim` 中包含：

    SET_MACRO $(ROOT_FILE_NAME)$ Run_36bbb28a-cc27-48a9-a3fd-60f74a5b6c47
    SET_MACRO $(WORK_DIR)$ E:\modelcodeE\carsim2022\CarSim2022.1_Data\
    INPUT $(WORK_DIR)$Results\$(ROOT_FILE_NAME)$\Run_all.par
    FILEBASE $(OUTPUT_FILE_PREFIX)$
    PORTS_IMP 1,6
    PORTS_EXP 1,61
    EXT_MODEL_STEP 0.00100000

这里**没有初始车速的直接数值**。不同文件夹名称不等于不同车速，
必须执行 `probe` 取得 CarSim Solver 初始化后的 `Vx`。

若八份 SIM 全部指向同一 Run_all.par 或同一 FILEBASE，本程序会直接报错。
处理方式：在 CarSim 建立真正不同的初始速度 Run，分别生成 SIM、
保留相应的 Results\Run_<GUID>\Run_all.par，检查各 Run 的输出位置互不冲突。
不能只改文件夹名，也不要盲目编辑 Solver 二进制或随意替换 PARAM 关键字。

## 第 1 步：安装

在 PyCharm Terminal (PowerShell) 执行：

    conda activate drl_mpc_carsim
    Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item .\wm_recovery\scenario_config.py ".\wm_recovery\scenario_config.py.$stamp.bak"
    Copy-Item .\wm_recovery\scenario_runner.py ".\wm_recovery\scenario_runner.py.$stamp.bak"
    if (Test-Path .\wm_recovery\tests\test_scenario_refactor.py) {
        Copy-Item .\wm_recovery\tests\test_scenario_refactor.py ".\wm_recovery\tests\test_scenario_refactor.py.$stamp.bak"
    }

    Expand-Archive -Path "$HOME\Downloads\wm_multispeed_18scenes_v1.zip" `
        -DestinationPath . -Force

安装后查看：

    python -c "from wm_recovery.scenario_config import SCENARIOS; print('场景数:',len(SCENARIOS))"
    python -m unittest wm_recovery.tests.test_multi_speed_runner -v
    python -m unittest discover -s wm_recovery/tests -v

预期场景数：18。
新增 10 项测试是 Python 模拟对象测试，不启动真实 CarSim DLL。

## 第 2 步：只读检查八份 SIM，必须先做

    python -m wm_recovery.multi_speed_runner audit `
        --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve" `
        --output artifacts/wm_recovery/multispeed_audit_01.json

audit 检查：
- 8 个 SIM 文件是否都存在；
- 宏展开后的 INPUT 是否均能找到 Run_all.par；
- 各 INPUT、FILEBASE、SIM SHA、PARAM SHA 是否互不冲突；
- Import 6、Export 61 和原来的通道顺序、Add/Replace 模式是否匹配。

任何冲突都不允许直接采集。
`audit` 不启动 CarSim，**也不能确认真实初始车速**。

## 第 3 步：真实初始化八个 Solver，检查初始速度

    python -m wm_recovery.multi_speed_runner probe `
        --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve" `
        --tolerance-kmh 1 `
        --output artifacts/wm_recovery/multispeed_probe_01.json

实际做的是：

    with CarSimBridge(sim_path) as bridge:
        actual_v0_kmh = float(bridge.latest.data.Vx)

必须满足：
    40kmh -> 40 ± 1 km/h
    60kmh -> 60 ± 1 km/h
    ...
    180kmh -> 180 ± 1 km/h

`probe` 启动并重置 Solver，不进行 6s 控制激励。
初始化过程仍可能写 CarSim 自己的日志，因此不同速度的 FILEBASE 必须不同。

## 第 4 步：先在 40/60/80 km/h 进行小批次测试

    python -m wm_recovery.multi_speed_runner capture `
        --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve" `
        --speeds 40 60 80 `
        --output artifacts/wm_recovery/multispeed_01

每个速度：
- 2 次 0.3g 参考前轮角标定；
- 18 条开环极限动力学轨迹（12 train / 3 val / 3 test）；
- 记录 CarSim 原始数组、SI 候选轨迹、覆盖率报告；
- 逐条检查实际第一帧速度、SIM/PAR SHA 指纹一致；
- 只有全部通过才写 `speed_capture_manifest.json` 成功标志。

如果 40/60/80 的效果和数值稳定性都合理，再执行剩下的速度。
120~180km/h 的极限转向仿真需要显式 `--allow-high-speed`：

    python -m wm_recovery.multi_speed_runner capture `
        --root "E:\modelcodeE\DRL_MPC\DRL_MPC\carsimruns\multi_ve" `
        --speeds all `
        --allow-high-speed `
        --resume `
        --output artifacts/wm_recovery/multispeed_01

`--resume` 只会跳过已经完整采集、完整性和配置指纹均通过的速度，
不会清空已有文件。某车速半途失败时，会留下不完整文件夹；
需要先人工检查/归档该文件夹，再使用新的目录重新采集，不要直接覆盖。
并不是每条强激励在高车速下都数值有效，CarSim 或 7DOF 可能报告域外状态。

## 第 5 步：合并八个车速的训练/验证/测试轨迹

    python -m wm_recovery.multi_speed_runner merge `
        --captures artifacts/wm_recovery/multispeed_01 `
        --speeds all `
        --output artifacts/wm_recovery/multispeed_workspace_01

合并结果：
- 8*12 = 96 条训练轨迹
- 8*3  = 24 条验证轨迹
- 8*3  = 24 条测试轨迹
- 合计 144 条独立的 6 秒轨迹
- 除此以外，16 条参考角标定轨迹不进入训练

注意：默认划分包含所有车速的训练样本，因此测试评估的是
**已见过车速范围内的不同控制波形**，不是“完全未见过车速的外推”。

合并只生成索引 JSON，`candidate_path` 仍指向先前各速度目录；
合并后**不能删除或移动**原 `multispeed_01/<speed>kmh/candidate_*`，
否则训练将找不到候选数据。

## 第 6 步：训练五网络残差 World Model

先检查：

    $coverage = Get-Content `
      "artifacts/wm_recovery/multispeed_workspace_01/multispeed_coverage_summary.json" `
      -Raw -Encoding UTF8 | ConvertFrom-Json
    $coverage.passed
    $coverage.speed_coverage_passed | Format-List

然后：

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"

    python -m wm_recovery.multi_speed_runner train `
        --workspace artifacts/wm_recovery/multispeed_workspace_01 `
        --output artifacts/wm_recovery/multispeed_model_100epochs_01 `
        --epochs 100

本命令使用已经存在的：
- `python -m wm_recovery check-data`
- `python -m wm_recovery train`
- `python -m wm_recovery calibrate`
- `python -m wm_recovery evaluate`

在不重采集 CarSim 的情况下训练、标定与分别测试。
如果某车速的覆盖率未通过，默认阻止“极限覆盖通过”的训练。
仅调试管线时可显式加 `--allow-underexcited`，并必须如实说明数据不足。

## 第 7 步：可选，验证未参加训练的车速

若希望研究“对完全未训练的车速能否泛化”，必须重新建分区，不能
直接使用全部车速参与训练的模型作未见车速测试。

    python -m wm_recovery.multi_speed_runner merge `
        --captures artifacts/wm_recovery/multispeed_01 `
        --speeds all `
        --validation-speed 160 `
        --test-speed 180 `
        --output artifacts/wm_recovery/multispeed_workspace_holdout_01

这样：
- 训练速度：40,60,80,100,120,140；每个速度 12 条训练轨迹 => 72 train
- 验证速度：160km/h；3 条原 validation 工况
- 测试速度：180km/h；3 条原 test 工况
- 其他没用到的候选轨迹不加入这一轮实验

对应单独训练：

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"
    python -m wm_recovery.multi_speed_runner train `
        --workspace artifacts/wm_recovery/multispeed_workspace_holdout_01 `
        --output artifacts/wm_recovery/multispeed_holdout_model_01 `
        --epochs 100

## 强制保留的工程边界

1. `scenario_config.py` 负责控制激励；`simfile.sim`/`Run_all.par` 决定真实车速、
   路面、初始状态。这两层不得混淆。
2. 原先 `CarSimBridge` 的 Import 顺序、61 Export 顺序、原生单位 -> SI 转换不变。
3. CarSim 同一速度的 18 条场景会共用其单个 Run 的 `FILEBASE`，
   所以原生 `LastRun.vs` 往往只保留最后一次计算；每条 Python `raw_capture.npz`
   会保存到独立目录，不受这个覆盖影响。
4. 当前 7DOF 使用名义 mu=0.8；不能把其他实际路面附着条件未经核验混进数据集。
5. `ready_for_wm_training=false` 仍旧保留；只有用户显式启用
   `WM_RECOVERY_EXPERIMENTAL_CARSIM=1` 才执行开发性训练。
6. 高速极限操纵下，7DOF 有可能出现模型域标志、无有效训练窗口等错误。
   不能通过随意跳过检查或修改标志伪装成有效数据。
7. 模型是否充分训练要看验证集表现、测试集单步与滚动误差，而不能仅看 epochs。
8. 多车速合并和训练只在 Python 模拟/隔离环境验证了数据契约形状，
   尚未在用户真实 Windows+CarSim 2022.1 环境完成 8 速度端到端运行。

