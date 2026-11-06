# CarSim / World Model 统一场景主程序（兼容现有项目的最小重构）

本补丁只增加两个新业务模块和一份单元测试，不覆盖原来的联仿、7DOF、数据加载或训练代码。

## 一、文件功能（给初学者）

| 文件 | 作用 | 你平时需要修改它吗？ |
|---|---|---|
| wm_recovery/scenario_config.py | 定义 train1/train2/validation/test 的持续时间和四轮转矩、前轮转向变化 | 经常修改 |
| wm_recovery/scenario_runner.py | 批量启动 CarSim、保存原始记录、转换候选数据、建立训练清单；还可以启动训练/标定/评价 | 一般不修改 |
| wm_recovery/carsim_bridge.py | 五维 Python 命令 → 六维 CarSim Import；CarSimEnv.control_step() → 61 维 Export | 保留原文件 |
| wm_recovery/dataset.py | 生成因果历史、7DOF 名义预测与残差训练样本 | 保留原文件 |
| wm_recovery/training.py | 使用 PyTorch 训练五个 World Model 网络 | 保留原文件 |
| wm_recovery/world_model.py | 网络结构、预测和模型参数保存/加载 | 保留原文件 |

**六个文件不能单独构成整个工程**。
还必须保留原来的：
- carsim_env.py / vs_solver.py：底层 CarSim DLL 调用
- wm_recovery/controls.py：五维指令与六维 Import 映射
- wm_recovery/inspection.py / wm_recovery/carsim_export_schema.py：端口与 61 维输出检查
- wm_recovery/carsim_si.py / wm_recovery/carsim_candidate.py：原生单位转 SI、候选轨迹
- wm_recovery/offline.py / wm_recovery/carsim_experimental.py：实验轨迹加载及指纹审计
- dynamics7dof/：原始物理模型
- wm_recovery/__main__.py / wm_recovery/evaluation.py：训练、标定、评价命令及评价逻辑

旧 validate_carsim_long.py 不必删除：它包含额外的加速度运动学诊断；
新统一场景入口主要负责采集训练轨迹，不自动生成那些诊断图表。

## 二、安装

在项目根目录 `E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl` 执行：

    Expand-Archive -Path "$HOME\Downloads\wm_scenario_refactor_additions.zip" -DestinationPath . -Force

此 ZIP 内没有任何原有业务模块，只有：
- `wm_recovery/scenario_config.py`
- `wm_recovery/scenario_runner.py`
- `wm_recovery/tests/test_scenario_refactor.py`

请先确认目标目录中不存在同名的新文件，或备份后覆盖。

## 三、检查配置与回归测试

    python -m wm_recovery.scenario_runner plan
    python -m unittest wm_recovery.tests.test_scenario_refactor -v
    python -m unittest discover -s wm_recovery/tests -v

`plan` 不加载 CarSim DLL，不产生数据。

## 四、只采集四条独立 CarSim 轨迹

    python -m wm_recovery.scenario_runner capture `
      --output artifacts/wm_recovery/scenario_batch_01

采集结束后生成：

    artifacts/wm_recovery/scenario_batch_01/
        raw_train1/raw_capture.npz
        raw_train2/raw_capture.npz
        raw_validation/raw_capture.npz
        raw_test/raw_capture.npz
        candidate_train1/candidate.npz
        candidate_train2/candidate.npz
        candidate_validation/candidate.npz
        candidate_test/candidate.npz
        workspace/registry.json
        workspace/episodes/train1/experimental_episode.json
        workspace/episodes/train2/experimental_episode.json
        workspace/episodes/validation/experimental_episode.json
        workspace/episodes/test/experimental_episode.json

注意：新流程自己重新采集 train1，**不依赖旧的 validate_carsim_long.py**。

## 五、在已有 workspace 上训练、标定、评价

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"

    python -m wm_recovery.scenario_runner train-existing `
      --workspace artifacts/wm_recovery/scenario_batch_01/workspace `
      --output artifacts/wm_recovery/scenario_batch_01/model_run_01 `
      --epochs 1

训练会调用旧版 `python -m wm_recovery check-data/train/calibrate/evaluate`，
在输出目录下形成 model/world_model.pt、calibration.json、evaluation/。

**全部新文件都在新的目录中保存，永不覆盖既有实验。**

## 六、一条命令完成采集与实验训练

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"

    python -m wm_recovery.scenario_runner full `
      --output artifacts/wm_recovery/scenario_batch_02 `
      --epochs 1

新手建议先 `plan` → `capture` → `train-existing` 三步，不要一开始就用 `full`。

## 七、怎样修改训练场景

打开 scenario_config.py：

    SCENARIOS = (
        Scenario(name="train1", split="train",
                 condition_id="scenario_train1_three_phase",
                 mode="three_phase"),
        Scenario(name="train2", split="train",
                 condition_id="scenario_train2_sine_049",
                 mode="sine", torque_nm=30.0, torque_hz=0.49,
                 difference_nm=9.0, difference_hz=0.23,
                 steer_deg=0.26, steer_hz=0.38),
        ...
    )

- torque_nm：基础车轮转矩幅值，单位 N·m
- torque_hz：基础转矩频率，Hz
- difference_nm：左右差动转矩幅值，N·m
- difference_hz：差动转矩频率，Hz
- steer_deg：前轮转角幅值，度；返回时自动转为 rad
- steer_hz：转向频率，Hz
- duration_s：轨迹时长，必须是 0.01 s 的整数倍（当前至少 3 秒）
- condition_id：实验工况标识，改变控制方案时请同时修改它

`train1` 三阶段参数在 TRAIN1_PHASES 字典中。

重要：控制激励定义在 Python，但道路附着、初始车速、车辆参数等
依然由 CarSim Run/数据集决定。改动这些条件后，须生成相应的 simfile，
且不能把不同工况的轨迹混用。

## 八、原有功能与物理验证边界

- 没有修改原始 7DOF、CarSim DLL 交互、World Model 架构、损失函数。
- 保留名义共同前轮转角与 CarSim 实测左右前轮转角的区别。
- `ready_for_wm_training=false` 仍然表示未完成严格物理接口核验。
- 只有显式打开 WM_RECOVERY_EXPERIMENTAL_CARSIM=1 才能进行实验训练。
- 目前四条轨迹仅保证不同控制激励/记录，不等于跨车辆、跨道路附着的独立外推验证。
- 旧版 `carsim_experimental._candidate_facts()` 被新入口复用，所以暂时不要删除旧模块。

## 九、测试范围

新代码在隔离环境通过了 8 项无需 DLL 的测试，
并验证了生成的 experimental_episode.json 的字段兼容形状。
**尚未在用户的实际 Windows + CarSim 2022.1 环境中运行完整联仿**；
请先运行上述单元测试和 plan，再进行真实采集。

