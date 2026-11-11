# CarSim 多车速 WM：采集后质量清洗（连续前缀版）

## 1. 放在哪里？

**在所有 CarSim 场景采集结束、`multi_speed_runner merge` 之后，
但在 `multi_speed_runner train` 之前。**

流水线：

    capture -> merge -> data_cleaning audit -> data_cleaning clean -> train

无需修改原始 `carsim_bridge.py`、`dataset.py`、`world_model.py` 或 DLL。
新模块会为清洗结果创建**全新的**实验 Workspace，
继续使用原有 `experimental_episode.json` 和 `candidate.npz` 契约。

## 2. 安装

将本 ZIP 下载到 Windows Downloads 文件夹。
PyCharm Terminal：

    conda activate drl_mpc_carsim
    Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

    Expand-Archive `
      -Path "$HOME\Downloads\wm_postcapture_data_cleaning_v1.zip" `
      -DestinationPath . -Force

安装后包含：

    wm_recovery/data_cleaning.py
    wm_recovery/tests/test_data_cleaning.py

仅新增两个文件，既不覆盖 `scenario_config.py`、
`multi_speed_runner.py`，也不修改现有训练模型。

验证：

    python -m unittest wm_recovery.tests.test_data_cleaning -v

无 DLL 测试包括：正确时序、负 Vx、过大 beta、实际前轮角的
速率和加速度、可选的转矩速率、截断后的数组形状、生成可训练
Workspace 索引、报告及样本分区保留等。已通过 10 项模拟测试。
未在你的真实 CarSim 环境运行清洗或训练。

## 3. 先合并已经采集完成的 40/60kmh 数据

若尚未 merge：

    python -m wm_recovery.multi_speed_runner merge `
      --captures artifacts/wm_recovery/multispeed_40_60_01 `
      --speeds 40 60 `
      --output artifacts/wm_recovery/workspace_40_60_01

如果 `workspace_40_60_01` 已存在且 merge 完整，跳过。

## 4. 只读审计：不要立即真正裁剪

    python -m wm_recovery.data_cleaning audit `
      --workspace artifacts/wm_recovery/workspace_40_60_01 `
      --min-vx-kmh 5 `
      --beta-limit-deg 15 `
      --max-steer-rate-deg-s 400 `
      --max-steer-accel-deg-s2 20000

会逐条显示：
- 保留时长 vs 原时长
- 首次违规的时间
- 主要原因：vx_too_low/beta_too_large/
             steer_rate_too_large/steer_accel_too_large
- 是否还有至少 1.2 秒的连续前缀

`--min-vx-kmh 5` 更严格，拒绝 `Vx < 5km/h`；
若只想过滤 `Vx<0`，改成 `--min-vx-kmh 0`。

`Vx` 为车身坐标纵向分量。它变负不一定代表世界坐标倒车。

## 5. 生成全新的清洗后 Workspace

确认审计规则后：

    python -m wm_recovery.data_cleaning clean `
      --workspace artifacts/wm_recovery/workspace_40_60_01 `
      --output artifacts/wm_recovery/workspace_40_60_clean_01 `
      --min-vx-kmh 5 `
      --beta-limit-deg 15 `
      --max-steer-rate-deg-s 400 `
      --max-steer-accel-deg-s2 20000 `
      --guard-s 0.05 `
      --min-keep-s 1.20

输出：

    workspace_40_60_clean_01/
      registry.json
      quality_report.json
      multispeed_coverage_summary.json
      episodes/
        v40_train_.../experimental_episode.json
        v60_test_.../experimental_episode.json
      clean_candidates/
        v60_train_brake_turn_left/
          candidate.npz
          candidate_report.json

如果某条轨迹全部正常，清洗 Workspace 会直接引用原来的
候选文件，不会额外复制；若首次越界，则只复制此前连续前缀，
不得跨越中间失稳段拼接另一段。

`quality_report.json` 中会记录全部原场景及剔除理由。
短于 1.2 秒的前缀会从新数据清单排除，原始文件仍在原目录。

清洗后必须**重新计算**数据覆盖率；
`multispeed_coverage_summary.json` 不能沿用原报告的 passed=True。

## 6. 训练前核对数据质量和覆盖率

    $clean = "artifacts/wm_recovery/workspace_40_60_clean_01"

    $quality = Get-Content "$clean/quality_report.json" `
      -Raw -Encoding UTF8 | ConvertFrom-Json

    $quality.retained_episodes
    $quality.rejected_episodes
    $quality.total_kept_intervals

    $coverage = Get-Content "$clean/multispeed_coverage_summary.json" `
      -Raw -Encoding UTF8 | ConvertFrom-Json

    $coverage.passed
    $coverage.checks | Format-List

`passed` 仅是清洗后达到项目定义的横向响应代理门槛。
绝不是 CarSim/7DOF 的全部动力学接口已经物理验证。

## 7. 试训练 10 epochs

确保修订的模型仍适用于保留段，测试分区足够，
并且清洗后覆盖率及诊断通过：

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"

    python -m wm_recovery.multi_speed_runner train `
      --workspace artifacts/wm_recovery/workspace_40_60_clean_01 `
      --output artifacts/wm_recovery/model_40_60_clean_10ep_01 `
      --epochs 10

只有 10 epochs 的管线、验证与独立测试都成功，
才建议在**新模型目录**做 100 epochs 正式实验。

## 8. 需要你理解的关键边界

- 该版本的策略是严格保守的「第一处违规后全部不再用于本轮训练」；
  这样才能沿用已有 7DOF 的连续 `LoadMemory`，不用伪造状态重置。
- 如果希望复用多个不连续的安全片段，必须明确片段间重新初始化
  7DOF 的载荷记忆及历史窗口，同时补充训练和评估测试。
- `max-steer-rate-deg-s=400`、`max-steer-accel-deg-s2=20000`
  是项目调试阈值，不是 CarSim 执行器规格或法规限值。
  请根据正常的 step、chirp、SWD 轨迹实测上界校准。
- `reported_external_torque_nm` 当前是 `My_US` 外加转矩回显；
  有意设置的突变不能简单判作执行器噪声。
  转矩变化率筛选默认关闭；需要时才加：
      --max-torque-rate-nm-s 100000
  上述 100000 只是演示，不是推荐物理阈值。
- 本版本只过滤明显异常的状态和执行器信号；不能
  保证侧向 15°、Vx 5km/h 的边界就是可控极限。
- 若将验证/测试中严重失稳时间段裁掉，那些指标
  **只说明模型对保留状态范围的性能**，不代表可预测
  已被裁掉的甩尾/倒车状态；原始测试集必须保留。
- 筛选规则应在最终测试评估前固定，避免按测试误差
  反复修改阈值造成选择偏差。
- 清洗后的数据对车速/侧偏分布会产生偏置，适合初步 WM-MPC，
  后续研究甩尾恢复需独立设计大侧偏数据建模与验证。
