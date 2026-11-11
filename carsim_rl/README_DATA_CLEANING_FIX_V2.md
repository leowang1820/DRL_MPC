# CarSim 多车速 WM 数据清洗 v2：严格候选格式兼容修复

## 发生了什么？

`data_cleaning clean` 在写入裁剪候选后执行

    new_facts = _candidate_facts(new_cand_path)

原始读取器 `wm_recovery/offline.py` 的
`CandidateEpisode.load()` 对 `candidate_report.json` 的要求为：

    status == "candidate_created_not_training_ready"
    ready_for_wm_training == false

旧版清洗模块错误地将 `status` 改成
`"candidate_cleaned_not_physically_validated"`，因而报错：

    ValueError: 候选轨迹状态与只读数据契约不一致

这不是 CarSim 计算失败，不是 0.01s 时间轴错误，也不是你的数据本身
因为 beta>15° 而无法清洗。

## 本版修复

1. 清洗后候选仍保留旧版原始 status 字段和严格 CandidateEpisode.load
   所要求的全部数值/语义/元数据校验。
2. 在候选报告额外增加 `quality_cleaned=true`、
   `quality_parent_candidate_sha256` 等审计字段，
   **不把 ready_for_wm_training 标记为 true**。
3. 保留原 registry 的 `status`，并额外加入质量清洗信息。
4. 保留所有状态、命令及采样索引的连续前缀裁剪，
   用原始 `_fingerprint` 重算 SHA256，仍要求旧读取器完整通过。
5. 清洗后的横向覆盖率按实际保留段重算，并保留原多车速训练入口
   使用的 `speed_coverage_passed` 字段。
6. 原始 `candidate.npz`、`raw_capture.npz`、SIM 配置和 World Model
   均不修改。

## 本次包内文件

    wm_recovery/data_cleaning.py
    wm_recovery/tests/test_data_cleaning.py
    wm_recovery/tests/test_data_cleaning_contract.py
    README_DATA_CLEANING_FIX_V2.md

请只覆盖 `data_cleaning.py`（及其单测），
**不要修改 `wm_recovery/offline.py` 或 `carsim_experimental.py`**。

## 安装（PowerShell）

    conda activate drl_mpc_carsim
    Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item .\wm_recovery\data_cleaning.py `
      ".\wm_recovery\data_cleaning.py.$stamp.bak"

    Expand-Archive `
      -Path "$HOME\Downloads\wm_postcapture_data_cleaning_fix_v2.zip" `
      -DestinationPath . -Force

## 测试（不启动 CarSim）

    python -m unittest wm_recovery.tests.test_data_cleaning -v
    python -m unittest wm_recovery.tests.test_data_cleaning_contract -v

新增单测直接调用项目原来的 CandidateEpisode.load()，
检查裁剪后的样本数量、6/5 维输入时间对齐、
原始模型输入语义、SHA256 和 `status` 等严格契约。
在隔离 Python 环境中已运行 11 项无 DLL 单测，全数通过。
由于无法访问你的真实 Windows/CarSim 文件，这不等于你本机全链路通过。

## 再次执行 clean

**上次失败的 `workspace_40_60_clean_01` 已建立了部分文件，
不是完整训练 Workspace。不要用它训练。**

最简单、最安全的方式是换新的输出路径 `_02`，不要手工覆盖。

    python -m wm_recovery.data_cleaning clean `
      --workspace artifacts/wm_recovery/workspace_40_60_01 `
      --output artifacts/wm_recovery/workspace_40_60_clean_02 `
      --min-vx-kmh 5 `
      --beta-limit-deg 15 `
      --max-steer-rate-deg-s 400 `
      --max-steer-accel-deg-s2 20000 `
      --guard-s 0.05 `
      --min-keep-s 1.20

如果原始数据集和参数均未变化，预期保持：
    保留 36/36 条，保留 20542/21600 区间。

另外应出现：

    清洗后的 Workspace： ...workspace_40_60_clean_02
    清洗后横向激励覆盖率： True 或 False

此时无 traceback 才表示完成。覆盖率取决于清洗后真实 ay 数据，
不能在运行前保证一定是 True。

## 验证输出并准备训练

    $clean = "artifacts/wm_recovery/workspace_40_60_clean_02"

    $quality = Get-Content "$clean/quality_report.json" `
      -Raw -Encoding UTF8 | ConvertFrom-Json
    $quality.retained_episodes
    $quality.rejected_episodes
    $quality.total_kept_intervals

    $coverage = Get-Content "$clean/multispeed_coverage_summary.json" `
      -Raw -Encoding UTF8 | ConvertFrom-Json
    $coverage.passed
    $coverage.speed_coverage_passed
    $coverage.checks | Format-List

如果清洗成功、覆盖率符合实验目标、物理输入域检查没有异常，
先做 10 epochs 的数据管线试训：

    $env:WM_RECOVERY_EXPERIMENTAL_CARSIM = "1"
    python -m wm_recovery.multi_speed_runner train `
      --workspace artifacts/wm_recovery/workspace_40_60_clean_02 `
      --output artifacts/wm_recovery/model_40_60_clean_10ep_02 `
      --epochs 10

如实验性覆盖率未通过，不建议为了正式训练而绕过；可单独使用
`--allow-underexcited` 仅调试训练流程，并明确记录该条件。
