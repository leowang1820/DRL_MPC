# 第一阶段代码验证记录：2026-11-04

## 交付范围

按本次会话最终约定，新建 `carsim_rl/wm_recovery` 独立模块，保留原论文物理参数与旧代码。没有修改或运行 CarSim，也未调整已有 7DOF、SAC 和旧残差学习代码。工程根 README 增加新模块入口。

## 已执行检查

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s wm_recovery/tests -v` | 23 项通过 |
| 原 `dynamics7dof/tests/test_reference.py` | 18 项通过 |
| 原 `dynamics7dof/tests/test_rollout_learning.py` | 11 项通过 |
| 默认配置合成端到端流程 | 完成：五成员训练 1 轮、独立验证标定、名义与未知 μ 测试 |
| 模型保存/加载 | 预测一致；拒绝覆盖检查点 |
| 历史与数据隔离 | 过去窗口、完整轨迹 Bootstrap、重压缩去重、跨分区排除通过 |
| 物理一致性 | 零残差时状态与载荷记忆对照原 7DOF 通过 |
| 可信度反例 | 分歧为零但成员共同预测错误时，残差监测可触发 |
| HTML 报告 | 两份报告均在浏览器加载，SVG 图片完整、无水平溢出；名义测试报告目视检查 |

最终示例目录为 `carsim_rl/artifacts/wm_recovery/framework_v1_20261104`：

- `model/world_model.pt`：五成员模型，64 隐藏单元，10 ms 周期、10 步历史、5 步训练滚动。
- `confidence.json`：单独合成验证轨迹的阈值与模型指纹。
- `test_evaluation/report.html`：独立合成名义条件测试。
- `unknown_mu_evaluation/report.html`：独立合成 μ=0.4 条件测试，名义物理预测仍为 μ=0.8。
- 各评价目录含 `evaluation.json`、`predictions.npz`、曲线 SVG 与浏览器预览 PNG。

两条测试各有 3 个精确 1 s 滚动窗口，连续预测有效覆盖率均为 100%。这仅是程序运行检查，不是模型性能验收；合成轨迹短、操纵温和、训练仅 1 轮，不能作为真实预测精度、未知附着识别能力或闭环稳定性的证据。

## 尚待实际实验完成的工作

目标 CarSim 的附着、转向、整车参数及输入/输出时序尚未对齐；本模块没有自动转换旧录制数据。真实采集需遵守 DATA_FORMAT 中的区间输入定义，确认输入保持或扩展子步执行器输入接口。

真实训练后的阈值标定、1 s 预测精度、未知工况误报/检测延迟、在线计算耗时均待测。SE、π_s、Risk 标签和 NMPC 集成属于后续阶段，尚未实现或声明通过。
