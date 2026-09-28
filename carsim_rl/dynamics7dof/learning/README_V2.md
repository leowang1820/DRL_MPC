# v2：直行一致性、镜像结构和多工况训练

本次只修改第一层残差模型。原 7DOF 参数、轮胎拟合、CarSim 配置及 SAC 未改动。
所有仿真结果来自已有采集的离线回放；代码没有替用户启动新的 CarSim 求解。

## 为什么修改

v1 在 3° / 0.6 Hz 滑行回合改善了整体 RMSE，但转向前凭空产生横向/横摆运动。
且滑行训练不能覆盖驱动、制动。新的结构约束解决前者，多工况训练和误差平衡改善后者，
但是否真正有益仍必须看连续预测，不能仅凭训练损失下降判断。

## 1. 精确的左右镜像约束

令 Mx 表示输入镜像：vy、r、前轮角和 memory_ay 变号；左右轮速、左右转角与左右扭矩交换。
vx、mu、memory_ax 保持不变。轮速、扭矩只交换左右，不因镜像而反转正负。
My 表示输出导数的对应镜像。每个网络的实际预测为：

    h(z) = 0.5 * [g(z) + My g(Mx z)]

因此 h(Mx z)=My h(z)。左右对称的直行状态中，每个成员的横向、横摆残差严格为零，
而纵向及车轮加减速补偿仍然保留。并非一见零方向盘就强行清零：已有侧向状态、左右轮速差、
差动扭矩和载荷记忆仍参与预测，转向结束后的横向运动也不会被硬截断。

Torch 训练、NumPy 推断、模型保存加载及在线候选训练都使用同一结构。
标准化和覆盖范围只由拟合样本及其镜像计算。历史保护集和开发/验证集不进入标准化或梯度。
镜像样本不是新增实测工况；模型检查点记录 `symmetry=mirror`。

适用前提：平路、均匀附着、左右对称车辆；不适用于分离附着、路面横坡、侧风、左右轮胎/定位
不一致等未建模非对称环境。该约束不能证明闭环稳定或真实车辆完全对称。
在完全直行的对称状态，横向网络分歧也会被结构压为零，这不是该状态真实误差已校准的证据。

## 2. 多工况误差平衡

训练采用两组已有回合：5° / 0.5 Hz 滑行，以及 0.5° / 0.3 Hz 驱动制动。
bootstrap 仍按回合内 50 ms 小块分组，独立训练 5 个成员。

默认 `--loss-weighting episode-rms`：以每个训练回合、每个输出的残差 RMS 调整归一化误差权重，
使用原物理噪声下限，权重限制在 0.5～4。权重从 FIT 行计算，不使用对照集误差。
目的是让小转角回合不被大转角回合的高幅值导数残差淹没。
这仍是导数监督，不是多步 rollout 损失；低导数误差不能保证长时积分稳定。
在线更新目前维持冻结的全局输出尺度和候选验证门槛，保留镜像结构，但不会重算历史工况权重。

## 3. 不能继续把已看过的回合叫作独立测试

3° / 0.6 Hz 回合已经用于诊断和选择改进方向，因此创建数据集时使用：

    --validation-role development

输出会标为 `DEVELOPMENT_EPISODE_NOT_INDEPENDENT`。
它不参与梯度和标准化，但成绩仅供开发回归使用。与训练回合共有的确定性直行启动片段也会
计入重叠比例；开发角色允许保留该片段以检查直行偏差，并明确说明不能声称独立。
默认 held_out 模式仍拒绝大量共享标签窗口，完全重复的物理录制在所有模式下都拒绝。
下一轮正式验收必须另外采集未见工况，而且不得反复调参后仍称它为独立测试集。

## 4. 输出与检查重点

本轮结果保存在 `artifacts/dynamics7dof/residual_framework_v2/`，v1 完整保留。
本轮具体数值、候选版本与限制见该目录 `RUN_REPORT.md`。

- `dataset/`：两组训练回合 + 一组开发对照，含来源和重叠比例。
- `model/`：首轮镜像候选，100 轮，尚未按回合平衡；保留作开发记录。
- `model_balanced/`：按工况平衡的镜像候选，150 轮。
- `balanced_development_3deg/comparison.html`：3° 开发对照。
- `balanced_training_5deg/comparison.html`：5° 训练回合回归。
- `balanced_training_drive_brake/comparison.html`：驱动制动训练回合回归。

本轮候选不是全工况合格模型：直行与滑行整体误差改善，但驱动制动的横摆角速度 RMSE
由原模型 0.002719 增至 0.004289 rad/s。不得只看开发回合的改善就将其接管控制。

每张报告新增：全程 RMSE 和 MAE、转向前/转向期间/转向结束 0.5 s 后的误差、连续积分回退比例。
末段不一定已经稳定。实测状态上的逐点回退与自由积分回退必须分开看。
训练回合改善不能证明泛化；前两版网络是否改善必须看同一工况、同一有效区间。

## 5. 如何重跑（不会启动 CarSim）

PyCharm 解释器为 drl_mpc_carsim，工作目录为：
`E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl`。

```powershell
$learningV2Root = ".\artifacts\dynamics7dof\residual_v2_my_run_01"
python -m dynamics7dof.learning build --train ".\artifacts\dynamics7dof\open_loop_20260927_233218_823895\capture" --train ".\artifacts\dynamics7dof\open_loop_20260927_223412_368297\capture" --validation ".\artifacts\dynamics7dof\open_loop_20260928_094104_829627\capture" --validation-role development --output "$learningV2Root\dataset"
python -m dynamics7dof.learning train --dataset "$learningV2Root\dataset" --output "$learningV2Root\model" --epochs 150 --device cuda --symmetry mirror --loss-weighting episode-rms
python -m dynamics7dof.learning evaluate --checkpoint "$learningV2Root\model\ensemble.pt" --capture ".\artifacts\dynamics7dof\open_loop_20260928_094104_829627\capture" --output "$learningV2Root\development"
```

新结果目录必须不存在；不会覆盖已保存的实验。旧 checkpoint 仍用原结构读取，
不会因源代码更新而把旧网络自动套用新约束。要使用 v2，必须显式指定新 checkpoint。

## 6. 在线接入的边界

`open_loop --learning-model ...` 只进行在线预测和记录；不带 `--learn-online` 就不更新权重。
带该开关时，更新的依然只是旁路观察器，不影响车辆转向/油门/制动输入。
本轮没有验证真实在线更新后的总体收益，不建议直接把学习模型接入 MPC、SAC 或 Risk 标签生成。
