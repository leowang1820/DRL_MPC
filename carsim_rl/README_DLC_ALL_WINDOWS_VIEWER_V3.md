# CarSim DLC：训练后自动弹出全部预测窗口 Figure（v3）

## 修改目标

你的旧版本 `dlc_single_scene_experiment.py` 只绘制第一个 3s 和第一个 5s 窗口：
`evaluate_windows()` 中评估了全部窗口，但 `_save_example()` 只传入 `examples[0]`；
原图函数强制使用 `matplotlib.use("Agg")`，只能保存 PNG，不能交互弹窗。

这次提供两种使用方式：

### A. 不重新训练（推荐：你已经有 final.pt）

使用已保存的模型权重和同一个原始 DLC `dlc_exports.npz`，
重新计算每个 3s/5s 预测窗口，弹出交互式 Matplotlib Figure。

### B. 训练完成后自动弹窗

在旧训练命令末尾增加 `--show-all-windows`。
同一次 `evaluate_windows()` 生成的全部预测轨迹会留在内存中用于交互图，
无需再为绘图重复积分；训练完成并写入报告后弹窗。

## 会显示哪些 Figure？

默认打开三个交互式窗口（**不保存 PNG**）：

1. `3s DLC prediction`：每个 3s 窗口的 CarSim / Pure NMPC / NMPC+MLP，
   同时显示 **vy、r、vx** 三个子图；
2. `5s DLC prediction`：同理，浏览每个 5s 窗口；
3. `DLC all-window error overview`：3s、5s 的 Pure / Residual 四个误差热力图，
   以起始 CarSim 时间为纵坐标、预测时间为横坐标。

操作：
- 在 3s/5s 图底部拖动 `Window` 滑块；
- 使用 `< Previous`、`Next >` 按钮翻页；
- 在预测图激活的情况下用键盘左/右箭头；
- 在热力图的对应横行单击，可跳转到该行的预测窗口。
- 三状态图下方还显示整个 CarSim 横向位置 Y 随绝对时间变化，
  阴影突出当前预测窗口，方便区分正向 DLC、平直、负向 DLC。

每一个窗口标题显示：
`起止绝对时间 / 纯 NMPC RMSE / 残差 RMSE / 改善率`。

本版本图不自动保存至本地 PNG，但模型 `final.pt`、`report.json`
和既有的首窗口原始 NPZ 仍按旧训练流程保存，便于复现。
图形窗口关闭后命令结束（`plt.show(block=True)`）。

## 安装

请先确保有 v2 全量训练程序和 `dlc_si.py` / `dlc_actuation.py` 等文件。

在 Windows PowerShell：

```powershell
conda activate drl_mpc_carsim
Set-Location "E:\modelcodeE\DRL_MPC\DRL_MPC\carsim_rl"

# 可选：备份旧程序
Copy-Item ".\wm_recovery\dlc_single_scene_experiment.py" `
    ".\wm_recovery\dlc_single_scene_experiment_backup_before_viewer.py"

# 将下载的 zip 放在系统 Downloads 文件夹
Expand-Archive `
    -Path "$HOME\Downloads\wm_dlc_all_windows_fig_viewer_v3.zip" `
    -DestinationPath . -Force
```

Matplotlib 推荐 TkAgg 交互后端：

```powershell
python -c "import matplotlib; import tkinter; print('Matplotlib:', matplotlib.__version__, '| Tk OK')"
```

如 `ModuleNotFoundError: tkinter`：

```powershell
conda install tk matplotlib -y
```

注意：Windows PowerShell 本机图形桌面可以弹出窗口；SSH 无图形桌面或
某些远程解释器无法打开 GUI。不要在交互模式使用 `MPLBACKEND=Agg`。
如果 TkAgg 加载失败且机器已安装 PyQt/PySide，可以传 `--backend QtAgg`。

## 一：已经训练好了，直接看全部窗口（不用训练）

适合你刚运行完成的正向+负向 DLC：

```powershell
python -m wm_recovery.dlc_all_windows_viewer `
    --data "artifacts/wm_recovery/dlc_pos_neg_90_01/run_001_CarSim2022_1_Data/dlc_exports.npz" `
    --checkpoint "artifacts/wm_recovery/dlc_pos_neg_90_train_01/final.pt" `
    --horizons "300,500" `
    --anchor-stride 50 `
    --mu-hat 0.8 `
    --mz-mode zero `
    --lambda-mode fixed-half `
    --backend TkAgg
```

独立查看器**不重新训练**，只加载 `final.pt` 权重，重新计算全部 3s/5s 预测窗口。
如果原始 NPZ 与 checkpoint 保存的 SHA256 不一致，将直接拒绝绘图，
避免误用其他场景模型。`mu_hat`、`Mz`、`lambda` 也与训练记录比对。

例：若实际完整采集为 18s，采样步长 0.01s：
- 3s：31 个窗口，起点 0, 0.5, ..., 15s；
- 5s：27 个窗口，起点 0, 0.5, ..., 13s；
每个窗口都能在滑块里看到。真实个数以采集结束时间为准。

如果需要把不同窗口与 CarSim 真实 Y 轨迹对应起来，
在下方 Y-时间曲线查看阴影即可。

## 二：以后训练结束自动弹窗

在原来全数据训练命令中增加 `--show-all-windows`，且输出目录需使用
**新的目录**（本程序不会覆盖已有的模型结果）：

```powershell
python -m wm_recovery.dlc_single_scene_experiment `
    --data "artifacts/wm_recovery/dlc_pos_neg_90_01/run_001_CarSim2022_1_Data/dlc_exports.npz" `
    --output "artifacts/wm_recovery/dlc_pos_neg_90_train_02" `
    --full-data `
    --mu-hat 0.8 `
    --epochs 250 `
    --batch-size 128 `
    --lr 0.001 `
    --horizons "300,500" `
    --anchor-stride 50 `
    --mz-mode zero `
    --lambda-mode fixed-half `
    --show-all-windows `
    --backend TkAgg
```

启动 GUI 前先保存 `final.pt`、`report.json`、`normalization.json`。
本命令因为设置了 `--show-all-windows`，不再调用 Agg 保存首窗口 PNG。
但保留此前的 NPZ 结果文件（非图像），供追溯。

如果你不加 `--show-all-windows`，旧版运行行为完全不变：
继续只把首个窗口的图保存到本地 PNG，不弹窗。

## 三：如何测试？

```powershell
python -m unittest `
    wm_recovery.tests.test_dlc_single_scene `
    wm_recovery.tests.test_dlc_all_windows_viewer -v
```

在没有 CarSim DLL 的合成数据环境中：
- 旧版 9 项全量训练单元测试通过；
- 新增 5 项测试通过；
- 检查了批量自由滚动与原逐窗口滚动一致；
- 验证全部窗口的时间顺序、滑块跳转和热力图绘制逻辑；
- 验证训练后自动可视化时不会保存 PNG，且不重新积分；
- 验证已保存 checkpoint 可以独立读回并重算全部窗口；
- 验证物理假设不一致时提示错误。

**测试环境无法实际打开 Windows 桌面 GUI**，GUI 弹窗需用户本机检查。

## 物理和统计范围

- 仍然是**同数据训练+同数据回放**，不是独立泛化验证。
- 仍然假设已知 CarSim 未来实际的前轮转角和轮胎力；
  它们是离线 oracle 外生信号，不是可部署 NMPC 指令。
- 图上每个窗口都会包含 CarSim 真值与两模型自由滚动；
  只有预测窗口的初始 q 来自 CarSim，此后禁止真值回灌。
- 全部窗口之间高度重叠，不能当作互相独立的统计试验。
- 如果想比较新的正、负 DLC 动态范围，可以观察热力图中
  **起始 CarSim 时间**对应哪段轨迹，重点查看是否出现负向 DLC
  的预测误差峰值。

## 包内文件

```text
wm_recovery/
    dlc_single_scene_experiment.py       # v2 全量训练基础上新增 --show-all-windows
    dlc_all_windows_viewer.py            # 独立全部窗口交互浏览器
    tests/
        test_dlc_all_windows_viewer.py   # 新增图形与数据流测试
README_DLC_ALL_WINDOWS_VIEWER_V3.md
```

不覆盖 `nmpc_reference.py`、`nmpc_residual_model.py`、`nmpc_rollout.py`，
也不修改任何 CarSim 文件和已采集/已训练的数据。
