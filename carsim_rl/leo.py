
from pathlib import Path
import numpy as np

from wm_recovery.dlc_si import load_dlc_si
from wm_recovery.dlc_actuation import build_dlc_inputs
from wm_recovery.nmpc_reference import (
    NMPCReferenceParams,
    nmpc_force_balance,
)

# 读取原始 CarSim 数据
path = Path(
    "artifacts/wm_recovery/dlc_pos_neg_90_01/"
    "run_001_CarSim2022_1_Data/dlc_exports.npz"
)

dlc = load_dlc_si(path)
p = NMPCReferenceParams()

z = build_dlc_inputs(
    dlc,
    mu_hat=0.8,
    mz_mode="zero",
    lambda_mode="fixed-half",
).z_si

# NMPC 计算的前后轴轮胎力和合力
nom = nmpc_force_balance(z, p)

t = dlc.time_s

# 按 FL、FR、RL、RR 顺序读取四轮轮胎力
wheels = ("L1", "R1", "L2", "R2")

fx = np.column_stack([
    dlc.get(f"Fx_{w}") for w in wheels
])

fy = np.column_stack([
    dlc.get(f"Fy_{w}") for w in wheels
])

fz = np.column_stack([
    dlc.get(f"Fz_{w}") for w in wheels
])

# 第一轮近似：前轮采用实际转角，后轮不转向
# 这里假设 Fx/Fy 是轮胎局部坐标系的力
# 尚未包含外倾角、车身横滚、路面坡度等三维修正
zero = np.zeros_like(t)

steer = np.column_stack([
    dlc.delta_fl_rad,
    dlc.delta_fr_rad,
    zero,
    zero,
])

# 近似将四轮局部力旋转至车身平面
fx_body = (
    fx * np.cos(steer)
    - fy * np.sin(steer)
)

fy_body = (
    fx * np.sin(steer)
    + fy * np.cos(steer)
)

# CarSim 四轮力，按前后轴合并
fy_front_car = fy_body[:, :2].sum(axis=1)
fy_rear_car = fy_body[:, 2:].sum(axis=1)

fy_total_car = fy_front_car + fy_rear_car

# NMPC 前轴力转换至车身横向
fy_front_nmpc = (
    nom["fy_front_n"] * np.cos(z[:, 3])
    + z[:, 5] * z[:, 6] * np.sin(z[:, 3])
)

fy_rear_nmpc = nom["fy_rear_n"]

# 近似重建四轮轮胎平面力产生的横摆力矩
# 轮距先沿用项目中的假设值，尚未核对 CarSim
track_front = 1.605
track_rear = 1.605

mz_tire_approx = (
    p.lf_m * fy_front_car
    - p.lr_m * fy_rear_car
    + 0.5 * track_front
      * (fx_body[:, 1] - fx_body[:, 0])
    + 0.5 * track_rear
      * (fx_body[:, 3] - fx_body[:, 2])
)

# 从 CarSim 加速度反推等效合力和横摆力矩
fy_equiv = p.mass_kg * dlc.ay_mps2

mz_equiv = (
    p.inertia_z_kgm2
    * np.gradient(dlc.q_si[:, 1], t)
)

intervals = [
    (3.0, 3.5),
    (3.5, 4.0),
    (4.0, 4.5),
    (4.5, 5.0),
    (5.0, 5.5),
    (5.5, 6.5),
]

for start, stop in intervals:

    mask = (t >= start) & (t < stop)

    def mean(values):
        return float(np.mean(values[mask]))

    print(f"\n===== {start:.1f}-{stop:.1f}s =====")

    print("Front Fy (N):")
    print(
        "  CarSim tires:",
        round(mean(fy_front_car), 2)
    )
    print(
        "  NMPC model: ",
        round(mean(fy_front_nmpc), 2)
    )

    print("Rear Fy (N):")
    print(
        "  CarSim tires:",
        round(mean(fy_rear_car), 2)
    )
    print(
        "  NMPC model: ",
        round(mean(fy_rear_nmpc), 2)
    )

    print("Total lateral force (N):")
    print(
        "  CarSim m*Ay equivalent:",
        round(mean(fy_equiv), 2)
    )
    print(
        "  Rotated tire forces:  ",
        round(mean(fy_total_car), 2)
    )
    print(
        "  NMPC model:         ",
        round(mean(nom["sum_fy_n"]), 2)
    )

    # --------------------------------------------------
    # 前轴垂向载荷 Fz
    # --------------------------------------------------

    print("Front Fz (N):")

    # 将左右前轮的垂向载荷相加
    # 注意：这里不要提前使用 mask
    # mean() 函数内部已经按照时间区间筛选数据
    fz_front_car = fz[:, :2].sum(axis=1)

    print(
        "  CarSim tires:",
        round(mean(fz_front_car), 2)
    )

    print(
        "  NMPC model:",
        round(mean(nom["fz_front_n"]), 2)
    )

    # --------------------------------------------------
    # 后轴垂向载荷 Fz
    # --------------------------------------------------

    print("Rear Fz (N):")

    # 将左右后轮的垂向载荷相加
    fz_rear_car = fz[:, 2:].sum(axis=1)

    print(
        "  CarSim tires:",
        round(mean(fz_rear_car), 2)
    )

    print(
        "  NMPC model:",
        round(mean(nom["fz_rear_n"]), 2)
    )
    print(
        "  NMPC model: ",
        round(mean(nom["fz_rear_n"]), 2)
    )

    print("Yaw moment (Nm):")
    print(
        "  CarSim Iz*r_dot equivalent:",
        round(mean(mz_equiv), 2)
    )
    print(
        "  Rotated tire approximation:",
        round(mean(mz_tire_approx), 2)
    )
    print(
        "  NMPC model:              ",
        round(mean(nom["sum_mz_nm"]), 2)
    )

print("\nAudit completed.")


# ======================================================
# 实验 A：仅改变质量，检查 NMPC 名义动力学误差
# 不修改原 MATLAB 模型，不训练神经网络
# ======================================================

from dataclasses import replace
import numpy as np

from wm_recovery.nmpc_reference import (
    NMPCReferenceParams,
    nmpc_accelerations,
)

# 两组物理参数：只允许质量不同
p_original = NMPCReferenceParams()

p_carsim = replace(
    p_original,
    mass_kg=2026.0
)

# 使用之前已经读取的 dlc 和 z
t = dlc.time_s
q = dlc.q_si

# CarSim 真实状态对应的运动学导数
actual = np.column_stack([
    dlc.ay_mps2 - q[:, 2] * q[:, 1],
    np.gradient(q[:, 1], t),
    dlc.ax_mps2 + q[:, 0] * q[:, 1],
])

# 分别计算两种质量对应的名义模型预测
pred_original = nmpc_accelerations(
    z, p_original
)

pred_carsim_mass = nmpc_accelerations(
    z, p_carsim
)

intervals = [
    (3.0, 3.5),
    (3.5, 4.0),
    (4.0, 4.5),
    (4.5, 5.0),
    (5.0, 5.5),
    (5.5, 6.5),
]

print("\n=== Mass sensitivity analysis ===")

for start, stop in intervals:

    mask = (t >= start) & (t < stop)

    # 原质量下的状态导数 RMSE
    rmse_old = np.sqrt(np.mean(
        (pred_original[mask] - actual[mask])**2,
        axis=0
    ))

    # CarSim 质量下的状态导数 RMSE
    rmse_new = np.sqrt(np.mean(
        (pred_carsim_mass[mask] - actual[mask])**2,
        axis=0
    ))

    print(f"\nTime {start:.1f}-{stop:.1f}s")

    for i, name in enumerate(
        ["vy_dot", "r_dot", "vx_dot"]
    ):

        improvement = 100 * (
            rmse_old[i] - rmse_new[i]
        ) / max(rmse_old[i], 1e-12)

        print(
            f"{name}: "
            f"2053.8kg={rmse_old[i]:.6f}, "
            f"2026kg={rmse_new[i]:.6f}, "
            f"change={improvement:+.2f}%"
        )

    # ======================================================
    # 实验 B：检查轮胎侧偏角及潜在滞后
    # 当前 AlphL 为滞后侧偏角，不能直接等同于瞬时角
    # ======================================================

    import numpy as np

    p = NMPCReferenceParams()

    vy = z[:, 0]
    r = z[:, 1]
    vx = z[:, 2]
    delta = z[:, 3]

    vx_safe = np.sqrt(
        vx ** 2 + p.vx_safe_square_m2ps2
    )

    # NMPC 使用的准静态瞬时侧偏角，单位 deg
    alpha_f_nmpc = np.rad2deg(
        delta - (vy + p.lf_m * r) / vx_safe
    )

    alpha_r_nmpc = np.rad2deg(
        (p.lr_m * r - vy) / vx_safe
    )

    # 当前 CarSim 已记录的滞后侧偏角，单位暂按 deg
    alpha_f_lag = 0.5 * (
            dlc.get("AlphL_L1")
            + dlc.get("AlphL_R1")
    )

    alpha_r_lag = 0.5 * (
            dlc.get("AlphL_L2")
            + dlc.get("AlphL_R2")
    )

    intervals = [
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
        (5.0, 5.5),
        (5.5, 6.5),
    ]

    print("\n=== Tire slip angle diagnostic ===")

    for start, stop in intervals:

        mask = (t >= start) & (t < stop)

        print(f"\nTime {start:.1f}-{stop:.1f}s")

        for axle, nominal, lagged in [
            ("Front", alpha_f_nmpc, alpha_f_lag),
            ("Rear", alpha_r_nmpc, alpha_r_lag),
        ]:
            # 仅用于判断符号约定和差异大小
            # 不能将这个 RMSE 视为 NMPC 几何公式误差
            rmse_same = np.sqrt(np.mean(
                (nominal[mask] - lagged[mask]) ** 2
            ))

            rmse_reverse = np.sqrt(np.mean(
                (nominal[mask] + lagged[mask]) ** 2
            ))

            print(f"{axle}:")
            print(
                "  NMPC mean:",
                round(np.mean(nominal[mask]), 4),
                "deg"
            )
            print(
                "  CarSim lagged mean:",
                round(np.mean(lagged[mask]), 4),
                "deg"
            )
            print(
                "  RMSE with same sign:",
                round(rmse_same, 4),
                "deg"
            )
            print(
                "  RMSE with reverse sign:",
                round(rmse_reverse, 4),
                "deg"
            )