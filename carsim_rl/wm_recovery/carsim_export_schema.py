
"""CarSim 通用状态输出接口定义。

统一管理 CarSim 的 61 个 Export 通道。

主要功能：
1. 固定全部 61 个通道的名称与顺序。
2. 保留两个重复的 Steer_SW。
3. 检查 CarSim 输出接口是否发生变化。
4. 支持按照变量名称提取数据。
5. 支持通过 data.Fx_L1 等形式直接访问变量。
6. 保留原有 World Model 数据提取功能。

本文件不调用 CarSim，也不进行单位转换。
"""

from types import SimpleNamespace

import numpy as np


# ============================================================
# 一、CarSim 输出接口版本
# ============================================================

EXPORT_SCHEMA_VERSION = "carsim_export_v1_61"

EXPORT_COUNT = 61


# ============================================================
# 二、CarSim 全部 Export 通道
# ============================================================

# 严格按照当前 CarSim 的 61 个输出通道排列
# 不删除、不合并、不改变任何通道的顺序

EXPORT_NAMES = (
    "Xo",
    "Yo",
    "Vx",
    "Vy",
    "Yaw",
    "AVz",
    "Ax",

    "AlphL_L1",
    "AlphL_R1",
    "AlphL_L2",
    "AlphL_R2",

    "Vx_WC_L1",
    "Vx_WC_R1",
    "Vx_WC_L2",
    "Vx_WC_R2",

    "AVy_L1",
    "AVy_R1",
    "AVy_L2",
    "AVy_R2",

    "Fx_L1",
    "Fx_R1",
    "Fx_L2",
    "Fx_R2",

    "Fy_L1",
    "Fy_R1",
    "Fy_L2",
    "Fy_R2",

    "Fz_L1",
    "Fz_R1",
    "Fz_L2",
    "Fz_R2",

    "KappL_L1",
    "KappL_R1",
    "KappL_L2",
    "KappL_R2",

    "MuX_L1",
    "MuX_R1",
    "MuX_L2",
    "MuX_R2",

    # 第 40 个输出
    "Steer_SW",

    "Beta",
    "Ay",
    "Steer_L1",
    "Steer_R1",

    "My_Dr_L1",
    "My_Dr_R1",
    "My_Dr_L2",
    "My_Dr_R2",

    # 第 49 个输出，保留重复变量
    "Steer_SW",

    "My_US_L1",
    "My_US_R1",
    "My_US_L2",
    "My_US_R2",

    "MyRR_L1",
    "MyRR_R1",
    "MyRR_L2",
    "MyRR_R2",

    "My_WC_L1",
    "My_WC_R1",
    "My_WC_L2",
    "My_WC_R2",
)


# ============================================================
# 三、检查输出通道数量
# ============================================================

if len(EXPORT_NAMES) != EXPORT_COUNT:
    raise ValueError(
        f"CarSim Export 必须包含 {EXPORT_COUNT} 个通道"
    )


# ============================================================
# 四、建立名称与位置的对应关系
# ============================================================

_index_map = {}

for index, name in enumerate(EXPORT_NAMES):

    key = name.upper()

    if key not in _index_map:
        _index_map[key] = []

    _index_map[key].append(index)


EXPORT_INDEX_MAP = {
    name: tuple(indices)
    for name, indices in _index_map.items()
}


# 检查重复变量的位置
if EXPORT_INDEX_MAP.get("STEER_SW") != (39, 48):
    raise ValueError(
        "两个 Steer_SW 的位置与固定接口不一致"
    )


# 检查是否出现其他意外重复的变量
_duplicate_names = {
    name: positions
    for name, positions in EXPORT_INDEX_MAP.items()
    if len(positions) > 1
}

if _duplicate_names != {"STEER_SW": (39, 48)}:
    raise ValueError(
        "CarSim Export 存在非预期的重复通道"
    )


# ============================================================
# 五、定义各算法可能使用的变量组
# ============================================================

# World Model 七维状态
STATE_EXPORT_NAMES = (
    "Vx",
    "Vy",
    "AVz",
    "AVy_L1",
    "AVy_R1",
    "AVy_L2",
    "AVy_R2",
)


# World Model 六维实际输入候选通道
APPLIED_EXPORT_NAMES = (
    "Steer_L1",
    "Steer_R1",
    "My_US_L1",
    "My_US_R1",
    "My_US_L2",
    "My_US_R2",
)


# 车辆加速度
ACCEL_EXPORT_NAMES = (
    "Ax",
    "Ay",
)


# 车辆位置与姿态
POSE_EXPORT_NAMES = (
    "Xo",
    "Yo",
    "Yaw",
)


# 四轮轮胎力
TIRE_FORCE_EXPORT_NAMES = (
    "Fx_L1",
    "Fx_R1",
    "Fx_L2",
    "Fx_R2",
    "Fy_L1",
    "Fy_R1",
    "Fy_L2",
    "Fy_R2",
    "Fz_L1",
    "Fz_R1",
    "Fz_L2",
    "Fz_R2",
)


# ============================================================
# 六、查找指定变量的数组位置
# ============================================================

def get_export_index(name, occurrence=None):
    """根据变量名称查找 CarSim 数组下标。

    对于重复变量，需要指定第几次出现。
    出现次数从 1 开始。
    Python 数组下标从 0 开始。
    """

    key = str(name).upper()

    if key not in EXPORT_INDEX_MAP:
        raise KeyError(
            f"CarSim Export 不存在变量：{name}"
        )

    positions = EXPORT_INDEX_MAP[key]

    if occurrence is None:

        if len(positions) > 1:
            raise ValueError(
                f"变量 {name} 存在重复，"
                "请指定 occurrence"
            )

        return positions[0]

    if (
        not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or not 1 <= occurrence <= len(positions)
    ):
        raise ValueError(
            f"变量 {name} 的出现次数无效"
        )

    return positions[occurrence - 1]


# ============================================================
# 七、检查 CarSim 输出顺序
# ============================================================

def validate_export_order(actual_names):
    """检查 CarSim 实际输出名称与固定顺序是否一致。"""

    actual = tuple(
        str(name).upper()
        for name in actual_names
    )

    expected = tuple(
        name.upper()
        for name in EXPORT_NAMES
    )

    if len(actual) != EXPORT_COUNT:
        raise ValueError(
            f"Export 数量错误：预期 {EXPORT_COUNT}，"
            f"实际 {len(actual)}"
        )

    for index, (a, e) in enumerate(zip(actual, expected)):

        if a != e:
            raise ValueError(
                f"第 {index + 1} 个通道不匹配："
                f"预期 {e}，实际 {a}"
            )

    return True


# ============================================================
# 八、统一检查原始数值数组
# ============================================================

def _validate_raw_exports(raw_exports):
    """检查原始输出是否为有效的 61 通道数值数组。"""

    values = np.asarray(
        raw_exports,
        dtype=np.float64
    )

    if values.ndim < 1 or values.shape[-1] != EXPORT_COUNT:
        raise ValueError(
            "CarSim 原始数据的最后一维必须为 61"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "CarSim 输出数据中存在非有限数值"
        )

    return values


# ============================================================
# 九、按照名称提取指定变量
# ============================================================

def select_exports(raw_exports, names):
    """从原始数组中提取指定名称的多个变量。

    重复变量使用以下形式：
        Steer_SW#1
        Steer_SW#2
    """

    values = _validate_raw_exports(raw_exports)

    if isinstance(names, str):
        names = (names,)

    indices = []

    for selector in names:

        if not isinstance(selector, str):
            raise TypeError("变量名称必须为字符串")

        if "#" in selector:

            name, _, number = selector.rpartition("#")

            if not name or not number.isdigit():
                raise ValueError(
                    f"变量选择标识无效：{selector}"
                )

            index = get_export_index(
                name,
                occurrence=int(number)
            )

        else:
            index = get_export_index(selector)

        indices.append(index)

    return values[..., indices].copy()


# ============================================================
# 十、新增：直接获取单个变量的数值
# ============================================================

def get_export_value(raw_exports, name, occurrence=None):
    """根据变量名称直接返回对应数值。

    单个时刻返回一个浮点数。
    多个时刻返回对应变量的时间序列。
    """

    values = _validate_raw_exports(raw_exports)

    index = get_export_index(
        name,
        occurrence=occurrence
    )

    result = values[..., index]

    if values.ndim == 1:
        return float(result)

    return np.array(result, copy=True)


# ============================================================
# 十一、新增：将输出转换为按名称访问的对象
# ============================================================

def to_named_exports(raw_exports):
    """把 CarSim 输出转换为可以直接按名称访问的对象。

    例如：
        data.Vx
        data.Fx_L1
        data.Steer_L1
        data.My_US_L1

    重复变量使用：
        data.Steer_SW_1
        data.Steer_SW_2

    单个时刻对应单个浮点数。
    多个时刻对应各变量的数值数组。

    本函数不进行单位转换。
    """

    values = _validate_raw_exports(raw_exports)

    # 记录每个变量已经出现的次数
    occurrence_counts = {}

    # 保存变量名称与实际数值
    named_values = {}

    for index, name in enumerate(EXPORT_NAMES):

        key = name.upper()

        # 计算当前变量是第几次出现
        count = occurrence_counts.get(key, 0) + 1

        occurrence_counts[key] = count

        # 对重复变量增加序号，避免覆盖
        if len(EXPORT_INDEX_MAP[key]) > 1:
            field_name = f"{name}_{count}"
        else:
            field_name = name

        # 从完整数组中读取当前变量
        value = values[..., index]

        # 单个时刻返回浮点数
        # 多个时刻返回独立的数值数组
        if values.ndim == 1:
            named_values[field_name] = float(value)
        else:
            named_values[field_name] = np.array(
                value,
                copy=True
            )

    # 创建支持 data.Fx_L1 访问形式的对象
    return SimpleNamespace(**named_values)


# ============================================================
# 十二、保留原有 World Model 数据提取功能
# ============================================================

def extract_world_model_raw(raw_exports):
    """提取七维状态、六维输入候选量和两维加速度。

    保留此函数是为了兼容已有的数据处理代码。
    """

    states = select_exports(
        raw_exports,
        STATE_EXPORT_NAMES
    )

    applied_inputs = select_exports(
        raw_exports,
        APPLIED_EXPORT_NAMES
    )

    accelerations = select_exports(
        raw_exports,
        ACCEL_EXPORT_NAMES
    )

    return states, applied_inputs, accelerations


# ============================================================
# 十三、保留两个重复方向盘转角的提取功能
# ============================================================

def extract_steering_wheel_raw(raw_exports):
    """读取两个独立位置的 Steer_SW 输出。"""

    return select_exports(
        raw_exports,
        (
            "Steer_SW#1",
            "Steer_SW#2",
        )
    )