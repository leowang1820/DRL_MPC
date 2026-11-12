
"""只读检查 CarSim 生成的平铺配置文件。

本模块只负责读取和检查 CarSim 配置，不会调用 DLL、
修改数据库、修改车辆参数或启动 CarSim 仿真。
"""


from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import re

from dynamics7dof.model import VehicleParameters
from dynamics7dof.signals import _parse_ports, _read_native_text

from .controls import DIRECT_IMPORT_NAMES


# 定义 CarSim 六个 Import 通道及其预期输入模式
# 四个车轮外加转矩使用 Add 模式
# 左右前轮转角使用 Replace 模式
EXPECTED_IMPORT_MODES = {
    "IMP_MYUSM_L1": "ADD",
    "IMP_MYUSM_R1": "ADD",
    "IMP_MYUSM_L2": "ADD",
    "IMP_MYUSM_R2": "ADD",
    "IMP_STEER_L1": "REPLACE",
    "IMP_STEER_R1": "REPLACE",
}


def file_sha256(path):
    """计算文件的 SHA256 指纹，用于记录配置文件版本。"""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_simfile(path):
    """只读检查 CarSim SIM 文件及其关联的参数文件。

    检查内容包括：
    1. SIM 文件与参数文件的关联关系。
    2. Import 和 Export 通道数量。
    3. 六个 Import 通道的名称及排列定义。
    4. 六个 Import 通道的输入模式。
    5. Export 通道是否存在重复名称。
    6. CarSim 配置中的相关仿真参数。
    7. 原论文参考车辆参数。

    本函数不执行 CarSim 仿真，也不修改任何配置文件。
    """

    # 读取 CarSim SIM 文件
    sim = Path(path).expanduser().resolve(strict=True)
    text, encoding = _read_native_text(sim)

    # 初始化宏定义、输入文件和端口数量信息
    macros = {}
    inputs = []
    ports = {}

    # 逐行解析 SIM 文件
    for raw in text.splitlines():

        # 去除注释和首尾空白
        line = raw.split("!", 1)[0].strip()

        if not line or line.startswith("#"):
            continue

        # 读取宏定义
        macro = re.fullmatch(
            r"SET_MACRO\s+\$\(([^)]+)\)\$\s+(.+)",
            line,
            re.I
        )

        if macro:
            macros[macro[1]] = macro[2]
            continue

        # 读取关键字及其对应内容
        parts = line.split(None, 1)

        if len(parts) == 2:
            key = parts[0].upper()
            value = parts[1]

            if key == "INPUT":
                inputs.append(value)



            elif key in ("PORTS_IMP", "PORTS_EXP"):

                # 去除首尾空格及可能存在的分号

                port_value = value.strip().rstrip(";").strip()

                # CarSim 内置驾驶员模式可能不需要外部 Import

                # 例如：PORTS_IMP 0

                # 此时直接记录端口数量为 0

                if port_value == "0":

                    ports[key] = 0


                else:

                    # 非零端口声明继续使用原来的解析函数

                    # 保留原有六通道控制实验的兼容性

                    ports[key] = _parse_ports(value)

    # 检查 SIM 文件是否只有一个参数文件入口
    if len(inputs) != 1:
        raise ValueError(
            "SIM 文件必须包含且只能包含一个指向平铺参数文件的 INPUT"
        )

    # 解析 INPUT 中的宏定义
    target = inputs[0]

    for _ in range(20):

        expanded = re.sub(
            r"\$\(([^)]+)\)\$",
            lambda m: macros.get(m[1], m[0]),
            target
        )

        if expanded == target:
            break

        target = expanded

    # 检查是否存在无法解析的宏定义
    if "$(" in target:
        raise ValueError(
            "INPUT 中存在无法解析的宏定义或循环宏引用"
        )

    # 确定参数文件路径
    par = Path(target.strip('"'))

    par = (
        par if par.is_absolute()
        else sim.parent / par
    ).resolve(strict=True)

    # 读取 CarSim 参数文件
    par_text, par_encoding = _read_native_text(par)

    # 初始化 Import、Export 和仿真参数
    imports = []
    exports = []
    settings = {}

    # 逐行解析参数文件
    for number, raw in enumerate(par_text.splitlines(), 1):

        # 去除注释和首尾空白
        line = raw.split("!", 1)[0].strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        key = parts[0].upper()

        # 检查是否存在未解析的嵌套参数文件
        if key in ("INPUT", "INCLUDE", "PARSFILE") and len(parts) > 1:
            raise ValueError(
                f"第 {number} 行存在嵌套参数文件，需要单独解析"
            )

        # 读取 Import 通道
        elif key == "IMPORT":

            if len(parts) < 4:
                raise ValueError(
                    f"第 {number} 行的 IMPORT 定义不完整"
                )

            imports.append({
                "name": parts[1].upper(),
                "mode": parts[2].upper(),
                "initial": parts[3],
                "line": number,
            })

        # 读取 Export 通道
        elif key == "EXPORT":

            if len(parts) != 2:
                raise ValueError(
                    f"第 {number} 行的 EXPORT 格式不受支持"
                )

            exports.append(parts[1])

        # 读取需要记录的仿真设置
        elif key in (
            "TSTEP",
            "TSTART",
            "TSTOP",
            "MU_ROAD_CONSTANT",
            "MU_ROAD_COMBINE",
            "MU_ROAD_L_CONSTANT",
            "OPT_IO_UPDATE",
            "OPT_IO_SYNC_FM",
            "OPT_INT_METHOD",
        ):

            settings[key] = " ".join(parts[1:]).strip("= ;")

    # 获取实际 Import 通道名称
    names = [item["name"] for item in imports]

    # 初始化检查警告
    warnings = []

    # --------------------------------------------------
    # 一、检查 SIM 文件中的 Import 和 Export 数量
    # --------------------------------------------------

    counts_match = ports == {
        "PORTS_IMP": len(imports),
        "PORTS_EXP": len(exports),
    }

    if not counts_match:
        warnings.append(
            "SIM 文件中的端口数量与参数文件中的通道数量不一致"
        )

    # --------------------------------------------------
    # 二、检查六个 Import 通道的名称和数量
    # --------------------------------------------------

    expected_channels = (
        len(names) == 6
        and set(names) == set(DIRECT_IMPORT_NAMES)
    )

    if not expected_channels:
        warnings.append(
            "CarSim Import 通道不符合四轮转矩和左右前轮转角的六通道定义"
        )

    # --------------------------------------------------
    # 三、检查六个 Import 通道的 Mode
    # --------------------------------------------------

    # 建立实际通道名称与输入模式之间的对应关系
    actual_modes = {
        item["name"]: item["mode"]
        for item in imports
    }

    # 检查每个通道的输入模式是否符合要求
    modes_match = (
        expected_channels
        and all(
            actual_modes.get(name) == expected_mode
            for name, expected_mode in EXPECTED_IMPORT_MODES.items()
        )
    )

    if not modes_match:
        warnings.append(
            "CarSim Import 通道的 Mode 与预期设置不一致："
            "四轮转矩应为 Add，左右前轮转角应为 Replace"
        )

    # --------------------------------------------------
    # 四、检查 Import 通道是否存在重复名称
    # --------------------------------------------------

    duplicate_imports = {
        name: count
        for name, count in Counter(names).items()
        if count > 1
    }

    if duplicate_imports:
        warnings.append(
            "存在重复的 Import 通道名称，请检查参数文件"
        )

    # --------------------------------------------------
    # 五、检查 Export 通道是否存在重复名称
    # --------------------------------------------------

    duplicates = {
        name: count
        for name, count in Counter(
            name.upper() for name in exports
        ).items()
        if count > 1
    }

    if duplicates:
        warnings.append(
            "存在重复的 Export 通道名称，"
            "读取时必须检查通道位置，不能直接转换为名称与数值字典"
        )

    # --------------------------------------------------
    # 六、记录物理接口尚未完成实际验证的状态
    # --------------------------------------------------

    warnings.append(
        "Import 名称和 Mode 检查不能证明左右前轮实际转角相等，"
        "也不能验证转向系统动力学及四轮转矩平衡"
    )

    # --------------------------------------------------
    # 七、生成完整检查报告
    # --------------------------------------------------

    return {
        # 检查报告版本
        "format": "wm_recovery_configuration_inventory_v1",

        # 文件路径
        "sim_path": str(sim),
        "parameter_path": str(par),

        # 文件 SHA256 指纹
        "sim_sha256": file_sha256(sim),
        "parameter_sha256": file_sha256(par),

        # 文件编码
        "encodings": {
            "sim": encoding,
            "parameters": par_encoding,
        },

        # CarSim 端口数量
        "ports": ports,

        # Import 和 Export 通道
        "imports": imports,
        "exports": exports,

        # 重复通道检查
        "duplicate_imports": duplicate_imports,
        "duplicate_exports": duplicates,

        # 仿真设置
        "settings": settings,

        # Import 通道检查结果
        "port_counts_match": counts_match,
        "direct_channel_names_match": expected_channels,
        "direct_channel_modes_match": modes_match,

        # 物理模型参数
        "paper_reference_parameters": asdict(
            VehicleParameters()
        ),

        # 车辆参数一致性尚未验证
        "vehicle_parameter_alignment": "not_verified",

        # 物理接口尚未通过实际仿真验证
        "physical_interface_verified": False,

        # 检查过程中发现的问题
        "warnings": warnings,
    }