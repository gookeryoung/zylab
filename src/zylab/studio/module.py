"""工作流模块类型系统：端口类型、参数规格、模块规格与内置模块表.

模块类型（:class:`ModuleSpec`）是模板节点的「型」：声明输入/输出端口（类型化）、
参数 schema（:class:`ParamSpec`，驱动 GUI 表单自动生成与取值校验）以及进程执行
目标（``target`` 为 ``"module:func"`` 全限定名，与 core.executor.TaskSpec 对齐）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Mapping

from .errors import ModuleNotFoundError_, ParamError

__all__ = [
    "BUILTIN_MODULES",
    "ModuleCategory",
    "ModuleSpec",
    "ParamSpec",
    "ParamType",
    "PortSpec",
    "PortType",
    "all_modules",
    "module_spec",
]

#: 参数数值范围的默认端点（JSON 无 inf 表示，内部采用大数哨兵）
_INF = 1.0e300


@unique
class PortType(Enum):
    """端口载荷类型（连接校验依据：仅同类型端口可相连）."""

    MODEL = "model"  # bundle.ModelBundle（网格+材料+截面+工况）
    STATIC = "static"  # fea.StaticSolution
    MODAL = "modal"  # fea.ModalSolution
    HARMONIC = "harmonic"  # fea.HarmonicResponse
    TRANSIENT = "transient"  # fea.TransientSolution
    BUCKLING = "buckling"  # fea.BucklingSolution
    NONLINEAR = "nonlinear"  # fea.NonlinearSolution
    ET_MODEL = "et_model"  # bundle.ConductionBundle（电-热传导模型）
    ELECTROTHERMAL = "electrothermal"  # fea.ElectroThermalSolution


@unique
class ParamType(Enum):
    """参数控件类型（决定 GUI 生成的输入控件与取值收敛规则）."""

    FLOAT = "float"
    INT = "int"


@unique
class ModuleCategory(Enum):
    """模块类别（决定节点在工作流图中的角色）."""

    SOURCE = "source"  # 无输入端口，产出 MODEL
    ANALYSIS = "analysis"  # MODEL -> 分析解
    POST = "post"  # 分析解 -> 展示数据（R4 接入渲染）


@dataclass(frozen=True)
class ParamSpec:
    """单个参数的 UI 呈现与校验规格.

    :param key: 参数键（节点函数 params 字典的键，模板中以 ``"node_id.key"`` 引用）。
    :param label: 中文显示名（GUI 表单行标签）。
    :param param_type: 控件类型（浮点/整数输入框）。
    :param default: 默认值。
    :param minimum: 最小允许值（含端点）。
    :param maximum: 最大允许值（含端点）。
    :param step: GUI 输入框单步步长。
    :param unit: 单位提示（追加在标签后，如 ``"MPa"``）。
    :param doc: 参数说明（GUI tooltip）。
    """

    key: str
    label: str
    param_type: ParamType
    default: float | int
    minimum: float = -_INF
    maximum: float = _INF
    step: float = 0.1
    unit: str = ""
    doc: str = ""

    def coerce(self, value: object) -> float | int:
        """校验并收敛参数值；类型不符或越界抛 :class:`ParamError`."""
        if self.param_type is ParamType.INT:
            result: float | int = self._coerce_int(value)
        else:
            result = self._coerce_float(value)
        if not self.minimum <= float(result) <= self.maximum:
            raise ParamError(f"参数 {self.key!r} 取值 {result} 越界（允许 [{self.minimum}, {self.maximum}]）")
        return result

    def _coerce_float(self, value: object) -> float:
        """收敛为浮点数（拒绝布尔与不可转换值）."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"参数 {self.key!r} 应为数值，得到 {value!r}")
        return float(value)

    def _coerce_int(self, value: object) -> int:
        """收敛为整数（拒绝布尔与非整值浮点）."""
        if isinstance(value, bool):
            raise ParamError(f"参数 {self.key!r} 应为整数，得到 {value!r}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ParamError(f"参数 {self.key!r} 应为整数，得到 {value!r}")
            return int(value)
        if isinstance(value, int):
            return value
        raise ParamError(f"参数 {self.key!r} 应为整数，得到 {value!r}")


@dataclass(frozen=True)
class PortSpec:
    """端口规格.

    :param name: 端口名（模板 inputs 中以 ``"node_id.port_name"`` 引用）。
    :param port_type: 载荷类型。
    :param label: 中文显示名。
    :param required: 是否必须连接（可选端口如屈曲的预应力参考，缺省不连即可运行）。
    """

    name: str
    port_type: PortType
    label: str = ""
    required: bool = True


@dataclass(frozen=True)
class ModuleSpec:
    """模块类型描述（模板经 ``type_id`` 引用本规格）.

    :param type_id: 全局唯一类型 id（如 ``"example.cantilever_q4"``）。
    :param name: 中文显示名。
    :param category: 模块类别。
    :param target: 节点执行函数全限定名（``"zylab.studio.nodes:xxx"``）。
    :param inputs: 输入端口表。
    :param outputs: 输出端口表。
    :param params: 参数 schema 表。
    """

    type_id: str
    name: str
    category: ModuleCategory
    target: str
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    params: tuple[ParamSpec, ...] = ()

    def param(self, key: str) -> ParamSpec:
        """按 key 取参数规格；未知键抛 :class:`ParamError`."""
        for spec in self.params:
            if spec.key == key:
                return spec
        raise ParamError(f"模块 {self.type_id!r} 无参数 {key!r}")

    def defaults(self) -> dict[str, float | int]:
        """全部参数的默认值表."""
        return {spec.key: spec.default for spec in self.params}

    def coerce_params(self, params: Mapping[str, Any]) -> dict[str, float | int]:
        """合并默认值并逐项校验收敛；未知键抛 :class:`ParamError`."""
        result = self.defaults()
        for key, value in params.items():
            result[key] = self.param(key).coerce(value)  # param() 拦未知键
        # 默认值亦须合法（防 schema 自身越界）
        for key, value in result.items():
            if key not in params:
                result[key] = self.param(key).coerce(value)
        return result

    def input_port(self, name: str) -> PortSpec:
        """按名取输入端口；不存在抛 :class:`ModuleNotFoundError_`."""
        for port in self.inputs:
            if port.name == name:
                return port
        raise ModuleNotFoundError_(f"模块 {self.type_id!r} 无输入端口 {name!r}")

    def output_port(self, name: str) -> PortSpec:
        """按名取输出端口；不存在抛 :class:`ModuleNotFoundError_`."""
        for port in self.outputs:
            if port.name == name:
                return port
        raise ModuleNotFoundError_(f"模块 {self.type_id!r} 无输出端口 {name!r}")


# ------------------------------------------------------------------ 内置模块表

_MATERIAL_PARAMS = (
    ParamSpec("e_modulus", "弹性模量 E", ParamType.FLOAT, 2.1e5, 1.0e3, 1.0e12, 1.0e5, "MPa"),
    ParamSpec("density", "密度 ρ", ParamType.FLOAT, 7.85, 1.0e-9, 1.0e5, 0.1, "t/mm³", "动力学分析（模态/谐响应）必需"),
)

BUILTIN_MODULES: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        type_id="example.cantilever_q4",
        name="悬臂梁（Q4 平面应力）",
        category=ModuleCategory.SOURCE,
        target="zylab.studio.nodes:build_cantilever",
        outputs=(PortSpec("model", PortType.MODEL, "模型"),),
        params=(
            ParamSpec("length", "长度 L", ParamType.FLOAT, 40.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("height", "高度 H", ParamType.FLOAT, 8.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("nx", "纵向单元数", ParamType.INT, 40, 1, 400, 5),
            ParamSpec("ny", "横向单元数", ParamType.INT, 8, 1, 400, 1),
            ParamSpec("tip_load", "端部载荷", ParamType.FLOAT, -100.0, -1.0e9, 1.0e9, 10.0, "N"),
            *_MATERIAL_PARAMS,
            ParamSpec("poisson", "泊松比 ν", ParamType.FLOAT, 0.3, 0.0, 0.49, 0.05),
            ParamSpec("thickness", "厚度 t", ParamType.FLOAT, 1.0, 0.01, 100.0, 0.1, "mm"),
        ),
    ),
    ModuleSpec(
        type_id="example.column_beam2",
        name="悬臂柱（BEAM2）",
        category=ModuleCategory.SOURCE,
        target="zylab.studio.nodes:build_column",
        outputs=(PortSpec("model", PortType.MODEL, "模型"),),
        params=(
            ParamSpec("height", "柱高 H", ParamType.FLOAT, 10.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("n_elem", "单元数", ParamType.INT, 20, 1, 500, 5),
            ParamSpec("tip_load", "顶部轴力", ParamType.FLOAT, -1.0, -1.0e9, 1.0e9, 0.5, "N"),
            *_MATERIAL_PARAMS,
            ParamSpec("area", "截面积 A", ParamType.FLOAT, 1.0, 1.0e-6, 1.0e6, 0.1, "mm²"),
            ParamSpec("inertia", "惯性矩 I", ParamType.FLOAT, 1.0e-4, 1.0e-12, 1.0e6, 1.0e-4, "mm⁴"),
        ),
    ),
    ModuleSpec(
        type_id="example.truss2_two_bar",
        name="两杆浅桁架（TRUSS2）",
        category=ModuleCategory.SOURCE,
        target="zylab.studio.nodes:build_truss",
        outputs=(PortSpec("model", PortType.MODEL, "模型"),),
        params=(
            ParamSpec("half_span", "半跨 b", ParamType.FLOAT, 5.0, 0.1, 1.0e4, 0.5, "mm"),
            ParamSpec("rise", "矢高 h", ParamType.FLOAT, 0.5, 0.01, 1.0e3, 0.1, "mm"),
            ParamSpec("apex_load", "顶点载荷", ParamType.FLOAT, -60.0, -1.0e9, 1.0e9, 5.0, "N"),
            *_MATERIAL_PARAMS,
            ParamSpec("area", "截面积 A", ParamType.FLOAT, 1.0, 1.0e-6, 1.0e6, 0.1, "mm²"),
        ),
    ),
    ModuleSpec(
        type_id="analysis.static",
        name="静力分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_static",
        inputs=(PortSpec("model", PortType.MODEL, "模型"),),
        outputs=(PortSpec("solution", PortType.STATIC, "静力解"),),
    ),
    ModuleSpec(
        type_id="analysis.modal",
        name="模态分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_modal",
        inputs=(PortSpec("model", PortType.MODEL, "模型"),),
        outputs=(PortSpec("solution", PortType.MODAL, "模态解"),),
        params=(ParamSpec("n_modes", "模态阶数", ParamType.INT, 6, 1, 50, 1),),
    ),
    ModuleSpec(
        type_id="analysis.harmonic",
        name="谐响应分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_harmonic",
        inputs=(PortSpec("model", PortType.MODEL, "模型"),),
        outputs=(PortSpec("solution", PortType.HARMONIC, "频响解"),),
        params=(
            ParamSpec("f_max", "扫频上限 ω", ParamType.FLOAT, 3.0, 1.0e-6, 1.0e6, 0.5, "rad/s"),
            ParamSpec("n_freq", "扫频点数", ParamType.INT, 60, 10, 2000, 10),
            ParamSpec("alpha", "阻尼 α", ParamType.FLOAT, 0.1, 0.0, 1.0e6, 0.05, "", "Rayleigh 质量比例系数"),
            ParamSpec("beta", "阻尼 β", ParamType.FLOAT, 0.0, 0.0, 1.0e3, 0.01, "", "Rayleigh 刚度比例系数"),
        ),
    ),
    ModuleSpec(
        type_id="analysis.transient",
        name="瞬态动力分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_transient",
        inputs=(PortSpec("model", PortType.MODEL, "模型"),),
        outputs=(PortSpec("solution", PortType.TRANSIENT, "瞬态解"),),
        params=(
            ParamSpec("duration", "总时长", ParamType.FLOAT, 10.0, 1.0e-9, 1.0e6, 1.0, "s", "载荷时程总时长"),
            ParamSpec("n_steps", "积分步数", ParamType.INT, 200, 1, 20000, 50),
            ParamSpec("alpha", "阻尼 α", ParamType.FLOAT, 0.0, 0.0, 1.0e6, 0.05, "", "Rayleigh 质量比例系数"),
            ParamSpec("beta", "阻尼 β", ParamType.FLOAT, 0.0, 0.0, 1.0e3, 0.01, "", "Rayleigh 刚度比例系数"),
        ),
    ),
    ModuleSpec(
        type_id="analysis.buckling",
        name="屈曲分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_buckling",
        inputs=(
            PortSpec("model", PortType.MODEL, "模型"),
            PortSpec("reference", PortType.STATIC, "参考静力", required=False),
        ),
        outputs=(PortSpec("solution", PortType.BUCKLING, "屈曲解"),),
        params=(ParamSpec("n_modes", "模态阶数", ParamType.INT, 5, 1, 50, 1),),
    ),
    ModuleSpec(
        type_id="example.joule_plate_2d",
        name="通电加热板（Q4 电-热耦合）",
        category=ModuleCategory.SOURCE,
        target="zylab.studio.nodes:build_joule_plate",
        outputs=(PortSpec("model", PortType.ET_MODEL, "传导模型"),),
        params=(
            ParamSpec("length", "长度 L", ParamType.FLOAT, 40.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("height", "高度 H", ParamType.FLOAT, 10.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("nx", "纵向单元数", ParamType.INT, 40, 1, 400, 5),
            ParamSpec("ny", "横向单元数", ParamType.INT, 10, 1, 400, 1),
            ParamSpec(
                "voltage", "电极电压 V₀", ParamType.FLOAT, 1.0, -1.0e6, 1.0e6, 0.1, "V", "右端电极电压，左端接地 0"
            ),
            ParamSpec("electric_sigma", "电导率 σ", ParamType.FLOAT, 1.0, 1.0e-9, 1.0e9, 0.1, "S/mm", "稳态电传导系数"),
            ParamSpec("thermal_k", "导热系数 k", ParamType.FLOAT, 1.0, 1.0e-9, 1.0e6, 0.1, "W/mm·K", "稳态热传导系数"),
            ParamSpec("thickness", "厚度 t", ParamType.FLOAT, 1.0, 0.01, 100.0, 0.1, "mm"),
            ParamSpec("t_base", "底边温度", ParamType.FLOAT, 20.0, -1.0e4, 1.0e4, 1.0, "", "底边恒温边界"),
            ParamSpec(
                "h_conv",
                "对流系数 h",
                ParamType.FLOAT,
                1.0e-5,
                1.0e-9,
                1.0,
                1.0e-6,
                "W/mm²·K",
                "其余三边与环境对流换热",
            ),
            ParamSpec("t_ambient", "环境温度", ParamType.FLOAT, 20.0, -1.0e4, 1.0e4, 1.0, ""),
        ),
    ),
    ModuleSpec(
        type_id="example.joule_series_2d",
        name="多材料串联电加热板（Q4 电-热耦合）",
        category=ModuleCategory.SOURCE,
        target="zylab.studio.nodes:build_joule_series",
        outputs=(PortSpec("model", PortType.ET_MODEL, "传导模型"),),
        params=(
            ParamSpec("length", "板总长 L", ParamType.FLOAT, 30.0, 0.3, 1.0e4, 1.0, "mm"),
            ParamSpec("height", "板高 H", ParamType.FLOAT, 10.0, 0.1, 1.0e4, 1.0, "mm"),
            ParamSpec("nx", "纵向单元数", ParamType.INT, 30, 3, 400, 5),
            ParamSpec("ny", "横向单元数", ParamType.INT, 8, 1, 400, 1),
            ParamSpec(
                "electrode_len",
                "电极段长 a",
                ParamType.FLOAT,
                5.0,
                0.1,
                5.0e3,
                1.0,
                "mm",
                "左右电极段长（各一段，中间为电阻区）",
            ),
            ParamSpec(
                "sigma_conductor", "电极电导率 σ_c", ParamType.FLOAT, 50.0, 1.0e-9, 1.0e9, 0.1, "S/mm", "电极区材料"
            ),
            ParamSpec(
                "sigma_resistor",
                "电阻区电导率 σ_h",
                ParamType.FLOAT,
                0.5,
                1.0e-9,
                1.0e9,
                0.1,
                "S/mm",
                "电阻区材料（热点源）",
            ),
            ParamSpec("k_conductor", "电极导热系数 k_c", ParamType.FLOAT, 0.4, 1.0e-9, 1.0e6, 0.01, "W/mm·K"),
            ParamSpec("k_resistor", "电阻区导热系数 k_h", ParamType.FLOAT, 0.015, 1.0e-9, 1.0e6, 0.001, "W/mm·K"),
            ParamSpec(
                "voltage", "电极电压 V₀", ParamType.FLOAT, 1.0, -1.0e6, 1.0e6, 0.1, "V", "右端电极电压，左端接地 0"
            ),
            ParamSpec("thickness", "厚度 t", ParamType.FLOAT, 1.0, 0.01, 100.0, 0.1, "mm"),
            ParamSpec("t_base", "底边温度", ParamType.FLOAT, 20.0, -1.0e4, 1.0e4, 1.0, "", "底边恒温边界"),
            ParamSpec(
                "h_conv",
                "对流系数 h",
                ParamType.FLOAT,
                1.0e-4,
                1.0e-9,
                1.0,
                1.0e-6,
                "W/mm²·K",
                "其余三边与环境对流换热",
            ),
            ParamSpec("t_ambient", "环境温度", ParamType.FLOAT, 20.0, -1.0e4, 1.0e4, 1.0, ""),
        ),
    ),
    ModuleSpec(
        type_id="analysis.electrothermal",
        name="电-热耦合分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_electrothermal",
        inputs=(PortSpec("model", PortType.ET_MODEL, "传导模型"),),
        outputs=(PortSpec("solution", PortType.ELECTROTHERMAL, "电热耦合解"),),
    ),
    ModuleSpec(
        type_id="analysis.nonlinear",
        name="几何非线性分析",
        category=ModuleCategory.ANALYSIS,
        target="zylab.studio.nodes:run_nonlinear",
        inputs=(
            PortSpec("model", PortType.MODEL, "模型"),
            PortSpec("initial", PortType.STATIC, "初态静力", required=False),
        ),
        outputs=(PortSpec("solution", PortType.NONLINEAR, "非线性解"),),
        params=(
            ParamSpec("n_increments", "增量步数", ParamType.INT, 10, 1, 100, 5),
            ParamSpec("tolerance", "收敛容差", ParamType.FLOAT, 1.0e-8, 1.0e-14, 1.0e-2, 1.0e-8),
            ParamSpec("max_iterations", "单步迭代上限", ParamType.INT, 30, 5, 500, 5),
        ),
    ),
)

_MODULES_BY_ID: dict[str, ModuleSpec] = {spec.type_id: spec for spec in BUILTIN_MODULES}


def module_spec(type_id: str) -> ModuleSpec:
    """按类型 id 取模块规格；未注册抛 :class:`ModuleNotFoundError_`."""
    try:
        return _MODULES_BY_ID[type_id]
    except KeyError:
        raise ModuleNotFoundError_(f"未知模块类型: {type_id!r}") from None


def all_modules() -> tuple[ModuleSpec, ...]:
    """返回全部内置模块规格（定义序）."""
    return BUILTIN_MODULES
