"""fea 求解器元数据注册表：以 SolverSpec 声明 solve_* 的物理接口.

将 fea 包内散落的 13 个 solve_* 函数以统一元数据形式集中声明：求解器类型 id、
所属学科、输入 bundle 类型、输出解类型、求解控制参数 schema 等。flowchart 层
通过 :func:`build_module_spec` 由 SolverSpec 直接生成 ModuleSpec，消除
``flowchart.nodes`` 中的手写适配层（run_static/run_modal/... 10 个模板函数）。

第三方求解器包可通过 entry point ``zylab.solver`` 注册 SolverSpec（或工厂函数），
加载时自动转换为 flowchart.ModuleSpec 注入流程图类型系统。

本文件只做元数据声明，不导入 flowchart.ParamSpec / PortType 等 flowchart 符号，
避免 fea 反向依赖 flowchart（fea 必须保持 Qt-free + flowchart-free）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable

from .buckling import BucklingSolution, solve_buckling
from .electrothermal import (
    ElectroThermalSolution,
    ElectroThermalTransientSolution,
    solve_electrothermal,
    solve_electrothermal_transient,
)
from .harmonic import HarmonicResponse, solve_harmonic
from .modal import ModalSolution, solve_modal
from .nonlinear import NonlinearSolution, solve_nonlinear_static
from .static import StaticSolution, solve_static
from .transient import TransientSolution, solve_transient

__all__ = [
    "ALL_SOLVERS",
    "ET_SOLVERS",
    "FE_SOLVERS",
    "BundleKind",
    "ParamMeta",
    "PortMeta",
    "SolverCategory",
    "SolverSpec",
    "build_solver_target",
    "module_id_of",
]


@unique
class BundleKind(str, Enum):
    """求解器消费的模型 bundle 种类.

    - ``MODEL``: 结构分析求解器（solve_static/modal/transient/...）消费
      ``ModelBundle``（mesh + LinearElastic + Section + StaticCase）。
    - ``ET_MODEL``: 电/热/电-热耦合求解器消费 ``ConductionBundle``
      （mesh + ConductionMaterial + Section + ElectricCase/ThermalCase）。
    """

    MODEL = "ModelBundle"
    ET_MODEL = "ConductionBundle"


@unique
class SolverCategory(str, Enum):
    """求解器类别（决定 flowchart ModuleCategory 与 Toolbox 分组）."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    STABILITY = "stability"
    NONLINEAR = "nonlinear"
    ELECTRIC = "electric"
    THERMAL = "thermal"
    COUPLING = "coupling"


@dataclass(frozen=True)
class ParamMeta:
    """求解器控制参数元数据（驱动 flowchart.ParamSpec 生成）.

    :param name: 参数名（对应 solve_* 函数的 kwargs 键）。
    :param label: 中文显示名。
    :param kind: 类型（"int" / "float" / "str"）。
    :param default: 默认值（None 从 solve_* 签名反射）。
    :param minimum: 最小允许值。
    :param maximum: 最大允许值。
    :param step: GUI 步进。
    :param unit: 单位标识。
    :param doc: 参数说明。
    """

    name: str
    label: str
    kind: str = "float"  # "int" | "float" | "str"
    default: Any = None
    minimum: float = -1.0e300
    maximum: float = 1.0e300
    step: float = 0.1
    unit: str = ""
    doc: str = ""


@dataclass(frozen=True)
class PortMeta:
    """额外端口元数据（可选输入端口，如屈曲的 reference=StaticSolution）.

    :param name: 端口名。
    :param port_type: PortType 字符串（"static" / "modal" / ...）。
    :param label: 中文显示名。
    :param required: 是否必须连接。
    """

    name: str
    port_type: str
    label: str = ""
    required: bool = False


@dataclass(frozen=True)
class SolverSpec:
    """单个求解器的物理接口声明（fea 层原生 solve_* 的元数据包装）.

    :param type_id: flowchart 模块类型 id（如 ``"fea.static"``）。
    :param name: 中文显示名（如 ``"静力分析"``）。
    :param category: 求解器类别。
    :param solve_fn: fea 层求解函数（直接引用，不是字符串 target）。
    :param bundle: 需要的模型 bundle 种类。
    :param output_port_type: 输出解的 PortType 字符串（如 ``"static"`` / ``"modal"``）。
    :param result_cls: 输出解的 Python 类（用于类型断言）。
    :param params: 求解控制参数表（空元组表示无额外参数，如 solve_static）。
    :param extra_inputs: 可选额外输入端口（如 buckling 的 reference=StaticSolution）。
    :param arg_builder: 自定义位置参数构造器 ``(bundle, params, extras) -> tuple``，
        覆盖默认拆分逻辑。用于 modal（要 constraints 而非 case）、harmonic（要生成 frequencies）、
        ET 求解器（要双 case）等签名特殊场景。
    """

    type_id: str
    name: str
    category: SolverCategory
    solve_fn: Callable[..., Any]
    bundle: BundleKind
    output_port_type: str
    result_cls: type
    params: tuple[ParamMeta, ...] = ()
    extra_inputs: tuple[PortMeta, ...] = ()
    arg_builder: Callable[..., tuple] | None = None  # (bundle, params, extras) -> tuple 位置参数
    kwarg_renames: tuple[tuple[str, str], ...] = ()  # params 键 → solve_* kwargs 键（如 ("t_init","initial")）

    @property
    def target(self) -> str:
        """可 pickle 的 target 字符串（指向 fea 层函数，跳过 nodes.py 手写层）."""
        return build_solver_target(self.solve_fn)


# ------------------------------------------------------------------ 反射工具


def build_solver_target(fn: Callable[..., Any]) -> str:
    """由 solve_* 函数对象生成 target 字符串.

    格式 ``"zylab.fea.<module>:<func_name>"``，供 runner 跨进程导入执行。
    """
    module = getattr(fn, "__module__", "")
    if module.startswith("zylab.fea."):
        leaf = module.split(".")[-1]
        return f"zylab.fea.{leaf}:{fn.__name__}"
    # 外部插件：完整模块路径
    return f"{module}:{fn.__name__}"


def module_id_of(spec: SolverSpec) -> str:
    """取 flowchart ModuleCategory 对应模块 id 前缀."""
    return spec.type_id


# ------------------------------------------------------------------ 结构求解器


def _model_static_args(bundle: Any, _params: Any, _extras: Any) -> tuple:
    """默认 ModelBundle 拆包：(mesh, materials, sections, case) —— solve_static/transient/buckling/nonlinear 通用."""
    return (bundle.mesh, bundle.materials, bundle.sections, bundle.case)


def _model_modal_args(bundle: Any, _params: Any, _extras: Any) -> tuple:
    """模态分析特殊拆包：constraints = case.constraints 而非 case 本身."""
    return (bundle.mesh, bundle.materials, bundle.sections, bundle.case.constraints)


def _model_harmonic_args(bundle: Any, params: Any, _extras: Any) -> tuple:
    """谐响应特殊拆包：额外生成频率扫描序列.

    消费 params 中的 f_max / n_freq（pop 掉，避免 make_run_fn 重复传入）.
    """
    import numpy as np

    f_max = float(params.pop("f_max", 3.0))
    n_freq = int(params.pop("n_freq", 60))
    frequencies = np.linspace(0.0, f_max, n_freq)
    return (bundle.mesh, bundle.materials, bundle.sections, bundle.case, frequencies)


def _et_electrothermal_args(bundle: Any, _params: Any, _extras: Any) -> tuple:
    """电-热耦合双 case 拆包."""
    return (bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case, bundle.thermal_case)


def _et_electrothermal_transient_args(bundle: Any, params: Any, _extras: Any) -> tuple:
    """瞬态电-热耦合：返回 (位置参数元组, kwargs字典).

    initial 需要从 mesh.n_nodes + t_init 生成数组，由本函数直接构建。
    """
    import numpy as np

    t_init = float(params.pop("t_init", 20.0))
    duration = float(params.pop("duration", 1.0))
    n_steps = int(params.pop("n_steps", 50))
    initial = np.full(bundle.mesh.n_nodes, t_init)
    args = (bundle.mesh, bundle.materials, bundle.sections, bundle.electric_case, bundle.thermal_case)
    kwargs = {"initial": initial, "total_time": duration, "n_steps": n_steps}
    return (args, kwargs)


FE_SOLVERS: tuple[SolverSpec, ...] = (
    SolverSpec(
        type_id="fea.static",
        name="静力分析",
        category=SolverCategory.STATIC,
        solve_fn=solve_static,
        bundle=BundleKind.MODEL,
        output_port_type="static",
        result_cls=StaticSolution,
        arg_builder=_model_static_args,
    ),
    SolverSpec(
        type_id="fea.modal",
        name="模态分析",
        category=SolverCategory.DYNAMIC,
        solve_fn=solve_modal,
        bundle=BundleKind.MODEL,
        output_port_type="modal",
        result_cls=ModalSolution,
        params=(ParamMeta("n_modes", "模态阶数", "int", default=6, minimum=1, maximum=50, step=1),),
        arg_builder=_model_modal_args,
    ),
    SolverSpec(
        type_id="fea.harmonic",
        name="谐响应分析",
        category=SolverCategory.DYNAMIC,
        solve_fn=solve_harmonic,
        bundle=BundleKind.MODEL,
        output_port_type="harmonic",
        result_cls=HarmonicResponse,
        params=(
            ParamMeta(
                "f_max", "扫频上限 ω", "float", default=3.0, minimum=1.0e-6, maximum=1.0e6, step=0.5, unit="rad/s"
            ),
            ParamMeta("n_freq", "扫频点数", "int", default=60, minimum=10, maximum=2000, step=10),
            ParamMeta(
                "alpha",
                "阻尼 α",
                "float",
                default=0.1,
                minimum=0.0,
                maximum=1.0e6,
                step=0.05,
                doc="Rayleigh 质量比例系数",
            ),
            ParamMeta(
                "beta",
                "阻尼 β",
                "float",
                default=0.0,
                minimum=0.0,
                maximum=1.0e3,
                step=0.01,
                doc="Rayleigh 刚度比例系数",
            ),
        ),
        arg_builder=_model_harmonic_args,
    ),
    SolverSpec(
        type_id="fea.transient",
        name="瞬态动力分析",
        category=SolverCategory.DYNAMIC,
        solve_fn=solve_transient,
        bundle=BundleKind.MODEL,
        output_port_type="transient",
        result_cls=TransientSolution,
        params=(
            ParamMeta("duration", "总时长", "float", default=10.0, minimum=1.0e-9, maximum=1.0e6, step=1.0, unit="s"),
            ParamMeta("n_steps", "积分步数", "int", default=200, minimum=1, maximum=20000, step=50),
            ParamMeta("alpha", "阻尼 α", "float", default=0.0, minimum=0.0, maximum=1.0e6, step=0.05),
            ParamMeta("beta", "阻尼 β", "float", default=0.0, minimum=0.0, maximum=1.0e3, step=0.01),
        ),
        arg_builder=_model_static_args,
    ),
    SolverSpec(
        type_id="fea.buckling",
        name="屈曲分析",
        category=SolverCategory.STABILITY,
        solve_fn=solve_buckling,
        bundle=BundleKind.MODEL,
        output_port_type="buckling",
        result_cls=BucklingSolution,
        params=(ParamMeta("n_modes", "模态阶数", "int", default=5, minimum=1, maximum=50, step=1),),
        extra_inputs=(PortMeta("reference", "static", "参考静力", required=False),),
        arg_builder=_model_static_args,
    ),
    SolverSpec(
        type_id="fea.nonlinear",
        name="几何非线性分析",
        category=SolverCategory.NONLINEAR,
        solve_fn=solve_nonlinear_static,
        bundle=BundleKind.MODEL,
        output_port_type="nonlinear",
        result_cls=NonlinearSolution,
        params=(
            ParamMeta("n_increments", "增量步数", "int", default=10, minimum=1, maximum=100, step=5),
            ParamMeta("tolerance", "收敛容差", "float", default=1.0e-8, minimum=1.0e-14, maximum=1.0e-2, step=1.0e-8),
            ParamMeta("max_iterations", "单步迭代上限", "int", default=30, minimum=5, maximum=500, step=5),
        ),
        extra_inputs=(PortMeta("initial", "static", "初态静力", required=False),),
        arg_builder=_model_static_args,
    ),
)


# ------------------------------------------------------------------ 电-热求解器


ET_SOLVERS: tuple[SolverSpec, ...] = (
    SolverSpec(
        type_id="fea.electrothermal",
        name="电-热耦合分析",
        category=SolverCategory.COUPLING,
        solve_fn=solve_electrothermal,
        bundle=BundleKind.ET_MODEL,
        output_port_type="electrothermal",
        result_cls=ElectroThermalSolution,
        arg_builder=_et_electrothermal_args,
    ),
    SolverSpec(
        type_id="fea.electrothermal_transient",
        name="瞬态电-热耦合分析",
        category=SolverCategory.COUPLING,
        solve_fn=solve_electrothermal_transient,
        bundle=BundleKind.ET_MODEL,
        output_port_type="et_transient",
        result_cls=ElectroThermalTransientSolution,
        params=(
            ParamMeta("t_init", "初始温度", "float", default=20.0, minimum=-1.0e4, maximum=1.0e4, step=1.0),
            ParamMeta("duration", "总时长", "float", default=1.0, minimum=1.0e-9, maximum=1.0e6, step=1.0, unit="s"),
            ParamMeta("n_steps", "积分步数", "int", default=50, minimum=1, maximum=10000, step=10),
        ),
        arg_builder=_et_electrothermal_transient_args,
    ),
)


ALL_SOLVERS: tuple[SolverSpec, ...] = (*FE_SOLVERS, *ET_SOLVERS)
