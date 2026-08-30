"""工作流节点执行函数（模块级定义，供 ProcessExecutor 按 target 全限定名定位）.

节点协议::

    def node_fn(inputs, params, report=None) -> output

- ``inputs``：入端口名 -> 上游节点输出对象（源节点为空 Mapping）；
- ``params``：参数键 -> 取值（经 :meth:`ModuleSpec.coerce_params` 合并默认值并校验）；
- ``report``：进度回调 ``(progress, message)``，ProcessExecutor 按签名自动注入；
- 返回值为节点输出端口载荷（可 pickle，跨进程回传）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from zylab.fea import (
    BucklingSolution,
    Constraint,
    ElementBlock,
    ElementType,
    HarmonicResponse,
    LinearElastic,
    Mesh,
    ModalSolution,
    NodalLoad,
    NonlinearSolution,
    Section,
    StaticCase,
    StaticSolution,
    TransientSolution,
    solve_buckling,
    solve_harmonic,
    solve_modal,
    solve_nonlinear_static,
    solve_static,
    solve_transient,
)

from .bundle import ModelBundle
from .module import module_spec

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "build_cantilever",
    "build_column",
    "build_truss",
    "run_buckling",
    "run_harmonic",
    "run_modal",
    "run_nonlinear",
    "run_static",
    "run_transient",
]

#: 节点进度回调签名（与 core.executor 注入的 report 对齐）
ReportFn = Callable[[float, str], None]

#: 节点输入表（源节点为空 Mapping）
NodeInputs = Mapping[str, Any]

#: 节点参数表（键为参数 key，值为 coerce 后的数值）
NodeParams = Mapping[str, Any]


def _params(type_id: str, params: NodeParams) -> dict[str, float | int]:
    """按模块规格合并默认值并校验收敛参数."""
    return module_spec(type_id).coerce_params(params)


def _report(report: ReportFn | None, progress: float, message: str) -> None:
    """空安全地上报进度."""
    if report is not None:
        report(progress, message)


def _model_of(inputs: NodeInputs) -> ModelBundle:
    """取上游模型端口载荷."""
    model = inputs["model"]
    if not isinstance(model, ModelBundle):
        raise TypeError(f"输入端口 'model' 应为 ModelBundle，得到 {type(model).__name__}")
    return model


def _optional_static(inputs: NodeInputs, port: str) -> StaticSolution | None:
    """取可选静力解端口载荷（未连接返回 None）."""
    value = inputs.get(port)
    if value is None:
        return None
    if not isinstance(value, StaticSolution):
        raise TypeError(f"输入端口 {port!r} 应为 StaticSolution，得到 {type(value).__name__}")
    return value


def _material(p: Mapping[str, float | int]) -> LinearElastic:
    """由参数表构建线弹性材料（泊松比缺失时按杆系模型置 0）."""
    return LinearElastic(
        e_modulus=float(p["e_modulus"]),
        poisson=float(p.get("poisson", 0.0)),
        density=float(p["density"]),
    )


# ------------------------------------------------------------------ 源节点


def build_cantilever(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ModelBundle:
    """构建悬臂梁 Q4 平面应力模型（单元正方形，长宽比 1 以规避剪切自锁）."""
    del inputs  # 源节点无输入
    p = _params("example.cantilever_q4", params)
    length, height = float(p["length"]), float(p["height"])
    nx, ny = int(p["nx"]), int(p["ny"])

    _report(report, 0.2, "生成悬臂梁网格")
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            conn.append((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
    block = ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn), name="梁")
    mesh = Mesh(coords=coords, blocks=(block,))

    _report(report, 0.6, "生成边界与载荷")
    n_nodes = mesh.n_nodes
    fixed = tuple(Constraint(node=n, dofs=(0, 1)) for n in range(ny + 1))
    tip = tuple(NodalLoad(node=n, forces=(0.0, float(p["tip_load"]))) for n in range(n_nodes - (ny + 1), n_nodes))
    case = StaticCase(constraints=fixed, loads=tip)
    bundle = ModelBundle(
        mesh=mesh,
        materials=(_material(p),),
        sections=(Section(thickness=float(p["thickness"])),),
        case=case,
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def build_column(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ModelBundle:
    """构建竖直悬臂柱 BEAM2 模型（底部固支，顶部压缩轴力，屈曲基准）."""
    del inputs
    p = _params("example.column_beam2", params)
    height, n_elem = float(p["height"]), int(p["n_elem"])

    _report(report, 0.2, "生成悬臂柱网格")
    coords = np.array([[0.0, height * i / n_elem] for i in range(n_elem + 1)])
    conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=np.int64)
    block = ElementBlock(etype=ElementType.BEAM2, conn=conn, name="柱")
    mesh = Mesh(coords=coords, blocks=(block,))

    top = mesh.n_nodes - 1
    case = StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1, 2)),),
        loads=(NodalLoad(node=top, forces=(0.0, float(p["tip_load"]), 0.0)),),
    )
    bundle = ModelBundle(
        mesh=mesh,
        materials=(_material(p),),
        sections=(Section(area=float(p["area"]), inertia=float(p["inertia"])),),
        case=case,
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def build_truss(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ModelBundle:
    """构建两杆浅桁架 TRUSS2 模型（两支座固支，顶点竖向集中力，几何非线性经典算例）."""
    del inputs
    p = _params("example.truss2_two_bar", params)
    half_span, rise = float(p["half_span"]), float(p["rise"])

    _report(report, 0.2, "生成桁架网格")
    coords = np.array([[-half_span, 0.0], [0.0, rise], [half_span, 0.0]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int64)
    block = ElementBlock(etype=ElementType.TRUSS2, conn=conn, name="桁架")
    mesh = Mesh(coords=coords, blocks=(block,))

    apex = mesh.n_nodes - 2  # 3 节点模型的中间顶点
    case = StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=2, dofs=(0, 1))),
        loads=(NodalLoad(node=apex, forces=(0.0, float(p["apex_load"]))),),
    )
    bundle = ModelBundle(
        mesh=mesh,
        materials=(_material(p),),
        sections=(Section(area=float(p["area"])),),
        case=case,
    )
    _report(report, 1.0, "模型就绪")
    return bundle


# ------------------------------------------------------------------ 分析节点


def run_static(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> StaticSolution:
    """静力分析节点：MODEL -> StaticSolution."""
    model = _model_of(inputs)
    _params("analysis.static", params)  # 校验（当前无参数，防御 schema 漂移）
    return solve_static(model.mesh, model.materials, model.sections, model.case, report=report)


def run_modal(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ModalSolution:
    """模态分析节点：MODEL -> ModalSolution."""
    model = _model_of(inputs)
    p = _params("analysis.modal", params)
    return solve_modal(
        model.mesh,
        model.materials,
        model.sections,
        model.case.constraints,
        n_modes=int(p["n_modes"]),
        report=report,
    )


def run_harmonic(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> HarmonicResponse:
    """谐响应分析节点：MODEL -> HarmonicResponse（频率扫描序列由参数生成）."""
    model = _model_of(inputs)
    p = _params("analysis.harmonic", params)
    frequencies: list[float] = np.linspace(0.0, float(p["f_max"]), int(p["n_freq"])).tolist()
    return solve_harmonic(
        model.mesh,
        model.materials,
        model.sections,
        model.case,
        frequencies,
        alpha=float(p["alpha"]),
        beta=float(p["beta"]),
        report=report,
    )


def run_transient(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> TransientSolution:
    """瞬态动力分析节点：MODEL -> TransientSolution（阶跃载荷时程）.

    工况载荷作为空间分布整体施加，时程因子恒为 1（阶跃）；时间离散为
    均匀步长 ``duration / n_steps``，阻尼为 Rayleigh 模型。
    """
    model = _model_of(inputs)
    p = _params("analysis.transient", params)
    return solve_transient(
        model.mesh,
        model.materials,
        model.sections,
        model.case,
        duration=float(p["duration"]),
        n_steps=int(p["n_steps"]),
        alpha=float(p["alpha"]),
        beta=float(p["beta"]),
        report=report,
    )


def run_buckling(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> BucklingSolution:
    """屈曲分析节点：MODEL(+可选 STATIC 参考) -> BucklingSolution.

    接入 ``reference`` 上游静力解时复用其轴力（预应力链接），否则内部自算。
    """
    model = _model_of(inputs)
    p = _params("analysis.buckling", params)
    reference = _optional_static(inputs, "reference")
    return solve_buckling(
        model.mesh,
        model.materials,
        model.sections,
        model.case,
        n_modes=int(p["n_modes"]),
        reference=reference,
        report=report,
    )


def run_nonlinear(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> NonlinearSolution:
    """几何非线性分析节点：MODEL(+可选 STATIC 初态) -> NonlinearSolution.

    接入 ``initial`` 上游静力解时从其位移状态起算（初态链接），否则零位移起算。
    """
    model = _model_of(inputs)
    p = _params("analysis.nonlinear", params)
    initial = _optional_static(inputs, "initial")
    return solve_nonlinear_static(
        model.mesh,
        model.materials,
        model.sections,
        model.case,
        n_increments=int(p["n_increments"]),
        tolerance=float(p["tolerance"]),
        max_iterations=int(p["max_iterations"]),
        initial=initial,
        report=report,
    )


#: 频响曲线观察点选择（x 最大列中 y 居中节点），供 GUI 结果渲染复用
def tip_node(mesh: Mesh) -> int:
    """取末端中点节点（x 最大列中 y 居中者）."""
    coords: npt.NDArray[np.float64] = mesh.coords
    tip_mask = coords[:, 0] >= coords[:, 0].max() - 1e-9
    tip_rows = np.flatnonzero(tip_mask)
    return int(tip_rows[np.argmin(np.abs(coords[tip_rows, 1] - np.mean(coords[tip_rows, 1])))])
