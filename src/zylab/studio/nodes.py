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
    ConductionMaterial,
    Constraint,
    Convection,
    ElectricCase,
    ElectroThermalSolution,
    ElectroThermalTransientSolution,
    ElementBlock,
    ElementType,
    HarmonicResponse,
    LinearElastic,
    Mesh,
    ModalSolution,
    NodalLoad,
    NodalValue,
    NonlinearSolution,
    Section,
    StaticCase,
    StaticSolution,
    ThermalCase,
    TransientSolution,
    solve_buckling,
    solve_electrothermal,
    solve_electrothermal_transient,
    solve_harmonic,
    solve_modal,
    solve_nonlinear_static,
    solve_static,
    solve_transient,
)

from .batch import run_workflow
from .bundle import ConductionBundle, ModelBundle
from .dsl import substitute_refs
from .errors import ParamError, StudioError
from .expressions import ARRAY_MATH_NAMESPACE, safe_eval
from .meshing3d import cylinder_resistor_mesh, vfilm_resistor_mesh
from .module import module_spec
from .template import Template

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "build_cantilever",
    "build_column",
    "build_cylinder_resistor",
    "build_joule_hole",
    "build_joule_plate",
    "build_joule_series",
    "build_truss",
    "build_vfilm_resistor",
    "compute_expr",
    "compute_sweep",
    "post_static",
    "run_buckling",
    "run_electrothermal",
    "run_electrothermal_transient",
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


def _params(type_id: str, params: NodeParams) -> dict[str, Any]:
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


def build_joule_plate(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ConductionBundle:
    """构建通电加热板 Q4 电-热耦合模型（左右电极给定电压，底边恒温，其余三边对流）."""
    del inputs
    p = _params("example.joule_plate_2d", params)
    length, height = float(p["length"]), float(p["height"])
    nx, ny = int(p["nx"]), int(p["ny"])

    _report(report, 0.2, "生成板网格")
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            conn.append((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
    block = ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn), name="板")
    mesh = Mesh(coords=coords, blocks=(block,))

    _report(report, 0.6, "生成电-热边界")
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
    bottom = np.flatnonzero(coords[:, 1] <= 0.0)
    top = np.flatnonzero(coords[:, 1] >= height - 1e-9)
    # 对流边界折线：左（底→顶）→ 顶（左→右）→ 右（顶→底），沿边连续且角节点不重复
    conv_nodes = tuple(int(n) for n in (*left, *top[1:-1], *right[::-1]))
    bundle = ConductionBundle(
        mesh=mesh,
        materials=(
            ConductionMaterial(
                electric_sigma=float(p["electric_sigma"]),
                thermal_k=float(p["thermal_k"]),
            ),
        ),
        sections=(Section(thickness=float(p["thickness"])),),
        electric_case=ElectricCase(
            voltages=(
                *(NodalValue(int(n), 0.0) for n in left),
                *(NodalValue(int(n), float(p["voltage"])) for n in right),
            ),
        ),
        thermal_case=ThermalCase(
            temperatures=tuple(NodalValue(int(n), float(p["t_base"])) for n in bottom),
            convections=(Convection(nodes=conv_nodes, h_coeff=float(p["h_conv"]), t_ambient=float(p["t_ambient"])),),
        ),
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def build_joule_series(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ConductionBundle:
    """构建多材料串联电加热板 Q4 模型（电极/电阻区/电极三区，电流集浓热点）.

    三区各自 ElementBlock 引用独立材料（电极高电导、电阻区高阻抗），
    热点稳态出现在电阻区；电学边界 = 左右端面给定电压，热边界 = 底边
    恒温 + 其余三边对流（与 :func:`build_joule_plate` 同布局）。
    """
    del inputs
    p = _params("example.joule_series_2d", params)
    length, height = float(p["length"]), float(p["height"])
    nx, ny = int(p["nx"]), int(p["ny"])

    _report(report, 0.2, "生成板网格")
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # 电极段单元数（贴齐网格，保证区界落在单元边界上；每区至少 1 单元）
    dx = length / nx
    n_electrode = min(max(round(float(p["electrode_len"]) / dx), 1), nx // 2 - 1) if nx >= 3 else 1
    blocks = []
    for zone, (i_begin, i_end) in enumerate(
        ((0, n_electrode), (n_electrode, nx - n_electrode), (nx - n_electrode, nx))
    ):
        conn = []
        for j in range(ny):
            for i in range(i_begin, i_end):
                n00 = j * (nx + 1) + i
                conn.append((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
        blocks.append(ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn), material=zone % 2, name=f"区{zone}"))
    mesh = Mesh(coords=coords, blocks=tuple(blocks))

    _report(report, 0.6, "生成电-热边界")
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
    bottom = np.flatnonzero(coords[:, 1] <= 0.0)
    top = np.flatnonzero(coords[:, 1] >= height - 1e-9)
    # 对流边界折线：左（底→顶）→ 顶（左→右）→ 右（顶→底），沿边连续且角节点不重复
    conv_nodes = tuple(int(n) for n in (*left, *top[1:-1], *right[::-1]))
    bundle = ConductionBundle(
        mesh=mesh,
        materials=(
            # 材料表按块索引被引用：0 = 电极，1 = 电阻区
            ConductionMaterial(
                electric_sigma=float(p["sigma_conductor"]),
                thermal_k=float(p["k_conductor"]),
            ),
            ConductionMaterial(
                electric_sigma=float(p["sigma_resistor"]),
                thermal_k=float(p["k_resistor"]),
            ),
        ),
        sections=(Section(thickness=float(p["thickness"])),),
        electric_case=ElectricCase(
            voltages=(
                *(NodalValue(int(n), 0.0) for n in left),
                *(NodalValue(int(n), float(p["voltage"])) for n in right),
            ),
        ),
        thermal_case=ThermalCase(
            temperatures=tuple(NodalValue(int(n), float(p["t_base"])) for n in bottom),
            convections=(Convection(nodes=conv_nodes, h_coeff=float(p["h_conv"]), t_ambient=float(p["t_ambient"])),),
        ),
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def _boundary_edges(blocks_conns: tuple[np.ndarray, ...]) -> list[tuple[int, int]]:
    """提取网格边界边（恰好属于一个单元的边），用于任意形状边界分类.

    全量边编码为单整数后 ``np.unique`` 计数（向量化，大网格下远快于逐边字典）。
    """
    if not blocks_conns:
        return []
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    for conn in blocks_conns:
        following = np.roll(conn, -1, axis=1)
        lo_parts.append(np.minimum(conn, following).ravel())
        hi_parts.append(np.maximum(conn, following).ravel())
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    stride = int(hi.max()) + 1  # 编码基数（节点索引上界 + 1，保证单整数编码唯一）
    codes = lo.astype(np.int64) * stride + hi
    uniq, counts = np.unique(codes, return_counts=True)
    return [(int(code // stride), int(code % stride)) for code in uniq[counts == 1]]


def build_joule_hole(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ConductionBundle:
    """构建带圆孔多材料电加热板 Q4 模型（复杂形状：形心掩码挖孔）.

    矩形板按形心掩码挖去圆孔（孔缘呈阶梯状逼近圆弧），三区材料分区同
    :func:`build_joule_series`；挖孔后压缩孤立节点（防刚度矩阵奇异），
    并自动提取边界边分类：左右电极面给定电压、底边恒温、其余边界
    （顶边/侧边剩余段/孔缘）施加对流。
    """
    del inputs
    p = _params("example.joule_hole_2d", params)
    length, height = float(p["length"]), float(p["height"])
    nx, ny = int(p["nx"]), int(p["ny"])
    hole_r = float(p["hole_r"])

    _report(report, 0.15, "生成板网格并挖孔")
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    dx = length / nx
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # 单元形心落在圆孔内的单元被剔除（严格小于，形心恰在圆周上保留）
    hole_x, hole_y = float(p["hole_x"]), float(p["hole_y"])
    n_electrode = min(max(round(float(p["electrode_len"]) / dx), 1), nx // 2 - 1) if nx >= 3 else 1
    zone_conns: list[list[tuple[int, ...]]] = [[], [], []]
    for j in range(ny):
        for i in range(nx):
            xc, yc = (i + 0.5) * dx, (j + 0.5) * height / ny
            if (xc - hole_x) ** 2 + (yc - hole_y) ** 2 < hole_r**2:
                continue  # 孔内单元剔除
            n00 = j * (nx + 1) + i
            quad = (n00, n00 + 1, n00 + nx + 2, n00 + nx + 1)
            zone = 0 if i < n_electrode else (2 if i >= nx - n_electrode else 1)
            zone_conns[zone].append(quad)

    _report(report, 0.5, "压缩孤立节点")
    conn_arrays = tuple(np.asarray(conns, dtype=np.intp) for conns in zone_conns)
    used = np.unique(np.concatenate(conn_arrays))
    remap = np.full(coords.shape[0], -1, dtype=np.intp)
    remap[used] = np.arange(used.size)
    coords = coords[used]
    blocks = tuple(
        ElementBlock(
            etype=ElementType.QUAD4,
            conn=remap[conns],
            material=zone % 2,
            name=f"区{zone}",
        )
        for zone, conns in enumerate(conn_arrays)
        if conns.size > 0
    )
    mesh = Mesh(coords=coords, blocks=blocks)

    _report(report, 0.8, "提取边界并生成电-热工况")
    tol = min(dx, height / ny) * 1e-9
    left = np.flatnonzero(coords[:, 0] <= tol)
    right = np.flatnonzero(coords[:, 0] >= length - tol)
    bottom = np.flatnonzero(coords[:, 1] <= tol)
    # 边界边分类：电极面（左右端）与底边恒温面除外，其余边界（含孔缘）施加对流
    convections = []
    for a, b in _boundary_edges(tuple(block.conn for block in blocks)):
        pa, pb = coords[a], coords[b]
        on_left = pa[0] <= tol and pb[0] <= tol
        on_right = pa[0] >= length - tol and pb[0] >= length - tol
        on_bottom = pa[1] <= tol and pb[1] <= tol
        if not (on_left or on_right or on_bottom):
            convections.append(Convection(nodes=(a, b), h_coeff=float(p["h_conv"]), t_ambient=float(p["t_ambient"])))
    bundle = ConductionBundle(
        mesh=mesh,
        materials=(
            ConductionMaterial(
                electric_sigma=float(p["sigma_conductor"]),
                thermal_k=float(p["k_conductor"]),
            ),
            ConductionMaterial(
                electric_sigma=float(p["sigma_resistor"]),
                thermal_k=float(p["k_resistor"]),
            ),
        ),
        sections=(Section(thickness=float(p["thickness"])),),
        electric_case=ElectricCase(
            voltages=(
                *(NodalValue(int(n), 0.0) for n in left),
                *(NodalValue(int(n), float(p["voltage"])) for n in right),
            ),
        ),
        thermal_case=ThermalCase(
            temperatures=tuple(NodalValue(int(n), float(p["t_base"])) for n in bottom),
            convections=tuple(convections),
        ),
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


def run_electrothermal(
    inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None
) -> ElectroThermalSolution:
    """电-热耦合分析节点：ET_MODEL -> ElectroThermalSolution（稳态顺序耦合）."""
    model = inputs["model"]
    if not isinstance(model, ConductionBundle):
        raise TypeError(f"输入端口 'model' 应为 ConductionBundle，得到 {type(model).__name__}")
    _params("analysis.electrothermal", params)  # 校验（当前无参数，防御 schema 漂移）
    return solve_electrothermal(
        model.mesh,
        model.materials,
        model.sections,
        model.electric_case,
        model.thermal_case,
        report=report,
    )


def build_cylinder_resistor(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ConductionBundle:
    """构建圆柱电阻 HEX8 电-热耦合模型（两端面电极给定电压，全表面对流）.

    圆柱沿 z 轴极坐标结构化离散（轴心留 5% 半径中心孔规避单元退化），
    z=0 端面接地 0V、z=L 端面给定电压，外圆柱面与两端面对流散热；
    材料为常物性电阻合金（含体积热容，瞬态分析可直接使用）。
    """
    del inputs  # 源节点无输入
    p = _params("example.cylinder_resistor_3d", params)

    _report(report, 0.4, "生成圆柱网格")
    geo = cylinder_resistor_mesh(
        radius=float(p["radius"]),
        length=float(p["length"]),
        n_theta=int(p["n_theta"]),
        n_r=int(p["n_r"]),
        n_z=int(p["n_z"]),
    )
    _report(report, 0.8, "生成电-热边界")
    bundle = ConductionBundle(
        mesh=geo.mesh,
        materials=(
            ConductionMaterial(
                electric_sigma=float(p["electric_sigma"]),
                thermal_k=float(p["thermal_k"]),
                volumetric_heat_capacity=float(p["rho_cp"]),
            ),
        ),
        sections=(Section(),),
        electric_case=ElectricCase(
            voltages=(
                *(NodalValue(n, 0.0) for n in geo.end_low_nodes),
                *(NodalValue(n, float(p["voltage"])) for n in geo.end_high_nodes),
            ),
        ),
        thermal_case=ThermalCase(
            convections=(
                Convection(faces=geo.conv_faces, h_coeff=float(p["h_conv"]), t_ambient=float(p["t_ambient"])),
            ),
        ),
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def build_vfilm_resistor(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> ConductionBundle:
    """构建 V 形薄膜电阻 HEX8 电-热耦合模型（微米级厚膜 + 陶瓷基底）.

    俯视 V 形路径扫掠生成薄膜（引入/引出段为高电导电极、中间 V 形区为
    阻性厚膜），陶瓷基底与薄膜共享底面节点：热学上双向耦合、电学上基底
    电导率极小被绝缘过滤（孤立节点自动接地）。热边界 = 陶瓷底面恒温 +
    薄膜顶面对流。
    """
    del inputs  # 源节点无输入
    p = _params("example.vfilm_resistor_3d", params)

    _report(report, 0.4, "生成 V 形薄膜网格")
    geo = vfilm_resistor_mesh(
        span=float(p["span"]),
        depth=float(p["depth"]),
        width=float(p["width"]),
        thickness=float(p["thickness"]),
        substrate_h=float(p["substrate_h"]),
        lead_len=float(p["lead_len"]),
        n_lead=int(p["n_lead"]),
        n_diag=int(p["n_diag"]),
        n_width=int(p["n_width"]),
        n_sub=int(p["n_sub"]),
    )
    _report(report, 0.8, "生成电-热边界")
    # 材料表按块索引引用：0 = 阻性厚膜，1 = 电极，2 = 陶瓷基底（绝缘，电场装配跳过）
    bundle = ConductionBundle(
        mesh=geo.mesh,
        materials=(
            ConductionMaterial(
                electric_sigma=float(p["sigma_film"]),
                thermal_k=float(p["k_film"]),
                volumetric_heat_capacity=float(p["rho_cp_film"]),
            ),
            ConductionMaterial(
                electric_sigma=float(p["sigma_electrode"]),
                thermal_k=float(p["k_electrode"]),
                volumetric_heat_capacity=float(p["rho_cp_electrode"]),
            ),
            ConductionMaterial(
                electric_sigma=1.0e-12,  # 陶瓷绝缘（低于绝缘阈值即被电场过滤）
                thermal_k=float(p["k_ceramic"]),
                volumetric_heat_capacity=float(p["rho_cp_ceramic"]),
            ),
        ),
        sections=(Section(),),
        electric_case=ElectricCase(
            voltages=(
                *(NodalValue(n, 0.0) for n in geo.lead_low_nodes),
                *(NodalValue(n, float(p["voltage"])) for n in geo.lead_high_nodes),
            ),
        ),
        thermal_case=ThermalCase(
            temperatures=tuple(NodalValue(n, float(p["t_base"])) for n in geo.base_bottom_nodes),
            convections=(
                Convection(faces=geo.film_top_faces, h_coeff=float(p["h_conv"]), t_ambient=float(p["t_ambient"])),
            ),
        ),
    )
    _report(report, 1.0, "模型就绪")
    return bundle


def run_electrothermal_transient(
    inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None
) -> ElectroThermalTransientSolution:
    """瞬态电-热耦合分析节点：ET_MODEL -> ElectroThermalTransientSolution.

    稳态电场（常物性）+ backward Euler 瞬态温度场；初始温度为均匀 ``t_init``。
    """
    model = inputs["model"]
    if not isinstance(model, ConductionBundle):
        raise TypeError(f"输入端口 'model' 应为 ConductionBundle，得到 {type(model).__name__}")
    p = _params("analysis.electrothermal_transient", params)
    initial = np.full(model.mesh.n_nodes, float(p["t_init"]))
    return solve_electrothermal_transient(
        model.mesh,
        model.materials,
        model.sections,
        model.electric_case,
        model.thermal_case,
        initial=initial,
        total_time=float(p["duration"]),
        n_steps=int(p["n_steps"]),
        report=report,
    )


def tip_node(mesh: Mesh) -> int:
    """取末端中点节点（x 最大列中 y 居中者）."""
    coords: npt.NDArray[np.float64] = mesh.coords
    tip_mask = coords[:, 0] >= coords[:, 0].max() - 1e-9
    tip_rows = np.flatnonzero(tip_mask)
    return int(tip_rows[np.argmin(np.abs(coords[tip_rows, 1] - np.mean(coords[tip_rows, 1])))])


# ------------------------------------------------------------------ 计算与后处理节点


def compute_expr(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> Any:
    """公式计算节点：受限命名空间安全求值表达式，输出 DATA 载荷.

    命名空间 = 数组数学函数 + 上游 ``data`` 输入（映射时逐项合并，否则
    以 ``data`` 为名绑定）+ ``vars`` 绑定（列表/元组收敛为 numpy 数组）。
    """
    p = _params("compute.expr", params)
    namespace: dict[str, Any] = dict(ARRAY_MATH_NAMESPACE)
    data = inputs.get("data")
    if data is not None:
        if isinstance(data, Mapping):
            namespace.update(data)
        else:
            namespace["data"] = data
    for name, value in dict(p["vars"]).items():
        namespace[name] = np.asarray(value) if isinstance(value, (list, tuple)) else value
    _report(report, 1.0, "公式计算完成")
    return safe_eval(str(p["expr"]), namespace)


def compute_sweep(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> dict[str, Any]:
    """参数扫描节点：对 body 子图按扫描值逐次执行并汇总结果序列.

    每步以 ``{var: 当前值}`` 深度代入 body 节点参数的 ``$var`` 引用后，
    经 :func:`~zylab.studio.batch.run_workflow` 进程内执行整个子图
    （省去子进程 pickle 往返）；``collect`` 引用（``"节点id[.端口名]"``）
    收集各步输出组成序列，输出 ``{var, values, series}`` DATA 载荷。
    """
    p = _params("compute.sweep", params)
    del inputs  # 扫参节点无输入端口（body 子图自包含）
    var = str(p["var"])
    count = int(p["count"])
    values = np.linspace(float(p["from"]), float(p["to"]), count)
    body = dict(p["body"])
    nodes_raw = body.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ParamError("compute.sweep 的 body 须声明非空 nodes 节点列表")
    for raw in nodes_raw:
        if not isinstance(raw, Mapping):
            raise ParamError("compute.sweep body.nodes 项应为节点对象")
    collect = tuple(str(ref) for ref in body.get("collect", ()))
    if not collect:
        raise ParamError("compute.sweep 的 body 须声明非空 collect 结果引用列表")
    series: dict[str, list[Any]] = {ref: [] for ref in collect}
    for index, value in enumerate(values):
        _report(report, (index + 1) / count, f"参数扫描 {var}={value:.6g}")
        step_nodes = [
            {
                **dict(raw),
                "params": substitute_refs(dict(raw.get("params", {})), {var: float(value)}, "compute.sweep"),
            }
            for raw in nodes_raw
        ]
        template = Template.from_dict({"id": "compute.sweep.body", "name": "扫参子图", "nodes": step_nodes})
        outcome = run_workflow(template)
        if not outcome.succeeded:
            raise StudioError(f"参数扫描 {var}={value:.6g} 步失败: {outcome.first_error()}")
        for ref in collect:
            series[ref].append(outcome.outcome(ref.partition(".")[0]).result)
    return {"var": var, "values": values.tolist(), "series": series}


def post_static(inputs: NodeInputs, params: NodeParams, report: ReportFn | None = None) -> Any:
    """静力结果提取节点：STATIC -> DATA（表达式从位移/反力/应变能取值）.

    提取表达式支持只读下标（如 ``"displacements[-1, 1]"`` 取末端节点
    竖向位移），数组运算与数学函数同 compute.expr 命名空间。
    """
    solution = inputs["solution"]
    if not isinstance(solution, StaticSolution):
        raise TypeError(f"输入端口 'solution' 应为 StaticSolution，得到 {type(solution).__name__}")
    p = _params("post.static", params)
    namespace: dict[str, Any] = {
        **ARRAY_MATH_NAMESPACE,
        "displacements": solution.displacements,
        "reactions": dict(solution.reactions),
        "strain_energy": solution.strain_energy,
        "n_nodes": solution.mesh.n_nodes,
        "n_elements": solution.mesh.n_elements,
    }
    _report(report, 1.0, "结果提取完成")
    return safe_eval(str(p["expr"]), namespace)
