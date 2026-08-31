"""稳态热传导分析：装配导热矩阵（含对流边界）-> 求解温度场 -> 恢复热流.

控制方程 ``∇·(k∇T) + q = 0``，边界为给定温度（Dirichlet）、节点热源
（Neumann）与表面对流（Robin，``h(T - T∞)`` 沿边界折线逐段施加）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .conduction import (
    ConductionMaterial,
    NodalSource,
    NodalValue,
    _batch_gradients,
    assemble_conduction,
)
from .errors import MeshError
from .material import Section
from .mesh import Mesh
from .solve import solve_system

__all__ = ["Convection", "ThermalCase", "ThermalSolution", "solve_thermal"]


@dataclass(frozen=True)
class Convection:
    """边界对流（Robin）：2D 沿节点折线逐段施加，3D 沿四边形面片施加 ``h(T - T∞)``.

    Attributes:
        nodes: 边界节点索引序列（沿边连续排列，2D 折线口径）。
        h_coeff: 对流换热系数（W/mm²·K，> 0）。
        t_ambient: 环境温度（K 或 °C，与给定温度同基准）。
        faces: 四边形面片节点组（3D 网格口径，每组恰 4 个节点，
            HEX8 边界面按逆时针（从域外看）排列）。
    """

    nodes: tuple[int, ...] = ()
    h_coeff: float = 0.0
    t_ambient: float = 0.0
    faces: tuple[tuple[int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        """校验对流参数."""
        if len(self.nodes) < 2 and not self.faces:
            raise MeshError("对流边界至少需要 2 个节点或 1 个四边形面片")
        for face in self.faces:
            if len(face) != 4:
                raise MeshError(f"对流面片须为 4 节点四边形，实际 {len(face)} 节点")
        if self.h_coeff <= 0.0:
            raise MeshError(f"对流换热系数须为正，实际 h={self.h_coeff}")


@dataclass(frozen=True)
class ThermalCase:
    """稳态热传导工况：给定温度 + 节点热源 + 边界对流.

    Attributes:
        temperatures: 给定温度（Dirichlet）。
        heat_sources: 节点热源（Neumann，正值为注入，W）。
        convections: 对流边界表（Robin）。
    """

    temperatures: tuple[NodalValue, ...] = ()
    heat_sources: tuple[NodalSource, ...] = ()
    convections: tuple[Convection, ...] = ()

    def validate(self, mesh: Mesh) -> None:
        """按网格校验节点索引范围."""
        for item in (*self.temperatures, *self.heat_sources):
            if not 0 <= item.node < mesh.n_nodes:
                raise MeshError(f"热学边界引用节点 {item.node} 越界（共 {mesh.n_nodes} 节点）")
        for convection in self.convections:
            for node in (*convection.nodes, *(n for face in convection.faces for n in face)):
                if not 0 <= node < mesh.n_nodes:
                    raise MeshError(f"对流边界引用节点 {node} 越界（共 {mesh.n_nodes} 节点）")


@dataclass(frozen=True)
class ThermalSolution:
    """稳态热传导结果.

    Attributes:
        mesh: 参与求解的网格。
        temperatures: 节点温度 ``(n_nodes,)``。
        element_gradients: 逐单元温度梯度 ``∇T (n_elements, 2)``。
        element_heat_flux: 逐单元热流密度模长 ``k|∇T| (n_elements,)``（W/mm²）。
        t_min: 全场最低温度。
        t_max: 全场最高温度。
        convection_heat: 对流边界总换热量（W，正值 = 向环境散热）。
    """

    mesh: Mesh
    temperatures: np.ndarray
    element_gradients: np.ndarray
    element_heat_flux: np.ndarray
    t_min: float
    t_max: float
    convection_heat: float = 0.0


def solve_thermal(  # noqa: PLR0913  模型四要素 + 工况/附加热载荷/回调，语义不可合并
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    case: ThermalCase,
    *,
    extra_heat: np.ndarray | None = None,
    report: Callable[[float, str], None] | None = None,
) -> ThermalSolution:
    """稳态热传导分析主流程.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表（ElementBlock.material 索引引用）。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。
        case: 热学工况（给定温度 + 热源 + 对流）。
        extra_heat: 附加节点热载荷（如 Joule 热一致节点载荷），与 case 热源叠加。
        report: 进度回调 ``(progress, message)``（进程执行器自动注入）。

    Returns:
        :class:`ThermalSolution`，含节点温度、单元热流与温度范围。
    """
    progress = report if report is not None else _no_report
    progress(0.1, "校验热学工况")
    case.validate(mesh)
    progress(0.4, "装配导热矩阵")
    k_thermal = assemble_conduction(mesh, materials, sections, "thermal")
    force = np.zeros(mesh.n_nodes)
    for source in case.heat_sources:
        force[source.node] += source.value
    if extra_heat is not None:
        if extra_heat.shape != (mesh.n_nodes,):
            raise MeshError(f"附加热载荷维度 {extra_heat.shape} 与节点数 {mesh.n_nodes} 不符")
        force = force + extra_heat

    k_thermal, force = _apply_convections(mesh, case, k_thermal, force)

    seen: dict[int, float] = {}
    for prescribed in case.temperatures:
        seen.setdefault(prescribed.node, prescribed.value)
    fixed = np.fromiter(seen.keys(), dtype=np.intp)
    values = np.fromiter(seen.values(), dtype=float)

    progress(0.7, "求解温度场")
    t, _ = solve_system(k_thermal, force, fixed, values)

    progress(0.9, "恢复热流")
    gradients: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    for block in mesh.blocks:
        material = materials[block.material]
        grads = _batch_gradients(block.etype, mesh.coords[block.conn], t[block.conn])
        gradients.append(grads)
        fluxes.append(material.thermal_k * np.linalg.norm(grads, axis=1))
    convection_heat = _convection_total(mesh, case, t)
    progress(1.0, "温度场求解完成")
    return ThermalSolution(
        mesh=mesh,
        temperatures=t,
        element_gradients=np.concatenate(gradients) if gradients else np.empty((0, mesh.dim)),
        element_heat_flux=np.concatenate(fluxes) if fluxes else np.empty(0),
        t_min=float(t.min()),
        t_max=float(t.max()),
        convection_heat=convection_heat,
    )


#: 对流面片 2×2 高斯点（与 conduction 单元积分同阶）
_FACE_GAUSS = (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0))


def _face_shape_values(xi: float, eta: float) -> np.ndarray:
    """四边形面片形函数值 ``(4,)``（节点序：左下/右下/右上/左上）."""
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


def _face_shape_derivs(xi: float, eta: float) -> np.ndarray:
    """四边形面片形函数对自然坐标的导数 ``(2, 4)``."""
    return 0.25 * np.array(
        [
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ]
    )


def _face_convection_terms(coords: np.ndarray, h: float, t_inf: float) -> tuple[np.ndarray, np.ndarray]:
    """单个四边形面片的一致对流刚度 ``(4,4)`` 与载荷 ``(4,)``（2×2 高斯）.

    :param coords: 面片节点坐标 ``(4, 3)``。
    :param h: 对流换热系数（W/mm²·K）。
    :param t_inf: 环境温度。
    """
    ke = np.zeros((4, 4))
    fe = np.zeros(4)
    for xi in _FACE_GAUSS:
        for eta in _FACE_GAUSS:
            dn = _face_shape_derivs(xi, eta)
            jac = dn @ coords
            det = float(np.linalg.norm(np.cross(jac[0], jac[1])))
            if det <= 0.0:
                raise MeshError("对流面片退化（面积接近零）")
            shape = _face_shape_values(xi, eta)
            ke += h * det * np.outer(shape, shape)
            fe += h * t_inf * det * shape
    return ke, fe


def _face_area(coords: np.ndarray) -> float:
    """四边形面片面积（2×2 高斯 |detJ| 求和）."""
    area = 0.0
    for xi in _FACE_GAUSS:
        for eta in _FACE_GAUSS:
            jac = _face_shape_derivs(xi, eta) @ coords
            area += float(np.linalg.norm(np.cross(jac[0], jac[1])))
    return area


def _apply_convections(
    mesh: Mesh,
    case: ThermalCase,
    stiffness: csr_matrix,
    force: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """对流边界折入刚度与载荷.

    2D 折线：段级 ``h L/6 [2,1;1,2]`` 与 ``h T∞ L/2 [1,1]``；
    3D 面片：``∫ h NᵀN dS`` 与 ``∫ h T∞ N dS``（2×2 高斯）。
    """
    if not case.convections:
        return stiffness, force
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    load = force.copy()
    for convection in case.convections:
        for here, there in zip(convection.nodes[:-1], convection.nodes[1:]):
            length = float(np.linalg.norm(mesh.coords[there] - mesh.coords[here]))
            if length <= 0.0:
                raise MeshError("对流边界折线出现重复相邻节点，段长为零")
            h = convection.h_coeff
            edge_k = h * length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
            edge_f = h * convection.t_ambient * length / 2.0 * np.ones(2)
            for local_i, node_i in enumerate((here, there)):
                load[node_i] += edge_f[local_i]
                for local_j, node_j in enumerate((here, there)):
                    rows.append(node_i)
                    cols.append(node_j)
                    values.append(float(edge_k[local_i, local_j]))
        for face in convection.faces:
            face_k, face_f = _face_convection_terms(
                mesh.coords[np.asarray(face)], convection.h_coeff, convection.t_ambient
            )
            for local_i, node_i in enumerate(face):
                load[node_i] += face_f[local_i]
                for local_j, node_j in enumerate(face):
                    rows.append(node_i)
                    cols.append(node_j)
                    values.append(float(face_k[local_i, local_j]))
    # 段级/面片级贡献以 COO 累加折入原 CSR
    extra = csr_matrix((values, (rows, cols)), shape=stiffness.shape)
    return stiffness + extra, load


def _convection_total(mesh: Mesh, case: ThermalCase, temperatures: np.ndarray) -> float:
    """对流边界总换热量（正 = 向环境散热）：逐段 ``h(T_avg-T∞)L`` 或逐面片 ``h(T_avg-T∞)A``."""
    total = 0.0
    for convection in case.convections:
        for here, there in zip(convection.nodes[:-1], convection.nodes[1:]):
            length = float(np.linalg.norm(mesh.coords[there] - mesh.coords[here]))
            t_avg = 0.5 * (temperatures[here] + temperatures[there])
            total += convection.h_coeff * (t_avg - convection.t_ambient) * length
        for face in convection.faces:
            area = _face_area(mesh.coords[np.asarray(face)])
            t_avg = float(np.mean(temperatures[np.asarray(face)]))
            total += convection.h_coeff * (t_avg - convection.t_ambient) * area
    return total


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调."""
