"""标量场传导内核：稳态/瞬态电-热传导共用的单元矩阵、装配与边界描述.

标量场（电压 / 温度）每节点 1 个自由度，全局编号 = 节点索引（无宽度偏移），
与结构位移场相互独立。单元传导矩阵 ``∫ Bᵀ c B dΩ``（c 为电导率或导热系数），
TRIA3 常梯度解析公式，QUAD4 二点高斯全积分，HEX8 二点×二点×二点高斯
全积分（体积单元，截面厚度不参与换算）。热容矩阵 ``∫ ρc Nᵀ N dΩ``
供瞬态热求解（backward Euler）使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .errors import ElementError, MeshError
from .material import Section
from .mesh import ElementType, Mesh

__all__ = [
    "ConductionMaterial",
    "NodalSource",
    "NodalValue",
    "assemble_capacity",
    "assemble_conduction",
    "element_conductance",
    "element_field_load",
    "element_scalar_gradient",
]

#: 2 点高斯积分位置（与结构单元一致）
_GAUSS_ABSCISSA = 1.0 / np.sqrt(3.0)

#: HEX8 节点自然坐标符号表（与 elements.py 同序：ζ=-1 面 1-4，ζ=+1 面 5-8）
_HEX8_SIGNS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)

#: HEX8 八高斯点自然坐标（ξ/η/ζ 三重 ±1/√3，与逐点实现同序）
_HEX8_GAUSS = tuple(
    (xi, eta, zeta)
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA)
    for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA)
    for zeta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA)
)


def _quad4_gauss_points() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """QUAD4 四高斯点 ``(N, dN)`` 常量表（xi 外层、eta 内层，与逐单元实现同序）."""
    points = []
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            n_shape = 0.25 * np.array(
                [
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ]
            )
            d_n = 0.25 * np.array(
                [
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
                ]
            )
            points.append((n_shape, d_n))
    return tuple(points)


#: QUAD4 四高斯点形函数/导数常量（批量内核复用）
_QUAD4_GAUSS = _quad4_gauss_points()

_GEOM_TOL = 1.0e-12

#: 支持标量场传导的单元族（2D 连续体 + 3D 六面体）
_CONDUCTION_TYPES = frozenset({ElementType.TRIA3, ElementType.QUAD4, ElementType.HEX8})


@dataclass(frozen=True)
class ConductionMaterial:
    """传导材料属性（各向同性、常物性，电-热共用一份材料表）.

    Attributes:
        electric_sigma: 电导率 σ（S/mm，> 0；绝缘体取极小正值，电场装配自动跳过）。
        thermal_k: 导热系数 k（W/mm·K，> 0）。
        volumetric_heat_capacity: 体积热容 ρc（J/mm³·K，瞬态热必需；0 表示未提供，
            稳态分析忽略该字段）。
    """

    electric_sigma: float
    thermal_k: float
    volumetric_heat_capacity: float = 0.0

    def __post_init__(self) -> None:
        """校验传导系数为正."""
        if self.electric_sigma <= 0.0:
            raise ElementError(f"电导率须为正，实际 sigma={self.electric_sigma}")
        if self.thermal_k <= 0.0:
            raise ElementError(f"导热系数须为正，实际 k={self.thermal_k}")
        if self.volumetric_heat_capacity < 0.0:
            raise ElementError(f"体积热容须非负，实际 rho_cp={self.volumetric_heat_capacity}")


@dataclass(frozen=True)
class NodalValue:
    """标量场节点给定值（Dirichlet：电压或温度）.

    Attributes:
        node: 节点索引（0 基）。
        value: 给定值（同一节点重复给定时取首个）。
    """

    node: int
    value: float


@dataclass(frozen=True)
class NodalSource:
    """标量场节点源项（Neumann：注入电流或热源）.

    Attributes:
        node: 节点索引（0 基）。
        value: 源值（电流 A / 热源 W，正值为注入）。
    """

    node: int
    value: float


def _tria3_gradient_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """CST 标量场梯度矩阵 G (2, 3) 与面积.

    行 0 为 dN/dx、行 1 为 dN/dy；常梯度（与坐标无关）。
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area <= _GEOM_TOL:
        raise ElementError("TRIA3 单元退化为直线（面积接近零）")
    g = np.array(
        [
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1],
        ]
    ) / (2.0 * area)
    return g, area


def _quad4_gradient_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """Q4 标量场梯度矩阵 G (2, 4) 与 |detJ|（指定自然坐标处）."""
    dn = 0.25 * np.array(
        [
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ]
    )
    jacob = dn @ coords
    det_j = float(np.linalg.det(jacob))
    if abs(det_j) <= _GEOM_TOL:
        raise ElementError("QUAD4 单元雅可比行列式接近零（单元退化或节点序错误）")
    return np.linalg.inv(jacob) @ dn, abs(det_j)


def _hex8_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形函数对自然坐标的导数 (3, 8)（节点序与符号表一致）."""
    sx, se, sz = _HEX8_SIGNS[:, 0], _HEX8_SIGNS[:, 1], _HEX8_SIGNS[:, 2]
    return (
        np.stack(
            (
                sx * (1.0 + eta * se) * (1.0 + zeta * sz),
                se * (1.0 + xi * sx) * (1.0 + zeta * sz),
                sz * (1.0 + xi * sx) * (1.0 + eta * se),
            )
        )
        / 8.0
    )


def _hex8_shape_values(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形函数值 (8,)（节点序与符号表一致）."""
    sx, se, sz = _HEX8_SIGNS[:, 0], _HEX8_SIGNS[:, 1], _HEX8_SIGNS[:, 2]
    return (1.0 + xi * sx) * (1.0 + eta * se) * (1.0 + zeta * sz) / 8.0


def _hex8_gradient_matrix(coords: np.ndarray, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, float]:
    """HEX8 标量场梯度矩阵 G (3, 8) 与 |detJ|（指定自然坐标处）."""
    dn = _hex8_shape_derivs(xi, eta, zeta)
    jacob = dn @ coords
    det_j = float(np.linalg.det(jacob))
    if abs(det_j) <= _GEOM_TOL:
        raise ElementError("HEX8 单元雅可比行列式接近零（单元退化或节点序错误）")
    return np.linalg.inv(jacob) @ dn, abs(det_j)


#: HEX8 八高斯点形函数值常量 ``(8, 8)``（批量场载荷/热容复用）
_HEX8_SHAPE = np.stack([_hex8_shape_values(xi, eta, zeta) for xi, eta, zeta in _HEX8_GAUSS])


def _hex8_batch_data(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量 HEX8 高斯点数据.

    :param coords: 单元节点坐标 ``(n, 8, 3)``。
    :return: ``(G, det)`` —— 八高斯点梯度矩阵 ``G (8, n, 3, 8)`` 与 ``|detJ| (8, n)``。
    """
    n = coords.shape[0]
    g_all = np.empty((8, n, 3, 8))
    det_all = np.empty((8, n))
    for point, (xi, eta, zeta) in enumerate(_HEX8_GAUSS):
        dn = _hex8_shape_derivs(xi, eta, zeta)
        jacob = np.einsum("ij,njk->nik", dn, coords)
        det = np.linalg.det(jacob)
        if np.any(np.abs(det) <= _GEOM_TOL):
            raise ElementError("HEX8 单元雅可比行列式接近零（单元退化或节点序错误）")
        g_all[point] = np.einsum("nij,jk->nik", np.linalg.inv(jacob), dn)
        det_all[point] = np.abs(det)
    return g_all, det_all


def element_conductance(
    etype: ElementType,
    coords: np.ndarray,
    coefficient: float,
    thickness: float,
) -> np.ndarray:
    """标量场单元传导矩阵 ``∫ Bᵀ c B dΩ``（2D 按厚度换算体积，3D 取单元体积）.

    Args:
        etype: 单元类型（支持 TRIA3 / QUAD4 / HEX8）。
        coords: 单元节点坐标 ``(n_node, 2)`` 或 ``(n_node, 3)``。
        coefficient: 传导系数（电导率 S/mm 或导热系数 W/mm·K）。
        thickness: 厚度（mm，仅 2D 单元参与换算；HEX8 忽略）。

    Returns:
        单元传导矩阵 ``(n_node, n_node)``（对称正半定）。

    Raises:
        ElementError: 单元类型不支持或几何退化时抛出。
    """
    if coefficient <= 0.0:
        raise ElementError(f"传导系数须为正，实际 c={coefficient}")
    if etype is ElementType.TRIA3:
        g, area = _tria3_gradient_matrix(coords)
        return coefficient * thickness * area * (g.T @ g)
    if etype is ElementType.QUAD4:
        ke = np.zeros((4, 4))
        for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                g, det_j = _quad4_gradient_matrix(coords, xi, eta)
                ke += det_j * thickness * coefficient * (g.T @ g)
        return ke
    if etype is ElementType.HEX8:
        ke = np.zeros((8, 8))
        for xi, eta, zeta in _HEX8_GAUSS:
            g, det_j = _hex8_gradient_matrix(coords, xi, eta, zeta)
            ke += det_j * coefficient * (g.T @ g)
        return ke
    raise ElementError(f"标量场传导暂不支持该单元类型: {etype}")


def element_scalar_gradient(
    etype: ElementType,
    coords: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """标量场单元梯度（TRIA3 常梯度精确，QUAD4/HEX8 高斯点平均）.

    Args:
        etype: 单元类型（支持 TRIA3 / QUAD4 / HEX8）。
        coords: 单元节点坐标 ``(n_node, 2)`` 或 ``(n_node, 3)``。
        values: 单元节点标量值 ``(n_node,)``（电压或温度）。

    Returns:
        梯度向量 ``(2,)`` 或 ``(3,)``。

    Raises:
        ElementError: 单元类型不支持或几何退化时抛出。
    """
    values = np.asarray(values, dtype=float)
    if etype is ElementType.TRIA3:
        g, _ = _tria3_gradient_matrix(coords)
        return g @ values
    if etype is ElementType.QUAD4:
        grad = np.zeros(2)
        for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                g, _ = _quad4_gradient_matrix(coords, xi, eta)
                grad += g @ values
        return grad / 4.0
    if etype is ElementType.HEX8:
        grad = np.zeros(3)
        for xi, eta, zeta in _HEX8_GAUSS:
            g, _ = _hex8_gradient_matrix(coords, xi, eta, zeta)
            grad += g @ values
        return grad / 8.0
    raise ElementError(f"标量场传导暂不支持该单元类型: {etype}")


def element_field_load(
    etype: ElementType,
    coords: np.ndarray,
    coefficient: float,
    values: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """场能一致节点载荷 ``∫ Nᵀ (c|∇φ|²) dΩ``（Joule 热的单元贡献）.

    Args:
        etype: 单元类型（支持 TRIA3 / QUAD4 / HEX8）。
        coords: 单元节点坐标 ``(n_node, 2)`` 或 ``(n_node, 3)``。
        coefficient: 传导系数（电导率）。
        values: 单元节点标量值 ``(n_node,)``（电压）。
        thickness: 厚度（mm，仅 2D 单元参与换算；HEX8 忽略）。

    Returns:
        一致节点载荷 ``(n_node,)``（TRIA3 常梯度解析式，等参单元高斯积分）。

    Raises:
        ElementError: 单元类型不支持或几何退化时抛出。
    """
    values = np.asarray(values, dtype=float)
    if etype is ElementType.TRIA3:
        g, area = _tria3_gradient_matrix(coords)
        grad = g @ values
        q = coefficient * float(grad @ grad)
        return q * thickness * area / 3.0 * np.ones(3)
    if etype is ElementType.QUAD4:
        load = np.zeros(4)
        for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                g, det_j = _quad4_gradient_matrix(coords, xi, eta)
                grad = g @ values
                q = coefficient * float(grad @ grad)
                shape = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                load += det_j * thickness * q * shape
        return load
    if etype is ElementType.HEX8:
        load = np.zeros(8)
        for xi, eta, zeta in _HEX8_GAUSS:
            g, det_j = _hex8_gradient_matrix(coords, xi, eta, zeta)
            grad = g @ values
            q = coefficient * float(grad @ grad)
            load += det_j * q * _hex8_shape_values(xi, eta, zeta)
        return load
    raise ElementError(f"标量场传导暂不支持该单元类型: {etype}")


def _quad4_gauss_data(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量 QUAD4 高斯点数据.

    :param coords: 单元节点坐标 ``(n, 4, 2)``。
    :return: ``(G, det)`` —— 四高斯点梯度矩阵 ``G (4, n, 2, 4)`` 与 ``|detJ| (4, n)``。
    """
    n = coords.shape[0]
    g_all = np.empty((4, n, 2, 4))
    det_all = np.empty((4, n))
    for point, (_shape, d_n) in enumerate(_QUAD4_GAUSS):
        jacob = np.einsum("ij,njk->nik", d_n, coords)
        det = np.linalg.det(jacob)
        if np.any(np.abs(det) <= _GEOM_TOL):
            raise ElementError("QUAD4 单元雅可比行列式接近零（单元退化或节点序错误）")
        g_all[point] = np.einsum("nij,jk->nik", np.linalg.inv(jacob), d_n)
        det_all[point] = np.abs(det)
    return g_all, det_all


def _tria3_batch_data(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量 TRIA3 梯度矩阵与面积.

    :param coords: 单元节点坐标 ``(n, 3, 2)``。
    :return: ``(G, area)`` —— 常梯度矩阵 ``G (n, 2, 3)`` 与面积 ``(n,)``。
    """
    x, y = coords[:, :, 0], coords[:, :, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    if np.any(area <= _GEOM_TOL):
        raise ElementError("TRIA3 单元退化为直线（面积接近零）")
    g = (
        np.stack(
            (
                np.stack((y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]), axis=1),
                np.stack((x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]), axis=1),
            ),
            axis=1,
        )
        / (2.0 * area)[:, None, None]
    )
    return g, area


def _batch_conductance(etype: ElementType, coords: np.ndarray, coefficient: float, thickness: float) -> np.ndarray:
    """批量单元传导矩阵（与 :func:`element_conductance` 同数值，块内向量化；HEX8 不乘厚度）."""
    if etype is ElementType.TRIA3:
        g, area = _tria3_batch_data(coords)
        return coefficient * thickness * area[:, None, None] * np.einsum("nia,nja->nij", g, g)
    if etype is ElementType.HEX8:
        g_all, det_all = _hex8_batch_data(coords)
        return coefficient * np.einsum("pn,pnki,pnkj->nij", det_all, g_all, g_all)
    g_all, det_all = _quad4_gauss_data(coords)
    return coefficient * thickness * np.einsum("pn,pnki,pnkj->nij", det_all, g_all, g_all)


def _batch_gradients(etype: ElementType, coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """批量单元标量梯度（TRIA3 常梯度精确，QUAD4/HEX8 高斯点平均）."""
    if etype is ElementType.TRIA3:
        g, _ = _tria3_batch_data(coords)
        return np.einsum("nij,nj->ni", g, values)
    if etype is ElementType.HEX8:
        g_all, _ = _hex8_batch_data(coords)
        return np.einsum("pnki,ni->pnk", g_all, values).mean(axis=0)
    g_all, _ = _quad4_gauss_data(coords)
    return np.einsum("pnki,ni->pnk", g_all, values).mean(axis=0)


def _batch_field_load(
    etype: ElementType, coords: np.ndarray, coefficient: float, values: np.ndarray, thickness: float
) -> np.ndarray:
    """批量场能一致节点载荷（与 :func:`element_field_load` 同数值；HEX8 不乘厚度）."""
    if etype is ElementType.TRIA3:
        g, area = _tria3_batch_data(coords)
        grad = np.einsum("nij,nj->ni", g, values)
        q = coefficient * np.einsum("ni,ni->n", grad, grad)
        return q[:, None] * thickness * area[:, None] / 3.0 * np.ones((coords.shape[0], 3))
    if etype is ElementType.HEX8:
        g_all, det_all = _hex8_batch_data(coords)
        load = np.zeros((coords.shape[0], 8))
        for point in range(8):
            grad = np.einsum("nki,ni->nk", g_all[point], values)
            q = coefficient * np.einsum("ni,ni->n", grad, grad)
            load += det_all[point][:, None] * q[:, None] * _HEX8_SHAPE[point][None, :]
        return load
    g_all, det_all = _quad4_gauss_data(coords)
    load = np.zeros((coords.shape[0], 4))
    for point, (n_shape, _d_n) in enumerate(_QUAD4_GAUSS):
        grad = np.einsum("nki,ni->nk", g_all[point], values)
        q = coefficient * np.einsum("ni,ni->n", grad, grad)
        load += det_all[point][:, None] * thickness * q[:, None] * n_shape[None, :]
    return load


def _batch_measures(etype: ElementType, coords: np.ndarray) -> np.ndarray:
    """批量连续体单元度量（TRIA3 面积；QUAD4 高斯 |detJ| 求和；HEX8 高斯 |detJ| 求和即体积）."""
    if etype is ElementType.TRIA3:
        return _tria3_batch_data(coords)[1]
    if etype is ElementType.HEX8:
        return _hex8_batch_data(coords)[1].sum(axis=0)
    return _quad4_gauss_data(coords)[1].sum(axis=0)


#: 绝缘块判定阈值：电导率低于该值的块不参与电场装配（V 形电阻陶瓷基底等）
_INSULATOR_SIGMA = 1.0e-9


def _validate_blocks(mesh: Mesh, materials: Sequence[ConductionMaterial], sections: Sequence[Section]) -> None:
    """校验各单元块类型与材料/截面索引范围（维数匹配由 Mesh 构造先行保证）."""
    for block in mesh.blocks:
        if block.etype not in _CONDUCTION_TYPES:
            raise MeshError(f"标量场传导暂不支持单元类型 {block.etype.value}")
        if not 0 <= block.material < len(materials):
            raise MeshError(f"单元块 {block.etype.value} 材料索引 {block.material} 越界（共 {len(materials)} 项）")
        if not 0 <= block.section < len(sections):
            raise MeshError(f"单元块 {block.etype.value} 截面索引 {block.section} 越界（共 {len(sections)} 项）")


def assemble_conduction(
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    field: str,
) -> csr_matrix:
    """装配全局标量场传导矩阵（CSR，每节点 1 DOF，块内向量化）.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表（ElementBlock.material 索引引用）。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。
        field: 取值字段，``"electric"`` 用电导率、``"thermal"`` 用导热系数；
            电导率低于 ``_INSULATOR_SIGMA`` 的绝缘块跳过装配（浮动节点由求解器接地）。

    Returns:
        全局传导矩阵 ``(n_nodes, n_nodes)`` CSR 稀疏矩阵。

    Raises:
        MeshError: 单元类型不支持或材料/截面索引越界时抛出。
        ElementError: 传导系数非正时抛出（经材料校验先行拦截）。
    """
    _validate_blocks(mesh, materials, sections)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for block in mesh.blocks:
        material = materials[block.material]
        thickness = sections[block.section].thickness
        coefficient = material.electric_sigma if field == "electric" else material.thermal_k
        if field == "electric" and coefficient < _INSULATOR_SIGMA:
            continue
        ke = _batch_conductance(block.etype, mesh.coords[block.conn], coefficient, thickness)
        n_dof_elem = ke.shape[1]
        rows.append(np.repeat(block.conn, n_dof_elem, axis=1).ravel())
        cols.append(np.tile(block.conn, (1, n_dof_elem)).ravel())
        values.append(ke.reshape(-1))
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.intp)
    value = np.concatenate(values) if values else np.empty(0)
    return csr_matrix((value, (row, col)), shape=(mesh.n_nodes, mesh.n_nodes))


def _batch_capacity(etype: ElementType, coords: np.ndarray, rho_cp: float, thickness: float) -> np.ndarray:
    """批量单元热容矩阵 ``∫ ρc Nᵀ N dΩ``（TRIA3 解析式，等参单元高斯积分；HEX8 不乘厚度）."""
    if etype is ElementType.TRIA3:
        area = _tria3_batch_data(coords)[1]
        pattern = np.ones((3, 3)) + np.eye(3)
        return rho_cp * thickness * (area / 12.0)[:, None, None] * pattern[None, :, :]
    if etype is ElementType.HEX8:
        _, det_all = _hex8_batch_data(coords)
        mass = np.zeros((coords.shape[0], 8, 8))
        for point in range(8):
            mass += det_all[point][:, None, None] * np.outer(_HEX8_SHAPE[point], _HEX8_SHAPE[point])[None, :, :]
        return rho_cp * mass
    _, det_all = _quad4_gauss_data(coords)
    mass = np.zeros((coords.shape[0], 4, 4))
    for point, (n_shape, _d_n) in enumerate(_QUAD4_GAUSS):
        mass += det_all[point][:, None, None] * np.outer(n_shape, n_shape)[None, :, :]
    return rho_cp * thickness * mass


def assemble_capacity(
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
) -> csr_matrix:
    """装配全局热容矩阵 ``∫ ρc Nᵀ N dΩ``（CSR，瞬态热 backward Euler 使用）.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表，取 ``volumetric_heat_capacity``（J/mm³·K）；
            数值为 0（未提供）的块贡献零热容，瞬态求解器负责整体校验。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。

    Returns:
        全局热容矩阵 ``(n_nodes, n_nodes)`` CSR 稀疏矩阵（对称正半定）。

    Raises:
        MeshError: 单元类型不支持或材料/截面索引越界时抛出。
    """
    _validate_blocks(mesh, materials, sections)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for block in mesh.blocks:
        rho_cp = materials[block.material].volumetric_heat_capacity
        if rho_cp <= 0.0:
            continue
        thickness = sections[block.section].thickness
        me = _batch_capacity(block.etype, mesh.coords[block.conn], rho_cp, thickness)
        n_dof_elem = me.shape[1]
        rows.append(np.repeat(block.conn, n_dof_elem, axis=1).ravel())
        cols.append(np.tile(block.conn, (1, n_dof_elem)).ravel())
        values.append(me.reshape(-1))
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.intp)
    value = np.concatenate(values) if values else np.empty(0)
    return csr_matrix((value, (row, col)), shape=(mesh.n_nodes, mesh.n_nodes))
