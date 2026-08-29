"""单元库：刚度矩阵与应力恢复（v1 静力内核）.

支持单元族：
- TRUSS2：2 节点杆（平面/空间），每节点 dim 个平动自由度；
- TRIA3：常应变三角形（CST），厚度截面；
- QUAD4：4 节点等参四边形，2x2 高斯全积分；
- TET4：常应变四面体；
- HEX8：8 节点等参六面体，2x2x2 高斯全积分。

应变分量顺序：平面 (εxx, εyy, γxy)，空间 (εxx, εyy, εzz, γxy, γyz, γxz)。
单元自由度顺序：节点优先，节点内按 x/y/z 分量。
"""

from __future__ import annotations

import numpy as np

from .errors import ElementError
from .material import LinearElastic, Section
from .mesh import ElementType

__all__ = ["element_stiffness", "element_stress"]

_GEOM_TOL = 1.0e-12

# 2 点高斯积分位置与权（一维）
_GAUSS_ABSCISSA = 1.0 / np.sqrt(3.0)


# ---------------------------------------------------------------------------
# TRUSS2：2 节点杆
# ---------------------------------------------------------------------------


def _truss2_stiffness(coords: np.ndarray, e_modulus: float, area: float) -> np.ndarray:
    """杆单元刚度（全局坐标，自由度 = 每节点 dim 个）."""
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("杆单元两节点重合，长度为零")
    dim = coords.shape[1]
    c = delta / length  # 方向余弦
    k_local = e_modulus * area / length * np.array([[1.0, -1.0], [-1.0, 1.0]])
    # 变换矩阵 T（局部 2 自由度 -> 全局 2*dim）：每节点方向余弦重复
    t = np.zeros((2, 2 * dim))
    t[0, :dim] = c
    t[1, dim:] = c
    return t.T @ k_local @ t


def _truss2_axial_stress(coords: np.ndarray, e_modulus: float, u_elem: np.ndarray) -> float:
    """杆单元轴向应力（拉为正）."""
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("杆单元两节点重合，长度为零")
    dim = coords.shape[1]
    elongation = float(np.dot(delta / length, u_elem[dim:] - u_elem[:dim]))
    return e_modulus * elongation / length


# ---------------------------------------------------------------------------
# TRIA3：常应变三角形（CST）
# ---------------------------------------------------------------------------


def _tria3_b_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """CST 应变矩阵 B (3, 6) 与面积.

    Returns:
        (B, area)：B 为常应变矩阵，area 为三角形面积（取绝对值）。
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area <= _GEOM_TOL:
        raise ElementError("TRIA3 单元退化为直线（面积接近零）")
    b = np.array(
        [
            [y2 - y3, 0.0, y3 - y1, 0.0, y1 - y2, 0.0],
            [0.0, x3 - x2, 0.0, x1 - x3, 0.0, x2 - x1],
            [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2],
        ]
    ) / (2.0 * area)
    return b, area


def _tria3_stiffness(coords: np.ndarray, dmat: np.ndarray, thickness: float) -> np.ndarray:
    """CST 单元刚度（平面，自由度 = 每节点 2 个）."""
    b, area = _tria3_b_matrix(coords)
    return thickness * area * (b.T @ dmat @ b)


# ---------------------------------------------------------------------------
# QUAD4：4 节点等参四边形（2x2 高斯）
# ---------------------------------------------------------------------------


def _quad4_shape_derivs(xi: float, eta: float) -> np.ndarray:
    """Q4 形函数对自然坐标的导数 (2, 4)：行 0 为 dN/dxi，行 1 为 dN/deta."""
    return 0.25 * np.array(
        [
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ]
    )


def _quad4_b_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """Q4 在指定高斯点的应变矩阵 B (3, 8) 与 |detJ|.

    Returns:
        (B, det_j)：det_j 为雅可比行列式绝对值。
    """
    dn = _quad4_shape_derivs(xi, eta)
    jacob = dn @ coords
    det_j = float(np.linalg.det(jacob))
    if abs(det_j) <= _GEOM_TOL:
        raise ElementError("QUAD4 单元雅可比行列式接近零（单元退化或节点序错误）")
    dn_xy = np.linalg.inv(jacob) @ dn
    b = np.zeros((3, 8))
    b[0, 0::2] = dn_xy[0]
    b[1, 1::2] = dn_xy[1]
    b[2, 0::2] = dn_xy[1]
    b[2, 1::2] = dn_xy[0]
    return b, abs(det_j)


def _quad4_stiffness(coords: np.ndarray, dmat: np.ndarray, thickness: float) -> np.ndarray:
    """Q4 单元刚度（2x2 高斯全积分）."""
    ke = np.zeros((8, 8))
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            b, det_j = _quad4_b_matrix(coords, xi, eta)
            ke += det_j * thickness * (b.T @ dmat @ b)
    return ke


def _quad4_stress(coords: np.ndarray, dmat: np.ndarray, u_elem: np.ndarray) -> np.ndarray:
    """Q4 高斯点应力取平均（(3,) 向量）."""
    stress = np.zeros(3)
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            b, _ = _quad4_b_matrix(coords, xi, eta)
            stress += dmat @ (b @ u_elem)
    return stress / 4.0


# ---------------------------------------------------------------------------
# TET4：常应变四面体
# ---------------------------------------------------------------------------


def _tet4_geometry(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """TET4 应变矩阵 B (6, 12) 与体积（形函数梯度经逆矩阵求取）.

    Returns:
        (B, volume)：volume 为四面体体积（取绝对值）。
    """
    m = np.column_stack([np.ones(4), coords])  # (4, 4)：[1, x, y, z]
    det_m = float(np.linalg.det(m))
    volume = abs(det_m) / 6.0
    if volume <= _GEOM_TOL:
        raise ElementError("TET4 单元退化为平面（体积接近零）")
    # inv(M) 的第 i 列 = [a_i, b_i, c_i, d_i]，即 N_i = a_i + b_i*x + c_i*y + d_i*z 的系数
    coeffs = np.linalg.inv(m).T  # (4, 4)：行 i 为节点 i 的形函数系数
    grad = coeffs[:, 1:]  # (4, 3)：行 i 为 grad N_i = (b_i, c_i, d_i)
    b = np.zeros((6, 12))
    for i in range(4):
        bi, ci, di = grad[i]
        col = 3 * i
        b[0, col] = bi
        b[1, col + 1] = ci
        b[2, col + 2] = di
        b[3, col] = ci
        b[3, col + 1] = bi
        b[4, col + 1] = di
        b[4, col + 2] = ci
        b[5, col] = di
        b[5, col + 2] = bi
    return b, volume


def _tet4_stiffness(coords: np.ndarray, dmat: np.ndarray) -> np.ndarray:
    """TET4 单元刚度（常应变）."""
    b, volume = _tet4_geometry(coords)
    return volume * (b.T @ dmat @ b)


# ---------------------------------------------------------------------------
# HEX8：8 节点等参六面体（2x2x2 高斯）
# ---------------------------------------------------------------------------

# HEX8 节点自然坐标符号表（节点顺序：z 面 1-4（ζ=-1），z 面 5-8（ζ=+1））
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


def _hex8_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形函数对自然坐标的导数 (3, 8)."""
    signs = _HEX8_SIGNS
    dn = np.zeros((3, 8))
    for i in range(8):
        sx, se, sz = signs[i]
        dn[0, i] = sx * (1.0 + eta * se) * (1.0 + zeta * sz) / 8.0
        dn[1, i] = se * (1.0 + xi * sx) * (1.0 + zeta * sz) / 8.0
        dn[2, i] = sz * (1.0 + xi * sx) * (1.0 + eta * se) / 8.0
    return dn


def _hex8_b_matrix(coords: np.ndarray, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, float]:
    """HEX8 在指定高斯点的应变矩阵 B (6, 24) 与 |detJ|.

    Returns:
        (B, det_j)：det_j 为雅可比行列式绝对值。
    """
    dn = _hex8_shape_derivs(xi, eta, zeta)
    jacob = dn @ coords  # (3, 3)
    det_j = float(np.linalg.det(jacob))
    if abs(det_j) <= _GEOM_TOL:
        raise ElementError("HEX8 单元雅可比行列式接近零（单元退化或节点序错误）")
    dn_xyz = np.linalg.inv(jacob) @ dn
    b = np.zeros((6, 24))
    for i in range(8):
        gx, gy, gz = dn_xyz[:, i]
        col = 3 * i
        b[0, col] = gx
        b[1, col + 1] = gy
        b[2, col + 2] = gz
        b[3, col] = gy
        b[3, col + 1] = gx
        b[4, col + 1] = gz
        b[4, col + 2] = gy
        b[5, col] = gz
        b[5, col + 2] = gx
    return b, abs(det_j)


def _hex8_stiffness(coords: np.ndarray, dmat: np.ndarray) -> np.ndarray:
    """HEX8 单元刚度（2x2x2 高斯全积分）."""
    ke = np.zeros((24, 24))
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for zeta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                b, det_j = _hex8_b_matrix(coords, xi, eta, zeta)
                ke += det_j * (b.T @ dmat @ b)
    return ke


def _hex8_stress(coords: np.ndarray, dmat: np.ndarray, u_elem: np.ndarray) -> np.ndarray:
    """HEX8 高斯点应力取平均（(6,) 向量）."""
    stress = np.zeros(6)
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for zeta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                b, _ = _hex8_b_matrix(coords, xi, eta, zeta)
                stress += dmat @ (b @ u_elem)
    return stress / 8.0


# ---------------------------------------------------------------------------
# 公共分发入口
# ---------------------------------------------------------------------------


def element_stiffness(
    etype: ElementType,
    coords: np.ndarray,
    material: LinearElastic,
    section: Section,
) -> np.ndarray:
    """按单元类型计算单元刚度矩阵（全局坐标系）.

    Args:
        etype: 单元类型。
        coords: 单元节点坐标 ``(n_node, dim)``。
        material: 线弹性材料（平面单元须配置对应 StressState）。
        section: 截面属性（杆取面积，平面单元取厚度）。

    Returns:
        单元刚度矩阵，形状 ``(n_node*dim, n_node*dim)``。

    Raises:
        ElementError: 单元几何退化或雅可比非正时抛出。
    """
    if etype is ElementType.TRUSS2:
        return _truss2_stiffness(coords, material.e_modulus, section.area)
    if etype is ElementType.TRIA3:
        return _tria3_stiffness(coords, material.d_matrix(), section.thickness)
    if etype is ElementType.QUAD4:
        return _quad4_stiffness(coords, material.d_matrix(), section.thickness)
    if etype is ElementType.TET4:
        return _tet4_stiffness(coords, material.d_matrix())
    if etype is ElementType.HEX8:
        return _hex8_stiffness(coords, material.d_matrix())
    raise ElementError(f"不支持的单元类型: {etype}")  # pragma: no cover（枚举闭合）


def element_stress(
    etype: ElementType,
    coords: np.ndarray,
    material: LinearElastic,
    u_elem: np.ndarray,
) -> np.ndarray:
    """按单元类型计算单元应力（常应变单元直接给出，等参单元取高斯点平均）.

    Args:
        etype: 单元类型。
        coords: 单元节点坐标 ``(n_node, dim)``。
        material: 线弹性材料。
        u_elem: 单元节点位移向量（全局坐标系，长度 = n_node*dim）。

    Returns:
        应力向量：杆为 (1,)（轴向应力），平面单元为 (3,)，空间单元为 (6,)。
    """
    if etype is ElementType.TRUSS2:
        return np.array([_truss2_axial_stress(coords, material.e_modulus, u_elem)])
    if etype is ElementType.TRIA3:
        b, _ = _tria3_b_matrix(coords)
        return material.d_matrix() @ (b @ u_elem)
    if etype is ElementType.QUAD4:
        return _quad4_stress(coords, material.d_matrix(), u_elem)
    if etype is ElementType.TET4:
        b, _ = _tet4_geometry(coords)
        return material.d_matrix() @ (b @ u_elem)
    if etype is ElementType.HEX8:
        return _hex8_stress(coords, material.d_matrix(), u_elem)
    raise ElementError(f"不支持的单元类型: {etype}")  # pragma: no cover（枚举闭合）
