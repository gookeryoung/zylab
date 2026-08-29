"""单元库：刚度矩阵、质量矩阵与应力恢复（v1 静力/模态内核）.

支持单元族：
- TRUSS2：2 节点杆（平面/空间），每节点 dim 个平动自由度；
- BEAM2：2 节点平面 Euler-Bernoulli 梁，每节点 3 自由度（ux/uy/θz）；
- TRIA3：常应变三角形（CST），厚度截面；
- QUAD4：4 节点等参四边形，2x2 高斯全积分；
- TET4：常应变四面体；
- HEX8：8 节点等参六面体，2x2x2 高斯全积分。

应变分量顺序：平面 (εxx, εyy, γxy)，空间 (εxx, εyy, εzz, γxy, γyz, γxz)。
单元自由度顺序：节点优先，节点内按 x/y/z 分量（梁附转角分量）。
质量矩阵为一致质量（∫ ρ N^T N dV），杆/梁为解析公式，等参单元高斯数值积分。
"""

from __future__ import annotations

import numpy as np

from .errors import ElementError
from .material import LinearElastic, Section
from .mesh import ElementType

__all__ = ["element_mass", "element_measure", "element_stiffness", "element_stress", "element_stress_at"]

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


def _truss2_mass(coords: np.ndarray, density: float, area: float) -> np.ndarray:
    """杆单元一致质量（ρAL/6 [2,1;1,2] 各向同性块对角展开到全局）.

    质量无方向性，不能沿方向余弦投影变换（否则横向自由度质量为零，
    质量矩阵奇异）；与刚度矩阵的方向余弦变换不同，此处按节点块直接展开。
    """
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("杆单元两节点重合，长度为零")
    dim = coords.shape[1]
    m_scalar = density * area * length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return np.kron(m_scalar, np.eye(dim))


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
# BEAM2：2 节点平面 Euler-Bernoulli 梁（ux/uy/θz）
# ---------------------------------------------------------------------------


def _beam2_stiffness(coords: np.ndarray, e_modulus: float, area: float, inertia: float) -> np.ndarray:
    """平面梁单元刚度（全局坐标，6x6，自由度序 u1x/u1y/θ1/u2x/u2y/θ2）."""
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("梁单元两节点重合，长度为零")
    c, s = float(delta[0] / length), float(delta[1] / length)
    ea, ei = e_modulus * area, e_modulus * inertia
    axial = ea / length
    shear = 12.0 * ei / length**3
    bend = 6.0 * ei / length**2
    rot = 4.0 * ei / length
    rot2 = 2.0 * ei / length
    k_local = np.array(
        [
            [axial, 0.0, 0.0, -axial, 0.0, 0.0],
            [0.0, shear, bend, 0.0, -shear, bend],
            [0.0, bend, rot, 0.0, -bend, rot2],
            [-axial, 0.0, 0.0, axial, 0.0, 0.0],
            [0.0, -shear, -bend, 0.0, shear, -bend],
            [0.0, bend, rot2, 0.0, -bend, rot],
        ]
    )
    # 坐标变换（局部 -> 全局）：节点块 [c s 0; -s c 0; 0 0 1]
    t = np.zeros((6, 6))
    for node in (0, 1):
        base = 3 * node
        t[base, base] = c
        t[base, base + 1] = s
        t[base + 1, base] = -s
        t[base + 1, base + 1] = c
        t[base + 2, base + 2] = 1.0
    return t.T @ k_local @ t


def _beam2_mass(coords: np.ndarray, density: float, area: float) -> np.ndarray:
    """平面梁单元一致质量（轴向线性插值 + 横向 Hermite 插值，经坐标变换）.

    转动惯量项（回转半径平方乘质量）未计入（Euler-Bernoulli 细梁常规做法）。
    """
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("梁单元两节点重合，长度为零")
    c, s = float(delta[0] / length), float(delta[1] / length)
    r = density * area * length
    m_local = r * np.array(
        [
            [1.0 / 3.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0],
            [0.0, 13.0 / 35.0, 11.0 * length / 210.0, 0.0, 9.0 / 70.0, -13.0 * length / 420.0],
            [0.0, 11.0 * length / 210.0, length**2 / 105.0, 0.0, 13.0 * length / 420.0, -(length**2) / 210.0],
            [1.0 / 6.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0],
            [0.0, 9.0 / 70.0, 13.0 * length / 420.0, 0.0, 13.0 / 35.0, -11.0 * length / 210.0],
            [0.0, -13.0 * length / 420.0, -(length**2) / 210.0, 0.0, -11.0 * length / 210.0, length**2 / 105.0],
        ]
    )
    t = np.zeros((6, 6))
    for node in (0, 1):
        base = 3 * node
        t[base, base] = c
        t[base, base + 1] = s
        t[base + 1, base] = -s
        t[base + 1, base + 1] = c
        t[base + 2, base + 2] = 1.0
    return t.T @ m_local @ t


def _beam2_stress(
    coords: np.ndarray,
    e_modulus: float,
    inertia: float,
    u_elem: np.ndarray,
) -> np.ndarray:
    """梁单元内力：轴向应力与两端弯矩.

    Returns:
        (3,) 向量 = (轴向应力 σ, 端 1 弯矩 M1, 端 2 弯矩 M2)；弯矩逆时针为正。
    """
    delta = coords[1] - coords[0]
    length = float(np.linalg.norm(delta))
    if length <= _GEOM_TOL:
        raise ElementError("梁单元两节点重合，长度为零")
    c, s = float(delta[0] / length), float(delta[1] / length)
    t = np.zeros((6, 6))
    for node in (0, 1):
        base = 3 * node
        t[base, base] = c
        t[base, base + 1] = s
        t[base + 1, base] = -s
        t[base + 1, base + 1] = c
        t[base + 2, base + 2] = 1.0
    u_local = t @ np.asarray(u_elem, dtype=float)
    axial_stress = e_modulus * (u_local[3] - u_local[0]) / length
    # 端部弯矩 = EI * v''(端部)，由 Hermite 插值 v'' = 1/L² (-6v1 + 2Lθ1... )，
    # 等价于局部平衡方程 M1 = EI(4θ1 + 2θ2)/L - 6EI(v2 - v1)/L²
    ei = e_modulus * inertia
    curvature1 = (-6.0 * (u_local[4] - u_local[1]) + (4.0 * u_local[2] + 2.0 * u_local[5]) * length) / length**2
    curvature2 = (-6.0 * (u_local[4] - u_local[1]) + (2.0 * u_local[2] + 4.0 * u_local[5]) * length) / length**2
    return np.array([axial_stress, ei * curvature1, ei * curvature2])


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


def _tria3_mass(coords: np.ndarray, density: float, thickness: float) -> np.ndarray:
    """CST 一致质量（ρtA/12 [2,1,1;...] 标量块展开为 2 DOF 块对角）."""
    _, area = _tria3_b_matrix(coords)
    m_scalar = (
        density
        * thickness
        * area
        / 12.0
        * np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        )
    )
    return np.kron(m_scalar, np.eye(2))


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


def _quad4_shape_values(xi: float, eta: float) -> np.ndarray:
    """Q4 形函数值 (4,)（节点顺序与导数函数一致）."""
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
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


def _quad4_mass(coords: np.ndarray, density: float, thickness: float) -> np.ndarray:
    """Q4 一致质量（2x2 高斯积分 ρt Σ w |J| N^T N，标量块展开为 2 DOF）."""
    m_scalar = np.zeros((4, 4))
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            n = _quad4_shape_values(xi, eta)
            _, det_j = _quad4_b_matrix(coords, xi, eta)
            m_scalar += det_j * np.outer(n, n)
    return np.kron(density * thickness * m_scalar, np.eye(2))


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


def _tet4_mass(coords: np.ndarray, density: float) -> np.ndarray:
    """TET4 一致质量（ρV/20 [2,1,1,1;...] 标量块展开为 3 DOF）."""
    _, volume = _tet4_geometry(coords)
    m_scalar = density * volume / 20.0 * (np.eye(4) + np.ones((4, 4)))
    return np.kron(m_scalar, np.eye(3))


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


def _hex8_shape_values(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形函数值 (8,)（节点顺序与符号表一致）."""
    signs = _HEX8_SIGNS
    return np.array([(1.0 + xi * sx) * (1.0 + eta * se) * (1.0 + zeta * sz) / 8.0 for sx, se, sz in signs])


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


def _hex8_mass(coords: np.ndarray, density: float) -> np.ndarray:
    """HEX8 一致质量（2x2x2 高斯积分 ρ Σ w |J| N^T N，标量块展开为 3 DOF）."""
    m_scalar = np.zeros((8, 8))
    for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
        for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for zeta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                n = _hex8_shape_values(xi, eta, zeta)
                _, det_j = _hex8_b_matrix(coords, xi, eta, zeta)
                m_scalar += det_j * np.outer(n, n)
    return np.kron(density * m_scalar, np.eye(3))


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


def element_measure(etype: ElementType, coords: np.ndarray) -> float:
    """连续体单元度量（2D 为面积，3D 为体积；供体力等效节点载荷）.

    Args:
        etype: 单元类型（连续体族；杆/梁无度量，抛错）。
        coords: 单元节点坐标 ``(n_node, dim)``。

    Returns:
        单元面积或体积。

    Raises:
        ElementError: 非连续体单元或几何退化时抛出。
    """
    if etype is ElementType.TRIA3:
        _, area = _tria3_b_matrix(coords)
        return area
    if etype is ElementType.QUAD4:
        # 2x2 高斯 |detJ| 求和（直边时等于面积，曲边为积分近似）
        total = 0.0
        for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                _, det_j = _quad4_b_matrix(coords, xi, eta)
                total += det_j
        return total
    if etype is ElementType.TET4:
        _, volume = _tet4_geometry(coords)
        return volume
    if etype is ElementType.HEX8:
        total = 0.0
        for xi in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
            for eta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                for zeta in (-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA):
                    _, det_j = _hex8_b_matrix(coords, xi, eta, zeta)
                    total += det_j
        return total
    raise ElementError(f"单元类型 {etype} 不是连续体单元，无面积/体积度量")


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
    if etype is ElementType.BEAM2:
        return _beam2_stiffness(coords, material.e_modulus, section.area, section.inertia)
    if etype is ElementType.TRIA3:
        return _tria3_stiffness(coords, material.d_matrix(), section.thickness)
    if etype is ElementType.QUAD4:
        return _quad4_stiffness(coords, material.d_matrix(), section.thickness)
    if etype is ElementType.TET4:
        return _tet4_stiffness(coords, material.d_matrix())
    if etype is ElementType.HEX8:
        return _hex8_stiffness(coords, material.d_matrix())
    raise ElementError(f"不支持的单元类型: {etype}")  # pragma: no cover（枚举闭合）


def element_mass(
    etype: ElementType,
    coords: np.ndarray,
    material: LinearElastic,
    section: Section,
) -> np.ndarray:
    """按单元类型计算一致质量矩阵（全局坐标系）.

    杆/梁为解析公式，等参单元为高斯数值积分 ``∫ ρ N^T N dV``。

    Args:
        etype: 单元类型。
        coords: 单元节点坐标 ``(n_node, dim)``。
        material: 线弹性材料（须配置正的质量密度）。
        section: 截面属性（杆/梁取面积，平面单元取厚度）。

    Returns:
        单元质量矩阵，形状与单元刚度矩阵一致 ``(n_node*每节点 DOF, ...)``。

    Raises:
        ElementError: 密度非正或单元几何退化时抛出。
    """
    if material.density <= 0.0:
        raise ElementError(f"模态分析须提供正的质量密度，实际 rho={material.density}")
    if etype is ElementType.TRUSS2:
        return _truss2_mass(coords, material.density, section.area)
    if etype is ElementType.BEAM2:
        return _beam2_mass(coords, material.density, section.area)
    if etype is ElementType.TRIA3:
        return _tria3_mass(coords, material.density, section.thickness)
    if etype is ElementType.QUAD4:
        return _quad4_mass(coords, material.density, section.thickness)
    if etype is ElementType.TET4:
        return _tet4_mass(coords, material.density)
    if etype is ElementType.HEX8:
        return _hex8_mass(coords, material.density)
    raise ElementError(f"不支持的单元类型: {etype}")  # pragma: no cover（枚举闭合）


def element_stress(
    etype: ElementType,
    coords: np.ndarray,
    material: LinearElastic,
    u_elem: np.ndarray,
    section: Section | None = None,
) -> np.ndarray:
    """按单元类型计算单元应力（常应变单元直接给出，等参单元取高斯点平均）.

    Args:
        etype: 单元类型。
        coords: 单元节点坐标 ``(n_node, dim)``。
        material: 线弹性材料。
        u_elem: 单元节点位移向量（全局坐标系，长度 = n_node*每节点 DOF 数）。
        section: 截面属性（梁单元恢复弯矩须提供）。

    Returns:
        应力向量：杆为 (1,)（轴向应力），梁为 (3,)（轴向应力 + 两端弯矩），
        平面单元为 (3,)，空间单元为 (6,)。
    """
    if etype is ElementType.TRUSS2:
        return np.array([_truss2_axial_stress(coords, material.e_modulus, u_elem)])
    if etype is ElementType.BEAM2:
        if section is None:
            raise ElementError("梁单元应力恢复须提供截面属性（惯性矩）")
        return _beam2_stress(coords, material.e_modulus, section.inertia, u_elem)
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


def element_stress_at(
    etype: ElementType,
    coords: np.ndarray,
    material: LinearElastic,
    u_elem: np.ndarray,
    location: tuple[float, float],
) -> np.ndarray:
    """等参单元内指定自然坐标处的应力（v1 支持 QUAD4）.

    供边界/角点应力恢复使用（:func:`element_stress` 的高斯点平均在
    应力梯度大的边界角点处系统性低估，角点直接求值无此偏差）。

    Args:
        etype: 单元类型（v1 仅 QUAD4）。
        coords: 单元节点坐标 ``(4, 2)``。
        material: 线弹性材料。
        u_elem: 单元节点位移向量（长度 8）。
        location: 自然坐标 ``(xi, eta)``（节点 0 处为 (-1, -1)）。

    Returns:
        该点应力向量 ``(3,)``（εxx, εyy, γxy 序对应的应力）。

    Raises:
        ElementError: 单元类型不支持或雅可比退化时抛出。
    """
    if etype is not ElementType.QUAD4:
        raise ElementError(f"单元类型 {etype} 暂不支持指定点应力求值")
    b, _ = _quad4_b_matrix(coords, location[0], location[1])
    return material.d_matrix() @ (b @ u_elem)
