"""FEA 基准测试：悬臂梁 Q4 网格收敛性 + NAFEMS LE1 椭圆膜.

- 悬臂梁：δ_tip = P L³ / (3 E I)。CPS4 全积分单元存在剪切自锁，
  随网格细化位移单调收敛于解析解，用于验证整体求解链路正确性。
- NAFEMS LE1：椭圆膜外弧受外向拉力（1/4 对称模型），目标
  σyy(D) = 92.7 MPa（D 为内边界与 x 轴交点），验证边压力装配与
  平面应力求解的工程精度。
"""

from __future__ import annotations

import numpy as np

from zylab.fea import (
    Constraint,
    EdgePressure,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    StaticCase,
    StressState,
    element_stress_at,
    solve_static,
)


def _cantilever_mesh(nx: int, ny: int, length: float = 100.0, height: float = 1.0) -> Mesh:
    """悬臂梁规则 Q4 网格（左端固定，右端加载；细长梁 L/h=100 以弱化剪切自锁）."""
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    coords = np.array([[xs[i], ys[j]] for j in range(ny) for i in range(nx)])
    conn = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b = j * nx + i, j * nx + i + 1
            c, d = (j + 1) * nx + i + 1, (j + 1) * nx + i
            conn.append([a, b, c, d])
    return Mesh(coords, (ElementBlock(ElementType.QUAD4, np.array(conn)),))


def _solve_cantilever(nx: int, ny: int) -> float:
    """求解悬臂梁并返回端部中点竖向位移."""
    mesh = _cantilever_mesh(nx, ny)
    material = LinearElastic(1000.0, 0.0, StressState.PLANE_STRESS)
    load_p = 1.0
    constraints = tuple(Constraint(node, (0, 1)) for node in range(ny))
    tip_nodes = [j * nx + (nx - 1) for j in range(ny)]
    loads = tuple(NodalLoad(node, (0.0, -load_p / len(tip_nodes))) for node in tip_nodes)
    solution = solve_static(mesh, [material], [Section(thickness=1.0)], StaticCase(constraints, loads))
    mid_tip = tip_nodes[len(tip_nodes) // 2]
    return float(solution.node_displacement(mid_tip)[1])


class TestCantileverConvergence:
    def test_tip_displacement_converges_to_beam_theory(self) -> None:
        """细化网格端部挠度收敛于梁理论解析解.

        CPS4 全积分存在剪切自锁，须保持单元长宽比≈1 细化；
        实测误差 34%（101x2）→ 14%（201x4）→ 7%（401x8），呈 O(h) 收敛。
        """
        # 解析解：δ = P L³ / (3 E I)，I = t h³ / 12 = 1/12
        delta_exact = 1.0 * 100.0**3 / (3.0 * 1000.0 * (1.0 / 12.0))

        coarse = _solve_cantilever(nx=101, ny=2)
        medium = _solve_cantilever(nx=201, ny=4)
        fine = _solve_cantilever(nx=401, ny=8)

        # 单调收敛趋向解析解，细网格误差 < 10%
        assert abs(coarse + delta_exact) > abs(medium + delta_exact) > abs(fine + delta_exact)
        assert abs(fine - (-delta_exact)) / delta_exact < 0.10

    def test_strain_energy_positive(self) -> None:
        """应变能须为正（能量范数合理性检查）."""
        mesh = _cantilever_mesh(nx=8, ny=4)
        material = LinearElastic(1000.0, 0.0, StressState.PLANE_STRESS)
        constraints = tuple(Constraint(node, (0, 1)) for node in range(4))
        loads = tuple(NodalLoad(1 * 8 + 7, (0.0, -0.5)) for _ in range(1))
        solution = solve_static(mesh, [material], [Section(thickness=1.0)], StaticCase(constraints, loads))
        assert solution.strain_energy > 0.0

    def test_reaction_equilibrium(self) -> None:
        """固定端反力之和与外载荷平衡."""
        mesh = _cantilever_mesh(nx=8, ny=4)
        material = LinearElastic(1000.0, 0.0, StressState.PLANE_STRESS)
        constraints = tuple(Constraint(node, (0, 1)) for node in range(4))
        loads = (NodalLoad(1 * 8 + 7, (0.0, -1.0)),)
        solution = solve_static(mesh, [material], [Section(thickness=1.0)], StaticCase(constraints, loads))
        # 固定端节点 0..3 的 y 反力（dof = node*2+1）之和 = +1.0
        total = sum(solution.reactions.get(node * 2 + 1, 0.0) for node in range(4))
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# NAFEMS LE1：椭圆膜外向拉力（1/4 对称模型）
# ---------------------------------------------------------------------------

# 几何：内椭圆半轴 2.0x1.0，外椭圆半轴 3.25x2.75；外弧 10 MPa 外向拉力；
# 对称约束（x=0 面 ux=0，y=0 面 uy=0）；D=(2,0) 目标 σyy = 92.7 MPa。
_LE1_INNER = (2.0, 1.0)
_LE1_OUTER = (3.25, 2.75)
_LE1_TARGET = 92.7


def _le1_mesh(n_theta: int, n_radial: int) -> Mesh:
    """LE1 椭圆环 1/4 对称 Q4 网格（极角-径向双层参数化，径向线性插值）.

    节点编号 node(j, i) = j*(n_theta+1) + i：j 为径向层（0=内边界），
    i 为极角列（0=θ=0 即 y=0 对称面，n_theta=θ=π/2 即 x=0 对称面）。
    """
    thetas = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    fracs = np.linspace(0.0, 1.0, n_radial + 1)
    coords = np.empty(((n_radial + 1) * (n_theta + 1), 2))
    for j, s in enumerate(fracs):
        for i, theta in enumerate(thetas):
            xi, yi = _LE1_INNER[0] * np.cos(theta), _LE1_INNER[1] * np.sin(theta)
            xo, yo = _LE1_OUTER[0] * np.cos(theta), _LE1_OUTER[1] * np.sin(theta)
            coords[j * (n_theta + 1) + i] = [(1.0 - s) * xi + s * xo, (1.0 - s) * yi + s * yo]
    conn = []
    for j in range(n_radial):
        for i in range(n_theta):
            a = j * (n_theta + 1) + i
            b, d = a + 1, (j + 1) * (n_theta + 1) + i
            c = d + 1
            conn.append([a, b, c, d])
    return Mesh(coords, (ElementBlock(ElementType.QUAD4, np.array(conn)),))


def _solve_le1(n_theta: int, n_radial: int) -> float:
    """求解 LE1 并返回 D 点（内边界与 x 轴交点）的 σyy（角点直接求值）.

    D 为单元 (j=0, i=0) 的节点 0（自然坐标 -1, -1）；节点平均恢复
    （高斯点平均）在角点单单元参与下系统性低估，故用角点求值。
    """
    mesh = _le1_mesh(n_theta, n_radial)
    material = LinearElastic(210000.0, 0.3, StressState.PLANE_STRESS)
    section = Section(thickness=0.1)
    width = n_theta + 1
    constraints = (
        # y=0 对称面（i=0 列）：uy=0
        *(Constraint(j * width, (1,)) for j in range(n_radial + 1)),
        # x=0 对称面（i=n_theta 列）：ux=0
        *(Constraint(j * width + n_theta, (0,)) for j in range(n_radial + 1)),
    )
    # 外弧（j=n_radial 层）沿 θ 增向行进，材料在左；外向拉力取负压（10 MPa * t=0.1）
    outer_nodes = tuple(n_radial * width + i for i in range(n_theta + 1))
    case = StaticCase(
        constraints=constraints,
        edge_pressures=(EdgePressure(outer_nodes, -10.0 * 0.1),),
    )
    solution = solve_static(mesh, [material], [section], case)
    conn = mesh.blocks[0].conn[0]
    u_elem = solution.displacements[conn].ravel()
    return float(element_stress_at(ElementType.QUAD4, mesh.coords[conn], material, u_elem, (-1.0, -1.0))[1])


class TestNafemsLe1:
    def test_sigma_yy_at_d_converges(self) -> None:
        """D 点 σyy 收敛于 NAFEMS 目标 92.7 MPa（Q4 双线性单元 +2% 量级）.

        实测：8x4 网格 -3.3% → 32x8 网格 +2.2%，随细化稳定在
        +2.3% 附近（双线性单元对该应力集中的离散误差水平）。
        """
        coarse = _solve_le1(n_theta=8, n_radial=4)
        fine = _solve_le1(n_theta=32, n_radial=8)

        assert abs(coarse - _LE1_TARGET) / _LE1_TARGET < 0.05
        assert abs(fine - _LE1_TARGET) / _LE1_TARGET < 0.03
        assert fine > coarse  # 自下逼近目标
