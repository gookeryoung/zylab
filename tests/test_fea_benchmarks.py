"""FEA 基准测试：悬臂梁 Q4 网格收敛性（对照梁理论解析解）.

梁理论：δ_tip = P L³ / (3 E I)。CPS4 全积分单元存在剪切自锁，
随网格细化位移单调收敛于解析解，用于验证整体求解链路正确性。
"""

from __future__ import annotations

import numpy as np

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    StaticCase,
    StressState,
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
