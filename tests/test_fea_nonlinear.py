"""fea.nonlinear / 几何非线性 TRUSS2 单元与 Newton-Raphson 求解测试.

验证维度：Green-Lagrange 内力刚体转动不变性、切线刚度数值差分一致性、
单杆/两杆桁架大位移解析平衡、Newton 迭代收敛历程、错误路径。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Constraint,
    EdgePressure,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    SolverError,
    StaticCase,
    solve_nonlinear_static,
    truss2_internal_force,
    truss2_tangent_stiffness,
)

E_MOD, AREA = 1000.0, 2.0


def _two_bar_mesh(b: float, h: float) -> Mesh:
    """对称两杆桁架：A(-b,0) C(0,-h) B(b,0)，顶点 C 为节点 1."""
    coords = np.array([[-b, 0.0], [0.0, -h], [b, 0.0]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int64)
    block = ElementBlock(etype=ElementType.TRUSS2, conn=conn, name="两杆")
    return Mesh(coords=coords, blocks=(block,))


# ---------------------------------------------------------------------------
# 单元级：内力与切线刚度
# ---------------------------------------------------------------------------


class TestTruss2NonlinearElement:
    def test_rigid_rotation_zero_force(self) -> None:
        """刚体转动（L = L0）内力恰为 0（Green-Lagrange 几何精确性）."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        center = coords.mean(axis=0)
        for angle in (0.3, 1.2, 2.7):
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated = (coords - center) @ rot.T + center
            u_elem = (rotated - coords).ravel()
            f_int = truss2_internal_force(coords, u_elem, E_MOD, AREA)
            np.testing.assert_allclose(f_int, 0.0, atol=1e-10)

    def test_axial_stretch_analytical(self) -> None:
        """单杆拉伸位移 u：内力 = EA·u(2L0+u)(L0+u)/(2L0³)（精确解析）."""
        length0 = 2.0
        coords = np.array([[0.0, 0.0], [length0, 0.0]])
        u = 0.6  # 30% 大位移
        f_int = truss2_internal_force(coords, np.array([0.0, 0.0, u, 0.0]), E_MOD, AREA)
        expected = E_MOD * AREA * u * (2.0 * length0 + u) * (length0 + u) / (2.0 * length0**3)
        np.testing.assert_allclose(f_int[2], expected, rtol=1e-14)
        np.testing.assert_allclose(f_int[0], -expected, rtol=1e-14)
        # 横向分量不耦合
        np.testing.assert_allclose(f_int[1], 0.0, atol=1e-14)

    def test_tangent_matches_finite_difference(self) -> None:
        """切线刚度与内力中心差分一致（rtol 1e-6）."""
        coords = np.array([[0.0, 0.0], [3.0, 1.0]])
        u_elem = np.array([0.1, -0.2, 0.4, 0.15])
        kt = truss2_tangent_stiffness(coords, u_elem, E_MOD, AREA)
        eps = 1.0e-7
        for j in range(4):
            up = u_elem.copy()
            up[j] += eps
            um = u_elem.copy()
            um[j] -= eps
            column = (
                truss2_internal_force(coords, up, E_MOD, AREA) - truss2_internal_force(coords, um, E_MOD, AREA)
            ) / (2.0 * eps)
            np.testing.assert_allclose(kt[:, j], column, rtol=1e-6, atol=1e-8)

    def test_tangent_small_disp_limit_linear(self) -> None:
        """小位移极限：切线刚度退化为线性刚度（EA/L0·n nᵀ 各向扩展）."""
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        kt = truss2_tangent_stiffness(coords, np.zeros(4), E_MOD, AREA)
        from zylab.fea import element_stiffness

        ke = element_stiffness(ElementType.TRUSS2, coords, LinearElastic(E_MOD), Section(area=AREA))
        np.testing.assert_allclose(kt, ke, rtol=1e-12)

    def test_degenerate_current_length_raises(self) -> None:
        """当前构型长度为零（两节点重合）报错."""
        from zylab.fea import ElementError

        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match=r"长度为零|退化"):
            truss2_internal_force(coords, np.array([0.0, 0.0, -1.0, 0.0]), E_MOD, AREA)


# ---------------------------------------------------------------------------
# 求解级：解析平衡与收敛
# ---------------------------------------------------------------------------


class TestNonlinearSolve:
    def test_single_rod_large_displacement(self) -> None:
        """单杆拉伸：外力 F 的平衡位移 u 满足 F = EA·u(2L0+u)(L0+u)/(2L0³)."""
        length0 = 1.0
        coords = np.array([[0.0, 0.0], [length0, 0.0]])
        conn = np.array([[0, 1]], dtype=np.int64)
        mesh = Mesh(coords=coords, blocks=(ElementBlock(etype=ElementType.TRUSS2, conn=conn),))
        u_target = 0.25
        force = E_MOD * AREA * u_target * (2.0 * length0 + u_target) * (length0 + u_target) / (2.0 * length0**3)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(1, (1,))),  # 单杆横向为机构，约束 y
            loads=(NodalLoad(1, (force, 0.0)),),
        )
        solution = solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case, n_increments=4)
        assert solution.converged
        np.testing.assert_allclose(solution.displacements[1, 0], u_target, rtol=1e-8)
        np.testing.assert_allclose(solution.displacements[1, 1], 0.0, atol=1e-12)

    def test_two_bar_truss_equilibrium(self) -> None:
        """对称两杆桁架：平衡 F = EA·(2hw+w²)(h+w)/L0³，w 为顶点竖向位移."""
        b, h = 1.0, 1.0
        mesh = _two_bar_mesh(b, h)
        length0 = float(np.hypot(b, h))
        w_target = 0.3
        force = E_MOD * AREA * (2.0 * h * w_target + w_target**2) * (h + w_target) / length0**3
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -force)),),
        )
        solution = solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case, n_increments=5)
        assert solution.converged
        # 顶点向下位移 w（y 分量为负）
        np.testing.assert_allclose(solution.displacements[1, 1], -w_target, rtol=1e-10)
        # 对称性：顶点水平位移为零
        np.testing.assert_allclose(solution.displacements[1, 0], 0.0, atol=1e-12)
        # 两支座无位移
        np.testing.assert_allclose(solution.displacements[[0, 2]], 0.0, atol=1e-12)

    def test_newton_quadratic_convergence(self) -> None:
        """光滑问题 Newton 二次收敛：每增量步迭代次数少（<= 5）."""
        b, h = 1.0, 1.0
        mesh = _two_bar_mesh(b, h)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -40.0)),),
        )
        solution = solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case, n_increments=4)
        assert solution.converged
        assert max(solution.iterations) <= 5
        assert solution.total_iterations < 4 * 5

    def test_small_load_matches_linear(self) -> None:
        """小载荷极限：非线性解与线性静力解一致."""
        mesh = _two_bar_mesh(1.0, 1.0)
        material, section = LinearElastic(E_MOD), Section(area=AREA)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -1.0e-3)),),
        )
        nonlinear = solve_nonlinear_static(mesh, [material], [section], case, n_increments=1)
        from zylab.fea import solve_static

        linear = solve_static(mesh, [material], [section], case)
        # 小载荷下非线性二阶效应 O(w²) 残留，相对 1e-6 量级正常
        np.testing.assert_allclose(nonlinear.displacements, linear.displacements, rtol=1e-4, atol=1e-12)

    def test_progress_report_invoked(self) -> None:
        """进度回调按增量步推进（覆盖 report 注入路径）."""
        mesh = _two_bar_mesh(1.0, 1.0)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -5.0)),),
        )
        records: list[float] = []
        solution = solve_nonlinear_static(
            mesh,
            [LinearElastic(E_MOD)],
            [Section(area=AREA)],
            case,
            n_increments=3,
            report=lambda p, _msg: records.append(p),
        )
        assert solution.converged
        assert records == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 1.0])

    def test_not_converged_raises(self) -> None:
        """迭代上限过小且载荷大：报未收敛."""
        mesh = _two_bar_mesh(1.0, 1.0)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -2000.0)),),
        )
        with pytest.raises(SolverError, match="未收敛"):
            solve_nonlinear_static(
                mesh,
                [LinearElastic(E_MOD)],
                [Section(area=AREA)],
                case,
                n_increments=1,
                max_iterations=2,
            )


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestNonlinearErrors:
    def test_beam2_not_supported(self) -> None:
        """BEAM2 网格报仅支持 TRUSS2."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        conn = np.array([[0, 1]], dtype=np.int64)
        mesh = Mesh(coords=coords, blocks=(ElementBlock(etype=ElementType.BEAM2, conn=conn),))
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(1, (0.0, -1.0, 0.0)),),
        )
        with pytest.raises(SolverError, match="仅支持 TRUSS2"):
            solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case)

    def test_distributed_load_not_supported(self) -> None:
        """分布载荷（边压力）报限节点集中力."""
        mesh = _two_bar_mesh(1.0, 1.0)
        edge = EdgePressure(nodes=(0, 1), pressure=1.0)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(edge,),
        )
        with pytest.raises(SolverError, match="限节点集中力"):
            solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case)

    def test_nonzero_constraint_raises(self) -> None:
        """非零约束值报错."""
        mesh = _two_bar_mesh(1.0, 1.0)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1), 0.01), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -1.0)),),
        )
        with pytest.raises(SolverError, match="约束值须为 0"):
            solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case)

    def test_missing_constraints_raises(self) -> None:
        """无约束报错."""
        mesh = _two_bar_mesh(1.0, 1.0)
        case = StaticCase(loads=(NodalLoad(1, (0.0, -1.0)),))
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case)

    def test_zero_load_raises(self) -> None:
        """自由自由度外载荷范数为零报错."""
        mesh = _two_bar_mesh(1.0, 1.0)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(1, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(0, (1.0, 0.0)),),  # 载荷全部落在约束节点上
        )
        with pytest.raises(SolverError, match="缺少非零外载荷"):
            solve_nonlinear_static(mesh, [LinearElastic(E_MOD)], [Section(area=AREA)], case)

    def test_bad_step_parameters_raise(self) -> None:
        """增量步数/容差非法报错."""
        mesh = _two_bar_mesh(1.0, 1.0)
        material, section = LinearElastic(E_MOD), Section(area=AREA)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(2, (0, 1))),
            loads=(NodalLoad(1, (0.0, -1.0)),),
        )
        with pytest.raises(SolverError, match="增量步数"):
            solve_nonlinear_static(mesh, [material], [section], case, n_increments=0)
        with pytest.raises(SolverError, match="容差"):
            solve_nonlinear_static(mesh, [material], [section], case, tolerance=0.0)
