"""fea.modal / 单元质量矩阵测试.

验证维度：单元质量矩阵总质量守恒与对称性、悬臂杆/悬臂梁固有频率对解析解收敛、
模态正交性（Φ^T M Φ = I）、错误路径（密度缺失/约束缺失/非零约束/阶数越界）。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementError,
    ElementType,
    LinearElastic,
    Mesh,
    ModalSolution,
    Section,
    SolverError,
    StressState,
    assemble_mass,
    element_mass,
    solve_modal,
)

RHO = 7.85e3  # 钢密度量级（统一单位制）


# ---------------------------------------------------------------------------
# 单元质量矩阵：总质量守恒与对称性
# ---------------------------------------------------------------------------


class TestElementMass:
    """总质量守恒：ones^T M ones = ρ × 单元质量（连续体为 ρ × 度量 × 厚度）."""

    @pytest.mark.parametrize(
        ("etype", "coords", "section", "total"),
        [
            (ElementType.TRUSS2, np.array([[0.0, 0.0], [3.0, 4.0]]), Section(area=2.0), 5.0 * 2.0),
            (ElementType.TRIA3, np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]), Section(thickness=0.5), 1.0 * 0.5),
            (
                ElementType.QUAD4,
                np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
                Section(thickness=0.5),
                2.0 * 0.5,
            ),
            (
                ElementType.TET4,
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                Section(),
                1.0 / 6.0,
            ),
            (
                ElementType.HEX8,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
                Section(),
                1.0,
            ),
        ],
    )
    def test_total_mass_conserved(self, etype, coords, section, total) -> None:
        material = LinearElastic(210.0e9, 0.3, StressState.SOLID, density=RHO)
        me = element_mass(etype, coords, material, section)
        np.testing.assert_allclose(me, me.T)
        # 单一方向（x）刚体平动：v^T M v = ρ × 单元质量；全 1 向量会叠加各方向刚体质量
        dim = coords.shape[1]
        rigid_x = np.zeros(me.shape[0])
        rigid_x[::dim] = 1.0
        np.testing.assert_allclose(rigid_x @ me @ rigid_x, RHO * total, rtol=1e-12)

    def test_beam2_total_mass(self) -> None:
        """梁一致质量：轴向平动质量合计 = ρAL（转动分量不贡献刚体平动质量）."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        me = element_mass(ElementType.BEAM2, coords, LinearElastic(210.0e9, density=RHO), Section(area=3.0))
        np.testing.assert_allclose(me, me.T)
        # 沿 x 刚体平动（ux 分量全 1，转角分量不影响轴向质量）
        v = np.zeros(6)
        v[[0, 3]] = 1.0
        np.testing.assert_allclose(v @ me @ v, RHO * 3.0 * 2.0, rtol=1e-12)

    def test_zero_density_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="质量密度"):
            element_mass(ElementType.TRUSS2, coords, LinearElastic(1000.0), Section())

    def test_truss2_mass_diagonal_along_x(self) -> None:
        """沿 x 杆质量：对角块 ρAL/3、耦合块 ρAL/6（仅 ux 分量耦合）."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        me = element_mass(ElementType.TRUSS2, coords, LinearElastic(1000.0, density=RHO), Section(area=2.0))
        m = RHO * 2.0 * 2.0
        np.testing.assert_allclose(me[0, 0], m / 3.0)
        np.testing.assert_allclose(me[0, 2], m / 6.0)
        np.testing.assert_allclose(me[1, 1], m / 3.0)
        np.testing.assert_allclose(me[1, 2], 0.0)


# ---------------------------------------------------------------------------
# 悬臂杆纵向模态：解析解 ω_n = (2n-1)π/(2L) √(E/ρ)
# ---------------------------------------------------------------------------


def _cantilever_rod_setup(n_elem: int) -> tuple[Mesh, tuple]:
    """沿 x 的等分悬臂杆网格与约束（全节点约束 y 消除桁架横向无刚度自由度）."""
    xs = np.linspace(0.0, 1.0, n_elem + 1)
    coords = np.column_stack([xs, np.zeros_like(xs)])
    conn = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, conn),))
    constraints = (Constraint(0, (0, 1)), *(Constraint(i, (1,)) for i in range(1, n_elem + 1)))
    return mesh, constraints


class TestRodModal:
    def test_fundamental_frequency(self) -> None:
        """悬臂杆基频：16 单元一致质量误差 < 1%."""
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(16)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=3)
        omega_exact = np.pi / 2.0 * np.sqrt(material.e_modulus / RHO)
        np.testing.assert_allclose(solution.frequencies[0], omega_exact, rtol=0.01)

    def test_higher_modes_converge(self) -> None:
        """前 3 阶纵向模态：误差随阶数缓慢增大，基频 < 1%、第 3 阶 < 5%."""
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(32)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=3)
        for n in range(1, 4):
            omega_exact = (2 * n - 1) * np.pi / 2.0 * np.sqrt(material.e_modulus / RHO)
            np.testing.assert_allclose(solution.frequencies[n - 1], omega_exact, rtol=0.05)

    def test_mode_shape_orthogonality(self) -> None:
        """质量归一化振型：Φ^T M Φ = I（含固定自由度零分量）."""
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(8)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=4)
        m_global = assemble_mass(mesh, (material,), (Section(),)).toarray()
        gram = solution.mode_shapes.T @ m_global @ solution.mode_shapes
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-10)

    def test_frequencies_hz(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(4)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=1)
        np.testing.assert_allclose(solution.frequencies_hz[0], solution.frequencies[0] / (2.0 * np.pi))

    def test_mode_shape_api(self) -> None:
        """mode_shape 整形为 (n_nodes, dofs_per_node)，固定端位移为 0."""
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(4)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=2)
        shape = solution.mode_shape(0)
        assert shape.shape == (5, 2)
        np.testing.assert_allclose(shape[0], 0.0, atol=1e-15)
        assert solution.n_modes == 2


# ---------------------------------------------------------------------------
# 悬臂梁横向模态：βL = 1.875 / 4.694 / 7.855
# ---------------------------------------------------------------------------


def _cantilever_beam_mesh(n_elem: int) -> Mesh:
    """沿 x 的等分悬臂梁网格（每节点 3 DOF）."""
    xs = np.linspace(0.0, 1.0, n_elem + 1)
    coords = np.column_stack([xs, np.zeros_like(xs)])
    conn = np.array([[i, i + 1] for i in range(n_elem)])
    return Mesh(coords, (ElementBlock(ElementType.BEAM2, conn),))


class TestBeamModal:
    def test_fundamental_frequency(self) -> None:
        """悬臂梁一阶弯曲频率：16 单元一致质量误差 < 2%."""
        e, rho, area, inertia = 2.1e11, RHO, 1.0e-4, 8.333e-10
        material = LinearElastic(e, density=rho)
        section = Section(area=area, inertia=inertia)
        mesh = _cantilever_beam_mesh(16)
        solution = solve_modal(mesh, (material,), (section,), (Constraint(0, (0, 1, 2)),), n_modes=3)
        beta_l = 1.87510407
        omega_exact = beta_l**2 * np.sqrt(e * inertia / (rho * area))
        np.testing.assert_allclose(solution.frequencies[0], omega_exact, rtol=0.02)

    def test_second_bending_mode(self) -> None:
        """二阶弯曲频率：16 单元一致质量误差 < 3%."""
        e, rho, area, inertia = 2.1e11, RHO, 1.0e-4, 8.333e-10
        material = LinearElastic(e, density=rho)
        section = Section(area=area, inertia=inertia)
        mesh = _cantilever_beam_mesh(16)
        solution = solve_modal(mesh, (material,), (section,), (Constraint(0, (0, 1, 2)),), n_modes=5)
        # 弯曲 2 阶 βL = 4.694；取全部频率中最接近者
        beta_l = 4.69409113
        omega_bend2 = beta_l**2 * np.sqrt(e * inertia / (rho * area))
        closest = min(solution.frequencies, key=lambda w: abs(w - omega_bend2))
        np.testing.assert_allclose(closest, omega_bend2, rtol=0.03)

    def test_orthogonality_with_rotation_dofs(self) -> None:
        """含转角自由度的正交性校验（梁网格 dofs_per_node = 3）."""
        material = LinearElastic(2.1e11, density=RHO)
        section = Section()
        mesh = _cantilever_beam_mesh(6)
        solution = solve_modal(mesh, (material,), (section,), (Constraint(0, (0, 1, 2)),), n_modes=3)
        m_global = assemble_mass(mesh, (material,), (section,)).toarray()
        gram = solution.mode_shapes.T @ m_global @ solution.mode_shapes
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# 连续体模态（QUAD4）
# ---------------------------------------------------------------------------


class TestContinuumModal:
    def test_quad4_strip(self) -> None:
        """QUAD4 条带：频率升序、总质量守恒（x/y 两方向刚体质量各 ρV）."""
        material = LinearElastic(2.1e11, 0.3, StressState.PLANE_STRESS, density=RHO)
        section = Section(thickness=1.0)
        n_x, n_y = 8, 2
        xs = np.linspace(0.0, 1.0, n_x + 1)
        ys = np.linspace(0.0, 0.05, n_y + 1)
        coords = np.array([[x, y] for y in ys for x in xs])
        conn = []
        for j in range(n_y):
            for i in range(n_x):
                n1 = j * (n_x + 1) + i
                conn.append([n1, n1 + 1, n1 + n_x + 2, n1 + n_x + 1])
        mesh = Mesh(coords, (ElementBlock(ElementType.QUAD4, np.array(conn)),))
        # 左边缘固定
        constraints = tuple(Constraint(j * (n_x + 1), (0, 1)) for j in range(n_y + 1))
        solution = solve_modal(mesh, (material,), (section,), constraints, n_modes=3)
        assert solution.n_modes == 3
        assert np.all(np.diff(solution.frequencies) > 0.0)
        # M.sum() = x 与 y 两方向刚体平动质量之和 = 2ρV
        volume = 1.0 * 0.05 * 1.0
        np.testing.assert_allclose(assemble_mass(mesh, (material,), (section,)).sum(), 2.0 * RHO * volume, rtol=1e-10)

    def test_modal_solution_type(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh, constraints = _cantilever_rod_setup(4)
        solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=1)
        assert isinstance(solution, ModalSolution)
        assert solution.mesh is mesh


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestModalErrors:
    def test_missing_density_raises(self) -> None:
        material = LinearElastic(2.1e11)  # 未配置密度
        mesh = _cantilever_beam_mesh(4)
        with pytest.raises(ElementError, match="质量密度"):
            assemble_mass(mesh, (material,), (Section(),))

    def test_missing_constraints_raises(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh = _cantilever_beam_mesh(4)
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_modal(mesh, (material,), (Section(),), (), n_modes=1)

    def test_nonzero_constraint_raises(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh = _cantilever_beam_mesh(4)
        with pytest.raises(SolverError, match="约束值须为 0"):
            solve_modal(mesh, (material,), (Section(),), (Constraint(0, (0, 1, 2), value=0.01),), n_modes=1)

    def test_excessive_modes_raises(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh = _cantilever_beam_mesh(2)  # 9 DOF，固定 3 个，剩 6 个自由
        with pytest.raises(SolverError, match="模态阶数"):
            solve_modal(mesh, (material,), (Section(),), (Constraint(0, (0, 1, 2)),), n_modes=6)

    def test_invalid_dof_raises(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh = _cantilever_beam_mesh(2)
        with pytest.raises(SolverError, match="超出"):
            solve_modal(mesh, (material,), (Section(),), (Constraint(0, (0, 1, 5)),), n_modes=1)

    def test_node_out_of_range_raises(self) -> None:
        material = LinearElastic(2.1e11, density=RHO)
        mesh = _cantilever_beam_mesh(2)
        with pytest.raises(SolverError, match="越界"):
            solve_modal(mesh, (material,), (Section(),), (Constraint(99, (0, 1, 2)),), n_modes=1)
