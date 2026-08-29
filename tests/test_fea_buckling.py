"""fea.buckling / 几何刚度与线性屈曲测试.

验证维度：单元几何刚度值与对称性、悬臂柱/两端铰支柱欧拉临界载荷收敛、
屈曲振型归一、错误路径（拉伸无屈曲/连续体无贡献/长度不符/类型不支持）。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    BucklingSolution,
    Constraint,
    ElementBlock,
    ElementError,
    ElementType,
    LinearElastic,
    Mesh,
    MeshError,
    NodalLoad,
    Section,
    SolverError,
    StaticCase,
    assemble_geometric,
    element_geometric_stiffness,
    solve_buckling,
)

E_MOD = 2.1e5  # 统一单位制下的弹性模量


def _column_mesh(n_elem: int, length: float) -> Mesh:
    """沿 x 的等分梁柱网格（BEAM2，3 DOF/节点）."""
    coords = np.array([[i * length / n_elem, 0.0] for i in range(n_elem + 1)])
    conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=np.int64)
    block = ElementBlock(etype=ElementType.BEAM2, conn=conn, material=0, section=0)
    return Mesh(coords=coords, blocks=(block,))


# ---------------------------------------------------------------------------
# 单元几何刚度
# ---------------------------------------------------------------------------


class TestElementGeometricStiffness:
    def test_truss2_values_along_x(self) -> None:
        """沿 x 杆：横向对角 N/L、耦合 N/L，轴向分量全 0."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        kg = element_geometric_stiffness(ElementType.TRUSS2, coords, 6.0)
        np.testing.assert_allclose(kg, kg.T)
        np.testing.assert_allclose(kg[1, 1], 6.0 / 2.0)
        np.testing.assert_allclose(kg[1, 3], 6.0 / 2.0)
        np.testing.assert_allclose(kg[3, 3], 6.0 / 2.0)
        # 轴向自由度不贡献
        np.testing.assert_allclose(kg[0, :], 0.0)
        np.testing.assert_allclose(kg[2, :], 0.0)

    def test_truss2_inclined_symmetric(self) -> None:
        """斜杆几何刚度对称且秩 1（n n^T 结构）."""
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        kg = element_geometric_stiffness(ElementType.TRUSS2, coords, -5.0)
        np.testing.assert_allclose(kg, kg.T)
        # 载荷向量张成的秩 1 矩阵
        np.testing.assert_allclose(np.linalg.matrix_rank(kg, tol=1e-10), 1)

    def test_truss2_3d_raises(self) -> None:
        """3D 杆几何刚度 v1 不支持."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ElementError, match="平面杆"):
            element_geometric_stiffness(ElementType.TRUSS2, coords, 1.0)

    def test_beam2_values_along_x(self) -> None:
        """沿 x 梁：横向平动项 P·6/5/L、耦合 P·L/10，轴向与符号约定."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        p = 10.0
        kg = element_geometric_stiffness(ElementType.BEAM2, coords, p)
        np.testing.assert_allclose(kg, kg.T)
        np.testing.assert_allclose(kg[1, 1], p * 6.0 / 5.0 / 2.0)
        np.testing.assert_allclose(kg[1, 5], p / 10.0)
        np.testing.assert_allclose(kg[2, 2], p * 2.0 * 2.0 / 15.0)  # 2PL/15
        # 轴向平动不贡献
        np.testing.assert_allclose(kg[0, :], 0.0)
        np.testing.assert_allclose(kg[3, :], 0.0)

    def test_unsupported_type_raises(self) -> None:
        """连续体单元几何刚度 v1 不支持."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ElementError, match="暂不支持"):
            element_geometric_stiffness(ElementType.TRIA3, coords, 1.0)


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


class TestAssembleGeometric:
    def test_length_mismatch_raises(self) -> None:
        mesh = _column_mesh(2, 1.0)
        with pytest.raises(MeshError, match="不符"):
            assemble_geometric(mesh, [1.0])  # 应为 2 个单元

    def test_zero_forces_empty(self) -> None:
        """全零轴力 -> 全零几何刚度矩阵."""
        mesh = _column_mesh(3, 1.0)
        kg = assemble_geometric(mesh, [0.0, 0.0, 0.0])
        assert kg.shape == (mesh.n_dofs, mesh.n_dofs)
        assert kg.nnz == 0


# ---------------------------------------------------------------------------
# 欧拉柱解析解
# ---------------------------------------------------------------------------


class TestEulerColumns:
    def test_cantilever_column(self) -> None:
        """悬臂柱：P_cr = π²EI/(4L²)，离散 16 单元误差 < 1%."""
        n_elem, length, inertia = 16, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
        )
        solution = solve_buckling(mesh, [material], [section], case, n_modes=2)
        analytical = np.pi**2 * E_MOD * inertia / (4.0 * length**2)
        assert solution.load_factors[0] == pytest.approx(analytical, rel=0.01)

    def test_pinned_column(self) -> None:
        """两端铰支柱：P_cr = π²EI/L²，离散 16 单元误差 < 1.5%."""
        n_elem, length, inertia = 16, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(
                Constraint(0, (0, 1)),
                Constraint(n_elem, (1,)),
            ),
            loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
        )
        solution = solve_buckling(mesh, [material], [section], case, n_modes=1)
        analytical = np.pi**2 * E_MOD * inertia / length**2
        assert solution.load_factors[0] == pytest.approx(analytical, rel=0.015)

    def test_cantilever_second_mode(self) -> None:
        """悬臂柱二阶临界载荷 = 一阶的 9 倍（cos(kL)=0 谱：(3π/2)²/(π/2)²）."""
        n_elem, length, inertia = 24, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
        )
        solution = solve_buckling(mesh, [material], [section], case, n_modes=2)
        ratio = solution.load_factors[1] / solution.load_factors[0]
        assert ratio == pytest.approx(9.0, rel=0.02)

    def test_mode_shape_normalized(self) -> None:
        """屈曲振型最大分量绝对值为 1，约束自由度为 0."""
        n_elem, length, inertia = 8, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
        )
        solution = solve_buckling(mesh, [material], [section], case, n_modes=1)
        assert solution.n_modes == 1
        shape = solution.mode_shape(0)
        assert shape.shape == (mesh.n_nodes, mesh.dofs_per_node)
        np.testing.assert_allclose(np.max(np.abs(shape)), 1.0)
        np.testing.assert_allclose(shape[0], 0.0)  # 固支端

    def test_reference_solution_attached(self) -> None:
        """返回结果携带参考态静力解（轴力来源）."""
        n_elem, length, inertia = 4, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
        )
        solution = solve_buckling(mesh, [material], [section], case, n_modes=1)
        assert isinstance(solution, BucklingSolution)
        assert solution.reference.strain_energy > 0.0
        # 参考态压缩轴力：σ = -P/A
        stress = solution.reference.element_stresses(ElementType.BEAM2)[0].stress
        np.testing.assert_allclose(stress[0], -1.0 / 0.01, rtol=1e-9)


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestBucklingErrors:
    def test_tension_no_buckling(self) -> None:
        """拉伸参考态无正载荷因子."""
        n_elem, length, inertia = 4, 1.0, 1.0e-4
        mesh = _column_mesh(n_elem, length)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=inertia)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(n_elem, (1.0, 0.0, 0.0)),),  # 拉伸
        )
        with pytest.raises(SolverError, match="无压缩轴力"):
            solve_buckling(mesh, [material], [section], case)

    def test_missing_constraints(self) -> None:
        """无约束报错."""
        mesh = _column_mesh(4, 1.0)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=1e-4)
        case = StaticCase(loads=(NodalLoad(4, (-1.0, 0.0, 0.0)),))
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_buckling(mesh, [material], [section], case)

    def test_nonzero_constraint_raises(self) -> None:
        """非零约束值报错（参考态静力可为非零，扰动须为 0）."""
        mesh = _column_mesh(4, 1.0)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=1e-4)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2), 0.001),),
            loads=(NodalLoad(4, (-1.0, 0.0, 0.0)),),
        )
        with pytest.raises(SolverError, match="约束值须为 0"):
            solve_buckling(mesh, [material], [section], case)

    def test_mode_count_out_of_range(self) -> None:
        """阶数越界报错."""
        mesh = _column_mesh(2, 1.0)
        material = LinearElastic(E_MOD)
        section = Section(area=0.01, inertia=1e-4)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1, 2)),),
            loads=(NodalLoad(2, (-1.0, 0.0, 0.0)),),
        )
        with pytest.raises(SolverError, match="须小于自由自由度数"):
            solve_buckling(mesh, [material], [section], case, n_modes=100)

    def test_continuum_no_geometric_contribution(self) -> None:
        """纯连续体网格无几何刚度贡献 -> 特征值求解失败（K_G = 0）."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        block = ElementBlock(etype=ElementType.TRIA3, conn=conn, material=0, section=0)
        mesh = Mesh(coords=coords, blocks=(block,))
        material = LinearElastic(E_MOD, 0.3)
        section = Section(thickness=0.1)
        case = StaticCase(
            constraints=(Constraint(0, (0, 1)), Constraint(1, (0,)), Constraint(3, (1,))),
            loads=(NodalLoad(2, (-1.0, 0.0)),),
        )
        with pytest.raises(SolverError):
            solve_buckling(mesh, [material], [section], case)
