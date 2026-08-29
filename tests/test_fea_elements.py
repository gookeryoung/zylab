"""fea.elements / fea.assemble / fea.solve 单元测试.

单元级验证：刚度对称性、退化几何报错、线性位移场下的精确应力恢复
（常应变单元/等参单元高斯点平均的 B/D 正确性）。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from zylab.fea import (
    ElementError,
    ElementType,
    LinearElastic,
    Section,
    SolverError,
    StressState,
    element_measure,
    element_stiffness,
    element_stress,
    element_stress_at,
)
from zylab.fea.assemble import assemble_stiffness, element_dofs
from zylab.fea.mesh import ElementBlock, Mesh
from zylab.fea.solve import solve_system

E_PLANE = LinearElastic(210.0, 0.3, StressState.PLANE_STRESS)
E_SOLID = LinearElastic(100.0, 0.25, StressState.SOLID)
UNIT = Section()


# ---------------------------------------------------------------------------
# TRUSS2
# ---------------------------------------------------------------------------


class TestTruss2:
    def test_stiffness_symmetric_2d(self) -> None:
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        ke = element_stiffness(ElementType.TRUSS2, coords, LinearElastic(1000.0), Section(area=2.0))
        assert ke.shape == (4, 4)
        np.testing.assert_allclose(ke, ke.T)

    def test_stiffness_along_x(self) -> None:
        # 沿 x 的杆：k = EA/L，仅 ux 分量耦合
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        ke = element_stiffness(ElementType.TRUSS2, coords, LinearElastic(1000.0), Section(area=2.0))
        k = 1000.0 * 2.0 / 2.0
        np.testing.assert_allclose(ke[0, 0], k)
        np.testing.assert_allclose(ke[0, 2], -k)
        np.testing.assert_allclose(ke[1, 1], 0.0)

    def test_stiffness_3d_shape(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ke = element_stiffness(ElementType.TRUSS2, coords, LinearElastic(1000.0), UNIT)
        assert ke.shape == (6, 6)
        np.testing.assert_allclose(ke, ke.T)

    def test_axial_stress(self) -> None:
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        u = np.array([0.0, 0.0, 0.001, 0.0])
        stress = element_stress(ElementType.TRUSS2, coords, LinearElastic(1000.0), u)
        np.testing.assert_allclose(stress, [1000.0 * 0.001 / 2.0])

    def test_zero_length_raises(self) -> None:
        coords = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ElementError, match="长度为零"):
            element_stiffness(ElementType.TRUSS2, coords, LinearElastic(1000.0), UNIT)


# ---------------------------------------------------------------------------
# BEAM2：平面 Euler-Bernoulli 梁
# ---------------------------------------------------------------------------


class TestBeam2:
    def test_stiffness_symmetric(self) -> None:
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        ke = element_stiffness(ElementType.BEAM2, coords, LinearElastic(1000.0), Section(area=2.0, inertia=0.5))
        assert ke.shape == (6, 6)
        np.testing.assert_allclose(ke, ke.T, atol=1e-9)

    def test_stiffness_aligned_x(self) -> None:
        # 沿 x 的梁：轴向 EA/L、弯曲 12EI/L³ 分量解耦
        coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        e, a, i, length = 1000.0, 2.0, 0.5, 2.0
        ke = element_stiffness(ElementType.BEAM2, coords, LinearElastic(e), Section(area=a, inertia=i))
        np.testing.assert_allclose(ke[0, 0], e * a / length)
        np.testing.assert_allclose(ke[0, 3], -e * a / length)
        np.testing.assert_allclose(ke[1, 1], 12.0 * e * i / length**3)
        np.testing.assert_allclose(ke[2, 2], 4.0 * e * i / length)
        np.testing.assert_allclose(ke[2, 5], 2.0 * e * i / length)
        # 轴向与弯曲不耦合
        np.testing.assert_allclose(ke[0, 1], 0.0, atol=1e-12)

    def test_zero_length_raises(self) -> None:
        coords = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ElementError, match="长度为零"):
            element_stiffness(ElementType.BEAM2, coords, LinearElastic(1000.0), UNIT)

    def test_stress_requires_section(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="截面属性"):
            element_stress(ElementType.BEAM2, coords, LinearElastic(1000.0), np.zeros(6))

    def test_stress_shape(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        stress = element_stress(ElementType.BEAM2, coords, LinearElastic(1000.0), np.zeros(6), Section(inertia=0.5))
        assert stress.shape == (3,)
        np.testing.assert_allclose(stress, np.zeros(3), atol=1e-15)


# ---------------------------------------------------------------------------
# element_measure：连续体单元度量
# ---------------------------------------------------------------------------


class TestElementMeasure:
    def test_tria3_area(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(element_measure(ElementType.TRIA3, coords), 0.5)

    def test_quad4_area(self) -> None:
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(element_measure(ElementType.QUAD4, coords), 2.0)

    def test_tet4_volume(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(element_measure(ElementType.TET4, coords), 1.0 / 6.0)

    def test_hex8_volume(self) -> None:
        cube = np.array(
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
        )
        np.testing.assert_allclose(element_measure(ElementType.HEX8, cube), 1.0)

    def test_line_element_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="不是连续体单元"):
            element_measure(ElementType.TRUSS2, coords)


# ---------------------------------------------------------------------------
# TRIA3 / QUAD4：线性位移场精确应力
# ---------------------------------------------------------------------------


def _linear_field_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ux = 0.001 + 0.002x + 0.003y, uy = -0.001 + 0.004x - 0.002y."""
    ux = 0.001 + 0.002 * x + 0.003 * y
    uy = -0.001 + 0.004 * x - 0.002 * y
    return np.stack([ux, uy], axis=-1).ravel()


def _expected_stress_plane(d: np.ndarray) -> np.ndarray:
    """线性场对应的精确应变/应力（平面）."""
    strain = np.array([0.002, -0.002, 0.003 + 0.004])  # εxx, εyy, γxy
    return d @ strain


class TestTria3:
    def test_stiffness_symmetric(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.1], [0.2, 1.0]])
        ke = element_stiffness(ElementType.TRIA3, coords, E_PLANE, UNIT)
        assert ke.shape == (6, 6)
        np.testing.assert_allclose(ke, ke.T)

    def test_linear_field_exact_stress(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.1], [0.2, 1.0]])
        u = _linear_field_2d(coords[:, 0], coords[:, 1])
        stress = element_stress(ElementType.TRIA3, coords, E_PLANE, u)
        np.testing.assert_allclose(stress, _expected_stress_plane(E_PLANE.d_matrix()), rtol=1e-12)

    def test_degenerate_triangle_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(ElementError, match="退化为直线"):
            element_stiffness(ElementType.TRIA3, coords, E_PLANE, UNIT)


class TestQuad4:
    def test_stiffness_symmetric(self) -> None:
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.1, 1.2]])
        ke = element_stiffness(ElementType.QUAD4, coords, E_PLANE, UNIT)
        assert ke.shape == (8, 8)
        np.testing.assert_allclose(ke, ke.T)

    def test_linear_field_exact_stress(self) -> None:
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.1, 1.2]])
        u = _linear_field_2d(coords[:, 0], coords[:, 1])
        stress = element_stress(ElementType.QUAD4, coords, E_PLANE, u)
        np.testing.assert_allclose(stress, _expected_stress_plane(E_PLANE.d_matrix()), rtol=1e-12)

    def test_degenerate_quad_raises(self) -> None:
        # 三点共线导致雅可比奇异
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
        with pytest.raises(ElementError, match="雅可比"):
            element_stiffness(ElementType.QUAD4, coords, E_PLANE, UNIT)

    def test_stress_at_point_linear_field(self) -> None:
        """任意自然坐标处求值：线性位移场下与高斯点平均一致（应变处处相等）."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [0.1, 1.2]])
        u = _linear_field_2d(coords[:, 0], coords[:, 1])
        expected = _expected_stress_plane(E_PLANE.d_matrix())
        stress = element_stress_at(ElementType.QUAD4, coords, E_PLANE, u, (-1.0, -1.0))
        np.testing.assert_allclose(stress, expected, rtol=1e-12)
        stress_mid = element_stress_at(ElementType.QUAD4, coords, E_PLANE, u, (0.3, -0.5))
        np.testing.assert_allclose(stress_mid, expected, rtol=1e-12)

    def test_stress_at_point_rejects_other_types(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ElementError, match="暂不支持"):
            element_stress_at(ElementType.TRIA3, coords, E_PLANE, np.zeros(6), (0.0, 0.0))


# ---------------------------------------------------------------------------
# TET4 / HEX8：线性位移场精确应力
# ---------------------------------------------------------------------------


def _linear_field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """ux = 0.001+0.002x+0.003y+0.001z, uy = ..., uz = ...（非常向量场）."""
    ux = 0.001 + 0.002 * x + 0.003 * y + 0.001 * z
    uy = -0.001 + 0.004 * x - 0.002 * y + 0.002 * z
    uz = 0.002 - 0.001 * x + 0.005 * y + 0.003 * z
    return np.stack([ux, uy, uz], axis=-1).ravel()


def _expected_stress_solid(d: np.ndarray) -> np.ndarray:
    """线性 3D 场对应的精确应力.

    εxx=0.002, εyy=-0.002, εzz=0.003, γxy=0.003+0.004, γyz=0.002+0.005, γxz=0.001-0.001.
    """
    strain = np.array([0.002, -0.002, 0.003, 0.007, 0.007, 0.0])
    return d @ strain


class TestTet4:
    def test_stiffness_symmetric(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]])
        ke = element_stiffness(ElementType.TET4, coords, E_SOLID, UNIT)
        assert ke.shape == (12, 12)
        np.testing.assert_allclose(ke, ke.T)

    def test_linear_field_exact_stress(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]])
        u = _linear_field_3d(coords[:, 0], coords[:, 1], coords[:, 2])
        stress = element_stress(ElementType.TET4, coords, E_SOLID, u)
        np.testing.assert_allclose(stress, _expected_stress_solid(E_SOLID.d_matrix()), rtol=1e-12, atol=1e-15)

    def test_degenerate_tet_raises(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
        with pytest.raises(ElementError, match="退化为平面"):
            element_stiffness(ElementType.TET4, coords, E_SOLID, UNIT)


class TestHex8:
    def _unit_cube(self) -> np.ndarray:
        # 节点顺序与 _HEX8_SIGNS 对应：z=0 面 0-3，z=1 面 4-7
        return np.array(
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
        )

    def test_stiffness_symmetric(self) -> None:
        ke = element_stiffness(ElementType.HEX8, self._unit_cube(), E_SOLID, UNIT)
        assert ke.shape == (24, 24)
        np.testing.assert_allclose(ke, ke.T, atol=1e-9)

    def test_linear_field_exact_stress(self) -> None:
        coords = self._unit_cube()
        u = _linear_field_3d(coords[:, 0], coords[:, 1], coords[:, 2])
        stress = element_stress(ElementType.HEX8, coords, E_SOLID, u)
        np.testing.assert_allclose(stress, _expected_stress_solid(E_SOLID.d_matrix()), rtol=1e-12, atol=1e-15)

    def test_degenerate_hex_raises(self) -> None:
        # 节点共面：z 坐标全部为 0
        coords = self._unit_cube()
        coords[:, 2] = 0.0
        with pytest.raises(ElementError, match="雅可比"):
            element_stiffness(ElementType.HEX8, coords, E_SOLID, UNIT)


# ---------------------------------------------------------------------------
# 装配与求解
# ---------------------------------------------------------------------------


class TestAssemble:
    def test_element_dofs(self) -> None:
        mesh = Mesh(np.zeros((3, 2)))
        dofs = element_dofs(mesh, np.array([2, 0]))
        np.testing.assert_array_equal(dofs, [4, 5, 0, 1])

    def test_element_dofs_beam_width(self) -> None:
        # 梁网格每节点 3 DOF，全局编号按宽度 3 展开
        block = ElementBlock(ElementType.BEAM2, np.array([[0, 1]]))
        mesh = Mesh(np.zeros((3, 2)), (block,))
        dofs = element_dofs(mesh, np.array([1, 0]))
        np.testing.assert_array_equal(dofs, [3, 4, 5, 0, 1, 2])

    def test_assemble_single_quad(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mesh = Mesh(coords, (ElementBlock(ElementType.QUAD4, np.arange(4).reshape(1, 4)),))
        k = assemble_stiffness(mesh, [E_PLANE], [UNIT])
        assert k.shape == (8, 8)
        np.testing.assert_allclose(k.toarray(), k.toarray().T)


class TestSolveSystem:
    def test_two_spring_system(self) -> None:
        # 两自由度弹簧系统：k1=k2=10，节点 0 固定，节点 1 受力 5
        k = csr_matrix(np.array([[20.0, -10.0], [-10.0, 10.0]]))
        u, reactions = solve_system(k, np.array([0.0, 5.0]), np.array([0]), np.array([0.0]))
        np.testing.assert_allclose(u, [0.0, 0.5])
        np.testing.assert_allclose(reactions, [-5.0])

    def test_nonzero_prescribed_displacement(self) -> None:
        # 节点 0 给定位移 0.1，节点 1 自由无载荷：均匀弹簧拉伸
        k = csr_matrix(np.array([[10.0, -10.0], [-10.0, 10.0]]))
        u, _ = solve_system(k, np.array([0.0, 0.0]), np.array([0]), np.array([0.1]))
        np.testing.assert_allclose(u, [0.1, 0.1])

    def test_singular_system_raises(self) -> None:
        # 无约束刚度（刚体模式）应报奇异
        k = csr_matrix(np.array([[10.0, -10.0], [-10.0, 10.0]]))
        with pytest.raises(SolverError, match="奇异"):
            solve_system(k, np.array([1.0, 0.0]), np.array([], dtype=int), np.array([]))

    def test_shape_mismatch_raises(self) -> None:
        k = csr_matrix(np.eye(2))
        with pytest.raises(SolverError, match="不匹配"):
            solve_system(k, np.zeros(3), np.array([0]), np.array([0.0]))

    def test_constraint_out_of_range_raises(self) -> None:
        k = csr_matrix(np.eye(2))
        with pytest.raises(SolverError, match="越界"):
            solve_system(k, np.zeros(2), np.array([5]), np.array([0.0]))

    def test_fixed_values_length_mismatch(self) -> None:
        k = csr_matrix(np.eye(2))
        with pytest.raises(SolverError, match="不一致"):
            solve_system(k, np.zeros(2), np.array([0, 1]), np.array([0.0]))
