"""fea.conduction 标量场传导内核测试：单元矩阵解析解、梯度/场载荷恢复、装配校验."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    ElementBlock,
    ElementError,
    ElementType,
    Mesh,
    MeshError,
    Section,
    assemble_conduction,
    element_conductance,
    element_field_load,
    element_scalar_gradient,
)

__all__ = []

#: 直角三角形（0,0)-(1,0)-(0,1)，面积 0.5
TRIA_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

#: 单位方形（标准 Q4 节点序：左下/右下/右上/左上）
QUAD_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

MAT = ConductionMaterial(electric_sigma=2.0, thermal_k=5.0)
UNIT = Section()


def _single_block_mesh(etype: ElementType, coords: np.ndarray) -> Mesh:
    """单单元网格（连接 0..n-1）."""
    conn = np.arange(coords.shape[0])[None, :]
    return Mesh(coords, (ElementBlock(etype=etype, conn=conn),))


class TestConductionMaterial:
    """传导材料校验."""

    @pytest.mark.parametrize(
        ("sigma", "k", "match"),
        [(-1.0, 1.0, "电导率"), (0.0, 1.0, "电导率"), (1.0, 0.0, "导热系数"), (1.0, -2.0, "导热系数")],
        ids=["sigma-neg", "sigma-zero", "k-zero", "k-neg"],
    )
    def test_non_positive_rejected(self, sigma: float, k: float, match: str) -> None:
        """电导率/导热系数非正抛 ElementError."""
        with pytest.raises(ElementError, match=match):
            ConductionMaterial(sigma, k)


class TestElementConductance:
    """单元传导矩阵解析解."""

    def test_tria3_right_triangle(self) -> None:
        """直角三角形 CST：K = c·t·A·GᵀG 手算精确值，行和为零（常场无净流）."""
        ke = element_conductance(ElementType.TRIA3, TRIA_COORDS, 1.0, 1.0)
        expected = 0.5 * np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        np.testing.assert_allclose(ke, expected, rtol=1e-14)
        np.testing.assert_allclose(ke.sum(axis=1), np.zeros(3), atol=1e-14)

    def test_quad4_unit_square(self) -> None:
        """单位方形 Q4：K = t/6·[[4,-1,-2,-1],...] 标准双线性传导矩阵."""
        ke = element_conductance(ElementType.QUAD4, QUAD_COORDS, 1.0, 1.0)
        expected = (
            np.array(
                [
                    [4.0, -1.0, -2.0, -1.0],
                    [-1.0, 4.0, -1.0, -2.0],
                    [-2.0, -1.0, 4.0, -1.0],
                    [-1.0, -2.0, -1.0, 4.0],
                ]
            )
            / 6.0
        )
        np.testing.assert_allclose(ke, expected, rtol=1e-12)
        assert ke[0, 0] == pytest.approx(2.0 / 3.0, rel=1e-12)

    def test_coefficient_and_thickness_scale_linearly(self) -> None:
        """传导系数与厚度线性缩放矩阵."""
        base = element_conductance(ElementType.TRIA3, TRIA_COORDS, 1.0, 1.0)
        doubled = element_conductance(ElementType.TRIA3, TRIA_COORDS, 2.0, 3.0)
        np.testing.assert_allclose(doubled, 6.0 * base, rtol=1e-14)

    def test_unsupported_element_type(self) -> None:
        """不支持的结构单元类型抛 ElementError."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="暂不支持"):
            element_conductance(ElementType.TRUSS2, coords, 1.0, 1.0)

    def test_degenerate_tria3(self) -> None:
        """共线节点退化三角形抛 ElementError."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(ElementError, match="退化"):
            element_conductance(ElementType.TRIA3, coords, 1.0, 1.0)

    def test_degenerate_quad4(self) -> None:
        """共线节点退化 Q4（雅可比为零）抛 ElementError."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        with pytest.raises(ElementError, match="雅可比"):
            element_conductance(ElementType.QUAD4, coords, 1.0, 1.0)

    def test_non_positive_coefficient(self) -> None:
        """传导系数非正抛 ElementError."""
        with pytest.raises(ElementError, match="须为正"):
            element_conductance(ElementType.TRIA3, TRIA_COORDS, 0.0, 1.0)


class TestElementScalarGradient:
    """标量场单元梯度恢复."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (np.array([0.0, 1.0, 0.0]), (1.0, 0.0)),  # V = x
            (np.array([0.0, 0.0, 1.0]), (0.0, 1.0)),  # V = y
            (np.zeros(3), (0.0, 0.0)),  # 常场零梯度
        ],
        ids=["vx", "vy", "const"],
    )
    def test_tria3_exact_linear(self, values: np.ndarray, expected: tuple[float, float]) -> None:
        """CST 常梯度对线性场精确."""
        grad = element_scalar_gradient(ElementType.TRIA3, TRIA_COORDS, values)
        np.testing.assert_allclose(grad, expected, atol=1e-14)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (np.array([0.0, 1.0, 1.0, 0.0]), (1.0, 0.0)),  # V = x
            (np.array([0.0, 0.0, 1.0, 1.0]), (0.0, 1.0)),  # V = y
        ],
        ids=["vx", "vy"],
    )
    def test_quad4_exact_linear(self, values: np.ndarray, expected: tuple[float, float]) -> None:
        """Q4 高斯平均对线性场精确."""
        grad = element_scalar_gradient(ElementType.QUAD4, QUAD_COORDS, values)
        np.testing.assert_allclose(grad, expected, atol=1e-14)

    def test_unsupported_element_type(self) -> None:
        """不支持单元类型抛 ElementError."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="暂不支持"):
            element_scalar_gradient(ElementType.TRUSS2, coords, np.zeros(2))


class TestElementFieldLoad:
    """场能一致节点载荷（Joule 热单元贡献）."""

    def test_tria3_uniform_field(self) -> None:
        """V = x 场（梯度 1）：q = c，载荷均分 q·t·A/3 到三节点."""
        load = element_field_load(ElementType.TRIA3, TRIA_COORDS, 1.0, np.array([0.0, 1.0, 0.0]), 1.0)
        np.testing.assert_allclose(load, np.full(3, 1.0 / 6.0), rtol=1e-14)  # 1·1·0.5/3

    def test_quad4_linear_field_integral(self) -> None:
        """V = x 场：Q4 一致载荷合计 = q·t·A（积分守恒）."""
        load = element_field_load(ElementType.QUAD4, QUAD_COORDS, 1.0, np.array([0.0, 1.0, 1.0, 0.0]), 1.0)
        assert load.sum() == pytest.approx(1.0, rel=1e-12)  # q=1, t=1, A=1
        assert np.all(load > 0.0)

    def test_zero_field_zero_load(self) -> None:
        """常场（零梯度）无 Joule 热."""
        load = element_field_load(ElementType.TRIA3, TRIA_COORDS, 1.0, np.ones(3), 1.0)
        np.testing.assert_allclose(load, np.zeros(3), atol=1e-14)

    def test_unsupported_element_type(self) -> None:
        """不支持单元类型抛 ElementError."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ElementError, match="暂不支持"):
            element_field_load(ElementType.TRUSS2, coords, 1.0, np.zeros(2), 1.0)


class TestAssembleConduction:
    """全局装配."""

    def test_field_selects_coefficient(self) -> None:
        """field=electric 用电导率、thermal 用导热系数（矩阵按系数比例）."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        k_electric = assemble_conduction(mesh, (MAT,), (UNIT,), "electric")
        k_thermal = assemble_conduction(mesh, (MAT,), (UNIT,), "thermal")
        ratio = MAT.electric_sigma / MAT.thermal_k
        np.testing.assert_allclose(k_electric.toarray(), ratio * k_thermal.toarray(), rtol=1e-14)

    def test_single_element_matches_elemental(self) -> None:
        """单单元网格装配结果与单元矩阵一致."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        assembled = assemble_conduction(mesh, (ConductionMaterial(1.0, 1.0),), (UNIT,), "electric").toarray()
        elemental = element_conductance(ElementType.QUAD4, QUAD_COORDS, 1.0, 1.0)
        np.testing.assert_allclose(assembled, elemental, rtol=1e-14)

    def test_3d_mesh_rejected(self) -> None:
        """3D 网格抛 MeshError（v1 仅 2D）."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mesh = _single_block_mesh(ElementType.TRUSS2, coords)
        with pytest.raises(MeshError, match="仅支持 2D"):
            assemble_conduction(mesh, (MAT,), (UNIT,), "electric")

    def test_structural_element_rejected(self) -> None:
        """2D 结构单元（TRUSS2）抛 MeshError."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        mesh = _single_block_mesh(ElementType.TRUSS2, coords)
        with pytest.raises(MeshError, match="暂不支持单元类型"):
            assemble_conduction(mesh, (MAT,), (UNIT,), "electric")

    def test_material_index_out_of_range(self) -> None:
        """材料索引越界抛 MeshError."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        with pytest.raises(MeshError, match="材料索引"):
            assemble_conduction(mesh, (), (UNIT,), "electric")

    def test_section_index_out_of_range(self) -> None:
        """截面索引越界抛 MeshError."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        with pytest.raises(MeshError, match="截面索引"):
            assemble_conduction(mesh, (MAT,), (), "thermal")
