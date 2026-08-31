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
    assemble_capacity,
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

#: 单位立方体（HEX8 节点序：ζ=-1 面 1-4，ζ=+1 面 5-8）
HEX_COORDS = np.array(
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

#: HEX8 节点自然坐标符号表（与实现同序）
_HEX_SIGNS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)

MAT = ConductionMaterial(electric_sigma=2.0, thermal_k=5.0)
UNIT = Section()


def _hex8_conductance_reference(scale: float) -> np.ndarray:
    """单位立方体 HEX8 传导参考矩阵：对角 scale/3、面邻 0、棱邻 -scale/12、体对 -scale/12."""

    def entry(i: int, j: int) -> float:
        if i == j:
            return scale / 3.0
        flips = int(np.count_nonzero(_HEX_SIGNS[i] != _HEX_SIGNS[j]))
        return scale * {1: 0.0, 2: -1.0 / 12.0, 3: -1.0 / 12.0}[flips]

    return np.array([[entry(i, j) for j in range(8)] for i in range(8)])


def _hex8_capacity_reference(rho_cp: float) -> np.ndarray:
    """单位立方体 HEX8 一致热容参考矩阵：对角 ρc/27、面邻 ρc/54、棱邻 ρc/108、体对 ρc/216."""

    def entry(i: int, j: int) -> float:
        flips = 0 if i == j else int(np.count_nonzero(_HEX_SIGNS[i] != _HEX_SIGNS[j]))
        return rho_cp * {0: 1.0 / 27.0, 1: 1.0 / 54.0, 2: 1.0 / 108.0, 3: 1.0 / 216.0}[flips]

    return np.array([[entry(i, j) for j in range(8)] for i in range(8)])


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

    def test_hex8_unit_cube(self) -> None:
        """单位立方体 HEX8：解析传导矩阵（对角/3、面邻 0、棱邻-/12、体对-/12），行和为零."""
        ke = element_conductance(ElementType.HEX8, HEX_COORDS, 1.0, 5.0)  # 厚度 5 应被忽略
        np.testing.assert_allclose(ke, _hex8_conductance_reference(1.0), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(ke.sum(axis=1), np.zeros(8), atol=1e-12)

    def test_hex8_coefficient_scales_linearly(self) -> None:
        """传导系数线性缩放 HEX8 矩阵."""
        base = element_conductance(ElementType.HEX8, HEX_COORDS, 1.0, 1.0)
        np.testing.assert_allclose(
            element_conductance(ElementType.HEX8, HEX_COORDS, 3.0, 1.0), 3.0 * base, rtol=1e-12, atol=1e-14
        )


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

    def test_hex8_exact_linear(self) -> None:
        """HEX8 高斯平均对线性场精确（V = x / V = z）."""
        grad_x = element_scalar_gradient(ElementType.HEX8, HEX_COORDS, np.array([0.0, 1.0, 1.0, 0.0] * 2))
        np.testing.assert_allclose(grad_x, (1.0, 0.0, 0.0), atol=1e-13)
        grad_z = element_scalar_gradient(
            ElementType.HEX8, HEX_COORDS, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        )
        np.testing.assert_allclose(grad_z, (0.0, 0.0, 1.0), atol=1e-13)


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

    def test_hex8_linear_field_integral(self) -> None:
        """V = x 场（单位立方体 |∇V|²=1）：HEX8 一致载荷合计 = σ·V（积分守恒）."""
        values = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        load = element_field_load(ElementType.HEX8, HEX_COORDS, 2.0, values, 5.0)  # 厚度应被忽略
        assert load.sum() == pytest.approx(2.0, rel=1e-12)  # σ=2 × 体积 1
        assert np.all(load > 0.0)


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

    def test_3d_structural_element_rejected(self) -> None:
        """3D 网格中的结构单元（TRUSS2）抛 MeshError（传导限 TRIA3/QUAD4/HEX8）."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mesh = _single_block_mesh(ElementType.TRUSS2, coords)
        with pytest.raises(MeshError, match="暂不支持单元类型"):
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

    def test_hex8_single_element_matches_elemental(self) -> None:
        """单 HEX8 网格装配与单元矩阵一致."""
        mesh = _single_block_mesh(ElementType.HEX8, HEX_COORDS)
        assembled = assemble_conduction(mesh, (ConductionMaterial(1.0, 1.0),), (UNIT,), "electric").toarray()
        elemental = element_conductance(ElementType.HEX8, HEX_COORDS, 1.0, 1.0)
        np.testing.assert_allclose(assembled, elemental, rtol=1e-12, atol=1e-14)

    def test_dim_mismatch_rejected_at_mesh_level(self) -> None:
        """3D 单元出现在 2D 网格在 Mesh 构造层即被拦截（传导层无需重复校验）."""
        coords_2d = HEX_COORDS[:, :2]
        with pytest.raises(MeshError, match="不支持 2D 网格"):
            _single_block_mesh(ElementType.HEX8, coords_2d)

    def test_electric_insulator_block_skipped(self) -> None:
        """电场装配跳过绝缘块（σ 极小），热装配保留绝缘块."""
        coords = np.vstack([QUAD_COORDS, QUAD_COORDS + np.array([10.0, 0.0])])
        conductor = ElementBlock(etype=ElementType.QUAD4, conn=np.arange(4)[None, :], material=0, section=0)
        insulator = ElementBlock(etype=ElementType.QUAD4, conn=np.arange(4, 8)[None, :], material=1, section=0)
        mesh = Mesh(coords, (conductor, insulator))
        mats = (ConductionMaterial(2.0, 5.0), ConductionMaterial(1.0e-12, 5.0))
        k_electric = assemble_conduction(mesh, mats, (UNIT,), "electric").toarray()
        np.testing.assert_allclose(k_electric[4:, :], np.zeros((4, 8)), atol=1e-15)
        expected = element_conductance(ElementType.QUAD4, QUAD_COORDS, 2.0, 1.0)
        np.testing.assert_allclose(k_electric[:4, :4], expected, rtol=1e-14)
        k_thermal = assemble_conduction(mesh, mats, (UNIT,), "thermal").toarray()
        assert np.linalg.norm(k_thermal[4:, 4:]) > 0.0


class TestAssembleCapacity:
    """热容矩阵装配（瞬态热 backward Euler 使用）."""

    def test_tria3_unit_triangle(self) -> None:
        """直角三角形一致热容：ρc·t·A/12·[[2,1,1],...]（矩阵总和 = ρc·t·A）."""
        mesh = _single_block_mesh(ElementType.TRIA3, TRIA_COORDS)
        mat = ConductionMaterial(1.0, 1.0, volumetric_heat_capacity=3.0)
        capacity = assemble_capacity(mesh, (mat,), (UNIT,)).toarray()
        expected = 3.0 * 0.5 / 12.0 * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        np.testing.assert_allclose(capacity, expected, rtol=1e-14)
        assert capacity.sum() == pytest.approx(3.0 * 1.0 * 0.5, rel=1e-14)

    def test_quad4_unit_square_row_sum(self) -> None:
        """单位方形 Q4 一致热容矩阵总和 = ρc·t·A（积分守恒）."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        mat = ConductionMaterial(1.0, 1.0, volumetric_heat_capacity=2.0)
        capacity = assemble_capacity(mesh, (mat,), (UNIT,)).toarray()
        assert capacity.sum() == pytest.approx(2.0 * 1.0 * 1.0, rel=1e-12)

    def test_hex8_unit_cube(self) -> None:
        """单位立方体 HEX8 一致热容解析矩阵（总和质量 = ρc·V）."""
        mesh = _single_block_mesh(ElementType.HEX8, HEX_COORDS)
        mat = ConductionMaterial(1.0, 1.0, volumetric_heat_capacity=4.0)
        capacity = assemble_capacity(mesh, (mat,), (UNIT,)).toarray()
        np.testing.assert_allclose(capacity, _hex8_capacity_reference(4.0), rtol=1e-12)
        assert capacity.sum() == pytest.approx(4.0, rel=1e-12)

    def test_zero_heat_capacity_contributes_nothing(self) -> None:
        """ρc=0（未提供）的块贡献零热容."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        capacity = assemble_capacity(mesh, (MAT,), (UNIT,))
        assert capacity.nnz == 0

    def test_material_index_out_of_range(self) -> None:
        """材料索引越界抛 MeshError."""
        mesh = _single_block_mesh(ElementType.QUAD4, QUAD_COORDS)
        with pytest.raises(MeshError, match="材料索引"):
            assemble_capacity(mesh, (), (UNIT,))
