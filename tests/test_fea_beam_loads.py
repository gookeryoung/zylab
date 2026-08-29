"""梁单元解析解验证 + 边压力/体力等效节点力装配测试.

梁理论（Euler-Bernoulli）：
- 端部集中力：δ = PL³/(3EI)、θ = PL²/(2EI)（精确，Hermite 梁元对点载精确）；
- 端部集中力矩：δ = ML²/(2EI)、θ = ML/EI；
- 轴向：u = PL/(EA)；
- 均布载荷（经边压力折线施加，集总 pL/2）：随网格细化收敛于 qL⁴/(8EI)。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    BodyForce,
    Constraint,
    EdgePressure,
    ElementBlock,
    ElementError,
    ElementType,
    LinearElastic,
    Mesh,
    MeshError,
    NodalLoad,
    Section,
    StaticCase,
    StressState,
    solve_static,
)
from zylab.fea.assemble import assemble_loads
from zylab.fea.viewdata import deformed_coords, displacement_field

E_B = 1000.0
AREA_B = 2.0
INERTIA_B = 1.0
LENGTH_B = 10.0
EI_B = E_B * INERTIA_B


def _beam_mesh(n_elem: int, length: float = LENGTH_B) -> Mesh:
    """沿 x 轴的等分 BEAM2 梁网格."""
    coords = np.column_stack([np.linspace(0.0, length, n_elem + 1), np.zeros(n_elem + 1)])
    conn = np.column_stack([np.arange(n_elem), np.arange(1, n_elem + 1)])
    return Mesh(coords, (ElementBlock(ElementType.BEAM2, conn),))


def _beam_case(loads: tuple[NodalLoad, ...]) -> StaticCase:
    return StaticCase(constraints=(Constraint(0, (0, 1, 2)),), loads=loads)


def _solve_beam(n_elem: int, loads: tuple[NodalLoad, ...]):
    mesh = _beam_mesh(n_elem)
    material = LinearElastic(E_B)
    section = Section(area=AREA_B, inertia=INERTIA_B)
    return solve_static(mesh, [material], [section], _beam_case(loads))


# ---------------------------------------------------------------------------
# 梁解析解
# ---------------------------------------------------------------------------


class TestBeamCantilever:
    def test_tip_force_exact(self) -> None:
        """端部集中力：挠度/转角与梁理论精确一致（Hermite 梁元对点载精确）."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, -1.0, 0.0)),))
        tip = solution.node_displacement(1)
        np.testing.assert_allclose(tip[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(tip[1], -(LENGTH_B**3) / (3.0 * EI_B), rtol=1e-12)
        np.testing.assert_allclose(tip[2], -(LENGTH_B**2) / (2.0 * EI_B), rtol=1e-12)

    def test_tip_moment_exact(self) -> None:
        """端部集中力矩：挠度/转角与梁理论精确一致."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, 0.0, 1.0)),))
        tip = solution.node_displacement(1)
        np.testing.assert_allclose(tip[1], LENGTH_B**2 / (2.0 * EI_B), rtol=1e-12)
        np.testing.assert_allclose(tip[2], LENGTH_B / EI_B, rtol=1e-12)

    def test_axial_force_exact(self) -> None:
        """端部轴向力：u = PL/(EA)，σ = P/A."""
        solution = _solve_beam(1, (NodalLoad(1, (1.0, 0.0, 0.0)),))
        tip = solution.node_displacement(1)
        np.testing.assert_allclose(tip[0], LENGTH_B / (E_B * AREA_B), rtol=1e-12)
        np.testing.assert_allclose(tip[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(tip[2], 0.0, atol=1e-14)
        stress = solution.element_results[0].stress
        np.testing.assert_allclose(stress[0], 1.0 / AREA_B, rtol=1e-12)

    def test_tip_force_moment_recovery(self) -> None:
        """端部集中力弯矩恢复：固端 |M1| = PL、自由端 M2 = 0、轴向应力为 0."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, -1.0, 0.0)),))
        sigma, m1, m2 = solution.element_results[0].stress
        np.testing.assert_allclose(sigma, 0.0, atol=1e-12)
        np.testing.assert_allclose(m1, 1.0 * LENGTH_B, rtol=1e-10)
        np.testing.assert_allclose(m2, 0.0, atol=1e-10)

    def test_tip_moment_recovery(self) -> None:
        """端部集中力矩：全梁等弯矩 |M| = M."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, 0.0, 1.0)),))
        _, m1, m2 = solution.element_results[0].stress
        np.testing.assert_allclose(m1, -1.0, rtol=1e-10)
        np.testing.assert_allclose(m2, 1.0, rtol=1e-10)

    def test_strain_energy_exact(self) -> None:
        """应变能与卡氏定理一致：U = P²L³/(6EI)."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, -1.0, 0.0)),))
        np.testing.assert_allclose(solution.strain_energy, LENGTH_B**3 / (6.0 * EI_B), rtol=1e-10)

    def test_multi_element_matches_single(self) -> None:
        """多单元细分不改变解析解（梁元对点载精确，与单元数无关）."""
        solution = _solve_beam(4, (NodalLoad(4, (0.0, -1.0, 0.0)),))
        tip = solution.node_displacement(4)
        np.testing.assert_allclose(tip[1], -(LENGTH_B**3) / (3.0 * EI_B), rtol=1e-10)

    def test_viewdata_ignores_rotation(self) -> None:
        """变形坐标/位移模只取平动分量（忽略转角列）."""
        solution = _solve_beam(1, (NodalLoad(1, (0.0, -1.0, 0.0)),))
        field = displacement_field(solution)
        assert field.shape == (2,)
        deformed = deformed_coords(solution.mesh, solution.displacements)
        assert deformed.shape == (2, 2)
        np.testing.assert_allclose(deformed[1, 1], -(LENGTH_B**3) / (3.0 * EI_B), rtol=1e-12)


# ---------------------------------------------------------------------------
# 边压力：方向约定与集总节点力
# ---------------------------------------------------------------------------


def _square_mesh() -> Mesh:
    """单位正方形单 Q4 单元（节点逆时针）."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Mesh(coords, (ElementBlock(ElementType.QUAD4, np.arange(4).reshape(1, 4)),))


class TestEdgePressure:
    def test_positive_pressure_compressive(self) -> None:
        """正压力指向材料内部：底边（沿 +x 行进，材料在上）节点获得 +y 力 pL/2."""
        force = assemble_loads(_square_mesh(), StaticCase(edge_pressures=(EdgePressure((0, 1), 2.0),)))
        # 节点 0/1 各得 (0, +1.0)，其余为 0
        np.testing.assert_allclose(force[:2], [0.0, 1.0])
        np.testing.assert_allclose(force[2:4], [0.0, 1.0])
        np.testing.assert_allclose(force[4:], 0.0, atol=1e-15)

    def test_reversed_travel_flips_direction(self) -> None:
        """节点序反转（材料在右侧）：同号压力变为外向拉力."""
        force = assemble_loads(_square_mesh(), StaticCase(edge_pressures=(EdgePressure((1, 0), 2.0),)))
        np.testing.assert_allclose(force[:2], [0.0, -1.0])
        np.testing.assert_allclose(force[2:4], [0.0, -1.0])

    def test_polyline_segments(self) -> None:
        """三节点折线两段独立取法向：L 形底边 + 右边（材料在左）."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
        mesh = Mesh(coords, (ElementBlock(ElementType.QUAD4, np.arange(4).reshape(1, 4)),))
        force = assemble_loads(mesh, StaticCase(edge_pressures=(EdgePressure((0, 1, 2), 1.0),)))
        # 段 (0,1)：法向 (0,1)，长 1 → 节点 0/1 各 (0, 0.5)
        # 段 (1,2)：法向 (-1,0)，长 1 → 节点 1/2 各 (-0.5, 0)
        np.testing.assert_allclose(force[:2], [0.0, 0.5])
        np.testing.assert_allclose(force[2:4], [-0.5, 0.5])
        np.testing.assert_allclose(force[4:6], [-0.5, 0.0])

    def test_pressure_equilibrium_reaction(self) -> None:
        """边压力作用下约束反力合力与外载平衡（顶边外向拉力）."""
        mesh = _square_mesh()
        material = LinearElastic(1000.0, 0.3, StressState.PLANE_STRESS)
        case = StaticCase(
            constraints=(
                Constraint(0, (0, 1)),
                Constraint(3, (0, 1)),
            ),
            edge_pressures=(EdgePressure((2, 3), -1.0),),
        )
        solution = solve_static(mesh, [material], [Section(thickness=0.5)], case)
        # 顶边外向拉力 = 1.0（向上），左边缘固定端 y 反力（dof 1/7）合计 = -1.0
        total_y = solution.reactions.get(1, 0.0) + solution.reactions.get(7, 0.0)
        np.testing.assert_allclose(total_y, -1.0, rtol=1e-10)

    def test_beam_udl_converges(self) -> None:
        """梁上均布载荷（边压力折线，集总 pL/2）：收敛于 qL⁴/(8EI)."""
        exact = 1.0 * LENGTH_B**4 / (8.0 * EI_B)

        def tip_deflection(n_elem: int) -> float:
            mesh = _beam_mesh(n_elem)
            case = StaticCase(
                constraints=(Constraint(0, (0, 1, 2)),),
                edge_pressures=(EdgePressure(tuple(range(n_elem + 1)), -1.0),),
            )
            solution = solve_static(mesh, [LinearElastic(E_B)], [Section(area=AREA_B, inertia=INERTIA_B)], case)
            return float(solution.node_displacement(n_elem)[1])

        coarse = tip_deflection(2)
        fine = tip_deflection(16)
        # 2 单元集总解可手算：-65/48（节点 1 受 5、节点 2 受 2.5 的悬臂叠加，向下）
        np.testing.assert_allclose(coarse, -65.0 / 48.0, rtol=1e-12)
        assert abs(abs(fine) - exact) / exact < 0.02
        assert abs(abs(fine) - exact) < abs(abs(coarse) - exact)

    def test_duplicate_adjacent_nodes_raise(self) -> None:
        case = StaticCase(edge_pressures=(EdgePressure((0, 0), 1.0),))
        with pytest.raises(MeshError, match="段长为零"):
            assemble_loads(_square_mesh(), case)

    def test_requires_2d(self) -> None:
        coords = np.zeros((2, 3))
        mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 1]])),))
        case = StaticCase(edge_pressures=(EdgePressure((0, 1), 1.0),))
        with pytest.raises(MeshError, match="仅支持 2D"):
            assemble_loads(mesh, case)


# ---------------------------------------------------------------------------
# 体力：连续体一致节点力
# ---------------------------------------------------------------------------


class TestBodyForce:
    def test_square_gravity(self) -> None:
        """单位方板（厚 0.5）受 fy=-2：每节点 (0, -2·1·0.5/4)."""
        force = assemble_loads(_square_mesh(), StaticCase(body_forces=(BodyForce(fy=-2.0),)), [Section(thickness=0.5)])
        expected = np.zeros(8)
        expected[1::2] = -2.0 * 1.0 * 0.5 / 4.0
        np.testing.assert_allclose(force, expected)

    def test_hex8_volume_force(self) -> None:
        """单位立方体受 fx=1：每节点 1/8."""
        coords = np.array(
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
        mesh = Mesh(coords, (ElementBlock(ElementType.HEX8, np.arange(8).reshape(1, 8)),))
        force = assemble_loads(mesh, StaticCase(body_forces=(BodyForce(fx=1.0),)))
        np.testing.assert_allclose(force[0::3], 1.0 / 8.0)
        np.testing.assert_allclose(force[1::3], 0.0, atol=1e-15)
        np.testing.assert_allclose(force[2::3], 0.0, atol=1e-15)

    def test_skips_line_elements(self) -> None:
        """梁 + Q4 混合网格：体力只作用于连续体块，写入前 2 个平动 DOF."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]])
        beam = ElementBlock(ElementType.BEAM2, np.array([[0, 4]]))
        quad = ElementBlock(ElementType.QUAD4, np.arange(4).reshape(1, 4))
        mesh = Mesh(coords, (beam, quad))
        force = assemble_loads(mesh, StaticCase(body_forces=(BodyForce(fy=4.0),)), [Section(thickness=1.0)])
        # 仅 Q4（面积 1）受载：节点 0-3 各 (0, 1)，梁节点 4 的 3 个 DOF 全 0
        expected = np.zeros(15)
        expected[1] = 1.0
        expected[3 * 1 + 1] = 1.0
        expected[3 * 2 + 1] = 1.0
        expected[3 * 3 + 1] = 1.0
        np.testing.assert_allclose(force, expected)

    def test_missing_sections_raise(self) -> None:
        with pytest.raises(MeshError, match="体力装配须提供截面表"):
            assemble_loads(_square_mesh(), StaticCase(body_forces=(BodyForce(fy=1.0),)))

    def test_2d_fz_rejected(self) -> None:
        case = StaticCase(body_forces=(BodyForce(fz=1.0),))
        with pytest.raises(MeshError, match="z 分量须为 0"):
            assemble_loads(_square_mesh(), case)

    def test_gravity_cantilever_equilibrium(self) -> None:
        """方板自重：约束反力合力 = -总重力（体力过单元与节点载荷平衡）."""
        mesh = _square_mesh()
        material = LinearElastic(1000.0, 0.0, StressState.PLANE_STRESS)
        case = StaticCase(
            constraints=(
                Constraint(0, (0, 1)),
                Constraint(3, (0, 1)),
            ),
            body_forces=(BodyForce(fy=-10.0),),
        )
        solution = solve_static(mesh, [material], [Section(thickness=0.5)], case)
        # 总重力 = 10·1·0.5 = 5 向下，左边缘固定端 y 反力（dof 1/7）合计 = +5
        total_y = solution.reactions.get(1, 0.0) + solution.reactions.get(7, 0.0)
        np.testing.assert_allclose(total_y, 10.0 * 1.0 * 0.5, rtol=1e-10)


# ---------------------------------------------------------------------------
# 边界校验：梁网格宽度语义
# ---------------------------------------------------------------------------


class TestBeamCaseValidation:
    def test_beam_load_width(self) -> None:
        """梁网格节点载荷分量数须为 3（第 3 分量为弯矩）."""
        mesh = _beam_mesh(1)
        case = StaticCase(loads=(NodalLoad(1, (0.0, -1.0)),))
        with pytest.raises(MeshError, match="力分量数"):
            case.validate(mesh)

    def test_beam_constraint_rotation_dof(self) -> None:
        """梁网格可约束转角自由度（dof=2）."""
        mesh = _beam_mesh(1)
        case = StaticCase(constraints=(Constraint(0, (0, 1, 2)),))
        case.validate(mesh)

    def test_section_inertia_validation(self) -> None:
        with pytest.raises(ElementError, match="惯性矩"):
            Section(inertia=0.0)
