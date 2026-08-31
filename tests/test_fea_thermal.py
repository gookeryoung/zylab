"""fea.thermal 稳态热传导测试：线性场 patch、均匀热源抛物线解、对流能量守恒与工况校验."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    Convection,
    ElementBlock,
    ElementType,
    Mesh,
    MeshError,
    NodalSource,
    NodalValue,
    Section,
    ThermalCase,
    solve_thermal,
)

__all__ = []

MAT = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0)
UNIT = Section()

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


def _plate(nx: int, ny: int, length: float = 2.0, height: float = 1.0) -> Mesh:
    """矩形板 Q4 网格（节点编号 = j*(nx+1)+i）."""
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            conn.append((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
    return Mesh(coords, (ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn)),))


def _uniform_heat(mesh: Mesh, q: float) -> np.ndarray:
    """均匀体热源的一致节点载荷（每单元 q·A_e/4 均分四节点）."""
    load = np.zeros(mesh.n_nodes)
    for conn in mesh.blocks[0].conn:
        u = mesh.coords[conn[1]] - mesh.coords[conn[0]]
        v = mesh.coords[conn[3]] - mesh.coords[conn[0]]
        ae = abs(u[0] * v[1] - u[1] * v[0])  # 矩形单元面积（鞋带公式）
        load[conn] += q * ae / 4.0
    return load


def _closed_convection_nodes(mesh: Mesh) -> tuple[int, ...]:
    """四条边的闭合连续对流折线（沿逆时针，角节点不重复）."""
    coords = mesh.coords
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    top = np.flatnonzero(coords[:, 1] >= coords[:, 1].max() - 1e-9)
    right = np.flatnonzero(coords[:, 0] >= coords[:, 0].max() - 1e-9)
    bottom = np.flatnonzero(coords[:, 1] <= 0.0)
    return tuple(int(n) for n in (*left, *top[1:-1], *right[::-1], *bottom[::-1][1:-1]))


class TestSolveThermal:
    """稳态热传导求解."""

    def test_linear_field_patch(self) -> None:
        """四边给定线性温度场 T=x：内部节点精确复现、热流密度 = k|∇T|."""
        mesh = _plate(4, 2)
        coords = mesh.coords
        boundary = [
            n
            for n in range(mesh.n_nodes)
            if coords[n, 0] <= 0.0 or coords[n, 0] >= 2.0 - 1e-9 or coords[n, 1] <= 0.0 or coords[n, 1] >= 1.0 - 1e-9
        ]
        case = ThermalCase(temperatures=tuple(NodalValue(n, float(coords[n, 0])) for n in boundary))
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case)
        expected = coords[:, 0]
        np.testing.assert_allclose(solution.temperatures, expected, rtol=1e-11)
        np.testing.assert_allclose(solution.element_heat_flux, np.full(mesh.n_elements, MAT.thermal_k), rtol=1e-10)
        assert solution.t_max == pytest.approx(2.0, rel=1e-11)
        assert solution.t_min == pytest.approx(0.0, abs=1e-14)

    def test_uniform_source_parabolic(self) -> None:
        """两端恒温 + 均匀热源：抛物线温度场，T_max = qL²/8k（等距网格精确）."""
        length, q = 2.0, 0.25
        mesh = _plate(4, 2, length, 1.0)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        case = ThermalCase(temperatures=tuple(NodalValue(int(n), 0.0) for n in (*left, *right)))
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case, extra_heat=_uniform_heat(mesh, q))
        assert solution.t_max == pytest.approx(q * length**2 / (8.0 * MAT.thermal_k), rel=1e-8)
        assert solution.t_min == pytest.approx(0.0, abs=1e-14)
        assert solution.convection_heat == pytest.approx(0.0, abs=1e-14)
        # 抛物线对称：中点截面温度最高，两端为零
        mid = solution.temperatures.reshape(3, 5)  # (ny+1, nx+1)
        assert mid[:, 2].min() == pytest.approx(solution.t_max, rel=1e-8)

    def test_node_heat_source_balance(self) -> None:
        """节点热源工况：单节点注入热量经恒温边界流出，注入点温度最高."""
        mesh = _plate(2, 1)
        case = ThermalCase(
            temperatures=(NodalValue(0, 0.0), NodalValue(2, 0.0), NodalValue(3, 0.0), NodalValue(5, 0.0)),
            heat_sources=(NodalSource(1, 1.0),),  # 内部节点注入 1 W
        )
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case)
        assert solution.t_max > 0.0
        assert solution.temperatures[1] == solution.t_max  # 注入点温度最高

    def test_convection_energy_balance(self) -> None:
        """全边对流 + 均匀热源：稳态全部热量经对流散出（离散精确守恒）."""
        mesh = _plate(2, 2, 1.0, 1.0)
        q_total = 1.0  # q·t·A = 1·1·1
        case = ThermalCase(
            convections=(Convection(nodes=_closed_convection_nodes(mesh), h_coeff=1.0, t_ambient=0.0),),
        )
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case, extra_heat=_uniform_heat(mesh, q_total))
        assert solution.convection_heat == pytest.approx(q_total, rel=1e-10)
        assert solution.t_max > 0.0
        assert solution.t_min > 0.0  # 无 Dirichlet，全场高于环境温度

    def test_pure_convection_decays_to_ambient(self) -> None:
        """无热源全边对流：温度场收敛到环境温度."""
        mesh = _plate(2, 2, 1.0, 1.0)
        case = ThermalCase(
            convections=(Convection(nodes=_closed_convection_nodes(mesh), h_coeff=0.5, t_ambient=30.0),),
        )
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case)
        np.testing.assert_allclose(solution.temperatures, np.full(mesh.n_nodes, 30.0), rtol=1e-10)
        assert solution.convection_heat == pytest.approx(0.0, abs=1e-10)

    def test_extra_heat_shape_mismatch(self) -> None:
        """附加热载荷维度不符抛 MeshError."""
        mesh = _plate(1, 1)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        with pytest.raises(MeshError, match="不符"):
            solve_thermal(mesh, (MAT,), (UNIT,), case, extra_heat=np.zeros(3))

    def test_report_progress(self) -> None:
        """进度回调按阶段上报并以 1.0 收尾."""
        mesh = _plate(1, 1)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        events: list[tuple[float, str]] = []
        solve_thermal(mesh, (MAT,), (UNIT,), case, report=lambda p, m: events.append((p, m)))
        assert events[-1] == (1.0, "温度场求解完成")
        progresses = [p for p, _ in events]
        assert progresses == sorted(progresses)

    def test_3d_face_convection_steady(self) -> None:
        """单 HEX8 立方体：底面恒温 100、顶面对流（h=1, T∞=0）→ 顶面 T=50（热阻串联解析）."""
        mesh = Mesh(HEX_COORDS, (ElementBlock(etype=ElementType.HEX8, conn=np.arange(8)[None, :]),))
        case = ThermalCase(
            temperatures=tuple(NodalValue(n, 100.0) for n in (0, 1, 2, 3)),
            convections=(Convection(faces=((4, 5, 6, 7),), h_coeff=1.0, t_ambient=0.0),),
        )
        solution = solve_thermal(mesh, (MAT,), (UNIT,), case)
        np.testing.assert_allclose(solution.temperatures[4:], np.full(4, 50.0), rtol=1e-12)
        assert solution.t_max == pytest.approx(100.0, rel=1e-12)
        assert solution.t_min == pytest.approx(50.0, rel=1e-12)
        # 对流散热 = h·(T_top - 0)·A = 1×50×1
        assert solution.convection_heat == pytest.approx(50.0, rel=1e-12)
        # 顶面 4 节点温度相同 → 顶面等温，热流沿 z 一维
        assert solution.element_gradients.shape == (1, 3)
        np.testing.assert_allclose(solution.element_gradients, [[0.0, 0.0, -50.0]], rtol=1e-10, atol=1e-10)


class TestThermalCaseValidation:
    """热学工况校验."""

    def test_temperature_node_out_of_range(self) -> None:
        """给定温度节点越界抛 MeshError."""
        mesh = _plate(1, 1)
        case = ThermalCase(temperatures=(NodalValue(99, 0.0),))
        with pytest.raises(MeshError, match="越界"):
            solve_thermal(mesh, (MAT,), (UNIT,), case)

    def test_heat_source_node_out_of_range(self) -> None:
        """热源节点越界抛 MeshError."""
        mesh = _plate(1, 1)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),), heat_sources=(NodalSource(99, 1.0),))
        with pytest.raises(MeshError, match="越界"):
            solve_thermal(mesh, (MAT,), (UNIT,), case)

    def test_convection_node_out_of_range(self) -> None:
        """对流折线节点越界抛 MeshError."""
        mesh = _plate(1, 1)
        case = ThermalCase(
            temperatures=(NodalValue(0, 0.0),),
            convections=(Convection(nodes=(0, 99), h_coeff=1.0, t_ambient=0.0),),
        )
        with pytest.raises(MeshError, match="越界"):
            solve_thermal(mesh, (MAT,), (UNIT,), case)

    def test_convection_face_node_out_of_range(self) -> None:
        """对流面片节点越界抛 MeshError."""
        mesh = Mesh(HEX_COORDS, (ElementBlock(etype=ElementType.HEX8, conn=np.arange(8)[None, :]),))
        case = ThermalCase(
            temperatures=(NodalValue(0, 0.0),),
            convections=(Convection(faces=((4, 5, 6, 99),), h_coeff=1.0, t_ambient=0.0),),
        )
        with pytest.raises(MeshError, match="越界"):
            solve_thermal(mesh, (MAT,), (UNIT,), case)


class TestConvectionValidation:
    """对流边界参数校验."""

    def test_too_few_nodes(self) -> None:
        """对流边界节点数不足抛 MeshError."""
        with pytest.raises(MeshError, match="至少需要 2 个节点"):
            Convection(nodes=(0,), h_coeff=1.0, t_ambient=0.0)

    def test_non_positive_coefficient(self) -> None:
        """对流换热系数非正抛 MeshError."""
        with pytest.raises(MeshError, match="须为正"):
            Convection(nodes=(0, 1), h_coeff=0.0, t_ambient=0.0)

    def test_face_with_wrong_node_count(self) -> None:
        """对流面片节点数非 4 抛 MeshError."""
        with pytest.raises(MeshError, match="4 节点四边形"):
            Convection(faces=((0, 1, 2),), h_coeff=1.0, t_ambient=0.0)
