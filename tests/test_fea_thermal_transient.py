"""fea.thermal_transient 瞬态热传导测试：backward Euler 收敛稳态、对流衰减、参数校验."""

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
    NodalValue,
    Section,
    ThermalCase,
    solve_thermal,
    solve_thermal_transient,
)

__all__ = []

#: 提供热容的材料（ρc = 1 J/mm³·K）
CAP_MAT = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0, volumetric_heat_capacity=1.0)

#: 无热容材料（瞬态应拒绝）
MAT = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0)

UNIT = Section()


def _plate(nx: int, ny: int, length: float = 2.0, height: float = 1.0) -> Mesh:
    """矩形板 Q4 网格（节点编号 = j*(nx+1)+i，单元逆时针）."""
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


def _closed_convection_nodes(mesh: Mesh) -> tuple[int, ...]:
    """四条边的闭合连续对流折线（沿逆时针，角节点不重复）."""
    coords = mesh.coords
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    top = np.flatnonzero(coords[:, 1] >= coords[:, 1].max() - 1e-9)
    right = np.flatnonzero(coords[:, 0] >= coords[:, 0].max() - 1e-9)
    bottom = np.flatnonzero(coords[:, 1] <= 0.0)
    return tuple(int(n) for n in (*left, *top[1:-1], *right[::-1], *bottom[::-1][1:-1]))


class TestSolveThermalTransient:
    """瞬态热传导求解."""

    def test_long_time_matches_steady(self) -> None:
        """两端恒温阶跃：总时长充分大时瞬态解收敛到稳态线性温度场."""
        length = 2.0
        mesh = _plate(4, 2, length, 1.0)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        case = ThermalCase(
            temperatures=(
                *(NodalValue(int(n), 100.0) for n in left),
                *(NodalValue(int(n), 0.0) for n in right),
            ),
        )
        steady = solve_thermal(mesh, (CAP_MAT,), (UNIT,), case)
        solution = solve_thermal_transient(
            mesh, (CAP_MAT,), (UNIT,), case, initial=np.zeros(mesh.n_nodes), total_time=200.0, n_steps=40
        )
        assert solution.temperatures.shape == (41, mesh.n_nodes)
        np.testing.assert_allclose(solution.times, np.linspace(0.0, 200.0, 41), rtol=1e-12)
        np.testing.assert_allclose(solution.temperatures[-1], steady.temperatures, rtol=1e-8, atol=1e-8)
        assert solution.t_max == pytest.approx(100.0, rel=1e-12)
        assert solution.t_min == pytest.approx(0.0, abs=1e-12)

    def test_convection_decay_to_ambient(self) -> None:
        """均匀初温 + 全边对流：温度单调衰减且末帧均值高于环境温度."""
        mesh = _plate(2, 2, 1.0, 1.0)
        case = ThermalCase(
            convections=(Convection(nodes=_closed_convection_nodes(mesh), h_coeff=0.5, t_ambient=30.0),),
        )
        solution = solve_thermal_transient(
            mesh, (CAP_MAT,), (UNIT,), case, initial=np.full(mesh.n_nodes, 100.0), total_time=2.0, n_steps=10
        )
        means = solution.temperatures.mean(axis=1)
        assert np.all(np.diff(means) < 0.0)  # 单调降温
        assert means[-1] > 30.0
        # 首帧均匀场换热量解析值：h·(T0-T∞)·折线总长（闭合折线长 3.5）
        assert solution.convection_heat[0] == pytest.approx(0.5 * 70.0 * 3.5, rel=1e-10)
        assert solution.convection_heat[-1] > 0.0

    def test_dirichlet_nodes_pinned_to_case_values(self) -> None:
        """给定温度节点在所有帧（含初始帧）恒为工况值（覆盖用户初值）."""
        mesh = _plate(2, 1, 2.0, 1.0)
        left = np.flatnonzero(mesh.coords[:, 0] <= 0.0)
        case = ThermalCase(temperatures=tuple(NodalValue(int(n), 50.0) for n in left))
        initial = np.full(mesh.n_nodes, 20.0)
        solution = solve_thermal_transient(mesh, (CAP_MAT,), (UNIT,), case, initial=initial, total_time=1.0, n_steps=5)
        for frame in solution.temperatures:
            np.testing.assert_allclose(frame[left], np.full(left.size, 50.0), rtol=1e-12)
        assert solution.temperatures[0].max() == pytest.approx(50.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("total_time", "n_steps", "match"),
        [(0.0, 5, "总时长"), (-1.0, 5, "总时长"), (1.0, 0, "时间步数"), (1.0, -3, "时间步数")],
        ids=["zero-time", "neg-time", "zero-steps", "neg-steps"],
    )
    def test_invalid_time_arguments(self, total_time: float, n_steps: int, match: str) -> None:
        """总时长/步数非法抛 MeshError."""
        mesh = _plate(2, 1, 1.0, 1.0)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        with pytest.raises(MeshError, match=match):
            solve_thermal_transient(
                mesh,
                (CAP_MAT,),
                (UNIT,),
                case,
                initial=np.zeros(mesh.n_nodes),
                total_time=total_time,
                n_steps=n_steps,
            )

    def test_initial_shape_mismatch(self) -> None:
        """初始温度维度不符抛 MeshError."""
        mesh = _plate(2, 1, 1.0, 1.0)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        with pytest.raises(MeshError, match="初始温度维度"):
            solve_thermal_transient(mesh, (CAP_MAT,), (UNIT,), case, initial=np.zeros(3), total_time=1.0, n_steps=2)

    def test_extra_heat_shape_mismatch(self) -> None:
        """附加热载荷维度不符抛 MeshError."""
        mesh = _plate(2, 1, 1.0, 1.0)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        with pytest.raises(MeshError, match="附加热载荷维度"):
            solve_thermal_transient(
                mesh,
                (CAP_MAT,),
                (UNIT,),
                case,
                initial=np.zeros(mesh.n_nodes),
                total_time=1.0,
                n_steps=2,
                extra_heat=np.zeros(4),
            )

    def test_missing_heat_capacity_rejected(self) -> None:
        """材料表未提供体积热容抛 MeshError."""
        mesh = _plate(2, 1, 1.0, 1.0)
        case = ThermalCase(temperatures=(NodalValue(0, 0.0),))
        with pytest.raises(MeshError, match="缺少热容"):
            solve_thermal_transient(
                mesh, (MAT,), (UNIT,), case, initial=np.zeros(mesh.n_nodes), total_time=1.0, n_steps=2
            )

    def test_final_frame_flux_recovered(self) -> None:
        """末帧单元热流与末帧温度梯度一致（k|∇T|）."""
        length = 2.0
        mesh = _plate(2, 1, length, 1.0)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        case = ThermalCase(
            temperatures=(
                *(NodalValue(int(n), 10.0) for n in left),
                *(NodalValue(int(n), 0.0) for n in right),
            ),
        )
        solution = solve_thermal_transient(
            mesh, (CAP_MAT,), (UNIT,), case, initial=np.zeros(mesh.n_nodes), total_time=100.0, n_steps=20
        )
        # 长时间后接近线性场：热流密度 → k·ΔT/L
        expected = CAP_MAT.thermal_k * 10.0 / length
        np.testing.assert_allclose(solution.element_heat_flux, np.full(mesh.n_elements, expected), rtol=5e-3)
