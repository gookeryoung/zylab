"""fea.electrothermal 电-热耦合测试：解析解互证、对流能量守恒、瞬态收敛与进度回调."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    Convection,
    ElectricCase,
    ElementBlock,
    ElementType,
    Mesh,
    NodalValue,
    Section,
    ThermalCase,
    solve_electrothermal,
    solve_electrothermal_transient,
)

__all__ = []

MAT = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0)

#: 提供热容的材料（ρc = 1 J/mm³·K）
CAP_MAT = ConductionMaterial(electric_sigma=1.0, thermal_k=1.0, volumetric_heat_capacity=1.0)

UNIT = Section()


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


def _closed_convection_nodes(mesh: Mesh) -> tuple[int, ...]:
    """四条边闭合连续对流折线（角节点不重复）."""
    coords = mesh.coords
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    top = np.flatnonzero(coords[:, 1] >= coords[:, 1].max() - 1e-9)
    right = np.flatnonzero(coords[:, 0] >= coords[:, 0].max() - 1e-9)
    bottom = np.flatnonzero(coords[:, 1] <= 0.0)
    return tuple(int(n) for n in (*left, *top[1:-1], *right[::-1], *bottom[::-1][1:-1]))


class TestSolveElectrothermal:
    """电-热顺序耦合求解."""

    def test_joule_heating_analytic(self) -> None:
        """两端电压 + 两端恒温：电压线性、P = σtH·V0²/L、T_max = qL²/8k."""
        length, height, v0 = 2.0, 1.0, 1.0
        mesh = _plate(4, 2, length, height)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        electric_case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
        )
        thermal_case = ThermalCase(
            temperatures=tuple(NodalValue(int(n), 0.0) for n in (*left, *right)),
        )
        solution = solve_electrothermal(mesh, (MAT,), (UNIT,), electric_case, thermal_case)

        # 电压场线性：V = V0·x/L
        np.testing.assert_allclose(solution.voltages, v0 * coords[:, 0] / length, rtol=1e-11)
        # 总电功率 = σ·t·H·V0²/L = 0.5 W
        assert solution.total_power == pytest.approx(MAT.electric_sigma * 1.0 * height * v0**2 / length, rel=1e-11)
        # 均匀 Joule 热源 q = σ(V0/L)²，两端恒温抛物线温度场
        q = MAT.electric_sigma * (v0 / length) ** 2
        assert solution.t_max == pytest.approx(q * length**2 / (8.0 * MAT.thermal_k), rel=1e-8)
        assert solution.t_min == pytest.approx(0.0, abs=1e-14)
        # property 委托
        np.testing.assert_array_equal(solution.temperatures, solution.thermal.temperatures)
        np.testing.assert_array_equal(solution.voltages, solution.electric.voltages)
        assert solution.t_max == solution.thermal.t_max
        assert solution.t_min == solution.thermal.t_min

    def test_convection_energy_balance(self) -> None:
        """全边对流 + 左右电极：稳态全部电功率经对流散出（能量守恒）."""
        length, height, v0 = 1.0, 1.0, 1.0
        mesh = _plate(2, 2, length, height)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        electric_case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
        )
        thermal_case = ThermalCase(
            convections=(Convection(nodes=_closed_convection_nodes(mesh), h_coeff=1.0, t_ambient=0.0),),
        )
        solution = solve_electrothermal(mesh, (MAT,), (UNIT,), electric_case, thermal_case)
        assert solution.thermal.convection_heat == pytest.approx(solution.total_power, rel=1e-10)
        assert solution.t_max > 0.0

    def test_report_progress(self) -> None:
        """进度回调按阶段上报并以 1.0 收尾."""
        mesh = _plate(1, 1)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= 2.0 - 1e-9)
        electric_case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), 1.0) for n in right)),
        )
        thermal_case = ThermalCase(
            temperatures=tuple(NodalValue(int(n), 0.0) for n in (*left, *right)),
        )
        events: list[tuple[float, str]] = []
        solve_electrothermal(
            mesh, (MAT,), (UNIT,), electric_case, thermal_case, report=lambda p, m: events.append((p, m))
        )
        assert events[-1] == (1.0, "电-热耦合求解完成")
        progresses = [p for p, _ in events]
        assert progresses == sorted(progresses)


class TestSolveElectrothermalTransient:
    """瞬态电-热顺序耦合求解."""

    def test_long_time_matches_steady(self) -> None:
        """两端电压 + 两端恒温：总时长充分大时瞬态末帧收敛到稳态耦合解."""
        length, height, v0 = 2.0, 1.0, 1.0
        mesh = _plate(4, 2, length, height)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        electric_case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
        )
        thermal_case = ThermalCase(
            temperatures=tuple(NodalValue(int(n), 0.0) for n in (*left, *right)),
        )
        steady = solve_electrothermal(mesh, (CAP_MAT,), (UNIT,), electric_case, thermal_case)
        solution = solve_electrothermal_transient(
            mesh,
            (CAP_MAT,),
            (UNIT,),
            electric_case,
            thermal_case,
            initial=np.zeros(mesh.n_nodes),
            total_time=200.0,
            n_steps=40,
        )
        # 常物性：电场与稳态一致
        np.testing.assert_allclose(solution.voltages, steady.voltages, rtol=1e-12)
        assert solution.total_power == pytest.approx(steady.total_power, rel=1e-12)
        # 末帧温度收敛稳态
        np.testing.assert_allclose(solution.temperatures, steady.temperatures, rtol=1e-8, atol=1e-8)
        assert solution.thermal.temperatures.shape == (41, mesh.n_nodes)
        # property 委托
        np.testing.assert_array_equal(solution.temperatures, solution.thermal.temperatures[-1])
        assert solution.t_max == solution.thermal.t_max

    def test_monotonic_heating_from_cold_start(self) -> None:
        """冷启动（初温 = 恒温边界温度）：峰值温度从零单调升且未达稳态值."""
        length, height, v0 = 2.0, 1.0, 1.0
        mesh = _plate(4, 2, length, height)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        electric_case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
        )
        thermal_case = ThermalCase(
            temperatures=tuple(NodalValue(int(n), 0.0) for n in (*left, *right)),
        )
        solution = solve_electrothermal_transient(
            mesh,
            (CAP_MAT,),
            (UNIT,),
            electric_case,
            thermal_case,
            initial=np.zeros(mesh.n_nodes),
            total_time=4.0,
            n_steps=20,
        )
        peaks = solution.thermal.temperatures.max(axis=1)
        assert np.all(np.diff(peaks) > 0.0)  # 峰值单调上升
        assert peaks[-1] > 0.0
        # 中间帧峰值介于初值与稳态解析值之间
        q = CAP_MAT.electric_sigma * (v0 / length) ** 2
        steady_peak = q * length**2 / (8.0 * CAP_MAT.thermal_k)
        assert peaks[-1] < steady_peak
