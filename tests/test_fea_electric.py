"""fea.electric 稳态电传导测试：电压线性分布、耗散功率解析解、电流注入与工况校验."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    ElectricCase,
    ElementBlock,
    ElementType,
    Mesh,
    MeshError,
    NodalSource,
    NodalValue,
    Section,
    solve_electric,
)

__all__ = []

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


def _electrode_case(mesh: Mesh, length: float, v0: float) -> ElectricCase:
    """左右电极电压工况（左 0 右 V0）."""
    coords = mesh.coords
    left = np.flatnonzero(coords[:, 0] <= 0.0)
    right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
    return ElectricCase(
        voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
    )


class TestSolveElectric:
    """稳态电传导求解."""

    def test_voltage_linear_and_power(self) -> None:
        """两端给定电压：电压线性分布、总功率 = σtH·V0²/L 解析解."""
        length, height, v0 = 2.0, 1.0, 1.0
        mesh = _plate(4, 2, length, height)
        solution = solve_electric(mesh, (MAT,), (UNIT,), _electrode_case(mesh, length, v0))
        expected = v0 * mesh.coords[:, 0] / length
        np.testing.assert_allclose(solution.voltages, expected, rtol=1e-11)
        # P = V0²/R = σ·t·H·V0²/L
        assert solution.total_power == pytest.approx(1.0 * 1.0 * 1.0 * v0**2 / length, rel=1e-11)
        assert solution.element_power.sum() == pytest.approx(solution.total_power, rel=1e-12)
        # 单元梯度 ∇V = (V0/L, 0)（电场强度 E = -∇V 沿 -x）
        np.testing.assert_allclose(
            solution.element_gradients, np.tile((v0 / length, 0.0), (mesh.n_elements, 1)), rtol=1e-10, atol=1e-12
        )

    def test_current_injection(self) -> None:
        """单单元板电流注入：右端电压 = I·R（Galerkin 对线性电流场精确）."""
        length, height, current = 2.0, 1.0, 1.0
        mesh = _plate(1, 1, length, height)
        coords = mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= length - 1e-9)
        case = ElectricCase(
            voltages=tuple(NodalValue(int(n), 0.0) for n in left),
            currents=tuple(NodalSource(int(n), current / right.size) for n in right),
        )
        solution = solve_electric(mesh, (MAT,), (UNIT,), case)
        resistance = length / (MAT.electric_sigma * UNIT.thickness * height)
        expected_v = current * resistance
        np.testing.assert_allclose(solution.voltages[right], expected_v, rtol=1e-12)
        # 输入电功率 P = V·I
        assert solution.total_power == pytest.approx(expected_v * current, rel=1e-11)

    def test_report_progress(self) -> None:
        """进度回调按阶段上报并以 1.0 收尾."""
        mesh = _plate(2, 2)
        events: list[tuple[float, str]] = []
        solve_electric(
            mesh, (MAT,), (UNIT,), _electrode_case(mesh, 2.0, 1.0), report=lambda p, m: events.append((p, m))
        )
        assert events[0][0] < 1.0
        assert events[-1] == (1.0, "电场求解完成")
        progresses = [p for p, _ in events]
        assert progresses == sorted(progresses)

    def test_insulator_floating_nodes_grounded(self) -> None:
        """绝缘块（σ 极小）浮动节点自动接地 0V，导体区电压场不受影响."""
        length, v0 = 2.0, 1.0
        plate = _plate(2, 1, length, 1.0)
        # 绝缘区远置（节点 6-9），与导体区无共享节点
        ins_coords = np.array([[10.0, 0.0], [11.0, 0.0], [11.0, 1.0], [10.0, 1.0]])
        coords = np.vstack([plate.coords, ins_coords])
        conductor = ElementBlock(etype=ElementType.QUAD4, conn=plate.blocks[0].conn, material=0, section=0)
        insulator = ElementBlock(etype=ElementType.QUAD4, conn=np.arange(6, 10)[None, :], material=1, section=0)
        mesh = Mesh(coords, (conductor, insulator))
        # 电极仅作用于导体区左右边界（绝缘区不加电压，留给浮动节点逻辑）
        left = np.flatnonzero(plate.coords[:, 0] <= 0.0)
        right = np.flatnonzero(plate.coords[:, 0] >= length - 1e-9)
        case = ElectricCase(
            voltages=(*(NodalValue(int(n), 0.0) for n in left), *(NodalValue(int(n), v0) for n in right)),
        )
        mats = (MAT, ConductionMaterial(electric_sigma=1.0e-12, thermal_k=1.0))
        solution = solve_electric(mesh, mats, (UNIT,), case)
        # 导体区电压线性分布
        np.testing.assert_allclose(solution.voltages[:6], v0 * plate.coords[:, 0] / length, rtol=1e-11)
        # 浮动节点接地 0V，绝缘块零功率
        np.testing.assert_allclose(solution.voltages[6:], np.zeros(4), atol=1e-12)
        insulator_power = solution.element_power[plate.n_elements :]
        np.testing.assert_allclose(insulator_power, np.zeros(1), atol=1e-24)


class TestElectricCaseValidation:
    """电学工况校验."""

    def test_missing_voltage(self) -> None:
        """无给定电压（缺电位基准）抛 MeshError."""
        mesh = _plate(1, 1)
        with pytest.raises(MeshError, match="缺少给定电压"):
            solve_electric(mesh, (MAT,), (UNIT,), ElectricCase())

    def test_voltage_node_out_of_range(self) -> None:
        """给定电压节点越界抛 MeshError."""
        mesh = _plate(1, 1)
        case = ElectricCase(voltages=(NodalValue(99, 1.0),))
        with pytest.raises(MeshError, match="越界"):
            solve_electric(mesh, (MAT,), (UNIT,), case)

    def test_current_node_out_of_range(self) -> None:
        """注入电流节点越界抛 MeshError."""
        mesh = _plate(1, 1)
        case = ElectricCase(
            voltages=(NodalValue(0, 0.0),),
            currents=(NodalSource(99, 1.0),),
        )
        with pytest.raises(MeshError, match="越界"):
            solve_electric(mesh, (MAT,), (UNIT,), case)

    def test_duplicate_voltage_takes_first(self) -> None:
        """同节点重复给定电压取首个（后值忽略）."""
        mesh = _plate(1, 1)
        case = ElectricCase(voltages=(NodalValue(0, 0.0), NodalValue(0, 5.0), NodalValue(2, 1.0)))
        solution = solve_electric(mesh, (MAT,), (UNIT,), case)
        assert solution.voltages[0] == pytest.approx(0.0, abs=1e-14)
