"""fea.export 结果导出测试：六类解 CSV 内容、目录创建与错误路径."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from zylab.fea import (
    BucklingSolution,
    ElectricSolution,
    ElectroThermalSolution,
    HarmonicResponse,
    Mesh,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    ThermalSolution,
    TransientSolution,
    export_csv,
)
from zylab.fea.mesh import ElementBlock, ElementType

__all__ = []


def _mesh() -> Mesh:
    """两节点单杆网格（2 DOF/节点）."""
    return Mesh(
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        blocks=(ElementBlock(etype=ElementType.TRUSS2, conn=np.array([[0, 1]])),),
    )


def _read(path: Path) -> list[list[str]]:
    """读 CSV 全部行."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_static_displacement_table(tmp_path: Path) -> None:
    """静力解导出逐节点位移表（表头 + 行数 + 数值）."""
    solution = StaticSolution(
        mesh=_mesh(),
        displacements=np.array([[0.0, 0.0], [1.5, -2.5]]),
        reactions={0: 1.0},
    )
    rows = _read(export_csv(solution, tmp_path / "static.csv"))
    assert rows[0] == ["node", "u0", "u1"]
    assert rows[1] == ["0", "0", "0"]
    assert rows[2][0] == "1"
    assert float(rows[2][1]) == 1.5
    assert float(rows[2][2]) == -2.5


def test_modal_frequency_table(tmp_path: Path) -> None:
    """模态解导出阶次/圆频率/频率表."""
    solution = ModalSolution(mesh=_mesh(), frequencies=np.array([2.0, 6.0]), mode_shapes=np.zeros((4, 2)))
    rows = _read(export_csv(solution, tmp_path / "modal.csv"))
    assert rows[0] == ["mode", "omega_rad_s", "freq_hz"]
    assert len(rows) == 3
    assert float(rows[1][1]) == 2.0
    assert float(rows[1][2]) == pytest.approx(2.0 / (2.0 * np.pi), rel=1e-9)


def test_buckling_factor_table(tmp_path: Path) -> None:
    """屈曲解导出阶次/载荷因子表."""
    reference = StaticSolution(mesh=_mesh(), displacements=np.zeros((2, 2)), reactions={})
    solution = BucklingSolution(
        mesh=_mesh(),
        load_factors=np.array([2.5, 9.0]),
        mode_shapes=np.zeros((4, 2)),
        reference=reference,
    )
    rows = _read(export_csv(solution, tmp_path / "buckling.csv"))
    assert rows[0] == ["mode", "load_factor"]
    assert float(rows[2][1]) == 9.0


def test_harmonic_amplitude_series(tmp_path: Path) -> None:
    """谐响应导出频率/全场峰值幅值序列."""
    solution = HarmonicResponse(
        mesh=_mesh(),
        frequencies=np.array([1.0, 2.0]),
        displacements=np.array([[1.0, 0.0], [3.0j, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    rows = _read(export_csv(solution, tmp_path / "harmonic.csv"))
    assert rows[0] == ["omega_rad_s", "max_amplitude"]
    assert float(rows[1][1]) == 3.0  # |3j|
    assert float(rows[2][1]) == 0.0


def test_transient_time_series(tmp_path: Path) -> None:
    """瞬态导出时间/全场最大位移序列."""
    solution = TransientSolution(
        mesh=_mesh(),
        times=np.array([0.0, 0.1]),
        displacements=np.array([[0.0, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        velocities=np.zeros((4, 2)),
        accelerations=np.zeros((4, 2)),
    )
    rows = _read(export_csv(solution, tmp_path / "transient.csv"))
    assert rows[0] == ["t", "max_abs_u"]
    assert float(rows[2][1]) == 0.2


def test_nonlinear_history_series(tmp_path: Path) -> None:
    """非线性导出载荷因子/最大位移历程（含零位移起始帧）."""
    solution = NonlinearSolution(
        mesh=_mesh(),
        displacements=np.array([[0.0, 0.0], [0.0, 0.5]]),
        load_factor=1.0,
        iterations=(2,),
        residual_norm=1e-10,
        converged=True,
        history_factors=np.array([0.0, 1.0]),
        history_displacements=np.zeros((2, 2, 2)),
    )
    solution.history_displacements[1, 1, 1] = 0.5
    rows = _read(export_csv(solution, tmp_path / "nonlinear.csv"))
    assert rows[0] == ["load_factor", "max_abs_u"]
    assert len(rows) == 3
    assert float(rows[2][1]) == 0.5


def test_electrothermal_node_table(tmp_path: Path) -> None:
    """电热耦合解导出逐节点电压/温度表."""
    mesh = _mesh()
    electric = ElectricSolution(
        mesh=mesh,
        voltages=np.array([0.0, 1.0]),
        element_gradients=np.zeros((1, 2)),
        element_power=np.array([0.5]),
        total_power=0.5,
    )
    thermal = ThermalSolution(
        mesh=mesh,
        temperatures=np.array([20.0, 25.0]),
        element_gradients=np.zeros((1, 2)),
        element_heat_flux=np.zeros(1),
        t_min=20.0,
        t_max=25.0,
    )
    solution = ElectroThermalSolution(mesh=mesh, electric=electric, thermal=thermal, total_power=0.5)
    rows = _read(export_csv(solution, tmp_path / "et.csv"))
    assert rows[0] == ["node", "voltage", "temperature"]
    assert len(rows) == 3
    assert float(rows[1][1]) == 0.0
    assert float(rows[1][2]) == 20.0
    assert float(rows[2][1]) == 1.0
    assert float(rows[2][2]) == 25.0


def test_creates_parent_directory(tmp_path: Path) -> None:
    """目标父目录不存在时自动创建."""
    solution = ModalSolution(mesh=_mesh(), frequencies=np.array([1.0]), mode_shapes=np.zeros((4, 1)))
    path = export_csv(solution, tmp_path / "a" / "b" / "out.csv")
    assert path.exists()


def test_unsupported_type_raises(tmp_path: Path) -> None:
    """不支持的结果类型抛 ValueError."""
    with pytest.raises(ValueError, match="不支持导出"):
        export_csv(object(), tmp_path / "x.csv")
