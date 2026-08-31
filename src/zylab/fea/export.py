"""结果导出：将各类求解结果写为 UTF-8 CSV（Qt-free，GUI 与 CLI 共用）.

导出语义按解类型分化：
- 场量型（静力）：逐节点位移表；
- 序列型（模态/屈曲/谐响应/瞬态/非线性）：逐条目标量序列（频率、载荷因子、
  峰值幅值、时程等），取全场统计量（不依赖观察点选择）.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .buckling import BucklingSolution
from .electrothermal import ElectroThermalSolution, ElectroThermalTransientSolution
from .harmonic import HarmonicResponse
from .modal import ModalSolution
from .nonlinear import NonlinearSolution
from .static import StaticSolution
from .transient import TransientSolution

__all__ = ["export_csv"]


def export_csv(solution: object, path: Path) -> Path:
    """将求解结果写为 CSV 文件，返回路径；不支持类型抛 :class:`ValueError`.

    :param solution: 求解结果对象（六类解之一）。
    :param path: 目标文件路径（建议 ``.csv`` 后缀）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: tuple[Sequence[object], ...]
    if isinstance(solution, StaticSolution):
        rows = _static_rows(solution)
    elif isinstance(solution, ModalSolution):
        rows = _modal_rows(solution)
    elif isinstance(solution, BucklingSolution):
        rows = _buckling_rows(solution)
    elif isinstance(solution, HarmonicResponse):
        rows = _harmonic_rows(solution)
    elif isinstance(solution, TransientSolution):
        rows = _transient_rows(solution)
    elif isinstance(solution, NonlinearSolution):
        rows = _nonlinear_rows(solution)
    elif isinstance(solution, ElectroThermalSolution):
        rows = _electrothermal_rows(solution)
    elif isinstance(solution, ElectroThermalTransientSolution):
        rows = _electrothermal_transient_rows(solution)
    else:
        raise ValueError(f"不支持导出的结果类型: {type(solution).__name__}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return path


def _static_rows(solution: StaticSolution) -> tuple[Sequence[object], ...]:
    """静力解：表头 + 逐节点位移行."""
    u = solution.displacements
    header: list[object] = ["node", *(f"u{i}" for i in range(u.shape[1]))]
    rows: list[Sequence[object]] = [header]
    for node in range(u.shape[0]):
        rows.append([node, *(f"{value:.10g}" for value in u[node])])
    return tuple(rows)


def _modal_rows(solution: ModalSolution) -> tuple[Sequence[object], ...]:
    """模态解：阶次 / 圆频率 / 频率（Hz）."""
    rows: list[Sequence[object]] = [("mode", "omega_rad_s", "freq_hz")]
    for i in range(solution.n_modes):
        rows.append([i + 1, f"{solution.frequencies[i]:.10g}", f"{solution.frequencies_hz[i]:.10g}"])
    return tuple(rows)


def _buckling_rows(solution: BucklingSolution) -> tuple[Sequence[object], ...]:
    """屈曲解：阶次 / 临界载荷因子."""
    rows: list[Sequence[object]] = [("mode", "load_factor")]
    for i in range(solution.n_modes):
        rows.append([i + 1, f"{float(solution.load_factors[i]):.10g}"])
    return tuple(rows)


def _harmonic_rows(solution: HarmonicResponse) -> tuple[Sequence[object], ...]:
    """谐响应：激励频率 / 全场峰值位移幅值."""
    rows: list[Sequence[object]] = [("omega_rad_s", "max_amplitude")]
    amplitudes = np.abs(solution.displacements).max(axis=0) if solution.displacements.size else np.zeros(0)
    for omega, amp in zip(solution.frequencies, amplitudes):
        rows.append([f"{omega:.10g}", f"{amp:.10g}"])
    return tuple(rows)


def _transient_rows(solution: TransientSolution) -> tuple[Sequence[object], ...]:
    """瞬态：时间站点 / 全场最大位移分量."""
    rows: list[Sequence[object]] = [("t", "max_abs_u")]
    peaks = np.abs(solution.displacements).max(axis=0) if solution.displacements.size else np.zeros(0)
    for t, peak in zip(solution.times, peaks):
        rows.append([f"{t:.10g}", f"{peak:.10g}"])
    return tuple(rows)


def _nonlinear_rows(solution: NonlinearSolution) -> tuple[Sequence[object], ...]:
    """非线性：载荷因子 / 全场最大位移模长（含零位移起始帧）."""
    rows: list[Sequence[object]] = [("load_factor", "max_abs_u")]
    if solution.history_displacements.size:
        norms = np.linalg.norm(solution.history_displacements, axis=2).max(axis=1)
    else:
        norms = np.zeros(0)
    for factor, peak in zip(solution.history_factors, norms):
        rows.append([f"{factor:.10g}", f"{peak:.10g}"])
    return tuple(rows)


def _electrothermal_rows(solution: ElectroThermalSolution) -> tuple[Sequence[object], ...]:
    """电-热耦合：逐节点电压与温度表."""
    rows: list[Sequence[object]] = [("node", "voltage", "temperature")]
    for node in range(solution.mesh.n_nodes):
        rows.append([node, f"{solution.voltages[node]:.10g}", f"{solution.temperatures[node]:.10g}"])
    return tuple(rows)


def _electrothermal_transient_rows(solution: ElectroThermalTransientSolution) -> tuple[Sequence[object], ...]:
    """瞬态电-热耦合：时间站点 / 全场温度峰值与谷值时程（电压常值不随时间变）."""
    rows: list[Sequence[object]] = [("t", "t_max", "t_min")]
    thermal = solution.thermal
    for t, frame in zip(thermal.times, thermal.temperatures):
        rows.append([f"{t:.10g}", f"{float(frame.max()):.10g}", f"{float(frame.min()):.10g}"])
    return tuple(rows)
