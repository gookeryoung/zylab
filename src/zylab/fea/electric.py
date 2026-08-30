"""稳态电传导分析：装配电导矩阵 -> 求解节点电压 -> 恢复电场与耗散功率.

控制方程 ``∇·(σ∇V) = 0``（无体电流源），边界为给定电压（Dirichlet）
与节点注入电流（Neumann）。逐单元恢复电场强度 ``E = -∇V``、电流密度
``J = σE`` 与 Joule 耗散功率 ``P_e = σ|∇V|²·t·A``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .conduction import (
    ConductionMaterial,
    NodalSource,
    NodalValue,
    _batch_gradients,
    _batch_measures,
    assemble_conduction,
)
from .errors import MeshError
from .material import Section
from .mesh import Mesh
from .solve import solve_system

__all__ = ["ElectricCase", "ElectricSolution", "solve_electric"]


@dataclass(frozen=True)
class ElectricCase:
    """稳态电传导工况：给定电压 + 节点注入电流.

    Attributes:
        voltages: 给定电压（Dirichlet；至少须一项，否则电位基准缺失）。
        currents: 节点注入电流（Neumann，正值为流入）。
    """

    voltages: tuple[NodalValue, ...] = ()
    currents: tuple[NodalSource, ...] = ()

    def validate(self, mesh: Mesh) -> None:
        """按网格校验节点索引范围与约束存在性."""
        for item in (*self.voltages, *self.currents):
            if not 0 <= item.node < mesh.n_nodes:
                raise MeshError(f"电学边界引用节点 {item.node} 越界（共 {mesh.n_nodes} 节点）")
        if not self.voltages:
            raise MeshError("电传导工况缺少给定电压（Dirichlet），电位基准缺失")


@dataclass(frozen=True)
class ElectricSolution:
    """稳态电传导结果.

    Attributes:
        mesh: 参与求解的网格。
        voltages: 节点电压 ``(n_nodes,)``。
        element_gradients: 逐单元电场梯度 ``∇V (n_elements, 2)``（块序 + 块内序展开）。
        element_power: 逐单元 Joule 耗散功率 ``(n_elements,)``（W）。
        total_power: 总耗散功率（W，恒等于输入电功率）。
    """

    mesh: Mesh
    voltages: np.ndarray
    element_gradients: np.ndarray
    element_power: np.ndarray
    total_power: float


def solve_electric(
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    case: ElectricCase,
    report: Callable[[float, str], None] | None = None,
) -> ElectricSolution:
    """稳态电传导分析主流程.

    Args:
        mesh: 网格（v1 限 2D 连续体 TRIA3/QUAD4）。
        materials: 传导材料表（ElementBlock.material 索引引用）。
        sections: 截面表（平面单元取厚度）。
        case: 电学工况（给定电压 + 注入电流）。
        report: 进度回调 ``(progress, message)``（进程执行器自动注入）。

    Returns:
        :class:`ElectricSolution`，含节点电压、单元电场与耗散功率。
    """
    progress = report if report is not None else _no_report
    progress(0.1, "校验电学工况")
    case.validate(mesh)
    progress(0.4, "装配电导矩阵")
    k_electric = assemble_conduction(mesh, materials, sections, "electric")
    force = np.zeros(mesh.n_nodes)
    for source in case.currents:
        force[source.node] += source.value
    seen: dict[int, float] = {}
    for prescribed in case.voltages:
        seen.setdefault(prescribed.node, prescribed.value)
    fixed = np.fromiter(seen.keys(), dtype=np.intp)
    values = np.fromiter(seen.values(), dtype=float)

    progress(0.7, "求解电压场")
    v, _ = solve_system(k_electric, force, fixed, values)

    progress(0.9, "恢复电场与耗散功率")
    gradients: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    for block in mesh.blocks:
        material = materials[block.material]
        thickness = sections[block.section].thickness
        coords_b = mesh.coords[block.conn]
        grads = _batch_gradients(block.etype, coords_b, v[block.conn])
        # 单元耗散功率 = σ|∇V|² × 体积（2D 度量 × 厚度），块内向量化
        measures = _batch_measures(block.etype, coords_b)
        gradients.append(grads)
        powers.append(material.electric_sigma * np.einsum("ni,ni->n", grads, grads) * measures * thickness)
    element_power = np.concatenate(powers) if powers else np.empty(0)
    total = float(element_power.sum())
    progress(1.0, "电场求解完成")
    return ElectricSolution(
        mesh=mesh,
        voltages=v,
        element_gradients=np.concatenate(gradients) if gradients else np.empty((0, 2)),
        element_power=element_power,
        total_power=total,
    )


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调."""
