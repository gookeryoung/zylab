"""电-热耦合分析（顺序耦合）：电场 -> Joule 热 -> 温度场（稳态或瞬态）.

常物性下单向耦合：先解稳态电传导 ``∇·(σ∇V) = 0``，逐单元计算 Joule
热源 ``q = σ|∇V|²`` 并化为一致节点热载荷 ``∫ Nᵀ q dΩ``，再解稳态热传导
``∇·(k∇T) + q = 0`` 或瞬态热传导（backward Euler）。温度对物性的反馈
（非线性迭代）推迟后续版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .conduction import ConductionMaterial, _batch_field_load
from .electric import ElectricCase, ElectricSolution, solve_electric
from .material import Section
from .mesh import Mesh
from .thermal import ThermalCase, ThermalSolution, solve_thermal
from .thermal_transient import ThermalTransientSolution, solve_thermal_transient

__all__ = [
    "ElectroThermalSolution",
    "ElectroThermalTransientSolution",
    "solve_electrothermal",
    "solve_electrothermal_transient",
]


@dataclass(frozen=True)
class ElectroThermalSolution:
    """电-热耦合求解结果（常物性顺序耦合）.

    Attributes:
        mesh: 参与求解的网格。
        electric: 电场子解（节点电压、单元电场、耗散功率）。
        thermal: 温度场子解（节点温度、单元热流、温度范围）。
        total_power: 总电功率（W，恒等于总 Joule 热源）。
    """

    mesh: Mesh
    electric: ElectricSolution
    thermal: ThermalSolution
    total_power: float

    @property
    def voltages(self) -> np.ndarray:
        """节点电压 ``(n_nodes,)``."""
        return self.electric.voltages

    @property
    def temperatures(self) -> np.ndarray:
        """节点温度 ``(n_nodes,)``."""
        return self.thermal.temperatures

    @property
    def t_max(self) -> float:
        """全场最高温度."""
        return self.thermal.t_max

    @property
    def t_min(self) -> float:
        """全场最低温度."""
        return self.thermal.t_min


def _joule_nodal_load(
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    electric: ElectricSolution,
) -> np.ndarray:
    """电场解的 Joule 热一致节点载荷 ``∫ Nᵀ σ|∇V|² dΩ``（HEX8 忽略厚度）."""
    joule = np.zeros(mesh.n_nodes)
    for block in mesh.blocks:
        material = materials[block.material]
        thickness = sections[block.section].thickness
        # np.add.at 无缓冲累加：跨单元共享节点正确求和（fancy 索引 += 会覆盖）
        load = _batch_field_load(
            block.etype, mesh.coords[block.conn], material.electric_sigma, electric.voltages[block.conn], thickness
        )
        np.add.at(joule, block.conn.ravel(), load.ravel())
    return joule


def solve_electrothermal(  # noqa: PLR0913  模型四要素 + 电/热双工况与回调，语义不可合并
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    electric_case: ElectricCase,
    thermal_case: ThermalCase,
    *,
    report: Callable[[float, str], None] | None = None,
) -> ElectroThermalSolution:
    """电-热耦合分析主流程（稳态顺序耦合）.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表（ElementBlock.material 索引引用）。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。
        electric_case: 电学工况（给定电压 + 注入电流）。
        thermal_case: 热学工况（给定温度 + 热源 + 对流）。
        report: 进度回调 ``(progress, message)``（进程执行器自动注入）。

    Returns:
        :class:`ElectroThermalSolution`，含电场与温度场子解及总电功率。
    """
    progress = report if report is not None else _no_report
    progress(0.1, "求解稳态电场")
    electric = solve_electric(mesh, materials, sections, electric_case)

    progress(0.5, "计算 Joule 热一致节点载荷")
    joule = _joule_nodal_load(mesh, materials, sections, electric)

    progress(0.7, "求解稳态温度场")
    thermal = solve_thermal(mesh, materials, sections, thermal_case, extra_heat=joule)
    progress(1.0, "电-热耦合求解完成")
    return ElectroThermalSolution(
        mesh=mesh,
        electric=electric,
        thermal=thermal,
        total_power=electric.total_power,
    )


@dataclass(frozen=True)
class ElectroThermalTransientSolution:
    """瞬态电-热耦合求解结果（常物性顺序耦合：稳态电场 + 瞬态温度场）.

    Attributes:
        mesh: 参与求解的网格。
        electric: 电场子解（常物性下电场不随温度变化，求解一次）。
        thermal: 瞬态温度场子解（各帧温度、末帧热流、温度范围）。
        total_power: 总电功率（W，恒等于总 Joule 热源）。
    """

    mesh: Mesh
    electric: ElectricSolution
    thermal: ThermalTransientSolution
    total_power: float

    @property
    def voltages(self) -> np.ndarray:
        """节点电压 ``(n_nodes,)``."""
        return self.electric.voltages

    @property
    def temperatures(self) -> np.ndarray:
        """末帧节点温度 ``(n_nodes,)``."""
        return self.thermal.temperatures[-1]

    @property
    def t_max(self) -> float:
        """全程最高温度."""
        return self.thermal.t_max

    @property
    def t_min(self) -> float:
        """全程最低温度."""
        return self.thermal.t_min


def solve_electrothermal_transient(  # noqa: PLR0913  模型四要素 + 双工况/初值/时长/步数/回调，语义不可合并
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    electric_case: ElectricCase,
    thermal_case: ThermalCase,
    *,
    initial: np.ndarray,
    total_time: float,
    n_steps: int,
    report: Callable[[float, str], None] | None = None,
) -> ElectroThermalTransientSolution:
    """瞬态电-热耦合分析主流程（稳态电场 + backward Euler 瞬态温度场）.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表，瞬态须提供 ``volumetric_heat_capacity``。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。
        electric_case: 电学工况（给定电压 + 注入电流，常物性下求解一次）。
        thermal_case: 热学工况（给定温度 + 热源 + 对流，时间不变）。
        initial: 初始温度 ``(n_nodes,)``。
        total_time: 总时长（> 0）。
        n_steps: 时间步数（≥ 1，均匀步长）。
        report: 进度回调 ``(progress, message)``（进程执行器自动注入）。

    Returns:
        :class:`ElectroThermalTransientSolution`，含电场子解与瞬态温度场子解。
    """
    progress = report if report is not None else _no_report
    progress(0.1, "求解稳态电场")
    electric = solve_electric(mesh, materials, sections, electric_case)
    progress(0.2, "计算 Joule 热一致节点载荷")
    joule = _joule_nodal_load(mesh, materials, sections, electric)
    thermal = solve_thermal_transient(
        mesh,
        materials,
        sections,
        thermal_case,
        initial=initial,
        total_time=total_time,
        n_steps=n_steps,
        extra_heat=joule,
    )
    progress(1.0, "瞬态电-热耦合求解完成")
    return ElectroThermalTransientSolution(
        mesh=mesh,
        electric=electric,
        thermal=thermal,
        total_power=electric.total_power,
    )


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调."""
