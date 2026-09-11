"""MODEL 端口载荷：模型数据包（结构/传导两族，网格 + 材料表 + 截面表 + 工况）."""

from __future__ import annotations

from dataclasses import dataclass

from zylab.fea import (
    ConductionMaterial,
    ElectricCase,
    LinearElastic,
    Mesh,
    Section,
    StaticCase,
    ThermalCase,
)

__all__ = ["ConductionBundle", "ModelBundle"]


@dataclass(frozen=True)
class ModelBundle:
    """源节点产出、分析节点消费的模型四要素（全字段可 pickle，可跨进程传输）.

    :param mesh: 网格（节点坐标 + 单元块）。
    :param materials: 材料表（单元块按索引引用）。
    :param sections: 截面表（单元块按索引引用）。
    :param case: 静力工况（模态分析取其 ``constraints``）。
    """

    mesh: Mesh
    materials: tuple[LinearElastic, ...]
    sections: tuple[Section, ...]
    case: StaticCase


@dataclass(frozen=True)
class ConductionBundle:
    """传导模型数据包（标量场电-热分析，全字段可 pickle）.

    :param mesh: 网格（v1 限 2D 连续体 TRIA3/QUAD4）。
    :param materials: 传导材料表（单元块按索引引用）。
    :param sections: 截面表（平面单元取厚度）。
    :param electric_case: 电学工况（给定电压 + 注入电流）。
    :param thermal_case: 热学工况（给定温度 + 热源 + 对流）。
    """

    mesh: Mesh
    materials: tuple[ConductionMaterial, ...]
    sections: tuple[Section, ...]
    electric_case: ElectricCase
    thermal_case: ThermalCase
