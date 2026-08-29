"""MODEL 端口载荷：模型数据包（网格 + 材料表 + 截面表 + 工况）."""

from __future__ import annotations

from dataclasses import dataclass

from zylab.fea import LinearElastic, Mesh, Section, StaticCase

__all__ = ["ModelBundle"]


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
