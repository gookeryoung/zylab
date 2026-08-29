"""边界条件：节点约束（Dirichlet）、节点集中载荷、边压力与体力.

约束/载荷均以「节点 + 节点内局部自由度」描述，装配阶段再映射到全局编号，
避免调用方直接操作全局自由度索引。边压力按节点折线逐段取几何法向，
体力为均匀单位体积力（连续体单元一致节点载荷）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MeshError
from .mesh import Mesh

__all__ = ["BodyForce", "Constraint", "EdgePressure", "NodalLoad", "StaticCase"]


@dataclass(frozen=True)
class Constraint:
    """节点位移约束（同一节点同一自由度重复约束时取首个值）.

    Attributes:
        node: 节点索引（0 基）。
        dofs: 节点内局部自由度索引元组（0 基；2D 连续体为 0/1，含梁网格为 0/1/2，3D 为 0/1/2）。
        value: 约定位移值（全部自由度共用；非零约束时按此值施加）。
    """

    node: int
    dofs: tuple[int, ...]
    value: float = 0.0


@dataclass(frozen=True)
class NodalLoad:
    """节点集中载荷.

    Attributes:
        node: 节点索引（0 基）。
        forces: 各自由度方向分量元组（长度须等于网格每节点自由度数；梁网格第 3 分量为弯矩）。
    """

    node: int
    forces: tuple[float, ...]


@dataclass(frozen=True)
class EdgePressure:
    """2D 边界法向压力（沿节点折线逐段施加）.

    节点顺序须使材料位于折线行进方向左侧；正压力指向材料内部（压缩），
    负值为外向拉力。每段直线独立取法向，曲边由折线逼近。

    Attributes:
        nodes: 边界节点索引序列（沿边连续排列）。
        pressure: 法向压力集度（力/长度）。
    """

    nodes: tuple[int, ...]
    pressure: float


@dataclass(frozen=True)
class BodyForce:
    """均匀体力（单位体积力密度，作用于全部连续体单元）.

    Attributes:
        fx: x 方向分量。
        fy: y 方向分量。
        fz: z 方向分量（2D 网格忽略）。
    """

    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0


@dataclass(frozen=True)
class StaticCase:
    """静力分析工况：节点约束 + 节点载荷 + 边压力 + 体力."""

    constraints: tuple[Constraint, ...] = ()
    loads: tuple[NodalLoad, ...] = ()
    edge_pressures: tuple[EdgePressure, ...] = ()
    body_forces: tuple[BodyForce, ...] = ()

    def validate(self, mesh: Mesh) -> None:
        """按网格校验节点索引与自由度编号范围."""
        width = mesh.dofs_per_node
        dim = mesh.dim
        for constraint in self.constraints:
            _check_node(mesh, constraint.node, "约束")
            for dof in constraint.dofs:
                if not 0 <= dof < width:
                    raise MeshError(f"约束节点 {constraint.node} 的自由度 {dof} 超出 [0, {width})")
        for load in self.loads:
            _check_node(mesh, load.node, "载荷")
            if len(load.forces) != width:
                raise MeshError(f"载荷节点 {load.node} 力分量数 {len(load.forces)} != 每节点自由度数 {width}")
        for pressure in self.edge_pressures:
            if dim != 2:
                raise MeshError("边压力仅支持 2D 网格")
            if len(pressure.nodes) < 2:
                raise MeshError("边压力至少需要 2 个节点")
            for node in pressure.nodes:
                _check_node(mesh, node, "边压力")
        for force in self.body_forces:
            if dim == 2 and force.fz != 0.0:
                raise MeshError("2D 网格的体力 z 分量须为 0")


def _check_node(mesh: Mesh, node: int, kind: str) -> None:
    """校验节点索引不越界."""
    if not 0 <= node < mesh.n_nodes:
        raise MeshError(f"{kind}引用节点 {node} 越界（共 {mesh.n_nodes} 节点）")
