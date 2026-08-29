"""网格数据结构：节点坐标 + 单元块（按类型分块）.

设计要点：
- 坐标统一存 ``(n_nodes, dim)`` float 数组，dim 取 2（平面）或 3（空间）；
- 单元按 ``ElementBlock`` 分块组织，同一块内类型/材料/截面一致，支持混合网格；
- 每节点全局自由度数 = 网格内最宽单元族宽度（连续体 = 网格维数，
  含平面梁时为 3：ux/uy/θz），见 :func:`element_dofs_per_node`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .errors import MeshError

__all__ = ["ElementBlock", "ElementType", "Mesh"]


class ElementType(Enum):
    """单元类型（v1 静力内核支持的单元族）."""

    TRUSS2 = "truss2"  # 2 节点空间/平面桁架杆
    BEAM2 = "beam2"  # 2 节点平面 Euler-Bernoulli 梁（每节点 3 DOF：ux/uy/θz）
    TRIA3 = "tria3"  # 3 节点常应变三角形（CST）
    QUAD4 = "quad4"  # 4 节点等参四边形（全积分）
    TET4 = "tet4"  # 4 节点常应变四面体
    HEX8 = "hex8"  # 8 节点等参六面体（全积分）


# 各单元类型的节点数（闭合映射，新增单元须同步登记）
_NODE_COUNTS: dict[ElementType, int] = {
    ElementType.TRUSS2: 2,
    ElementType.BEAM2: 2,
    ElementType.TRIA3: 3,
    ElementType.QUAD4: 4,
    ElementType.TET4: 4,
    ElementType.HEX8: 8,
}

# 各单元类型允许的网格维数（TRUSS2 平面/空间均可，其余固定）
_ALLOWED_DIMS: dict[ElementType, frozenset[int]] = {
    ElementType.TRUSS2: frozenset({2, 3}),
    ElementType.BEAM2: frozenset({2}),
    ElementType.TRIA3: frozenset({2}),
    ElementType.QUAD4: frozenset({2}),
    ElementType.TET4: frozenset({3}),
    ElementType.HEX8: frozenset({3}),
}

# 各单元类型每节点自由度数（None 表示与网格维数一致）
_ELEMENT_DOFS: dict[ElementType, int | None] = {
    ElementType.TRUSS2: None,
    ElementType.BEAM2: 3,
    ElementType.TRIA3: None,
    ElementType.QUAD4: None,
    ElementType.TET4: None,
    ElementType.HEX8: None,
}


def element_dofs_per_node(etype: ElementType, dim: int) -> int:
    """单元类型的每节点自由度数（无转角单元 = 网格维数）."""
    width = _ELEMENT_DOFS[etype]
    return dim if width is None else width


@dataclass(frozen=True)
class ElementBlock:
    """单元块：同一类型/材料/截面的一组单元.

    Attributes:
        etype: 单元类型。
        conn: 连接表 ``(n_elem, n_node_per_elem)`` int 数组，节点索引 0 基。
        material: 材料表索引（对应 ``materials`` 列表）。
        section: 截面表索引（对应 ``sections`` 列表；杆取面积，平面单元取厚度）。
    """

    etype: ElementType
    conn: np.ndarray
    material: int = 0
    section: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        """校验并规范化连接表（转 int 二维数组）."""
        conn = np.asarray(self.conn, dtype=np.intp)
        if conn.ndim != 2:
            raise MeshError(f"单元块 {self.etype.value} 连接表须为二维数组，实际 ndim={conn.ndim}")
        expected = _NODE_COUNTS[self.etype]
        if conn.shape[1] != expected:
            raise MeshError(f"单元 {self.etype.value} 需要 {expected} 个节点，连接表每行 {conn.shape[1]} 个")
        if conn.size == 0:
            raise MeshError(f"单元块 {self.etype.value} 连接表为空")
        if conn.min() < 0:
            raise MeshError(f"单元块 {self.etype.value} 连接表出现负节点索引")
        object.__setattr__(self, "conn", conn)

    @property
    def count(self) -> int:
        """块内单元数."""
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class Mesh:
    """网格：节点坐标 + 单元块元组.

    Attributes:
        coords: 节点坐标 ``(n_nodes, dim)`` float 数组。
        blocks: 单元块元组（可为混合类型）。
    """

    coords: np.ndarray
    blocks: tuple[ElementBlock, ...] = field(default=())

    def __post_init__(self) -> None:
        """校验坐标形状与各单元块的维数/越界."""
        coords = np.asarray(self.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] not in (2, 3):
            raise MeshError(f"节点坐标须为 (n, 2) 或 (n, 3) 数组，实际 shape={coords.shape}")
        if coords.shape[0] == 0:
            raise MeshError("网格至少需要一个节点")
        object.__setattr__(self, "coords", coords)
        dim = coords.shape[1]
        n_nodes = coords.shape[0]
        for block in self.blocks:
            if dim not in _ALLOWED_DIMS[block.etype]:
                raise MeshError(f"单元 {block.etype.value} 不支持 {dim}D 网格")
            if block.conn.max() >= n_nodes:
                raise MeshError(
                    f"单元块 {block.etype.value} 连接表最大索引 {block.conn.max()} 越界（共 {n_nodes} 节点）"
                )

    @property
    def dim(self) -> int:
        """网格维数（2 或 3）."""
        return int(self.coords.shape[1])

    @property
    def dofs_per_node(self) -> int:
        """每节点全局自由度数（含梁等带转角单元时大于网格维数）."""
        widest = self.dim
        for block in self.blocks:
            widest = max(widest, element_dofs_per_node(block.etype, self.dim))
        return widest

    @property
    def n_nodes(self) -> int:
        """节点总数."""
        return int(self.coords.shape[0])

    @property
    def n_elements(self) -> int:
        """单元总数（各块之和）."""
        return sum(block.count for block in self.blocks)

    @property
    def n_dofs(self) -> int:
        """自由度总数（每节点 dofs_per_node 个）."""
        return self.n_nodes * self.dofs_per_node
