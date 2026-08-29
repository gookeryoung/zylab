"""FEA 可视化数据提取（Qt-free）：网格线框、变形坐标、节点标量与颜色映射.

GUI 渲染层（pyqtgraph 等）只消费本模块产出的纯 NumPy 数组，
着色/几何计算下沉至此便于单元测试与跨渲染后端复用。
"""

from __future__ import annotations

import numpy as np

from .mesh import ElementType, Mesh
from .static import StaticSolution

__all__ = [
    "deformed_coords",
    "displacement_field",
    "edge_segments",
    "mesh_edges",
    "nodal_stress_field",
    "scalar_colors",
]

# 各单元类型的边（节点局部索引对）：杆为单边，闭合单元取环绕边，实体取棱边
_EDGE_TABLE: dict[ElementType, tuple[tuple[int, int], ...]] = {
    ElementType.TRUSS2: ((0, 1),),
    ElementType.BEAM2: ((0, 1),),
    ElementType.TRIA3: ((0, 1), (1, 2), (2, 0)),
    ElementType.QUAD4: ((0, 1), (1, 2), (2, 3), (3, 0)),
    ElementType.TET4: ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    ElementType.HEX8: (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
}

# jet 风格色带控制点（归一化标量 -> RGB）
_JET_STOPS: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.0, (0.0, 0.0, 0.9)),
    (0.35, (0.0, 0.9, 0.9)),
    (0.5, (0.1, 0.9, 0.1)),
    (0.65, (0.9, 0.9, 0.0)),
    (1.0, (0.9, 0.0, 0.0)),
)


def mesh_edges(mesh: Mesh) -> np.ndarray:
    """提取去重后的网格边（节点索引对）.

    Args:
        mesh: 网格。

    Returns:
        ``(n_edges, 2)`` 整型数组，每行为一条边的两端节点索引。
    """
    pairs: list[np.ndarray] = []
    for block in mesh.blocks:
        local = np.asarray(_EDGE_TABLE[block.etype], dtype=np.intp)
        for conn in block.conn:
            pairs.append(conn[local])
    if not pairs:
        return np.zeros((0, 2), dtype=np.intp)
    stacked = np.concatenate(pairs, axis=0)
    # 无向边去重：按 (min, max) 排序后 unique
    ordered = np.sort(stacked, axis=1)
    return np.unique(ordered, axis=0)


def edge_segments(coords: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """边索引对转为线段端点坐标（供线框绘制）.

    Args:
        coords: 节点坐标 ``(n_nodes, dim)``。
        edges: :func:`mesh_edges` 输出的 ``(n_edges, 2)``。

    Returns:
        ``(n_edges, 2, dim)`` 数组，``[i, 0]``/``[i, 1]`` 为第 i 条边两端坐标。
    """
    if edges.size == 0:
        return np.zeros((0, 2, coords.shape[1]), dtype=float)
    return coords[edges]


def deformed_coords(mesh: Mesh, displacements: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """变形后节点坐标（位移放大 ``scale`` 倍叠加原坐标）.

    Args:
        mesh: 网格。
        displacements: 节点位移 ``(n_nodes, dofs_per_node)`` 或 ``(n_nodes, dim)``
            （梁网格取前 dim 列平动分量参与叠加）。
        scale: 位移放大系数（1.0 为真实变形）。

    Returns:
        ``(n_nodes, dim)`` 变形坐标。
    """
    disp = np.asarray(displacements, dtype=float)[:, : mesh.dim]
    return mesh.coords + scale * disp


def displacement_field(solution: StaticSolution, component: int | None = None) -> np.ndarray:
    """节点位移标量场（供云图着色）.

    Args:
        solution: 静力求解结果。
        component: 位移分量索引；None 取平动位移模（前 dim 列，忽略转角）。

    Returns:
        ``(n_nodes,)`` 标量数组。
    """
    if component is None:
        return np.linalg.norm(solution.displacements[:, : solution.mesh.dim], axis=1)
    return solution.displacements[:, component].copy()


def nodal_stress_field(solution: StaticSolution, component: int = 0) -> np.ndarray:
    """单元应力外推节点平均（面积/体积未加权，v1 简单平均）.

    Args:
        solution: 静力求解结果。
        component: 应力分量索引（杆为 0，平面 0-2，空间 0-5）。

    Returns:
        ``(n_nodes,)`` 节点平均应力数组。
    """
    mesh = solution.mesh
    totals = np.zeros(mesh.n_nodes, dtype=float)
    counts = np.zeros(mesh.n_nodes, dtype=float)
    for result in solution.element_results:
        conn = mesh.blocks[result.block].conn[result.index]
        value = float(result.stress[component])
        totals[conn] += value
        counts[conn] += 1.0
    # 悬空节点（不属于任何单元）保持 0
    return np.divide(totals, counts, out=totals, where=counts > 0)


def scalar_colors(values: np.ndarray) -> np.ndarray:
    """标量场映射 jet 风格 RGB 颜色（供云图着色）.

    Args:
        values: ``(n,)`` 标量数组。

    Returns:
        ``(n, 3)`` 浮点 RGB 数组，取值 ``[0, 1]``。
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return np.zeros((0, 3), dtype=float)
    vmin, vmax = float(np.min(data)), float(np.max(data))
    if vmax - vmin < 1e-15:  # 常值场全部映射中点色
        t = np.full(data.shape, 0.5)
    else:
        t = (data - vmin) / (vmax - vmin)
    stops = np.array([s for s, _ in _JET_STOPS], dtype=float)
    colors = np.array([c for _, c in _JET_STOPS], dtype=float)
    rgb = np.empty((data.size, 3), dtype=float)
    for channel in range(3):
        rgb[:, channel] = np.interp(t, stops, colors[:, channel])
    return rgb
