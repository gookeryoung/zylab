"""FEA 网格与 meshio 文件格式的双向互转.

支持的读入格式（取决于 meshio 版本）：
- Gmsh ``.msh``（2.x / 4.x）
- Abaqus ``.inp``
- ANSYS ``.msh``
- VTK legacy ``.vtk`` 与 VTU ``.vtu``
- 其他 meshio 支持的格式（Nastran, CGNS 等）

设计要点：
- 仅做 **几何** 互转（节点坐标 + 单元连接），不保留材料/截面属性；
  meshio 物理组信息（cell_data）用于区分同类型的多个单元块。
- 坐标统一映射到 zylab ``Mesh.coords``，维度由 ``coords.shape[1]`` 判定。
- ``"line"`` 类型无法自动区分 TRUSS2 / BEAM2，默认归为 TRUSS2（桁架）；
  梁单元须在求解前显式指定。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from zylab.core.errors import ZylabError
from zylab.fea.mesh import ElementBlock, ElementType, Mesh

logger = logging.getLogger(__name__)

__all__ = ["MeshIOError", "read_mesh", "write_mesh"]


class MeshIOError(ZylabError):
    """网格文件读写错误（格式不支持、文件损坏、缺少依赖等）."""


# meshio cells 名称 → zylab ElementType
# meshio 5.x 的标准单元名：line, triangle, quad, tetra, hexahedron
# vertex 与 polygon 不参与 FEA，跳过
_MESHIO_TO_ETYPE: dict[str, ElementType] = {
    "line": ElementType.TRUSS2,
    "triangle": ElementType.TRIA3,
    "quad": ElementType.QUAD4,
    "tetra": ElementType.TET4,
    "hexahedron": ElementType.HEX8,
}

# zylab ElementType → meshio cells 名称（写文件时用）
_ETYPE_TO_MESHIO: dict[ElementType, str] = {v: k for k, v in _MESHIO_TO_ETYPE.items()}


def _import_meshio() -> Any:
    """延迟导入 meshio（避免对 CLI/worker 路径形成硬依赖）."""
    try:
        import meshio  # type: ignore
    except ImportError as exc:  # pragma: no cover - 运行时缺依赖才触发
        raise MeshIOError("meshio 未安装，请先 `uv add meshio>=5.3` 后重试") from exc
    return meshio


def read_mesh(path: str | Path) -> Mesh:
    """从 meshio 支持的网格文件读入并转换为 zylab Mesh.

    Args:
        path: 网格文件路径。

    Returns:
        转换好的 zylab Mesh 对象（仅含几何，无材料属性）。

    Raises:
        MeshIOError: 文件不存在、格式不支持、读取失败或网格为空。
    """
    p = Path(path)
    if not p.is_file():
        raise MeshIOError(f"网格文件不存在: {p}")
    meshio = _import_meshio()
    try:
        m = meshio.read(str(p))
    except BaseException as exc:
        # meshio 对格式错误的文件可能抛非预期异常（含 SystemExit），统一转 MeshIOError
        raise MeshIOError(f"读取网格文件失败 {p}: {exc}") from exc

    coords = np.asarray(m.points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise MeshIOError(f"网格坐标须为 (n, 2) 或 (n, 3) 数组，实际 shape={coords.shape}")
    # meshio 写 2D 网格到 VTU 时会自动补 z=0，读回来后如果 z 列全 0 则降维
    if coords.shape[1] == 3 and np.all(coords[:, 2] == 0.0):
        coords = coords[:, :2].copy()
    if coords.shape[0] == 0:
        raise MeshIOError("网格不含任何节点")

    blocks: list[ElementBlock] = []
    # meshio 5.x 的 cells 是 CellBlock 对象列表（.type / .data）
    for cellblock in m.cells:
        cell_type = cellblock.type
        cell_data = cellblock.data
        if cell_type not in _MESHIO_TO_ETYPE:
            logger.debug("跳过 meshio 不支持的单元类型: %s (%d 个)", cell_type, cell_data.shape[0])
            continue
        etype = _MESHIO_TO_ETYPE[cell_type]
        conn = np.asarray(cell_data, dtype=np.intp)
        blocks.append(ElementBlock(etype=etype, conn=conn))

    if not blocks:
        raise MeshIOError("网格未包含任何 FEA 支持的单元类型（line/triangle/quad/tetra/hexahedron）")

    # 压缩坐标到实际用到的范围（去除 meshio 中未被任何单元引用的悬空节点）
    all_indices = np.concatenate([b.conn.ravel() for b in blocks])
    used_indices = np.unique(all_indices)
    if used_indices.size != coords.shape[0]:
        logger.info(
            "网格含 %d 个悬空节点，压缩至 %d 个",
            coords.shape[0],
            used_indices.size,
        )
        coords = coords[used_indices]
        # 重映射连接表索引
        old_to_new = np.full(all_indices.max() + 1, -1, dtype=np.intp)
        old_to_new[used_indices] = np.arange(used_indices.size, dtype=np.intp)
        remapped: list[ElementBlock] = []
        for b in blocks:
            remapped.append(
                ElementBlock(
                    etype=b.etype,
                    conn=old_to_new[b.conn],
                    material=b.material,
                    section=b.section,
                )
            )
        blocks = remapped

    return Mesh(coords=coords, blocks=tuple(blocks))


def write_mesh(mesh: Mesh, path: str | Path) -> None:
    """将 zylab Mesh 写出为 meshio 支持的网格文件.

    Args:
        mesh: 待写出的网格（必须通过 Mesh 构造校验）。
        path: 输出文件路径；格式由扩展名决定（``.msh`` / ``.inp`` / ``.vtu`` 等）。

    Raises:
        MeshIOError: 扩展名无对应 meshio 写器、写出失败。
    """
    p = Path(path)
    meshio = _import_meshio()

    cells: list[tuple[str, np.ndarray]] = []
    for block in mesh.blocks:
        if block.etype not in _ETYPE_TO_MESHIO:
            raise MeshIOError(f"单元类型 {block.etype.value} 无法写出为 meshio 格式")
        cells.append((_ETYPE_TO_MESHIO[block.etype], block.conn.copy()))

    # meshio 要求 points 为 (n, 3)，z=0 填充
    points = mesh.coords
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(points.shape[0], dtype=float)))

    m = meshio.Mesh(points=points, cells=cells)
    try:
        meshio.write(str(p), m)
    except (OSError, ValueError) as exc:
        raise MeshIOError(f"写出网格文件失败 {p}: {exc}") from exc
    logger.info("网格已写出: %s (%d 节点, %d 单元)", p, mesh.n_nodes, mesh.n_elements)
