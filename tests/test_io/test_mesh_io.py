"""网格 I/O 模块测试（纯 Python，不需要 Qt 环境）."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea.mesh import ElementBlock, ElementType, Mesh
from zylab.io.mesh_io import MeshIOError, read_mesh, write_mesh


def _make_2d_tria_mesh() -> Mesh:
    """构建一个 2D 三角形网格：2x2 矩形，沿对角线分成 2 个三角形."""
    coords = np.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    block = ElementBlock(
        etype=ElementType.TRIA3,
        conn=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=np.intp,
        ),
    )
    return Mesh(coords=coords, blocks=(block,))


def _make_3d_tetra_mesh() -> Mesh:
    """构建一个 3D 四面体网格：8 顶点立方体，用 6 个四面体填充."""
    coords = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=float,
    )
    block = ElementBlock(
        etype=ElementType.TET4,
        conn=np.array(
            [
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 4, 5, 6],
                [0, 4, 6, 7],
                [0, 2, 6, 7],
                [0, 1, 5, 6],
            ],
            dtype=np.intp,
        ),
    )
    return Mesh(coords=coords, blocks=(block,))


class TestMeshIORoundtrip:
    """write_mesh → read_mesh 往返验证."""

    def test_roundtrip_2d_tria_vtu(self, tmp_path) -> None:
        """2D 三角形网格 VTU 格式往返."""
        mesh = _make_2d_tria_mesh()
        p = tmp_path / "test_tria.vtu"
        write_mesh(mesh, p)
        assert p.is_file()

        loaded = read_mesh(p)
        assert loaded.dim == 2
        assert loaded.n_nodes == mesh.n_nodes
        assert loaded.n_elements == mesh.n_elements
        np.testing.assert_allclose(loaded.coords, mesh.coords)
        assert len(loaded.blocks) == len(mesh.blocks)
        assert loaded.blocks[0].etype == mesh.blocks[0].etype
        np.testing.assert_array_equal(loaded.blocks[0].conn, mesh.blocks[0].conn)

    def test_roundtrip_3d_tetra_vtu(self, tmp_path) -> None:
        """3D 四面体网格 VTU 格式往返."""
        mesh = _make_3d_tetra_mesh()
        p = tmp_path / "test_tetra.vtu"
        write_mesh(mesh, p)

        loaded = read_mesh(p)
        assert loaded.dim == 3
        assert loaded.n_nodes == mesh.n_nodes
        assert loaded.n_elements == mesh.n_elements
        np.testing.assert_allclose(loaded.coords, mesh.coords)
        np.testing.assert_array_equal(loaded.blocks[0].conn, mesh.blocks[0].conn)

    def test_roundtrip_3d_tetra_vtk(self, tmp_path) -> None:
        """3D 四面体网格 legacy VTK 格式往返."""
        mesh = _make_3d_tetra_mesh()
        p = tmp_path / "test_tetra.vtk"
        write_mesh(mesh, p)

        loaded = read_mesh(p)
        assert loaded.dim == 3
        assert loaded.n_nodes == mesh.n_nodes
        np.testing.assert_allclose(loaded.coords, mesh.coords)


class TestMeshIOEdge:
    """边界与异常场景."""

    def test_nonexistent_file(self, tmp_path) -> None:
        """读取不存在的文件 → MeshIOError."""
        p = tmp_path / "not_exist.vtu"
        with pytest.raises(MeshIOError, match="网格文件不存在"):
            read_mesh(p)

    def test_corrupted_file(self, tmp_path) -> None:
        """写入非法内容 → MeshIOError."""
        p = tmp_path / "corrupted.vtu"
        p.write_text("garbage", encoding="utf-8")
        with pytest.raises(MeshIOError):
            read_mesh(p)

    def test_mixed_types(self, tmp_path) -> None:
        """混合单元类型网格（tria3 + quad4）往返."""
        coords = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [2, 0],
                [3, 0],
                [3, 1],
                [2, 1],
            ],
            dtype=float,
        )
        tria = ElementBlock(
            etype=ElementType.TRIA3,
            conn=np.array([[0, 1, 2]], dtype=np.intp),
        )
        quad = ElementBlock(
            etype=ElementType.QUAD4,
            conn=np.array([[4, 5, 6, 7]], dtype=np.intp),
        )
        mesh = Mesh(coords=coords, blocks=(tria, quad))
        p = tmp_path / "mixed.vtu"
        write_mesh(mesh, p)

        loaded = read_mesh(p)
        assert loaded.n_elements == 2
        # meshio 可能会合并同类型单元块，所以按 etype 分组计数
        etype_counts: dict[ElementType, int] = {}
        for b in loaded.blocks:
            etype_counts[b.etype] = etype_counts.get(b.etype, 0) + b.count
        assert etype_counts.get(ElementType.TRIA3, 0) == 1
        assert etype_counts.get(ElementType.QUAD4, 0) == 1

    def test_dangling_nodes_compressed(self, tmp_path) -> None:
        """有悬空节点时 read_mesh 自动压缩."""
        coords = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [99, 99],  # 悬空节点，不被任何单元引用
            ],
            dtype=float,
        )
        block = ElementBlock(
            etype=ElementType.TRIA3,
            conn=np.array([[0, 1, 2]], dtype=np.intp),
        )
        mesh = Mesh(coords=coords, blocks=(block,))
        p = tmp_path / "dangling.vtu"
        write_mesh(mesh, p)  # meshio 不会自动压缩，悬空节点会被写出

        # 重新读入时应自动压缩
        loaded = read_mesh(p)
        assert loaded.n_nodes == 3  # 悬空节点被剔除
        assert loaded.n_elements == 1
        np.testing.assert_array_equal(loaded.coords, coords[:3])
