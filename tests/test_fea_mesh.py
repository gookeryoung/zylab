"""fea.mesh / fea.boundary 单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    Mesh,
    MeshError,
    NodalLoad,
    StaticCase,
)


def _square_quad_block() -> ElementBlock:
    """单位正方形单 Q4 单元块."""
    return ElementBlock(ElementType.QUAD4, np.arange(4).reshape(1, 4))


class TestMesh:
    def test_mesh_properties(self) -> None:
        mesh = Mesh(np.zeros((5, 2)), (_square_quad_block(),))
        assert mesh.dim == 2
        assert mesh.n_nodes == 5
        assert mesh.n_elements == 1
        assert mesh.n_dofs == 10

    def test_mesh_rejects_bad_coord_shape(self) -> None:
        with pytest.raises(MeshError, match="坐标"):
            Mesh(np.zeros((5, 1)))

    def test_mesh_rejects_empty_nodes(self) -> None:
        with pytest.raises(MeshError, match="至少"):
            Mesh(np.zeros((0, 2)))

    def test_mesh_rejects_conn_out_of_range(self) -> None:
        block = ElementBlock(ElementType.QUAD4, np.array([[0, 1, 2, 9]]))
        with pytest.raises(MeshError, match="越界"):
            Mesh(np.zeros((4, 2)), (block,))

    def test_mesh_rejects_dim_mismatch(self) -> None:
        block = ElementBlock(ElementType.HEX8, np.arange(8).reshape(1, 8))
        with pytest.raises(MeshError, match="不支持"):
            Mesh(np.zeros((8, 2)), (block,))

    def test_mesh_blocks_may_be_empty_tuple(self) -> None:
        mesh = Mesh(np.zeros((3, 3)))
        assert mesh.n_elements == 0

    def test_dofs_per_node_continuum_equals_dim(self) -> None:
        mesh = Mesh(np.zeros((5, 2)), (_square_quad_block(),))
        assert mesh.dofs_per_node == 2
        assert mesh.n_dofs == 10

    def test_dofs_per_node_beam_width(self) -> None:
        block = ElementBlock(ElementType.BEAM2, np.array([[0, 1], [1, 2]]))
        mesh = Mesh(np.zeros((3, 2)), (block,))
        assert mesh.dofs_per_node == 3
        assert mesh.n_dofs == 9

    def test_dofs_per_node_mixed_beam_quad(self) -> None:
        # 梁与 Q4 混合网格：宽度取最宽单元族（3）
        beam = ElementBlock(ElementType.BEAM2, np.array([[0, 1]]))
        quad = ElementBlock(ElementType.QUAD4, np.array([[1, 2, 3, 4]]))
        mesh = Mesh(np.zeros((5, 2)), (beam, quad))
        assert mesh.dofs_per_node == 3
        assert mesh.n_dofs == 15

    def test_beam_rejected_in_3d(self) -> None:
        block = ElementBlock(ElementType.BEAM2, np.array([[0, 1]]))
        with pytest.raises(MeshError, match="不支持"):
            Mesh(np.zeros((2, 3)), (block,))


class TestElementBlock:
    def test_block_validates_node_count(self) -> None:
        with pytest.raises(MeshError, match="节点"):
            ElementBlock(ElementType.TRIA3, np.array([[0, 1]]))

    def test_block_rejects_negative_index(self) -> None:
        with pytest.raises(MeshError, match="负节点索引"):
            ElementBlock(ElementType.TRUSS2, np.array([[0, -1]]))

    def test_block_rejects_empty_conn(self) -> None:
        with pytest.raises(MeshError, match="为空"):
            ElementBlock(ElementType.TRUSS2, np.zeros((0, 2), dtype=int))

    def test_block_count(self) -> None:
        block = ElementBlock(ElementType.TRUSS2, np.array([[0, 1], [1, 2]]))
        assert block.count == 2


class TestStaticCase:
    def test_case_validates_node_index(self) -> None:
        mesh = Mesh(np.zeros((3, 2)))
        case = StaticCase(constraints=(Constraint(node=5, dofs=(0,)),))
        with pytest.raises(MeshError, match="越界"):
            case.validate(mesh)

    def test_case_validates_dof_range(self) -> None:
        mesh = Mesh(np.zeros((3, 2)))
        case = StaticCase(constraints=(Constraint(node=0, dofs=(2,)),))
        with pytest.raises(MeshError, match="自由度"):
            case.validate(mesh)

    def test_case_validates_load_dimension(self) -> None:
        mesh = Mesh(np.zeros((3, 2)))
        case = StaticCase(loads=(NodalLoad(node=0, forces=(1.0, 2.0, 3.0)),))
        with pytest.raises(MeshError, match="力分量数"):
            case.validate(mesh)

    def test_case_valid_pass(self) -> None:
        mesh = Mesh(np.zeros((3, 3)))
        case = StaticCase(
            constraints=(Constraint(node=0, dofs=(0, 1, 2)),),
            loads=(NodalLoad(node=1, forces=(1.0, 0.0, 0.0)),),
        )
        case.validate(mesh)
