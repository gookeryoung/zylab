"""fea.viewdata 可视化数据提取测试."""

from __future__ import annotations

import numpy as np

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    StaticCase,
    solve_static,
)
from zylab.fea.viewdata import (
    deformed_coords,
    displacement_field,
    edge_segments,
    mesh_edges,
    nodal_stress_field,
    scalar_colors,
)


def _make_mesh() -> Mesh:
    """构建 2x1 Q4 网格（两单元共享一条边）."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    conn = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    block = ElementBlock(etype=ElementType.QUAD4, conn=conn)
    return Mesh(coords=coords, blocks=(block,))


def _make_solution() -> object:
    """悬臂小模型静力求解（左端固支、右上节点加载）."""
    mesh = _make_mesh()
    materials = [LinearElastic(e_modulus=1000.0, poisson=0.3)]
    sections = [Section(thickness=1.0)]
    case = StaticCase(
        constraints=(Constraint(0, (0, 1)), Constraint(3, (0, 1))),
        loads=(NodalLoad(5, (0.0, -10.0)),),
    )
    return solve_static(mesh, materials, sections, case)


def test_mesh_edges_dedup_shared() -> None:
    """相邻单元共享边应去重（2 个 Q4 共 7 条边）."""
    mesh = _make_mesh()
    edges = mesh_edges(mesh)
    assert edges.shape == (7, 2)
    # 每条边端点有序且全局唯一
    assert np.all(edges[:, 0] < edges[:, 1])


def test_mesh_edges_mixed_types() -> None:
    """混合网格（杆 + 四边形）边合并去重."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.5]])
    quad = ElementBlock(etype=ElementType.QUAD4, conn=np.array([[0, 1, 2, 3]]))
    truss = ElementBlock(etype=ElementType.TRUSS2, conn=np.array([[1, 4]]))
    mesh = Mesh(coords=coords, blocks=(quad, truss))
    # quad 四条边 + 杆边 (1,4) = 5 条
    assert mesh_edges(mesh).shape == (5, 2)


def test_mesh_edges_empty_blocks() -> None:
    """无单元块的网格返回空边表."""
    mesh = Mesh(coords=np.array([[0.0, 0.0], [1.0, 0.0]]))
    edges = mesh_edges(mesh)
    assert edges.shape == (0, 2)


def test_edge_segments_shape() -> None:
    """边线段坐标形状为 (n_edges, 2, dim)."""
    mesh = _make_mesh()
    segments = edge_segments(mesh.coords, mesh_edges(mesh))
    assert segments.shape == (7, 2, 2)
    assert segments[0, 0].tolist() == [0.0, 0.0]


def test_edge_segments_empty() -> None:
    """空边表返回 (0, 2, dim)."""
    segments = edge_segments(np.zeros((2, 3)), np.zeros((0, 2), dtype=np.intp))
    assert segments.shape == (0, 2, 3)


def test_deformed_coords_scaling() -> None:
    """变形坐标 = 原坐标 + scale * 位移."""
    mesh = _make_mesh()
    disp = np.ones((mesh.n_nodes, 2), dtype=float)
    deformed = deformed_coords(mesh, disp, scale=2.0)
    np.testing.assert_allclose(deformed, mesh.coords + 2.0, atol=1e-15)


def test_displacement_field_component_and_norm() -> None:
    """位移场可取分量或模."""
    solution = _make_solution()
    norm = displacement_field(solution)
    assert norm.shape == (solution.mesh.n_nodes,)
    assert np.all(norm >= 0.0)
    comp = displacement_field(solution, component=1)
    np.testing.assert_allclose(comp, solution.displacements[:, 1], atol=1e-15)


def test_nodal_stress_field_averages() -> None:
    """节点应力为相邻单元应力简单平均."""
    solution = _make_solution()
    field = nodal_stress_field(solution, component=0)
    assert field.shape == (solution.mesh.n_nodes,)
    # 节点 0 只属于单元 0，其值等于该单元应力分量
    elem0 = solution.element_results[0].stress[0]
    np.testing.assert_allclose(field[0], elem0, atol=1e-12)
    # 内部节点 1/4 属于两个单元，取平均
    s0 = solution.element_results[0].stress[0]
    s1 = solution.element_results[1].stress[0]
    np.testing.assert_allclose(field[1], (s0 + s1) / 2.0, atol=1e-12)


def test_scalar_colors_range() -> None:
    """颜色映射：最小值偏蓝、最大值偏红、形状正确."""
    values = np.array([0.0, 0.5, 1.0])
    colors = scalar_colors(values)
    assert colors.shape == (3, 3)
    assert colors[0, 2] > colors[0, 0]  # 蓝通道占优
    assert colors[2, 0] > colors[2, 2]  # 红通道占优
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0)


def test_scalar_colors_constant_field() -> None:
    """常值场映射统一中点色."""
    colors = scalar_colors(np.full(4, 3.14))
    assert np.allclose(colors, colors[0])
    mid = colors[0]
    assert mid[1] >= mid[0] and mid[1] >= mid[2]  # 中点偏绿


def test_scalar_colors_empty() -> None:
    """空数组返回 (0, 3)."""
    assert scalar_colors(np.zeros(0)).shape == (0, 3)
