"""fea.static 端到端测试：patch test、桁架解析解、Hex8 单轴受拉.

patch test 是 FEA 验证的黄金标准：边界节点施加线性位移场，内部节点自由，
精确单元应给出内部节点位移 = 线性场值、单元应力 = D @ ε 精确常应力。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    SolverError,
    StaticCase,
    StressState,
    solve_static,
)

MAT_PLANE = LinearElastic(210.0, 0.3, StressState.PLANE_STRESS)
MAT_SOLID = LinearElastic(100.0, 0.25, StressState.SOLID)
UNIT = Section()


def _field_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """线性位移场 ux = 0.001+0.002x+0.003y, uy = -0.001+0.004x-0.002y."""
    ux = 0.001 + 0.002 * x + 0.003 * y
    uy = -0.001 + 0.004 * x - 0.002 * y
    return np.stack([ux, uy], axis=-1)


def _field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """线性位移场（3D，含剪切分量）."""
    ux = 0.001 + 0.002 * x + 0.003 * y + 0.001 * z
    uy = -0.001 + 0.004 * x - 0.002 * y + 0.002 * z
    uz = 0.002 - 0.001 * x + 0.005 * y + 0.003 * z
    return np.stack([ux, uy, uz], axis=-1)


def _grid_coords_2d(nx: int, ny: int, size: float = 1.0) -> np.ndarray:
    """nx*ny 规则网格节点坐标（编号 = j*nx + i）."""
    xs = np.linspace(0.0, size, nx)
    ys = np.linspace(0.0, size, ny)
    return np.array([[xs[i], ys[j]] for j in range(ny) for i in range(nx)])


def _tria_mesh(nx: int, ny: int) -> Mesh:
    """规则三角网格（每方格两三角，逆时针）."""
    coords = _grid_coords_2d(nx, ny)
    conn = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b = j * nx + i, j * nx + i + 1
            c, d = (j + 1) * nx + i + 1, (j + 1) * nx + i
            conn.append([a, b, c])
            conn.append([a, c, d])
    return Mesh(coords, (ElementBlock(ElementType.TRIA3, np.array(conn)),))


def _quad_mesh(nx: int, ny: int) -> Mesh:
    """规则四边形网格（逆时针）."""
    coords = _grid_coords_2d(nx, ny)
    conn = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b = j * nx + i, j * nx + i + 1
            c, d = (j + 1) * nx + i + 1, (j + 1) * nx + i
            conn.append([a, b, c, d])
    return Mesh(coords, (ElementBlock(ElementType.QUAD4, np.array(conn)),))


def _hex_mesh(n: int) -> Mesh:
    """n*n*n 节点/边的规则六面体网格（节点编号 = k*n*n + j*n + i）."""
    axis = np.linspace(0.0, 1.0, n)
    coords = np.array([[axis[i], axis[j], axis[k]] for k in range(n) for j in range(n) for i in range(n)])
    conn = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                n0 = k * n * n + j * n + i
                conn.append(
                    [
                        n0,
                        n0 + 1,
                        n0 + n + 1,
                        n0 + n,  # z = k 层
                        n0 + n * n,
                        n0 + n * n + 1,
                        n0 + n * n + n + 1,
                        n0 + n * n + n,  # z = k+1 层
                    ]
                )
    return Mesh(coords, (ElementBlock(ElementType.HEX8, np.array(conn)),))


def _boundary_nodes_2d(nx: int, ny: int) -> list[int]:
    """网格边界节点编号（内部节点除外）."""
    return [j * nx + i for j in range(ny) for i in range(nx) if i in (0, nx - 1) or j in (0, ny - 1)]


def _linear_constraints_2d(mesh: Mesh, nodes: list[int]) -> tuple:
    """按线性场为指定节点构造逐分量约束."""
    constraints = []
    for node in nodes:
        ux, uy = _field_2d(*mesh.coords[node])
        constraints.append(Constraint(node, (0,), ux))
        constraints.append(Constraint(node, (1,), uy))
    return tuple(constraints)


def _linear_constraints_3d(mesh: Mesh, nodes) -> tuple:
    """按线性场为指定节点构造逐分量约束（3D）."""
    constraints = []
    for node in nodes:
        ux, uy, uz = _field_3d(*mesh.coords[node])
        constraints.append(Constraint(node, (0,), ux))
        constraints.append(Constraint(node, (1,), uy))
        constraints.append(Constraint(node, (2,), uz))
    return tuple(constraints)


class TestPatch2D:
    def test_tria3_patch(self) -> None:
        nx = ny = 3
        mesh = _tria_mesh(nx, ny)
        interior = [j * nx + i for j in range(1, ny - 1) for i in range(1, nx - 1)]
        constraints = _linear_constraints_2d(mesh, _boundary_nodes_2d(nx, ny))
        solution = solve_static(mesh, [MAT_PLANE], [UNIT], StaticCase(constraints=constraints))

        # 内部节点位移精确再现线性场
        expected = _field_2d(*mesh.coords[interior[0]])
        np.testing.assert_allclose(solution.node_displacement(interior[0]), expected, rtol=1e-12, atol=1e-15)

        # 全部单元应力 = D @ ε 精确值
        strain = np.array([0.002, -0.002, 0.007])
        stress_expected = MAT_PLANE.d_matrix() @ strain
        for result in solution.element_stresses(ElementType.TRIA3):
            np.testing.assert_allclose(result.stress, stress_expected, rtol=1e-12)

    def test_quad4_patch(self) -> None:
        nx = ny = 3
        mesh = _quad_mesh(nx, ny)
        interior = [j * nx + i for j in range(1, ny - 1) for i in range(1, nx - 1)]
        constraints = _linear_constraints_2d(mesh, _boundary_nodes_2d(nx, ny))
        solution = solve_static(mesh, [MAT_PLANE], [UNIT], StaticCase(constraints=constraints))

        expected = _field_2d(*mesh.coords[interior[0]])
        np.testing.assert_allclose(solution.node_displacement(interior[0]), expected, rtol=1e-12, atol=1e-15)

        strain = np.array([0.002, -0.002, 0.007])
        stress_expected = MAT_PLANE.d_matrix() @ strain
        for result in solution.element_stresses(ElementType.QUAD4):
            np.testing.assert_allclose(result.stress, stress_expected, rtol=1e-12)


class TestPatch3D:
    def test_tet4_patch(self) -> None:
        # 四面体 patch：4 角点 + 1 中心节点，划分为 4 个四面体
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.25, 0.25, 0.25]])
        conn = np.array([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
        mesh = Mesh(coords, (ElementBlock(ElementType.TET4, conn),))
        constraints = _linear_constraints_3d(mesh, range(4))
        solution = solve_static(mesh, [MAT_SOLID], [UNIT], StaticCase(constraints=constraints))

        # 中心节点（4 号）位移精确再现线性场
        expected = _field_3d(*mesh.coords[4])
        np.testing.assert_allclose(solution.node_displacement(4), expected, rtol=1e-12, atol=1e-15)

        strain = np.array([0.002, -0.002, 0.003, 0.007, 0.007, 0.0])
        stress_expected = MAT_SOLID.d_matrix() @ strain
        for result in solution.element_stresses(ElementType.TET4):
            np.testing.assert_allclose(result.stress, stress_expected, rtol=1e-12, atol=1e-15)

    def test_hex8_patch(self) -> None:
        n = 3
        mesh = _hex_mesh(n)
        center = 1 * n * n + 1 * n + 1
        nodes = [node for node in range(mesh.n_nodes) if node != center]
        constraints = _linear_constraints_3d(mesh, nodes)
        solution = solve_static(mesh, [MAT_SOLID], [UNIT], StaticCase(constraints=constraints))

        # 中心节点位移精确再现线性场
        expected = _field_3d(*mesh.coords[center])
        np.testing.assert_allclose(solution.node_displacement(center), expected, rtol=1e-12, atol=1e-15)

        strain = np.array([0.002, -0.002, 0.003, 0.007, 0.007, 0.0])
        stress_expected = MAT_SOLID.d_matrix() @ strain
        for result in solution.element_stresses(ElementType.HEX8):
            np.testing.assert_allclose(result.stress, stress_expected, rtol=1e-10, atol=1e-14)


class TestTrussStatic:
    def test_single_bar_tension(self) -> None:
        # 竖直杆下端固定、上端受拉：u = FL/EA，σ = F/A
        coords = np.array([[0.0, 0.0], [0.0, 5.0]])
        mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 1]])),))
        material = LinearElastic(1000.0)
        case = StaticCase(
            # 桁架节点横向（垂直杆轴）无刚度，须额外约束 x 方向
            constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=1, dofs=(0,))),
            loads=(NodalLoad(node=1, forces=(0.0, 10.0)),),
        )
        solution = solve_static(mesh, [material], [Section(area=2.0)], case)

        np.testing.assert_allclose(solution.node_displacement(1), [0.0, 10.0 * 5.0 / 2000.0])
        np.testing.assert_allclose(solution.element_results[0].stress, [10.0 / 2.0])
        # 反力与外载荷平衡
        np.testing.assert_allclose(solution.reactions[1], -10.0)  # 节点 0 的 y 反力（dof=1）

    def test_two_bar_symmetric(self) -> None:
        # 对称两杆桁架：A/B 固定，C 受竖直向下载荷 P
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 2], [1, 2]])),))
        material = LinearElastic(1000.0)
        load_p = 10.0
        case = StaticCase(
            constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=1, dofs=(0, 1))),
            loads=(NodalLoad(node=2, forces=(0.0, -load_p)),),
        )
        solution = solve_static(mesh, [material], [Section(area=1.0)], case)

        # 解析：δ = P L / (2 E A sin²45°)，L = sqrt(2)
        length = np.sqrt(2.0)
        delta = load_p * length / (2.0 * 1000.0 * 1.0 * 0.5)
        np.testing.assert_allclose(solution.node_displacement(2), [0.0, -delta], rtol=1e-12)

        # 轴力 N = E A δ sinθ / L，两杆相等；C 受压载荷 → 杆受压（应力为负）
        axial = 1000.0 * delta * (1.0 / np.sqrt(2.0)) / length
        np.testing.assert_allclose(solution.element_results[0].stress, [-abs(axial)], rtol=1e-12)

    def test_missing_constraint_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 1]])),))
        case = StaticCase(loads=(NodalLoad(node=1, forces=(1.0, 0.0)),))
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_static(mesh, [LinearElastic(1000.0)], [UNIT], case)


class TestHex8Uniaxial:
    def test_unit_cube_tension(self) -> None:
        # 单 Hex8 立方体单轴受拉（ν=0）：σxx = E·δ，端面反力均布
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        mesh = Mesh(coords, (ElementBlock(ElementType.HEX8, np.arange(8).reshape(1, 8)),))
        material = LinearElastic(100.0, 0.0, StressState.SOLID)
        delta = 0.001
        constraints = (
            # x=0 面 ux=0，x=1 面 ux=δ
            Constraint(0, (0, 1, 2), 0.0),
            Constraint(3, (0,), 0.0),
            Constraint(4, (0,), 0.0),
            Constraint(7, (0,), 0.0),
            Constraint(1, (0,), delta),
            Constraint(2, (0,), delta),
            Constraint(5, (0,), delta),
            Constraint(6, (0,), delta),
            # 防刚体：节点 1 约束 uy/uz，节点 3 约束 uz，节点 4 约束 uy
            Constraint(1, (1, 2), 0.0),
            Constraint(3, (2,), 0.0),
            Constraint(4, (1,), 0.0),
        )
        solution = solve_static(mesh, [material], [UNIT], StaticCase(constraints=constraints))

        # 均匀应力状态：σxx = E δ，其余分量为 0（ν=0）
        stress = solution.element_results[0].stress
        np.testing.assert_allclose(stress[0], 100.0 * delta, rtol=1e-12)
        np.testing.assert_allclose(stress[1:], 0.0, atol=1e-12)

        # x=0 面每个节点反力 fx = -σ A / 4
        np.testing.assert_allclose(solution.reactions[0], -100.0 * delta / 4.0, rtol=1e-12)
        np.testing.assert_allclose(solution.reactions[3 * 3 + 0], -100.0 * delta / 4.0, rtol=1e-12)

        # 泊松比为 0 → uy/uz 全零
        np.testing.assert_allclose(solution.displacements[:, 1:], 0.0, atol=1e-12)

        # 应变能 = 0.5 σ ε V
        np.testing.assert_allclose(solution.strain_energy, 0.5 * 100.0 * delta * delta, rtol=1e-12)
