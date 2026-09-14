"""FEA 有限元计算正确性数据驱动校核（测试数据见 tests/data/fea_analytical_cases.json）.

每组数据给出教科书闭式解析解金标准，测试侧按参数自动建模并断言：

- ``truss_tension``：单杆拉/压 u=PL/(EA)、σ=P/A、固定端反力 −P（含极端
- 小截面大柔度与反向载荷）；
- ``two_bar_truss``：对称两杆 δ=PL/(2EA sin²θ)（典型 45°、极端浅角/陡角）；
- ``cantilever_rod_modal``：ω_n=(2n-1)π/(2L)√(E/ρ)，含尺度律（L 加倍频率减半）；
- ``cantilever_beam_modal``：ω=β₁²√(EI/(ρAL⁴))，L 加倍按 1/16 缩放；
- ``euler_column``：悬臂 π²EI/(4L²)、铰支 π²EI/L² 及 L^-2 尺度律；
- ``thermal_linear_gradient``：Q4 条带线性温度场精确恢复（梯度/热流/反向/小温差）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    NodalValue,
    Section,
    StaticCase,
    ThermalCase,
    solve_buckling,
    solve_modal,
    solve_static,
    solve_thermal,
)

_DATA = json.loads((Path(__file__).parent / "data" / "fea_analytical_cases.json").read_text(encoding="utf-8"))


def _ids(cases: list[dict]) -> list[str]:
    return [case["name"] for case in cases]


def _rod_mesh(length: float, n_elem: int) -> Mesh:
    """沿 x 等分杆网格（TRUSS2）."""
    xs = np.linspace(0.0, length, n_elem + 1)
    coords = np.column_stack([xs, np.zeros_like(xs)])
    conn = np.array([[i, i + 1] for i in range(n_elem)])
    return Mesh(coords, (ElementBlock(ElementType.TRUSS2, conn),))


def _beam_mesh(length: float, n_elem: int) -> Mesh:
    """沿 x 等分梁网格（BEAM2，3 DOF/节点）."""
    xs = np.linspace(0.0, length, n_elem + 1)
    coords = np.column_stack([xs, np.zeros_like(xs)])
    conn = np.array([[i, i + 1] for i in range(n_elem)])
    return Mesh(coords, (ElementBlock(ElementType.BEAM2, conn),))


def _thermal_strip_mesh(length: float, height: float, nx: int) -> Mesh:
    """矩形条带 Q4 网格（两行节点，编号 node(j,i)=j*(nx+1)+i）."""
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, 2)
    coords = np.array([[x, y] for y in ys for x in xs])
    conn = []
    for i in range(nx):
        n0 = i
        conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return Mesh(coords, (ElementBlock(ElementType.QUAD4, np.array(conn)),))


# ---------------------------------------------------------------------------
# 静力学：桁架解析解（桁架单元为常应变，FEM 精确）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _DATA["truss_tension"], ids=_ids(_DATA["truss_tension"]))
def test_truss_tension_analytical(case: dict) -> None:
    coords = np.array([[0.0, 0.0], [0.0, case["L"]]])
    mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 1]])),))
    static_case = StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=1, dofs=(0,))),
        loads=(NodalLoad(node=1, forces=(0.0, case["P"])),),
    )
    solution = solve_static(mesh, [LinearElastic(case["E"])], [Section(area=case["A"])], static_case)

    assert solution.node_displacement(1)[1] == pytest.approx(case["u_tip"], rel=case["rtol"])
    assert solution.element_results[0].stress[0] == pytest.approx(case["stress"], rel=case["rtol"])
    # 固定端 y 向反力与外载荷平衡
    assert solution.reactions[1] == pytest.approx(case["reaction"], rel=case["rtol"])
    # 能量守恒：U = 1/2 P·u
    assert solution.strain_energy == pytest.approx(0.5 * case["P"] * case["u_tip"], rel=case["rtol"])


@pytest.mark.parametrize("case", _DATA["two_bar_truss"], ids=_ids(_DATA["two_bar_truss"]))
def test_two_bar_truss_analytical(case: dict) -> None:
    a, h = case["a"], case["h"]
    coords = np.array([[0.0, 0.0], [2.0 * a, 0.0], [a, h]])
    mesh = Mesh(coords, (ElementBlock(ElementType.TRUSS2, np.array([[0, 2], [1, 2]])),))
    static_case = StaticCase(
        constraints=(Constraint(node=0, dofs=(0, 1)), Constraint(node=1, dofs=(0, 1))),
        loads=(NodalLoad(node=2, forces=(0.0, -case["P"])),),
    )
    solution = solve_static(mesh, [LinearElastic(case["E"])], [Section(area=case["A"])], static_case)

    displacement = solution.node_displacement(2)
    assert displacement[0] == pytest.approx(0.0, abs=1.0e-12)
    assert displacement[1] == pytest.approx(case["dy_tip"], rel=case["rtol"])
    # 两杆对称、轴力相等（顶点受向下载荷 -> 压杆，应力为负）
    for result in solution.element_results:
        assert result.stress[0] == pytest.approx(case["stress"], rel=case["rtol"])


# ---------------------------------------------------------------------------
# 模态分析：杆纵向 / 梁弯曲固有频率
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _DATA["cantilever_rod_modal"], ids=_ids(_DATA["cantilever_rod_modal"]))
def test_cantilever_rod_modal_analytical(case: dict) -> None:
    material = LinearElastic(case["E"], density=case["rho"])
    mesh = _rod_mesh(case["L"], case["n_elem"])
    # 全节点约束横向自由度（杆单元仅提供轴向刚度）
    constraints = (Constraint(0, (0, 1)), *(Constraint(i, (1,)) for i in range(1, case["n_elem"] + 1)))
    solution = solve_modal(mesh, (material,), (Section(),), constraints, n_modes=len(case["omegas"]))
    for omega, expected, rtol in zip(solution.frequencies, case["omegas"], case["rtol"], strict=True):
        assert omega == pytest.approx(expected, rel=rtol)


@pytest.mark.parametrize("case", _DATA["cantilever_beam_modal"], ids=_ids(_DATA["cantilever_beam_modal"]))
def test_cantilever_beam_modal_analytical(case: dict) -> None:
    material = LinearElastic(case["E"], density=case["rho"])
    section = Section(area=case["A"], inertia=case["I"])
    mesh = _beam_mesh(case["L"], case["n_elem"])
    solution = solve_modal(mesh, (material,), (section,), (Constraint(0, (0, 1, 2)),), n_modes=1)
    assert solution.frequencies[0] == pytest.approx(case["omega"], rel=case["rtol"])


# ---------------------------------------------------------------------------
# 线性屈曲：欧拉临界载荷
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _DATA["euler_column"], ids=_ids(_DATA["euler_column"]))
def test_euler_column_analytical(case: dict) -> None:
    n_elem = case["n_elem"]
    mesh = _beam_mesh(case["L"], n_elem)
    material = LinearElastic(case["E"])
    section = Section(area=0.01, inertia=case["I"])
    if case["bc"] == "cantilever":
        constraints = (Constraint(0, (0, 1, 2)),)
    else:
        constraints = (Constraint(0, (0, 1)), Constraint(n_elem, (1,)))
    static_case = StaticCase(
        constraints=constraints,
        loads=(NodalLoad(n_elem, (-1.0, 0.0, 0.0)),),
    )
    solution = solve_buckling(mesh, [material], [section], static_case, n_modes=1)
    assert solution.load_factors[0] == pytest.approx(case["p_cr"], rel=case["rtol"])


# ---------------------------------------------------------------------------
# 稳态热传导：线性温度场（Q4 patch 级精确解）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _DATA["thermal_linear_gradient"], ids=_ids(_DATA["thermal_linear_gradient"]))
def test_thermal_linear_gradient_analytical(case: dict) -> None:
    nx = case["nx"]
    mesh = _thermal_strip_mesh(case["L"], case["h"], nx)
    material = ConductionMaterial(electric_sigma=1.0, thermal_k=case["k"])
    section = Section(thickness=case["thickness"])
    thermal_case = ThermalCase(
        temperatures=(
            NodalValue(0, case["t0"]),
            NodalValue(nx + 1, case["t0"]),
            NodalValue(nx, case["t1"]),
            NodalValue(2 * nx + 1, case["t1"]),
        )
    )
    solution = solve_thermal(mesh, [material], [section], thermal_case)

    # 节点温度沿 x 线性、沿 y 不变
    expected_t = case["t0"] + (case["t1"] - case["t0"]) * mesh.coords[:, 0] / case["L"]
    np.testing.assert_allclose(solution.temperatures, expected_t, rtol=case["rtol"])
    # 各单元温度梯度恒为 (dT/dx, 0)，热流密度模长 = k|dT/dx|
    np.testing.assert_allclose(solution.element_gradients[:, 0], case["grad_x"], rtol=case["rtol"])
    np.testing.assert_allclose(solution.element_gradients[:, 1], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(solution.element_heat_flux, case["flux"], rtol=case["rtol"])
    assert solution.t_min == pytest.approx(min(case["t0"], case["t1"]), rel=case["rtol"])
    assert solution.t_max == pytest.approx(max(case["t0"], case["t1"]), rel=case["rtol"])
