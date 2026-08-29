"""fea.harmonic / 谐响应分析测试.

验证维度：Rayleigh 阻尼矩阵组合、静力极限收敛、共振峰位置与相位、
直接法与模态叠加法互证（质量归一完备模态基下精确一致）、错误路径。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    HarmonicResponse,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    SolverError,
    StaticCase,
    solve_harmonic,
    solve_modal,
)
from zylab.fea.assemble import assemble_loads, assemble_mass, assemble_stiffness
from zylab.fea.harmonic import rayleigh_damping

E_MOD, RHO, AREA, LENGTH, N_ELEM = 1.0e7, 1.0, 1.0, 1.0, 4
FORCE = 1.0  # 末端 x 向简谐力幅值


def _cantilever_rod() -> tuple[Mesh, tuple[LinearElastic, ...], tuple[Section, ...], StaticCase]:
    """沿 x 的等分悬臂杆（4 单元）：全部节点锁 y 消除桁架横向无刚度自由度."""
    coords = np.array([[x * LENGTH / N_ELEM, 0.0] for x in range(N_ELEM + 1)])
    conn = np.array([[i, i + 1] for i in range(N_ELEM)], dtype=int)
    mesh = Mesh(
        coords=coords,
        blocks=(ElementBlock(ElementType.TRUSS2, conn, material=0, section=0),),
    )
    materials = (LinearElastic(E_MOD, density=RHO),)
    sections = (Section(area=AREA),)
    constraints = (
        Constraint(0, (0, 1)),
        *(Constraint(i, (1,)) for i in range(1, N_ELEM + 1)),
    )
    case = StaticCase(
        constraints=constraints,
        loads=(NodalLoad(N_ELEM, (FORCE, 0.0)),),
    )
    return mesh, materials, sections, case


def _rod_frequencies() -> np.ndarray:
    """覆盖前两阶固有频率的扫描序列（ω1 ≈ π/2L·√(E/ρ)）."""
    omega1 = np.pi / (2.0 * LENGTH) * np.sqrt(E_MOD / RHO)
    return np.linspace(0.0, 1.6 * omega1, 81)


# ---------------------------------------------------------------------------
# Rayleigh 阻尼矩阵
# ---------------------------------------------------------------------------


class TestRayleighDamping:
    def test_composition(self) -> None:
        """C = αM + βK 逐项线性叠加."""
        mesh, materials, sections, _ = _cantilever_rod()
        k = assemble_stiffness(mesh, materials, sections)
        m = assemble_mass(mesh, materials, sections)
        c = rayleigh_damping(2.0, 3.0, m, k).toarray()
        np.testing.assert_allclose(c, 2.0 * m.toarray() + 3.0 * k.toarray(), rtol=1e-12)

    def test_negative_coefficient_raises(self) -> None:
        mesh, materials, sections, _ = _cantilever_rod()
        k = assemble_stiffness(mesh, materials, sections)
        m = assemble_mass(mesh, materials, sections)
        with pytest.raises(SolverError, match="非负"):
            rayleigh_damping(-0.1, 0.0, m, k)


# ---------------------------------------------------------------------------
# 悬臂杆谐响应
# ---------------------------------------------------------------------------


class TestCantileverRodHarmonic:
    def test_static_limit(self) -> None:
        """ω→0 时复位移退化为静力解 u = FL/EA（实部主导、虚部近零）."""
        mesh, materials, sections, case = _cantilever_rod()
        response = solve_harmonic(mesh, materials, sections, case, [1.0e-6])
        u_end = response.node_response(N_ELEM, 0)[0]
        assert u_end.real == pytest.approx(FORCE * LENGTH / (E_MOD * AREA), rel=1e-6)
        assert abs(u_end.imag) < 1e-12

    def test_resonant_peak_and_phase(self) -> None:
        """共振峰位于基频附近，峰值点相位约 90°（滞后）."""
        mesh, materials, sections, case = _cantilever_rod()
        freqs = _rod_frequencies()
        response = solve_harmonic(mesh, materials, sections, case, freqs, alpha=3.0e2)
        tip = np.array([response.node_response(N_ELEM, j)[0] for j in range(freqs.size)])
        modal = solve_modal(mesh, materials, sections, case.constraints, n_modes=1)
        omega1 = modal.frequencies[0]
        peak_index = int(np.argmax(np.abs(tip)))
        # 峰值频率与基频偏差不超过一个扫描步长（离散采样容差）
        assert abs(freqs[peak_index] - omega1) <= (freqs[1] - freqs[0]) * 1.01
        # 有阻尼共振点相位约 -90°（位移滞后激励）
        assert abs(np.angle(tip[peak_index]) + np.pi / 2) < 0.35

    def test_matches_modal_superposition(self) -> None:
        """直接法与完备模态叠加法精确一致（质量归一模态基下解析等价）.

        eigsh 无法提取全部自由度模态，完备基改用稠密广义特征分解构造。
        """
        from scipy.linalg import eigh

        mesh, materials, sections, case = _cantilever_rod()
        alpha, beta = 5.0e1, 1.0e-5
        freqs = _rod_frequencies()[10:40]
        response = solve_harmonic(mesh, materials, sections, case, freqs, alpha=alpha, beta=beta)
        # 划块后自由子空间的完备模态基（稠密 eigh，质量归一）
        width = mesh.dofs_per_node
        fixed = {0 * width + 0, 0 * width + 1, *(i * width + 1 for i in range(1, N_ELEM + 1))}
        free = np.array([d for d in range(mesh.n_dofs) if d not in fixed])
        k_global = assemble_stiffness(mesh, materials, sections)
        m_global = assemble_mass(mesh, materials, sections)
        omegas, basis = eigh(
            k_global[free][:, free].toarray(),
            m_global[free][:, free].toarray(),
        )
        shapes = np.zeros((mesh.n_dofs, free.size))
        shapes[free, :] = basis
        # 质量归一振型：C = αM + βK 的模态阻尼 = α + βω_i²（模态基完备时叠加精确）
        force = assemble_loads(mesh, case, sections)
        for j, omega in enumerate(freqs):
            u_modal = np.zeros(mesh.n_dofs, dtype=complex)
            for i in range(free.size):
                phi = shapes[:, i]
                # eigh 返回广义特征值 λ = ω_i²，模态阻尼 = α + βω_i² = α + βλ_i
                denom = omegas[i] - omega**2 + 1j * omega * (alpha + beta * omegas[i])
                u_modal += phi * (phi @ force) / denom
            np.testing.assert_allclose(response.displacements[:, j], u_modal, rtol=1e-8, atol=1e-12)

    def test_result_api(self) -> None:
        """amplitude/phase/n_frequencies 访问器."""
        mesh, materials, sections, case = _cantilever_rod()
        response = solve_harmonic(mesh, materials, sections, case, [0.0, 100.0])
        assert isinstance(response, HarmonicResponse)
        assert response.n_frequencies == 2
        amp = response.amplitude(0)
        phase = response.phase(0)
        assert amp.shape == phase.shape == (mesh.n_dofs,)
        # ω=0 无阻尼：位移为实数静力解，相位全 0
        np.testing.assert_allclose(phase, 0.0, atol=1e-12)
        # 约束自由度位移为 0
        np.testing.assert_allclose(response.node_response(0, 0), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestHarmonicErrors:
    def test_empty_frequencies(self) -> None:
        mesh, materials, sections, case = _cantilever_rod()
        with pytest.raises(SolverError, match="频率序列为空"):
            solve_harmonic(mesh, materials, sections, case, [])

    def test_negative_frequency(self) -> None:
        mesh, materials, sections, case = _cantilever_rod()
        with pytest.raises(SolverError, match="非负"):
            solve_harmonic(mesh, materials, sections, case, [100.0, -1.0])

    def test_negative_damping(self) -> None:
        mesh, materials, sections, case = _cantilever_rod()
        with pytest.raises(SolverError, match="非负"):
            solve_harmonic(mesh, materials, sections, case, [100.0], alpha=-1.0)

    def test_missing_density(self) -> None:
        mesh, _, sections, case = _cantilever_rod()
        materials = (LinearElastic(E_MOD),)  # 无密度
        with pytest.raises(Exception, match="质量密度"):
            solve_harmonic(mesh, materials, sections, case, [100.0])

    def test_nonzero_constraint(self) -> None:
        mesh, materials, sections, case = _cantilever_rod()
        bad = StaticCase(
            constraints=(Constraint(0, (0, 1), value=0.1), *(Constraint(i, (1,)) for i in range(1, N_ELEM + 1))),
            loads=case.loads,
        )
        with pytest.raises(SolverError, match="须为 0"):
            solve_harmonic(mesh, materials, sections, bad, [100.0])

    def test_no_constraint(self) -> None:
        mesh, materials, sections, case = _cantilever_rod()
        free_case = StaticCase(loads=case.loads)
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_harmonic(mesh, materials, sections, free_case, [100.0])
