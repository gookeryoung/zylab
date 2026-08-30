"""fea.transient / 瞬态动力分析测试.

验证维度：SDOF 阶跃载荷解析解（1-cos ωt）、初始速度自由振动、
无阻尼能量守恒（平均加速度法线性系统精确守恒）、多自由度模态纯激励、
Rayleigh 阻尼衰减包络、载荷时程因子、逐步平衡方程回代、
时间步二阶收敛、访问器与错误路径。
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
    TransientSolution,
    solve_modal,
    solve_transient,
)
from zylab.fea.assemble import assemble_mass, assemble_stiffness

E_MOD, RHO, AREA, LENGTH = 1.0e7, 1.0, 1.0, 1.0


def _sdof_rod(load: float = 0.0) -> tuple[Mesh, tuple[LinearElastic, ...], tuple[Section, ...], StaticCase]:
    """单杆 SDOF 模型：节点 0 全固支、节点 1 锁 y，自由度仅节点 1 的 x.

    一致质量下 m = ρAL/3，k = EA/L，ω = √(3E/ρL²)。
    """
    mesh = Mesh(
        coords=np.array([[0.0, 0.0], [LENGTH, 0.0]]),
        blocks=(ElementBlock(ElementType.TRUSS2, np.array([[0, 1]], dtype=int), material=0, section=0),),
    )
    materials = (LinearElastic(E_MOD, density=RHO),)
    sections = (Section(area=AREA),)
    loads = (NodalLoad(1, (load, 0.0)),) if load else ()
    case = StaticCase(
        constraints=(Constraint(0, (0, 1)), Constraint(1, (1,))),
        loads=loads,
    )
    return mesh, materials, sections, case


def _sdof_omega() -> float:
    """SDOF 固有圆频率（一致质量）."""
    return float(np.sqrt(3.0 * E_MOD / (RHO * LENGTH**2)))


def _cantilever_rod(n_elem: int = 4) -> tuple[Mesh, tuple[LinearElastic, ...], tuple[Section, ...]]:
    """沿 x 的等分悬臂杆（全部节点锁 y 消除桁架横向无刚度自由度）."""
    coords = np.array([[x * LENGTH / n_elem, 0.0] for x in range(n_elem + 1)])
    conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(
        coords=coords,
        blocks=(ElementBlock(ElementType.TRUSS2, conn, material=0, section=0),),
    )
    return mesh, (LinearElastic(E_MOD, density=RHO),), (Section(area=AREA),)


# ---------------------------------------------------------------------------
# SDOF 解析解验证
# ---------------------------------------------------------------------------


class TestSdofAnalytic:
    def test_step_load_response(self) -> None:
        """阶跃载荷（λ≡1）精确解 u = (F/k)(1 - cos ωt)."""
        force = 2.0
        mesh, materials, sections, case = _sdof_rod(load=force)
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        n_steps = 320  # 每周期 160 步
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=2.0 * period,
            n_steps=n_steps,
        )
        k_spr = E_MOD * AREA / LENGTH
        exact = force / k_spr * (1.0 - np.cos(omega * solution.times))
        u_tip = solution.displacements[mesh.dofs_per_node + 0, :]
        # 按响应峰值归一化比较（rtol 管峰谷相位、atol 管过零邻域）
        amplitude = 2.0 * force / k_spr
        np.testing.assert_allclose(u_tip / amplitude, exact / amplitude, rtol=5e-3, atol=2e-3)

    def test_initial_velocity_free_vibration(self) -> None:
        """零载荷 + 初始速度：u = (v0/ω) sin ωt，振幅无衰减."""
        mesh, materials, sections, case = _sdof_rod()
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        v0 = np.zeros(mesh.n_dofs)
        v0[mesh.dofs_per_node + 0] = 0.8
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=2.0 * period,
            n_steps=320,
            initial_velocity=v0,
        )
        exact = v0[mesh.dofs_per_node + 0] / omega * np.sin(omega * solution.times)
        u_tip = solution.displacements[mesh.dofs_per_node + 0, :]
        # 按响应峰值归一化比较（rtol 管峰谷相位、atol 管过零邻域）
        amplitude = v0[mesh.dofs_per_node + 0] / omega
        np.testing.assert_allclose(u_tip / amplitude, exact / amplitude, rtol=5e-3, atol=2e-3)
        # 平均加速度法无数值阻尼：峰值幅值不应系统性衰减
        assert np.max(np.abs(u_tip)) == pytest.approx(np.max(np.abs(exact)), rel=5e-3)

    def test_second_order_convergence(self) -> None:
        """位移误差随步长减半按 O(dt²) 收缩（误差比约 4）."""
        force = 2.0
        mesh, materials, sections, case = _sdof_rod(load=force)
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        k_spr = E_MOD * AREA / LENGTH
        errors = []
        for n_steps in (40, 80):
            solution = solve_transient(
                mesh,
                materials,
                sections,
                case,
                duration=2.0 * period,
                n_steps=n_steps,
            )
            exact = force / k_spr * (1.0 - np.cos(omega * solution.times))
            u_tip = solution.displacements[mesh.dofs_per_node + 0, :]
            errors.append(float(np.max(np.abs(u_tip - exact))))
        ratio = errors[0] / errors[1]
        assert 3.0 < ratio < 5.0

    def test_energy_conservation(self) -> None:
        """无阻尼线性系统平均加速度法精确守恒能量 E = ½vᵀMv + ½uᵀKu."""
        mesh, materials, sections, case = _sdof_rod()
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        v0 = np.zeros(mesh.n_dofs)
        v0[mesh.dofs_per_node + 0] = 0.5
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=period,
            n_steps=80,
            initial_velocity=v0,
        )
        k_mat = assemble_stiffness(mesh, materials, sections)
        m_mat = assemble_mass(mesh, materials, sections)
        energy = 0.5 * np.einsum(
            "ij,ij->j",
            solution.velocities,
            m_mat @ solution.velocities,
        ) + 0.5 * np.einsum("ij,ij->j", solution.displacements, k_mat @ solution.displacements)
        np.testing.assert_allclose(energy, energy[0], rtol=1e-9)

    def test_damped_free_vibration_envelope(self) -> None:
        """质量比例阻尼 α：峰值包络按 exp(-ξωt) 衰减，ξ = α/(2ω)."""
        mesh, materials, sections, case = _sdof_rod()
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        alpha = 0.05 * 2.0 * omega  # ξ = 0.05
        v0 = np.zeros(mesh.n_dofs)
        v0[mesh.dofs_per_node + 0] = 1.0
        n_cycles = 8
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=n_cycles * period,
            n_steps=80 * n_cycles,
            initial_velocity=v0,
            alpha=alpha,
        )
        u_tip = np.abs(solution.displacements[mesh.dofs_per_node + 0, :])
        times = solution.times
        # 初速激励下峰值出现在 t_k = (k + 1/4)T，逐峰与解析包络 v0/ω·exp(-ξωt) 比较
        for k in (1, 4, n_cycles - 2):
            t_peak = (k + 0.25) * period
            lo = np.searchsorted(times, t_peak - 0.25 * period)
            hi = np.searchsorted(times, t_peak + 0.25 * period)
            peak = float(np.max(u_tip[lo:hi]))
            envelope = 1.0 / omega * np.exp(-0.05 * omega * t_peak)
            assert peak == pytest.approx(envelope, rel=0.05)


# ---------------------------------------------------------------------------
# 多自由度与载荷时程
# ---------------------------------------------------------------------------


class TestMultiDofAndLoads:
    def test_modal_pure_excitation(self) -> None:
        """初始速度取一阶振型时响应保持纯模态：u(t) = (c/ω1) sin(ω1 t) φ1."""
        mesh, materials, sections = _cantilever_rod()
        constraints = (
            Constraint(0, (0, 1)),
            *(Constraint(i, (1,)) for i in range(1, 5)),
        )
        case = StaticCase(constraints=constraints)
        modal = solve_modal(mesh, materials, sections, constraints, n_modes=1)
        omega1 = float(modal.frequencies[0])
        scale = 0.01
        v0 = scale * modal.mode_shapes[:, 0]
        period = 2.0 * np.pi / omega1
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=period,
            n_steps=240,
            initial_velocity=v0,
        )
        tip = mesh.n_nodes - 1
        phi_tip = modal.mode_shapes[tip * mesh.dofs_per_node + 0, 0]
        exact_tip = scale / omega1 * np.sin(omega1 * solution.times) * phi_tip
        u_tip = solution.displacements[tip * mesh.dofs_per_node + 0, :]
        # 按响应峰值归一化比较（rtol 管峰谷相位、atol 管过零邻域）
        amplitude = scale / omega1 * abs(phi_tip)
        np.testing.assert_allclose(u_tip / amplitude, exact_tip / amplitude, rtol=5e-3, atol=2e-3)

    def test_load_fn_scales_amplitude(self) -> None:
        """λ(t) ≡ 2 与静力载荷翻倍给出同一时程（分布不变仅幅值缩放）."""
        force = 1.0
        mesh, materials, sections, case = _sdof_rod(load=force)
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        base = solve_transient(mesh, materials, sections, case, duration=period, n_steps=80)
        doubled = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=period,
            n_steps=80,
            load_fn=lambda _t: 2.0,
        )
        np.testing.assert_allclose(
            doubled.displacements,
            2.0 * base.displacements,
            rtol=1e-12,
        )

    def test_equilibrium_residual(self) -> None:
        """半正弦脉冲下逐步平衡方程 M ü + C u̇ + K u = λ f 精确成立."""
        force = 3.0
        mesh, materials, sections, case = _sdof_rod(load=force)
        omega = _sdof_omega()
        period = 2.0 * np.pi / omega
        t_pulse = 0.75 * period
        alpha, beta = 4.0e2, 1.0e-7
        load_fn = lambda t: np.sin(np.pi * t / t_pulse) if t < t_pulse else 0.0  # noqa: E731  局部时程函数
        solution = solve_transient(
            mesh,
            materials,
            sections,
            case,
            duration=1.5 * period,
            n_steps=120,
            load_fn=load_fn,
            alpha=alpha,
            beta=beta,
        )
        k_mat = assemble_stiffness(mesh, materials, sections)
        m_mat = assemble_mass(mesh, materials, sections)
        c_mat = alpha * m_mat + beta * k_mat
        force_vec = np.zeros(mesh.n_dofs)
        force_vec[mesh.dofs_per_node + 0] = force
        # 平衡校验仅覆盖自由自由度：约束自由度由约束反力平衡，残差非零是预期行为
        free = np.array([mesh.dofs_per_node + 0])
        for step in range(solution.times.size):
            factor = 1.0 if load_fn is None else load_fn(solution.times[step])
            residual = (
                m_mat @ solution.accelerations[:, step]
                + c_mat @ solution.velocities[:, step]
                + k_mat @ solution.displacements[:, step]
                - force_vec * factor
            )
            scale = max(float(np.linalg.norm(force_vec * factor)), 1.0)
            assert float(np.linalg.norm(residual[free])) < 1e-6 * scale


# ---------------------------------------------------------------------------
# 结果访问器与元数据
# ---------------------------------------------------------------------------


class TestSolutionAccessors:
    def test_shapes_and_metadata(self) -> None:
        """时间站点含 t=0 共 n_steps+1 个；三个场形状一致；dt 一致."""
        mesh, materials, sections, case = _sdof_rod()
        solution = solve_transient(mesh, materials, sections, case, duration=0.4, n_steps=20)
        assert isinstance(solution, TransientSolution)
        assert solution.n_steps == 20
        assert solution.times[0] == pytest.approx(0.0)
        assert solution.times[-1] == pytest.approx(0.4)
        assert solution.dt == pytest.approx(0.02)
        assert solution.displacements.shape == (mesh.n_dofs, 21)
        assert solution.velocities.shape == (mesh.n_dofs, 21)
        assert solution.accelerations.shape == (mesh.n_dofs, 21)
        # 约束自由度全时程为零
        np.testing.assert_allclose(solution.displacements[[0, 1], :], 0.0, atol=1e-15)

    def test_node_history(self) -> None:
        """node_history 返回指定节点分量的位移时程副本."""
        mesh, materials, sections, case = _sdof_rod(load=1.0)
        solution = solve_transient(mesh, materials, sections, case, duration=0.1, n_steps=10)
        history = solution.node_history(1, 0)
        np.testing.assert_allclose(history, solution.displacements[mesh.dofs_per_node + 0, :])
        history[0] = 999.0  # 副本修改不回写
        assert solution.displacements[mesh.dofs_per_node + 0, 0] == pytest.approx(0.0)

    def test_node_history_out_of_range(self) -> None:
        mesh, materials, sections, case = _sdof_rod()
        solution = solve_transient(mesh, materials, sections, case, duration=0.1, n_steps=5)
        with pytest.raises(SolverError, match="节点索引"):
            solution.node_history(2, 0)
        with pytest.raises(SolverError, match="自由度"):
            solution.node_history(1, 2)


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_time_parameters(self) -> None:
        mesh, materials, sections, case = _sdof_rod()
        with pytest.raises(SolverError, match="总时长"):
            solve_transient(mesh, materials, sections, case, duration=0.0, n_steps=10)
        with pytest.raises(SolverError, match="总时长"):
            solve_transient(mesh, materials, sections, case, duration=-1.0, n_steps=10)
        with pytest.raises(SolverError, match="积分步数"):
            solve_transient(mesh, materials, sections, case, duration=1.0, n_steps=0)

    def test_invalid_damping(self) -> None:
        mesh, materials, sections, case = _sdof_rod()
        with pytest.raises(SolverError, match="非负"):
            solve_transient(mesh, materials, sections, case, duration=1.0, n_steps=5, alpha=-1.0)

    def test_invalid_initial_velocity(self) -> None:
        mesh, materials, sections, case = _sdof_rod()
        with pytest.raises(SolverError, match="初始速度"):
            solve_transient(
                mesh,
                materials,
                sections,
                case,
                duration=1.0,
                n_steps=5,
                initial_velocity=np.zeros(3),
            )

    def test_constraints_required_and_zero(self) -> None:
        mesh, materials, sections, _ = _sdof_rod()
        with pytest.raises(SolverError, match="缺少位移约束"):
            solve_transient(mesh, materials, sections, StaticCase(), duration=1.0, n_steps=5)
        bad = StaticCase(constraints=(Constraint(0, (0,), value=0.1), Constraint(1, (1,))))
        with pytest.raises(SolverError, match="须为 0"):
            solve_transient(mesh, materials, sections, bad, duration=1.0, n_steps=5)
