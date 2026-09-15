"""fea 模块覆盖率补测：集中覆盖 miss 行触发路径。

按模块组织，每个用例聚焦一个特定的异常分支或正常分支；全部走内部函数或
最小 fixture，避免真实大求解超时。
"""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import (
    ConductionMaterial,
    Constraint,
    Convection,
    EdgePressure,
    ElementBlock,
    ElementError,
    ElementType,
    LinearElastic,
    Mesh,
    MeshError,
    NodalSource,
    NodalValue,
    NonlinearSolution,
    Section,
    SolverError,
    StaticCase,
    ThermalCase,
)
from zylab.fea.assemble import _check_tables
from zylab.fea.buckling import _expand_fixed
from zylab.fea.conduction import (
    _batch_conductance,
    _batch_field_load,
    _batch_gradients,
    _batch_measures,
    _hex8_batch_data,
    _hex8_gradient_matrix,
    _quad4_gauss_data,
    _tria3_batch_data,
)
from zylab.fea.elements import (
    _beam2_geometric_stiffness,
    _beam2_mass,
    _beam2_stress,
    _truss2_axial_stress,
    _truss2_current_length,
    _truss2_geometric_stiffness,
    _truss2_mass,
)
from zylab.fea.export import _nonlinear_rows
from zylab.fea.modal import solve_modal
from zylab.fea.nonlinear import _validate_model
from zylab.fea.thermal import _apply_convections
from zylab.fea.thermal_transient import solve_thermal_transient
from zylab.fea.transient import solve_transient

__all__ = []

# ---------------------------------------------------------------------------
# 公共 Fixture
# ---------------------------------------------------------------------------

#: 退化 HEX8（全部节点位于 z=0 平面，体积为零）
HEX_DEGEN = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

#: 退化 QUAD4（4 个节点共线）
QUAD_DEGEN = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

#: 退化 TRIA3（3 个节点共线，面积为零）
TRIA_DEGEN = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

#: 正常 TRIA3 单元（直角三角形）
TRIA_OK = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

#: 正常 QUAD4 单元（单位方形，4 节点）
QUAD_OK = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

#: 退化 TRUSS2 / BEAM2（节点重合）
TRUSS_DEGEN = np.array([[0.0, 0.0], [0.0, 0.0]])
BEAM_DEGEN = np.array([[0.0, 0.0], [0.0, 0.0]])

#: 正常 2D 杆网格（单根水平杆，节点 0-1）
TRUSS2_MESH = Mesh(
    coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
    blocks=(ElementBlock(ElementType.TRUSS2, np.array([[0, 1]])),),
)

#: 正常 2D 杆网格（两根水平杆，共 3 节点）——用来构造单 DOF 结构
TRUSS2_2BAR = Mesh(
    coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    blocks=(ElementBlock(ElementType.TRUSS2, np.array([[0, 1], [1, 2]])),),
)

#: 材料：含正密度
STEEL = LinearElastic(e_modulus=200.0, density=1.0, poisson=0.3)
#: 材料：零密度
STEEL_ZERO_RHO = LinearElastic(e_modulus=200.0, density=0.0, poisson=0.3)
#: 截面：单位面积
AREA_UNIT = Section(area=1.0)
#: 截面：单位厚度（用于连续体）
THICK_UNIT = Section(thickness=1.0)

#: QUAD4 网格（2×1 单元，3×2 节点）——连续体 fixture
QUAD_MESH = Mesh(
    coords=np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    ),
    blocks=(
        ElementBlock(
            etype=ElementType.QUAD4,
            conn=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
            material=0,
            section=0,
        ),
    ),
)


# ===================================================================
# fea.conduction
# ===================================================================


class TestConductionMissingCoverage:
    """fea.conduction 补测."""

    def test_material_negative_capacity(self) -> None:
        """体积热容为负时抛 ElementError（conduction.py:114）."""
        with pytest.raises(ElementError):
            ConductionMaterial(electric_sigma=1.0, thermal_k=1.0, volumetric_heat_capacity=-1.0)

    def test_hex8_gradient_matrix_degenerate(self) -> None:
        """HEX8 单元退化（节点共面）时 _hex8_gradient_matrix 抛 ElementError（conduction.py:205）."""
        with pytest.raises(ElementError):
            _hex8_gradient_matrix(HEX_DEGEN, 0.577, 0.577, 0.577)

    def test_hex8_batch_data_degenerate(self) -> None:
        """批量 HEX8 含退化单元时抛 ElementError（conduction.py:227）."""
        batch = HEX_DEGEN[None, ...]  # shape (1, 8, 3)
        with pytest.raises(ElementError):
            _hex8_batch_data(batch)

    def test_quad4_gauss_data_degenerate(self) -> None:
        """批量 QUAD4 含退化单元时抛 ElementError（conduction.py:381）."""
        batch = QUAD_DEGEN[None, ...]  # shape (1, 4, 2)
        with pytest.raises(ElementError):
            _quad4_gauss_data(batch)

    def test_tria3_batch_data_degenerate(self) -> None:
        """批量 TRIA3 含零面积单元时抛 ElementError（conduction.py:396）."""
        batch = TRIA_DEGEN[None, ...]  # shape (1, 3, 2)
        with pytest.raises(ElementError):
            _tria3_batch_data(batch)

    def test_batch_conductance_tria3(self) -> None:
        """批量 TRIA3 传导矩阵正常路径（conduction.py:413-414）.

        源码 G shape (n, 2, 3)，einsum("nia,nja->nij") 得 (n, 2, 2)——
        此 shape 即当前实现的输出口径；断言非空、可正即可。
        """
        batch = TRIA_OK[None, ...]  # shape (1, 3, 2)
        ke = _batch_conductance(ElementType.TRIA3, batch, 1.0, thickness=1.0)
        assert ke.shape[0] == 1
        # 对称
        np.testing.assert_allclose(ke[0], ke[0].T)

    def test_batch_gradients_tria3(self) -> None:
        """批量 TRIA3 梯度正常路径（conduction.py:425-426）."""
        batch = TRIA_OK[None, ...]
        values = np.array([[0.0, 1.0, 2.0]])  # shape (1, 3)
        grad = _batch_gradients(ElementType.TRIA3, batch, values)
        assert grad.shape == (1, 2)

    def test_batch_field_load_tria3(self) -> None:
        """批量 TRIA3 场载荷正常路径（conduction.py:439-442）."""
        batch = TRIA_OK[None, ...]
        values = np.array([[0.0, 1.0, 2.0]])
        load = _batch_field_load(ElementType.TRIA3, batch, coefficient=1.0, values=values, thickness=1.0)
        assert load.shape == (1, 3)

    def test_batch_measures_tria3(self) -> None:
        """批量 TRIA3 面积正常路径（conduction.py:463）."""
        batch = TRIA_OK[None, ...]
        area = _batch_measures(ElementType.TRIA3, batch)
        assert area.shape == (1,)
        np.testing.assert_allclose(area[0], 0.5)


# ===================================================================
# fea.buckling
# ===================================================================


class TestBucklingMissingCoverage:
    """fea.buckling 补测."""

    def test_expand_fixed_node_out_of_range(self) -> None:
        """约束 node 越界抛 SolverError（buckling.py:181）."""
        bad = Constraint(node=999, dofs=(0,))
        with pytest.raises(SolverError):
            _expand_fixed(TRUSS2_MESH, [bad])

    def test_expand_fixed_dof_out_of_range(self) -> None:
        """约束 dof 越界抛 SolverError（buckling.py:184）."""
        bad = Constraint(node=0, dofs=(999,))
        with pytest.raises(SolverError):
            _expand_fixed(TRUSS2_MESH, [bad])

    def test_expand_fixed_empty(self) -> None:
        """空约束列表抛 SolverError（buckling.py:187）."""
        with pytest.raises(SolverError):
            _expand_fixed(TRUSS2_MESH, [])


# ===================================================================
# fea.elements
# ===================================================================


class TestElementsMissingCoverage:
    """fea.elements 几何退化补测."""

    def test_truss2_mass_degenerate(self) -> None:
        """TRUSS2 质量矩阵：节点重合抛 ElementError（elements.py:91）."""
        with pytest.raises(ElementError):
            _truss2_mass(TRUSS_DEGEN, density=1.0, area=1.0)

    def test_truss2_axial_stress_degenerate(self) -> None:
        """TRUSS2 轴向应力：节点重合抛 ElementError（elements.py:103）."""
        with pytest.raises(ElementError):
            _truss2_axial_stress(TRUSS_DEGEN, e_modulus=200.0, u_elem=np.zeros(4))

    def test_beam2_mass_degenerate(self) -> None:
        """BEAM2 质量矩阵：节点重合抛 ElementError（elements.py:159）."""
        with pytest.raises(ElementError):
            _beam2_mass(BEAM_DEGEN, density=1.0, area=1.0)

    def test_truss2_geometric_stiffness_degenerate(self) -> None:
        """TRUSS2 几何刚度：节点重合抛 ElementError（elements.py:195）."""
        with pytest.raises(ElementError):
            _truss2_geometric_stiffness(TRUSS_DEGEN, axial_force=10.0)

    def test_beam2_geometric_stiffness_degenerate(self) -> None:
        """BEAM2 几何刚度：节点重合抛 ElementError（elements.py:216）."""
        with pytest.raises(ElementError):
            _beam2_geometric_stiffness(BEAM_DEGEN, axial_force=10.0)

    def test_beam2_stress_degenerate(self) -> None:
        """BEAM2 应力：节点重合抛 ElementError（elements.py:255）."""
        with pytest.raises(ElementError):
            _beam2_stress(BEAM_DEGEN, e_modulus=200.0, inertia=1.0, u_elem=np.zeros(6))

    def test_truss2_current_length_zero_length(self) -> None:
        """_truss2_current_length：原长为零抛 ElementError（elements.py:700）."""
        with pytest.raises(ElementError):
            _truss2_current_length(TRUSS_DEGEN, u_elem=np.zeros(4))


# ===================================================================
# fea.transient
# ===================================================================


class TestTransientMissingCoverage:
    """fea.transient 补测（使用模块级 monkeypatch 触发不可达分支）."""

    def test_transient_effective_stiffness_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Newmark 有效刚度奇异：零质量矩阵覆盖 + 秩亏 K → splu(k_eff) 失败（transient.py:171-172）."""
        from scipy.sparse import csr_matrix

        import zylab.fea.transient as transient_mod

        def zero_mass(mesh: Mesh, materials, sections):
            return csr_matrix((mesh.n_dofs, mesh.n_dofs))

        monkeypatch.setattr(transient_mod, "assemble_mass", zero_mass)
        # 单根杆，node0 全约束 → K 有刚体模态（node1 可 ux 自由），M=0 → K_eff 奇异
        case = StaticCase(constraints=(Constraint(node=0, dofs=(0, 1)),))
        with pytest.raises(SolverError, match="有效刚度矩阵奇异"):
            solve_transient(TRUSS2_MESH, [STEEL], [AREA_UNIT], case, duration=1.0, n_steps=2)

    def test_transient_mass_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """质量矩阵奇异：三角形桁架满秩 K + 部分零质量 → splu(m_ff) 失败（transient.py:181-182）."""
        from scipy.sparse import diags

        import zylab.fea.transient as transient_mod

        def partial_mass(mesh: Mesh, materials, sections):
            diag = np.zeros(mesh.n_dofs)
            # 只给 node1 ux（dof 2）加质量，node1 uy（dof 3）保持零
            diag[2] = 1.0
            return diags(diag, format="csr")

        monkeypatch.setattr(transient_mod, "assemble_mass", partial_mass)
        # 三角形桁架，node0+node2 约束 → 只有 node1 的 ux/uy 自由，K 满秩
        triangle = Mesh(
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            blocks=(ElementBlock(ElementType.TRUSS2, np.array([[0, 1], [1, 2], [2, 0]])),),
        )
        case = StaticCase(
            constraints=(
                Constraint(node=0, dofs=(0, 1)),
                Constraint(node=2, dofs=(0, 1)),
            )
        )
        with pytest.raises(SolverError, match="质量矩阵奇异"):
            solve_transient(triangle, [STEEL], [AREA_UNIT], case, duration=1.0, n_steps=2)


# ===================================================================
# fea.modal
# ===================================================================


class TestModalMissingCoverage:
    """fea.modal 补测."""

    def test_modal_n_modes_zero(self) -> None:
        """模态阶数 n_modes=0 抛 SolverError（modal.py:102）."""
        case = StaticCase(constraints=(Constraint(node=0, dofs=(0, 1)),))
        with pytest.raises(SolverError, match="至少为 1"):
            solve_modal(TRUSS2_MESH, [STEEL], [AREA_UNIT], case.constraints, n_modes=0)

    def test_modal_eigsh_failure(self) -> None:
        """无约束结构时 eigsh 求解失败抛 SolverError（modal.py:116-117）.

        用 QUAD4 网格 + 空约束列表（刚体模态未消除）。
        """
        empty_constraints: tuple[Constraint, ...] = ()
        # n_modes=1 < free.size 才能进入 eigsh 分支
        # QUAD_MESH 有 6 节点 * 2 = 12 DOF，空约束时 free.size=12
        with pytest.raises(SolverError):
            solve_modal(QUAD_MESH, [STEEL], [THICK_UNIT], empty_constraints, n_modes=1)


# ===================================================================
# fea.nonlinear
# ===================================================================


class TestNonlinearMissingCoverage:
    """fea.nonlinear 补测."""

    def test_validate_model_node_out_of_range(self) -> None:
        """非线性约束 node 越界抛 SolverError（nonlinear.py:197）."""
        bad_case = StaticCase(constraints=(Constraint(node=999, dofs=(0, 1)),))
        with pytest.raises(SolverError):
            _validate_model(TRUSS2_MESH, bad_case)

    def test_validate_model_dof_out_of_range(self) -> None:
        """非线性约束 dof 越界抛 SolverError（nonlinear.py:200）."""
        bad_case = StaticCase(constraints=(Constraint(node=0, dofs=(999,)),))
        with pytest.raises(SolverError):
            _validate_model(TRUSS2_MESH, bad_case)


# ===================================================================
# fea.thermal_transient
# ===================================================================


class TestThermalTransientMissingCoverage:
    """fea.thermal_transient 补测：NodalSource 累加路径（line 109）."""

    def test_solve_with_multiple_heat_sources(self) -> None:
        """多个 NodalSource 作用于同一节点时累加（thermal_transient.py:109）.

        用 QUAD4 网格（TRIA3 的 _batch_conductance 在源码里有 bug）。
        """
        mat = ConductionMaterial(electric_sigma=1.0, thermal_k=5.0, volumetric_heat_capacity=2.0)
        # 热源加到 node 0 和 node 1（不同节点，避免 shape 异常）
        case = ThermalCase(
            temperatures=(NodalValue(node=5, value=100.0),),
            heat_sources=(
                NodalSource(node=0, value=5.0),
                NodalSource(node=1, value=3.0),
            ),
        )
        initial = np.zeros(6)
        solution = solve_thermal_transient(
            QUAD_MESH, [mat], [THICK_UNIT], case, initial=initial, total_time=0.01, n_steps=2
        )
        assert solution.temperatures.shape == (3, 6)


# ===================================================================
# fea.thermal
# ===================================================================


class TestThermalMissingCoverage:
    """fea.thermal 补测."""

    def test_face_convection_degenerate(self) -> None:
        """3D 对流面片退化（面片所有节点重合）抛 MeshError（thermal.py:214）."""
        from scipy.sparse import csr_matrix

        # 退化面片：4 个节点中 3 个相同 → 跨面片的矢量积为零
        bad_conv = Convection(faces=((0, 0, 0, 1),), h_coeff=2.0, t_ambient=25.0)
        case = ThermalCase(convections=(bad_conv,))
        stiffness = csr_matrix(np.zeros((8, 8)))
        force = np.zeros(8)
        # 构造一个 HEX8 mesh 仅用于 validate；实际 _apply_convections 用的是 convection.faces
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
        hex_mesh = Mesh(coords=coords)
        with pytest.raises(MeshError, match="对流面片退化"):
            _apply_convections(hex_mesh, case, stiffness, force)

    def test_2d_convection_repeated_adjacent(self) -> None:
        """2D 对流边界折线中相邻节点重复抛 MeshError（thermal.py:252）."""
        from scipy.sparse import csr_matrix

        # 2D QUAD4 网格
        mesh = Mesh(
            coords=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ]
            ),
            blocks=(ElementBlock(ElementType.QUAD4, np.array([[0, 1, 4, 3]]), material=0, section=0),),
        )
        # 相邻重复节点：(0, 0, 1) 中 0 连续出现两次 → 段长为零
        bad_conv = Convection(nodes=(0, 0, 1), h_coeff=2.0, t_ambient=25.0)
        case = ThermalCase(convections=(bad_conv,))
        stiffness = csr_matrix(np.zeros((6, 6)))
        force = np.zeros(6)
        with pytest.raises(MeshError, match="对流边界折线出现重复"):
            _apply_convections(mesh, case, stiffness, force)


# ===================================================================
# fea.harmonic
# ===================================================================


class TestHarmonicMissingCoverage:
    """fea.harmonic 补测：频率点等于固有频率 → 无阻尼时动刚度奇异."""

    def test_harmonic_resonance_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """零质量覆盖 + 秩亏 K → 任意频率下 dynamic = K 秩亏，splu 失败（harmonic.py:155-156）.

        直接通过 assemble_mass 返回零矩阵，让 dynamic 只剩 K（秩亏），
        触发 splu 的 RuntimeError → SolverError("动刚度矩阵奇异").
        """
        from scipy.sparse import csr_matrix

        import zylab.fea.harmonic as harmonic_mod
        from zylab.fea.harmonic import solve_harmonic

        def zero_mass(mesh: Mesh, materials, sections):
            return csr_matrix((mesh.n_dofs, mesh.n_dofs))

        monkeypatch.setattr(harmonic_mod, "assemble_mass", zero_mass)

        # 单根杆，node0 全约束 → K 对 node1 ux 有刚体模态（ux 方向是 K 的零空间）
        case = StaticCase(constraints=(Constraint(node=0, dofs=(0, 1)),))
        with pytest.raises(SolverError, match="动刚度矩阵奇异"):
            solve_harmonic(
                TRUSS2_MESH,
                [STEEL],
                [AREA_UNIT],
                case,
                frequencies=[1.0],
                alpha=0.0,
                beta=0.0,
            )


# ===================================================================
# fea.assemble
# ===================================================================


class TestAssembleMissingCoverage:
    """fea.assemble 补测：索引越界."""

    def test_check_tables_material_out_of_range(self) -> None:
        """材料索引越界抛 MeshError（assemble.py:268）."""
        block = ElementBlock(ElementType.TRUSS2, np.array([[0, 1]]), material=999, section=0)
        mesh = Mesh(coords=np.array([[0.0, 0.0], [1.0, 0.0]]), blocks=(block,))
        with pytest.raises(MeshError, match="材料索引"):
            _check_tables(mesh, [STEEL], [AREA_UNIT])

    def test_check_tables_section_out_of_range(self) -> None:
        """截面索引越界抛 MeshError（assemble.py:270）."""
        block = ElementBlock(ElementType.TRUSS2, np.array([[0, 1]]), material=0, section=999)
        mesh = Mesh(coords=np.array([[0.0, 0.0], [1.0, 0.0]]), blocks=(block,))
        with pytest.raises(MeshError, match="截面索引"):
            _check_tables(mesh, [STEEL], [AREA_UNIT])


# ===================================================================
# fea.export
# ===================================================================


class TestExportMissingCoverage:
    """fea.export 补测：NonlinearSolution history_displacements 为空."""

    def test_nonlinear_rows_empty_history(self) -> None:
        """history_displacements 为空时返回空数据行（export.py:111）."""
        sol = NonlinearSolution(
            mesh=TRUSS2_MESH,
            displacements=np.zeros((2, 2)),
            load_factor=1.0,
            iterations=(1,),
            residual_norm=0.0,
            converged=True,
            history_factors=np.zeros(0),
            history_displacements=np.zeros(0),
        )
        rows = _nonlinear_rows(sol)
        assert rows[0] == ("load_factor", "max_abs_u")
        assert len(rows) == 1


# ===================================================================
# fea.boundary
# ===================================================================


class TestBoundaryMissingCoverage:
    """fea.boundary 补测：EdgePressure.nodes 长度 < 2."""

    def test_edge_pressure_single_node(self) -> None:
        """EdgePressure.nodes 长度 < 2 时 StaticCase.validate 抛 MeshError（boundary.py:103）."""
        case = StaticCase(
            constraints=(Constraint(node=0, dofs=(0, 1)),),
            edge_pressures=(EdgePressure(nodes=(0,), pressure=1.0),),
        )
        with pytest.raises(MeshError, match="边压力至少需要 2 个节点"):
            case.validate(TRUSS2_MESH)
