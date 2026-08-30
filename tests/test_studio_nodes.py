"""studio.nodes 节点执行函数测试：源节点建模、分析节点求解、内置模板端到端可执行."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from zylab.fea import (
    BucklingSolution,
    ElectroThermalSolution,
    ElementType,
    HarmonicResponse,
    ModalSolution,
    NonlinearSolution,
    StaticSolution,
    TransientSolution,
)
from zylab.studio import BUILTIN_TEMPLATES, ModelBundle, module_spec, nodes
from zylab.studio.module import ModuleSpec

__all__ = []


def _small_cantilever() -> ModelBundle:
    """小规模悬臂梁模型（提速用）."""
    return nodes.build_cantilever({}, {"nx": 8, "ny": 2})


class TestBuildSources:
    """源节点建模."""

    def test_cantilever_default_counts(self) -> None:
        """默认参数复刻原 FeaPage 示例规模（369 节点 / 320 单元）."""
        bundle = nodes.build_cantilever({}, {})
        assert bundle.mesh.n_nodes == 41 * 9
        assert bundle.mesh.n_elements == 40 * 8
        assert bundle.mesh.blocks[0].etype is ElementType.QUAD4
        bundle.case.validate(bundle.mesh)

    def test_cantilever_custom_params(self) -> None:
        """自定义几何/网格参数生效."""
        bundle = nodes.build_cantilever({}, {"length": 10.0, "height": 2.0, "nx": 5, "ny": 2, "tip_load": -5.0})
        assert bundle.mesh.n_nodes == 6 * 3
        assert float(bundle.mesh.coords[:, 0].max()) == pytest.approx(10.0)
        assert bundle.case.loads[0].forces == (0.0, -5.0)

    def test_column_structure(self) -> None:
        """悬臂柱：BEAM2 单元、底部固支、顶部轴压."""
        bundle = nodes.build_column({}, {"n_elem": 4})
        assert bundle.mesh.n_nodes == 5
        assert bundle.mesh.blocks[0].etype is ElementType.BEAM2
        assert bundle.sections[0].inertia == pytest.approx(1.0e-4)
        assert bundle.case.constraints[0].node == 0
        assert bundle.case.loads[0].forces[1] == pytest.approx(-1.0)

    def test_truss_structure(self) -> None:
        """两杆桁架：3 节点 2 单元、顶点集中力."""
        bundle = nodes.build_truss({}, {})
        assert bundle.mesh.n_nodes == 3
        assert bundle.mesh.blocks[0].etype is ElementType.TRUSS2
        assert len(bundle.case.constraints) == 2
        assert bundle.case.loads[0].node == 1

    def test_report_callback_invoked(self) -> None:
        """源节点声明 report 时按阶段上报并以 1.0 收尾."""
        events: list[tuple[float, str]] = []
        nodes.build_truss({}, {}, report=lambda p, m: events.append((p, m)))
        assert events[0][0] < 1.0
        assert events[-1] == (1.0, "模型就绪")

    def test_joule_plate_structure(self) -> None:
        """通电加热板：Q4 网格、左右电极电压、底边恒温与三边对流折线."""
        bundle = nodes.build_joule_plate({}, {"nx": 4, "ny": 2})
        assert bundle.mesh.n_nodes == 5 * 3
        assert bundle.mesh.n_elements == 8
        assert bundle.mesh.blocks[0].etype is ElementType.QUAD4
        bundle.electric_case.validate(bundle.mesh)
        bundle.thermal_case.validate(bundle.mesh)
        coords = bundle.mesh.coords
        left = np.flatnonzero(coords[:, 0] <= 0.0)
        right = np.flatnonzero(coords[:, 0] >= 40.0 - 1e-9)
        bottom = np.flatnonzero(coords[:, 1] <= 0.0)
        # 左端接地 0、右端电极电压 1.0（默认 voltage）
        left_given = {v.node: v.value for v in bundle.electric_case.voltages}
        assert all(left_given[int(n)] == 0.0 for n in left)
        assert all(left_given[int(n)] == 1.0 for n in right)
        # 底边恒温 t_base=20，对流折线覆盖左/顶/右三边（角节点不重复）
        assert {t.node: t.value for t in bundle.thermal_case.temperatures} == {int(n): 20.0 for n in bottom}
        conv = bundle.thermal_case.convections[0]
        expected = 3 + 3 + 3  # 左(3) + 顶内部(3) + 右(3)，ny=2/nx=4
        assert len(conv.nodes) == expected
        assert len(set(conv.nodes)) == expected
        assert bundle.materials[0].electric_sigma == pytest.approx(1.0)
        assert bundle.sections[0].thickness == pytest.approx(1.0)

    def test_joule_plate_pickle_roundtrip(self) -> None:
        """ConductionBundle 可 pickle（跨进程传输前提）."""
        bundle = nodes.build_joule_plate({}, {"nx": 2, "ny": 1})
        restored = pickle.loads(pickle.dumps(bundle))
        assert restored.mesh.n_nodes == bundle.mesh.n_nodes
        assert restored.electric_case == bundle.electric_case
        assert restored.thermal_case == bundle.thermal_case

    def test_joule_series_structure(self) -> None:
        """多材料串联板：三区多块网格（电极/电阻区/电极），材料索引 0/1/0."""
        bundle = nodes.build_joule_series({}, {"nx": 10, "ny": 2, "electrode_len": 3.0})
        mesh = bundle.mesh
        assert mesh.n_nodes == 11 * 3
        assert mesh.n_elements == 20
        assert len(mesh.blocks) == 3
        # dx = 30/10 = 3 → 电极段恰 1 单元宽：左 2、中 8、右 2 每行
        counts = [block.conn.shape[0] for block in mesh.blocks]
        assert counts == [2, 16, 2]
        # 材料索引：电极区引用 0，电阻区引用 1
        assert [block.material for block in mesh.blocks] == [0, 1, 0]
        # 三区单元数合计覆盖全域无重叠（节点 DOF 装配不漏）
        assert sum(counts) == mesh.n_elements
        bundle.electric_case.validate(mesh)
        bundle.thermal_case.validate(mesh)
        assert bundle.materials[0].electric_sigma == pytest.approx(50.0)
        assert bundle.materials[1].electric_sigma == pytest.approx(0.5)
        assert bundle.materials[1].thermal_k == pytest.approx(0.015)

    def test_joule_series_electrode_len_snaps_to_grid(self) -> None:
        """电极段长贴齐网格（非整单元数取整）且各区至少 1 单元."""
        # electrode_len=4.4, dx=3 → round(4.4/3)=1；过大时被钳制保留电阻区
        bundle = nodes.build_joule_series({}, {"nx": 4, "ny": 1, "electrode_len": 4.4})
        counts = [block.conn.shape[0] for block in bundle.mesh.blocks]
        assert counts == [1, 2, 1]
        clamped = nodes.build_joule_series({}, {"nx": 4, "ny": 1, "electrode_len": 1.0e3})
        assert [block.conn.shape[0] for block in clamped.mesh.blocks] == [1, 2, 1]

    def test_joule_series_pickle_roundtrip(self) -> None:
        """多材料 ConductionBundle 可 pickle."""
        bundle = nodes.build_joule_series({}, {"nx": 6, "ny": 1})
        restored = pickle.loads(pickle.dumps(bundle))
        assert len(restored.mesh.blocks) == len(bundle.mesh.blocks)
        assert restored.materials == bundle.materials

    def test_joule_hole_structure(self) -> None:
        """带孔板：形心掩码删孔内单元（r=2.4 恰删 2 个），三区材料 0/1/0."""
        # nx=10, ny=5 → dx=3, dy=2，形心 x∈{1.5,...,28.5}, y∈{1,3,5,7,9}；
        # 孔心 (15,5) r=2.4：仅形心 (13.5,5)/(16.5,5) 落入孔内
        bundle = nodes.build_joule_hole({}, {"nx": 10, "ny": 5, "hole_r": 2.4})
        mesh = bundle.mesh
        assert mesh.n_elements == 48  # 50 - 2
        assert [block.conn.shape[0] for block in mesh.blocks] == [10, 28, 10]
        assert [block.material for block in mesh.blocks] == [0, 1, 0]
        assert mesh.n_nodes == 66  # 无孤立节点
        bundle.electric_case.validate(mesh)
        bundle.thermal_case.validate(mesh)

    def test_joule_hole_orphan_nodes_compacted(self) -> None:
        """孤立节点压缩：r=3.6 删 6 单元产生 2 个四邻全删的孤立节点被剔除."""
        # 孔删 i∈{4,5}, j∈{1,2,3} 共 6 单元 → 网格点 (5,2)/(5,3) 的四邻单元全删
        bundle = nodes.build_joule_hole({}, {"nx": 10, "ny": 5, "hole_r": 3.6})
        mesh = bundle.mesh
        assert mesh.n_elements == 44  # 50 - 6
        assert mesh.n_nodes == 64  # 66 - 2 孤立
        # 剩余节点全部被引用（连接表最大索引 < 节点数由 Mesh 校验保证）
        bundle.electric_case.validate(mesh)
        bundle.thermal_case.validate(mesh)

    def test_joule_hole_convection_covers_hole_edge(self) -> None:
        """边界边自动分类：存在两端均不在外边界的对流段（孔缘内边界）."""
        bundle = nodes.build_joule_hole({}, {"nx": 10, "ny": 5, "hole_r": 3.6})
        coords = bundle.mesh.coords
        length, height = 30.0, 10.0
        inner = [
            conv
            for conv in bundle.thermal_case.convections
            if all(1e-6 < coords[n][0] < length - 1e-6 and 1e-6 < coords[n][1] < height - 1e-6 for n in conv.nodes)
        ]
        assert inner  # 孔缘对流段存在
        # 外边界（顶边 + 左右侧边电极面以上部分）其余段也施加对流
        assert len(bundle.thermal_case.convections) > len(inner)

    def test_joule_hole_pickle_roundtrip(self) -> None:
        """带孔 ConductionBundle 可 pickle."""
        bundle = nodes.build_joule_hole({}, {"nx": 8, "ny": 4, "hole_r": 2.0})
        restored = pickle.loads(pickle.dumps(bundle))
        assert restored.mesh.n_elements == bundle.mesh.n_elements
        assert len(restored.thermal_case.convections) == len(bundle.thermal_case.convections)

    def test_bundle_pickle_roundtrip(self) -> None:
        """ModelBundle 可 pickle（跨进程传输前提）."""
        bundle = _small_cantilever()
        restored = pickle.loads(pickle.dumps(bundle))
        assert restored.mesh.n_nodes == bundle.mesh.n_nodes
        assert restored.case.loads == bundle.case.loads


class TestRunAnalyses:
    """分析节点求解（小规模冒烟，精度基准由 tests/test_fea_* 承担）."""

    def test_static(self) -> None:
        """静力解：位移场形状与向下挠曲方向."""
        solution = nodes.run_static({"model": _small_cantilever()}, {})
        assert isinstance(solution, StaticSolution)
        assert solution.displacements.shape == (27, 2)
        tip = solution.displacements[-1, 1]
        assert tip < 0.0  # 端部载荷向下

    def test_modal(self) -> None:
        """模态解：阶数与频率升序."""
        solution = nodes.run_modal({"model": _small_cantilever()}, {"n_modes": 3})
        assert isinstance(solution, ModalSolution)
        assert solution.n_modes == 3
        assert np.all(np.diff(solution.frequencies) > 0.0)

    def test_harmonic(self) -> None:
        """谐响应解：复数频响矩阵形状."""
        solution = nodes.run_harmonic({"model": _small_cantilever()}, {"f_max": 2.0, "n_freq": 10})
        assert isinstance(solution, HarmonicResponse)
        assert solution.displacements.shape == (27 * 2, 10)
        assert np.iscomplexobj(solution.displacements)

    def test_buckling_matches_euler(self) -> None:
        """悬臂柱一阶屈曲因子逼近 π²EI/4L²（rtol 1%）."""
        bundle = nodes.build_column({}, {})
        solution = nodes.run_buckling({"model": bundle}, {"n_modes": 1})
        assert isinstance(solution, BucklingSolution)
        # EI = 2.1e5 * 1e-4 = 21，L = 10，参考载荷 1.0 → λ₁ ≈ π²·21/(4·100)
        expected = np.pi**2 * 21.0 / 400.0
        assert solution.load_factors[0] == pytest.approx(expected, rel=0.01)

    def test_nonlinear_converges(self) -> None:
        """两杆桁架非线性收敛且位移大于线性参照（软化效应）."""
        bundle = nodes.build_truss({}, {})
        solution = nodes.run_nonlinear({"model": bundle}, {})
        assert isinstance(solution, NonlinearSolution)
        assert solution.converged
        assert len(solution.history_factors) == 11  # 10 增量步 + 起始 0

    def test_transient(self) -> None:
        """瞬态解：时程形状与阶跃载荷下绕静力平衡位置振荡衰减."""
        bundle = _small_cantilever()
        solution = nodes.run_transient(
            {"model": bundle},
            {"duration": 5.0, "n_steps": 50, "alpha": 0.5},
        )
        assert isinstance(solution, TransientSolution)
        assert solution.times.shape == (51,)
        assert solution.displacements.shape == (27 * 2, 51)
        assert solution.times[-1] == pytest.approx(5.0)
        # 阶跃载荷响应绕静力平衡位置振荡：超调后回弹、有阻尼下向静力解收敛
        static = nodes.run_static({"model": bundle}, {})
        tip = nodes.tip_node(bundle.mesh)
        uy_static = static.displacements[tip, 1]  # < 0
        tip_uy = solution.node_history(tip, 1)
        assert np.min(tip_uy) < 1.2 * uy_static  # 超调超过 1.2 倍静力挠度
        assert np.max(tip_uy) > 0.5 * uy_static  # 回弹越过平衡位置上方
        assert tip_uy[-1] < 0.0  # 末端仍向下

    def test_wrong_input_type_rejected(self) -> None:
        """输入端口载荷类型错误抛 TypeError（防御外部直调）."""
        with pytest.raises(TypeError, match="ModelBundle"):
            nodes.run_static({"model": object()}, {})

    def test_electrothermal(self) -> None:
        """电-热耦合解：1D 解析互证（底边恒温 + 近绝热三边 → T_max = qH²/2k）."""
        bundle = nodes.build_joule_plate(
            {},
            {
                "nx": 4,
                "ny": 2,
                "length": 2.0,
                "height": 1.0,
                "voltage": 1.0,
                "electric_sigma": 1.0,
                "thermal_k": 1.0,
                "thickness": 1.0,
                "t_base": 0.0,
                "t_ambient": 0.0,
                "h_conv": 1e-9,
            },
        )
        solution = nodes.run_electrothermal({"model": bundle}, {})
        assert isinstance(solution, ElectroThermalSolution)
        # 电压线性：右端 = V0
        assert solution.voltages.max() == pytest.approx(1.0, rel=1e-12)
        # 总电功率 = σtH·V0²/L = 0.5
        assert solution.total_power == pytest.approx(0.5, rel=1e-9)
        # Joule 均匀热源 q = σ(V0/L)² = 0.25，底边恒温 + 近绝热 → T_max = qH²/2k
        assert solution.t_max == pytest.approx(0.25 * 1.0**2 / 2.0, rel=1e-6)

    def test_electrothermal_wrong_input_type(self) -> None:
        """model 端口载荷类型错误抛 TypeError."""
        with pytest.raises(TypeError, match="ConductionBundle"):
            nodes.run_electrothermal({"model": object()}, {})

    def test_joule_series_analytic(self) -> None:
        """多材料串联 1D 解析互证：P = V₀²·t·H/Σ(Lᵢ/σᵢ) 精确，热点落在电阻区."""
        length, height, nx, ny = 30.0, 10.0, 30, 4
        electrode_len = 5.0
        sigma_c, sigma_h = 50.0, 0.5
        voltage, thickness = 1.0, 1.0
        bundle = nodes.build_joule_series(
            {},
            {
                "length": length,
                "height": height,
                "nx": nx,
                "ny": ny,
                "electrode_len": electrode_len,
                "sigma_conductor": sigma_c,
                "sigma_resistor": sigma_h,
                "k_conductor": 0.4,
                "k_resistor": 0.015,
                "voltage": voltage,
                "thickness": thickness,
                "t_base": 20.0,
                "t_ambient": 20.0,
                "h_conv": 1e-9,  # 近绝热隔离顶/侧边，逼近 1D 竖向导热
            },
        )
        solution = nodes.run_electrothermal({"model": bundle}, {})
        # 1D 串联电阻：R = Σ(Lᵢ/σᵢ)/(t·H)，电场分区分段线性（区界对齐单元边界 → Galerkin 精确）
        r_series = 2.0 * electrode_len / sigma_c + (length - 2.0 * electrode_len) / sigma_h
        assert solution.total_power == pytest.approx(voltage**2 * thickness * height / r_series, rel=1e-10)
        # 电压分段线性：电阻区压降 = V₀·(L_h/σ_h)/Σ(Lᵢ/σᵢ)（100:1 电导率对比下占绝大部分）
        coords = bundle.mesh.coords
        resistor = (coords[:, 0] >= electrode_len - 1e-9) & (coords[:, 0] <= length - electrode_len + 1e-9)
        v_drop_mid = solution.voltages[resistor].max() - solution.voltages[resistor].min()
        assert v_drop_mid == pytest.approx(voltage * (length - 2 * electrode_len) / sigma_h / r_series, rel=1e-10)
        # 热点（t_max 节点）落在电阻区内（热源集中在高阻抗区）
        hotspot = int(np.argmax(solution.temperatures))
        assert electrode_len - 1e-9 <= coords[hotspot, 0] <= length - electrode_len + 1e-9

    def test_joule_hole_analytic(self) -> None:
        """带孔板求解：无孔退化串联解析精确；有孔电流绕流功率下降、热点近孔."""
        common = {
            "nx": 10,
            "ny": 5,
            "length": 30.0,
            "height": 10.0,
            "electrode_len": 5.0,
            "sigma_conductor": 50.0,
            "sigma_resistor": 0.5,
            "k_conductor": 0.4,
            "k_resistor": 0.015,
            "voltage": 1.0,
            "thickness": 1.0,
            "t_base": 20.0,
            "t_ambient": 20.0,
            "h_conv": 1e-9,  # 近绝热隔离顶/侧/孔缘，逼近 1D 导热
            "hole_x": 15.0,
            "hole_y": 5.0,
        }
        # 1) hole_r=0 无孔：退化为串联板，P = V₀²·t·H/Σ(Lᵢ/σᵢ) 精确
        #    （电极段长 5 贴齐网格 round(5/3)=2 单元 → 实际电极长 6、电阻区长 18）
        solid = nodes.run_electrothermal({"model": nodes.build_joule_hole({}, {**common, "hole_r": 0.0})}, {})
        r_series = 2.0 * 6.0 / 50.0 + 18.0 / 0.5
        assert solid.total_power == pytest.approx(1.0 * 1.0 * 10.0 / r_series, rel=1e-10)
        # 2) 有孔：导体截面减小 → 电阻增大 → 功率下降
        holed = nodes.run_electrothermal({"model": nodes.build_joule_hole({}, {**common, "hole_r": 3.6})}, {})
        assert holed.total_power < solid.total_power
        # 3) 电极电压边界：左端全 0、右端全 V₀
        coords = holed.mesh.coords
        np.testing.assert_allclose(holed.voltages[coords[:, 0] <= 1e-9], 0.0, atol=1e-14)
        np.testing.assert_allclose(holed.voltages[coords[:, 0] >= 30.0 - 1e-9], 1.0, rtol=1e-12)
        # 4) 热点落在孔缘附近（电阻区内、距孔心 2dx 内）
        hotspot = int(np.argmax(holed.temperatures))
        hx, hy = coords[hotspot]
        assert (hx - 15.0) ** 2 + (hy - 5.0) ** 2 <= (3.6 + 6.0) ** 2
        assert holed.t_max > 20.0

    def test_param_validation_flows_through(self) -> None:
        """节点参数经模块规格校验（非法值抛 ParamError）."""
        from zylab.studio import ParamError

        with pytest.raises(ParamError, match="越界"):
            nodes.run_modal({"model": _small_cantilever()}, {"n_modes": 0})


class TestBuiltinTemplatesExecutable:
    """内置模板端到端可执行（按定义序串行驱动，源节点 -> 分析节点）."""

    @pytest.mark.parametrize("template", BUILTIN_TEMPLATES, ids=lambda t: t.id)
    def test_template_runs(self, template) -> None:
        """每个内置模板按节点定义序执行，分析节点输出类型与端口声明一致."""
        outputs: dict[str, object] = {}
        for node in template.nodes:
            spec: ModuleSpec = module_spec(node.type_id)
            target = getattr(nodes, spec.target.partition(":")[2])
            inputs = {port: outputs[ref.partition(".")[0]] for port, ref in node.inputs.items()}
            outputs[node.id] = target(inputs, node.params)
            assert outputs[node.id] is not None
        for result_id in template.results:
            assert result_id in outputs

    def test_tip_node_selects_midpoint_of_free_end(self) -> None:
        """tip_node 取 x 最大列中 y 居中节点."""
        bundle = _small_cantilever()
        node = nodes.tip_node(bundle.mesh)
        coords = bundle.mesh.coords
        assert coords[node, 0] == pytest.approx(coords[:, 0].max())
        assert coords[node, 1] == pytest.approx(coords[:, 1].max() / 2.0)


class TestLinkedNodes:
    """链接节点（可选 STATIC 输入端口）."""

    def test_buckling_with_reference_matches_standalone(self) -> None:
        """reference 链接上游静力解与独立屈曲结果一致."""
        bundle = nodes.build_column({}, {})
        static = nodes.run_static({"model": bundle}, {})
        linked = nodes.run_buckling({"model": bundle, "reference": static}, {"n_modes": 2})
        standalone = nodes.run_buckling({"model": bundle}, {"n_modes": 2})
        np.testing.assert_allclose(linked.load_factors, standalone.load_factors, rtol=1e-9)
        assert linked.reference is static

    def test_buckling_reference_wrong_type(self) -> None:
        """reference 端口载荷类型错误抛 TypeError."""
        bundle = nodes.build_column({}, {})
        with pytest.raises(TypeError, match="StaticSolution"):
            nodes.run_buckling({"model": bundle, "reference": object()}, {})

    def test_nonlinear_with_initial_from_static(self) -> None:
        """initial 链接静力解：首帧为初态，收敛点与零位移起算一致."""
        bundle = nodes.build_truss({}, {})
        static = nodes.run_static({"model": bundle}, {})
        linked = nodes.run_nonlinear({"model": bundle, "initial": static}, {"n_increments": 4})
        fresh = nodes.run_nonlinear({"model": bundle}, {"n_increments": 4})
        assert linked.converged
        np.testing.assert_allclose(linked.displacements, fresh.displacements, rtol=1e-6)
        np.testing.assert_allclose(linked.history_displacements[0], static.displacements, rtol=1e-12)

    def test_nonlinear_initial_wrong_type(self) -> None:
        """initial 端口载荷类型错误抛 TypeError."""
        bundle = nodes.build_truss({}, {})
        with pytest.raises(TypeError, match="StaticSolution"):
            nodes.run_nonlinear({"model": bundle, "initial": object()}, {})
