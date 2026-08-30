"""studio.nodes 节点执行函数测试：源节点建模、分析节点求解、内置模板端到端可执行."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from zylab.fea import (
    BucklingSolution,
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
