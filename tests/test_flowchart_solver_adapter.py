"""fea.solvers + flowchart.solver_adapter 单元测试."""

from __future__ import annotations

import pytest

from zylab.fea.solvers import (
    ALL_SOLVERS,
    ET_SOLVERS,
    FE_SOLVERS,
    BundleKind,
    ParamMeta,
    PortMeta,
    SolverCategory,
    build_solver_target,
    module_id_of,
)
from zylab.flowchart.module import PortType, module_spec
from zylab.flowchart.solver_adapter import (
    RUN_FN_REGISTRY,
    build_all_module_specs,
    build_module_spec,
    extract_solver_params,
    get_run_fn,
    make_run_fn,
    param_meta_to_spec,
    port_meta_to_spec,
)


class TestSolverSpec:
    """SolverSpec 元数据完整性检查."""

    def test_all_solvers_registered(self) -> None:
        """ALL_SOLVERS 是 FE + ET 求解器的并集."""
        assert set(ALL_SOLVERS) == {*FE_SOLVERS, *ET_SOLVERS}
        assert len(FE_SOLVERS) == 6
        assert len(ET_SOLVERS) == 2
        assert len(ALL_SOLVERS) == 8

    def test_each_solver_has_required_fields(self) -> None:
        """每个 SolverSpec 的 type_id/solve_fn/result_cls 都是非空有效引用."""
        for solver in ALL_SOLVERS:
            assert solver.type_id  # 非空
            assert callable(solver.solve_fn)
            assert isinstance(solver.result_cls, type)
            assert solver.bundle in (BundleKind.MODEL, BundleKind.ET_MODEL)

    def test_unique_type_ids(self) -> None:
        """所有 type_id 不重复."""
        ids = [s.type_id for s in ALL_SOLVERS]
        assert len(ids) == len(set(ids))

    def test_categories_are_valid(self) -> None:
        """category 是合法枚举."""
        for solver in ALL_SOLVERS:
            assert isinstance(solver.category, SolverCategory)

    def test_param_meta_to_spec(self) -> None:
        """ParamMeta → ParamSpec 类型转换."""
        meta = ParamMeta("n_modes", "模态阶数", "int", default=6, minimum=1, maximum=50, step=1)
        spec = param_meta_to_spec(meta)
        assert spec.key == "n_modes"
        assert spec.default == 6

    def test_port_meta_to_spec(self) -> None:
        """PortMeta → PortSpec 类型转换."""
        meta = PortMeta("reference", "static", "参考静力", required=False)
        spec = port_meta_to_spec(meta)
        assert spec.name == "reference"
        assert spec.port_type is PortType.STATIC
        assert not spec.required

    def test_build_solver_target(self) -> None:
        """build_solver_target 生成可导入的 target 字符串."""
        spec = FE_SOLVERS[0]  # fea.static
        target = build_solver_target(spec.solve_fn)
        assert target == "zylab.fea.static:solve_static"

    def test_module_id_of(self) -> None:
        """module_id_of 返回 type_id."""
        spec = FE_SOLVERS[0]
        assert module_id_of(spec) == spec.type_id


class TestSolverAdapter:
    """solver_adapter 元数据到 ModuleSpec 转换."""

    def test_build_module_spec_static(self) -> None:
        """fea.static → ModuleSpec: MODEL 输入 + STATIC 输出 + 无额外参数."""
        spec = FE_SOLVERS[0]  # fea.static
        ms = build_module_spec(spec)
        assert ms.type_id == "fea.static"
        assert ms.category.value == "analysis"
        assert ms.inputs[0].port_type is PortType.MODEL
        assert ms.outputs[0].port_type is PortType.STATIC
        assert "solver_adapter" in ms.target

    def test_build_module_spec_modal_has_params(self) -> None:
        """fea.modal 有 n_modes 参数."""
        modal = next(s for s in ALL_SOLVERS if s.type_id == "fea.modal")
        ms = build_module_spec(modal)
        assert len(ms.params) == 1
        assert ms.params[0].key == "n_modes"

    def test_buckling_has_extra_input(self) -> None:
        """fea.buckling 有 reference 可选输入端口."""
        buckling = next(s for s in ALL_SOLVERS if s.type_id == "fea.buckling")
        ms = build_module_spec(buckling)
        port_names = [p.name for p in ms.inputs]
        assert "model" in port_names
        assert "reference" in port_names
        ref_port = next(p for p in ms.inputs if p.name == "reference")
        assert not ref_port.required

    def test_solver_modules_registered(self) -> None:
        """build_all_module_specs 把 8 个求解器都注册到 _MODULES_BY_ID."""
        specs = build_all_module_specs()
        assert len(specs) == 8
        for solver in ALL_SOLVERS:
            assert solver.type_id in specs

    def test_run_fn_registry_filled(self) -> None:
        """RUN_FN_REGISTRY 在 build_all_module_specs 后有 8 个条目."""
        build_all_module_specs()
        assert len(RUN_FN_REGISTRY) == 8
        for solver in ALL_SOLVERS:
            assert solver.type_id in RUN_FN_REGISTRY

    def test_module_spec_lookup(self) -> None:
        """module_spec 能查到新的 fea.static 模块."""
        ms = module_spec("fea.static")
        assert ms.target is not None

    def test_make_run_fn_produces_callable(self) -> None:
        """make_run_fn 返回可调用对象."""
        static = FE_SOLVERS[0]
        fn = make_run_fn(static)
        assert callable(fn)
        assert fn.__name__ == "_run_fea_static"

    def test_get_run_fn(self) -> None:
        """get_run_fn 从注册表取适配函数."""
        fn = get_run_fn("fea.static")
        assert callable(fn)

    def test_get_run_fn_unknown_raises(self) -> None:
        """未知 type_id 抛 FlowchartError."""
        from zylab.flowchart.errors import FlowchartError

        with pytest.raises(FlowchartError):
            get_run_fn("no.such.solver")

    def test_extract_solver_params(self) -> None:
        """extract_solver_params 合并默认值."""
        modal = next(s for s in ALL_SOLVERS if s.type_id == "fea.modal")
        params = extract_solver_params(modal, {"n_modes": 3})
        assert params["n_modes"] == 3  # 用户覆盖
        # 其他默认不存在（因为 modal 只有一个参数）

    def test_module_target_resolvable(self) -> None:
        """所有 SOLVER_MODULES 的 target 都能被 resolve_target 解析."""
        from zylab.flowchart.batch import resolve_target

        for solver in ALL_SOLVERS:
            ms = build_module_spec(solver)
            fn = resolve_target(ms.target)
            assert callable(fn)


class TestSolverIntegration:
    """端到端：用新的 fea.* 模块 id 创建 Template 并运行."""

    def test_fea_static_worfklow(self) -> None:
        """fea.static 端到端成功."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template

        t = Template.from_dict(
            {
                "id": "test.fea.static",
                "name": "静力测试",
                "nodes": [
                    {"id": "geo", "type": "example.cantilever_q4", "params": {}},
                    {"id": "solver", "type": "fea.static", "inputs": {"model": "geo.model"}},
                ],
            }
        )
        outcome = run_workflow(t)
        assert outcome.succeeded
        assert outcome.outcome("solver").result is not None

    def test_fea_modal_workflow(self) -> None:
        """fea.modal 端到端成功（constraints 特殊拆包）."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template

        t = Template.from_dict(
            {
                "id": "test.fea.modal",
                "name": "模态测试",
                "nodes": [
                    {"id": "geo", "type": "example.cantilever_q4", "params": {}},
                    {"id": "solver", "type": "fea.modal", "inputs": {"model": "geo.model"}, "params": {"n_modes": 4}},
                ],
            }
        )
        outcome = run_workflow(t)
        assert outcome.succeeded

    def test_fea_harmonic_workflow(self) -> None:
        """fea.harmonic 端到端成功（frequencies 特殊生成 + arg_builder pop）."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template

        t = Template.from_dict(
            {
                "id": "test.fea.harmonic",
                "name": "谐响应测试",
                "nodes": [
                    {"id": "geo", "type": "example.cantilever_q4", "params": {}},
                    {
                        "id": "solver",
                        "type": "fea.harmonic",
                        "inputs": {"model": "geo.model"},
                        "params": {"f_max": 2.0, "n_freq": 10, "alpha": 0.1},
                    },
                ],
            }
        )
        outcome = run_workflow(t)
        assert outcome.succeeded

    def test_et_electrothermal_workflow(self) -> None:
        """fea.electrothermal 双 case 拆包成功."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template

        t = Template.from_dict(
            {
                "id": "test.et",
                "name": "电热耦合测试",
                "nodes": [
                    {"id": "geo", "type": "example.joule_plate_2d", "params": {}},
                    {"id": "solver", "type": "fea.electrothermal", "inputs": {"model": "geo.model"}},
                ],
            }
        )
        outcome = run_workflow(t)
        assert outcome.succeeded

    def test_old_analysis_types_still_work(self) -> None:
        """旧 analysis.* 类型 id 向后兼容."""
        from zylab.flowchart.batch import run_workflow
        from zylab.flowchart.template import Template

        t = Template.from_dict(
            {
                "id": "test.old.compat",
                "name": "旧类型兼容",
                "nodes": [
                    {"id": "geo", "type": "example.cantilever_q4", "params": {}},
                    {"id": "solver", "type": "analysis.static", "inputs": {"model": "geo.model"}},
                ],
            }
        )
        outcome = run_workflow(t)
        assert outcome.succeeded

    def test_et_transient_arg_builder_tuple_and_dict(self) -> None:
        from zylab.fea.solvers import _et_electrothermal_transient_args

        class _M:
            n_nodes = 100

        class _B:
            mesh = _M()
            materials = ()
            sections = ()
            electric_case = object()
            thermal_case = object()

        p = {"t_init": 30.0, "duration": 2.0, "n_steps": 20, "extra": "keep"}
        result = _et_electrothermal_transient_args(_B(), p, {})
        args, kwargs = result
        assert len(args) == 5
        assert len(kwargs["initial"]) == 100
        assert kwargs["total_time"] == 2.0
        assert kwargs["n_steps"] == 20
        assert p == {"extra": "keep"}

    def test_make_run_fn_renames_and_extra(self) -> None:
        from zylab.fea.solvers import BundleKind, PortMeta, SolverSpec
        from zylab.flowchart.module import ModuleCategory
        from zylab.flowchart.solver_adapter import make_run_fn

        cap = {}

        def fake(*a, **kw):
            cap.update(args=a, kwargs=kw)
            return "ok"

        spec = SolverSpec(
            type_id="test.rn",
            name="T",
            category=ModuleCategory.ANALYSIS,
            solve_fn=fake,
            bundle=BundleKind.MODEL,
            output_port_type="t",
            result_cls=object,
            arg_builder=lambda _b, _p, _e: (),
            kwarg_renames=(("alpha", "damping_alpha"),),
            extra_inputs=(PortMeta("ref", "static", "ref"),),
        )
        fn = make_run_fn(spec)

        class _M:
            n_nodes = 1

        class _B:
            mesh = _M()
            materials = ()
            sections = ()
            case = object()

        fn({"model": _B(), "ref": "RV"}, {"alpha": 0.5, "beta": 0.1}, report=None)
        assert cap["kwargs"]["damping_alpha"] == 0.5
        assert cap["kwargs"]["beta"] == 0.1
        assert cap["kwargs"]["ref"] == "RV"

    def test_extract_model_legacy_function(self) -> None:
        from zylab.flowchart.bundle import ConductionBundle, ModelBundle
        from zylab.flowchart.errors import FlowchartError
        from zylab.flowchart.solver_adapter import _extract_model

        mb = ModelBundle(object(), (), (), object())
        args = _extract_model(mb, "ModelBundle")
        assert len(args) == 4

        cb = ConductionBundle(object(), (), (), object(), object())
        args = _extract_model(cb, "ConductionBundle")
        assert len(args) == 3

        with pytest.raises(TypeError):
            _extract_model("nope", "ModelBundle")
        with pytest.raises(TypeError):
            _extract_model("nope", "ConductionBundle")
        with pytest.raises(FlowchartError):
            _extract_model(mb, "NoSuch")

    def test_build_solver_target_missing_module(self) -> None:
        from zylab.fea.solvers import build_solver_target

        # 普通函数没有 __module__ 也能处理
        def bare():
            pass

        target = build_solver_target(bare)
        assert "bare" in target

    def test_solver_spec_target_property(self) -> None:
        spec = FE_SOLVERS[0]
        assert spec.target.endswith(":solve_static")
        assert "zylab.fea.static" in spec.target

    def test_make_run_fn_no_arg_builder_default_extract(self) -> None:
        from zylab.fea.solvers import BundleKind, SolverSpec
        from zylab.flowchart.bundle import ModelBundle
        from zylab.flowchart.module import ModuleCategory
        from zylab.flowchart.solver_adapter import make_run_fn

        cap = {}

        def fake(*a, **kw):
            cap.update(args=a, kwargs=kw)
            return "ok"

        spec = SolverSpec(
            type_id="test.noab",
            name="T",
            category=ModuleCategory.ANALYSIS,
            solve_fn=fake,
            bundle=BundleKind.MODEL,
            output_port_type="t",
            result_cls=object,
        )
        fn = make_run_fn(spec)
        mb = ModelBundle(object(), (), (), object())
        fn({"model": mb}, {"duration": 1.0}, report=None)
        assert len(cap["args"]) == 4
        assert cap["kwargs"]["duration"] == 1.0

    def test_build_solver_target_external_plugin_path(self) -> None:
        from zylab.fea.solvers import build_solver_target

        def fake_external():
            pass

        fake_external.__module__ = "third_party.solver.plugin"
        target = build_solver_target(fake_external)
        assert target == "third_party.solver.plugin:fake_external"

    def test_arg_builder_tuple_and_dict_form_via_make_run_fn(self) -> None:
        """覆盖 L90：arg_builder 返回 (args, extra_kwargs) 形式."""
        from zylab.fea.solvers import BundleKind, SolverSpec
        from zylab.flowchart.module import ModuleCategory
        from zylab.flowchart.solver_adapter import make_run_fn

        cap = {}

        def fake(*a, **kw):
            cap.update(args=a, kwargs=kw)
            return "ok"

        def ab(_b, _p, _e):
            return ("mesh_v",), {"extra_kw": "from_arg_builder"}

        spec = SolverSpec(
            type_id="test.ab2",
            name="T",
            category=ModuleCategory.ANALYSIS,
            solve_fn=fake,
            bundle=BundleKind.MODEL,
            output_port_type="t",
            result_cls=object,
            arg_builder=ab,
        )
        fn = make_run_fn(spec)
        fn({"model": object()}, {"duration": 1.0}, report=None)
        assert cap["args"] == ("mesh_v",)
        assert cap["kwargs"]["extra_kw"] == "from_arg_builder"

    def test_no_arg_builder_wrong_type_raises(self) -> None:
        """覆盖 L95：无 arg_builder 时 model 非 ModelBundle → TypeError."""
        from zylab.fea.solvers import BundleKind, SolverSpec
        from zylab.flowchart.module import ModuleCategory
        from zylab.flowchart.solver_adapter import make_run_fn

        spec = SolverSpec(
            type_id="test.wt",
            name="T",
            category=ModuleCategory.ANALYSIS,
            solve_fn=lambda *_a, **_kw: None,
            bundle=BundleKind.MODEL,
            output_port_type="t",
            result_cls=object,
        )
        fn = make_run_fn(spec)
        with pytest.raises(TypeError, match="应为 ModelBundle"):
            fn({"model": "not_a_bundle"}, {}, report=None)

    def test_solver_category_enum_values(self) -> None:
        from zylab.fea.solvers import SolverCategory

        assert SolverCategory.STATIC.value == "static"
        assert SolverCategory.DYNAMIC.value == "dynamic"
        assert SolverCategory.COUPLING.value == "coupling"

    def test_buckling_arg_builder(self) -> None:
        from zylab.fea.solvers import ALL_SOLVERS
        from zylab.flowchart.bundle import ModelBundle

        buckling = next(s for s in ALL_SOLVERS if s.type_id == "fea.buckling")
        assert buckling.arg_builder is not None

        class _FakeCase:
            constraints = ()

        mb = ModelBundle(object(), (), (), _FakeCase())
        params = {"n_modes": 3}
        # buckling 需要 reference case 作为 extra_input
        result = buckling.arg_builder(mb, params, {"reference": _FakeCase()})
        assert len(result) == 4
        assert "reference" not in params  # 被消费了
