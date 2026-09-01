"""studio.module 模块类型系统测试：ParamSpec 收敛、ModuleSpec 查询、内置模块表完整性."""

from __future__ import annotations

import pytest

from zylab.studio import (
    BUILTIN_MODULES,
    ModuleCategory,
    ModuleNotFoundError_,
    ParamError,
    ParamSpec,
    ParamType,
    PortType,
    all_modules,
    module_spec,
)

__all__ = []


class TestParamSpecCoerce:
    """参数值收敛（类型 + 范围校验）."""

    def test_float_accepts_int_and_float(self) -> None:
        """浮点参数接受 int/float 并统一转 float."""
        spec = ParamSpec("e_modulus", "弹性模量", ParamType.FLOAT, 2.1e5, 1.0e3, 1.0e12)
        assert spec.coerce(2.1e5) == 2.1e5
        assert spec.coerce(210000) == 210000.0

    def test_int_accepts_integral_float(self) -> None:
        """整数参数接受整值浮点（JSON 数值常以 float 载入）."""
        spec = ParamSpec("nx", "单元数", ParamType.INT, 40, 1, 400)
        assert spec.coerce(20.0) == 20
        assert isinstance(spec.coerce(20.0), int)

    def test_int_rejects_non_integral_float(self) -> None:
        """整数参数拒绝非整值浮点."""
        spec = ParamSpec("nx", "单元数", ParamType.INT, 40, 1, 400)
        with pytest.raises(ParamError, match="应为整数"):
            spec.coerce(2.5)

    @pytest.mark.parametrize("value", [True, "abc", None], ids=["bool", "str", "none"])
    def test_rejects_bad_types(self, value: object) -> None:
        """两种参数类型均拒绝布尔/字符串/None."""
        for param_type in (ParamType.FLOAT, ParamType.INT):
            spec = ParamSpec("k", "键", param_type, 1)
            with pytest.raises(ParamError, match="应为"):
                spec.coerce(value)

    def test_range_violation(self) -> None:
        """越界取值抛 ParamError（含边界允许）."""
        spec = ParamSpec("poisson", "泊松比", ParamType.FLOAT, 0.3, 0.0, 0.49)
        assert spec.coerce(0.0) == 0.0
        assert spec.coerce(0.49) == 0.49
        with pytest.raises(ParamError, match="越界"):
            spec.coerce(0.5)


class TestModuleSpec:
    """模块规格查询与参数合并."""

    def test_param_unknown_key(self) -> None:
        """未知参数键抛 ParamError."""
        spec = module_spec("analysis.modal")
        with pytest.raises(ParamError, match="无参数"):
            spec.param("nope")

    def test_defaults(self) -> None:
        """默认值表覆盖全部参数."""
        spec = module_spec("analysis.harmonic")
        assert spec.defaults() == {"f_max": 3.0, "n_freq": 60, "alpha": 0.1, "beta": 0.0}

    def test_coerce_params_merges_defaults(self) -> None:
        """coerce_params 合并默认值并转换覆盖值."""
        spec = module_spec("analysis.modal")
        assert spec.coerce_params({}) == {"n_modes": 6}
        assert spec.coerce_params({"n_modes": 10.0}) == {"n_modes": 10}

    def test_coerce_params_rejects_unknown_key(self) -> None:
        """coerce_params 拒绝未知键."""
        spec = module_spec("analysis.static")
        with pytest.raises(ParamError, match="无参数"):
            spec.coerce_params({"ghost": 1.0})

    def test_port_lookup(self) -> None:
        """端口按名查询；缺失端口抛 ModuleNotFoundError_."""
        spec = module_spec("analysis.static")
        assert spec.input_port("model").port_type is PortType.MODEL
        assert spec.output_port("solution").port_type is PortType.STATIC
        with pytest.raises(ModuleNotFoundError_, match="无输入端口"):
            spec.input_port("ghost")
        with pytest.raises(ModuleNotFoundError_, match="无输出端口"):
            spec.output_port("ghost")


class TestBuiltinModules:
    """内置模块表完整性（schema 自身合法性）."""

    def test_type_id_unique(self) -> None:
        """类型 id 全局唯一."""
        ids = [spec.type_id for spec in BUILTIN_MODULES]
        assert len(ids) == len(set(ids))

    def test_targets_point_to_nodes_module(self) -> None:
        """执行目标均为 zylab.studio.nodes 下的函数（spawn 子进程可定位）."""
        for spec in BUILTIN_MODULES:
            assert spec.target.startswith("zylab.studio.nodes:"), spec.type_id

    def test_source_has_no_input_and_expected_output(self) -> None:
        """源模块无输入端口，输出 MODEL/ET_MODEL（CAE）或 DATA（可靠性）."""
        expected_outputs = {
            "model": {PortType.MODEL, PortType.ET_MODEL},
            "data": {PortType.DATA},
        }
        for spec in BUILTIN_MODULES:
            if spec.category is ModuleCategory.SOURCE:
                assert spec.inputs == ()
                assert len(spec.outputs) == 1
                (port,) = spec.outputs
                assert port.name in expected_outputs, spec.type_id
                assert port.port_type in expected_outputs[port.name], spec.type_id

    def test_analysis_takes_model_input(self) -> None:
        """分析模块输入 MODEL/ET_MODEL、输出对应解类型."""
        expected = {
            "analysis.static": PortType.STATIC,
            "analysis.modal": PortType.MODAL,
            "analysis.harmonic": PortType.HARMONIC,
            "analysis.transient": PortType.TRANSIENT,
            "analysis.buckling": PortType.BUCKLING,
            "analysis.nonlinear": PortType.NONLINEAR,
            "analysis.electrothermal": PortType.ELECTROTHERMAL,
        }
        model_ports = {PortType.MODEL, PortType.ET_MODEL}
        for type_id, port_type in expected.items():
            spec = module_spec(type_id)
            assert spec.category is ModuleCategory.ANALYSIS
            assert spec.input_port("model").port_type in model_ports
            assert spec.output_port("solution").port_type is port_type

    def test_electrothermal_ports(self) -> None:
        """电-热耦合模块端口契约：ET_MODEL 输入、ELECTROTHERMAL 输出."""
        source = module_spec("example.joule_plate_2d")
        assert source.output_port("model").port_type is PortType.ET_MODEL
        analysis = module_spec("analysis.electrothermal")
        assert analysis.input_port("model").port_type is PortType.ET_MODEL
        assert analysis.output_port("solution").port_type is PortType.ELECTROTHERMAL

    def test_optional_link_ports(self) -> None:
        """屈曲/非线性声明可选 STATIC 链接端口（reference/initial），model 必填."""
        buckling = module_spec("analysis.buckling")
        assert buckling.input_port("model").required
        reference = buckling.input_port("reference")
        assert not reference.required
        assert reference.port_type is PortType.STATIC
        nonlinear = module_spec("analysis.nonlinear")
        initial = nonlinear.input_port("initial")
        assert not initial.required
        assert initial.port_type is PortType.STATIC

    def test_param_defaults_within_range(self) -> None:
        """全部数值参数默认值落在自身范围内（STR/MAP 结构参数无范围语义）."""
        for spec in all_modules():
            for param in spec.params:
                if param.param_type not in (ParamType.FLOAT, ParamType.INT):
                    continue
                assert param.minimum <= float(param.default) <= param.maximum, f"{spec.type_id}.{param.key}"

    def test_module_spec_unknown_raises(self) -> None:
        """未知类型 id 抛 ModuleNotFoundError_."""
        with pytest.raises(ModuleNotFoundError_, match="未知模块类型"):
            module_spec("no.such.module")

    def test_all_modules_returns_builtin(self) -> None:
        """all_modules 返回内置表."""
        assert all_modules() is BUILTIN_MODULES
