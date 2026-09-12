"""求解器适配层：fea.solvers.SolverSpec → flowchart.ModuleSpec + 运行时拆包适配器.

fea.solve_* 的物理接口各异（有的要 StaticCase，有的要 ElectricCase+ThermalCase），
runner 的统一调用协议是 ``fn(inputs: Mapping[str, Any], params: Mapping[str, Any])``。
本模块为每个 SolverSpec 生成一个薄适配器，在 runner 与 fea 求解器之间完成：

1. bundle 拆包（ModelBundle.mesh/materials/sections/case → solve_* 的参数序列）；
2. kwargs 组装（SolverSpec.params 的键 → solve_* 的 kwargs）；
3. 可选端口转发（buckling 的 reference / nonlinear 的 initial）；
4. 可选 report 回调透传（solve_* 的最后一个参数）。

生成的适配函数存到模块级注册表 ``RUN_FN_REGISTRY``，target 字符串格式为
``"zylab.flowchart.solver_adapter:_run_<type_id>"``，可被 runner 进程内/进程间
按全限定名导入执行。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .bundle import ConductionBundle, ModelBundle
from .errors import FlowchartError
from .module import ModuleCategory, ModuleSpec, ParamSpec, ParamType, PortSpec, PortType

__all__ = [
    "RUN_FN_REGISTRY",
    "build_all_module_specs",
    "build_module_spec",
    "extract_solver_params",
    "get_run_fn",
    "make_run_fn",
    "param_meta_to_spec",
    "port_meta_to_spec",
]

#: 节点输入表（源节点为空 Mapping；与 nodes.py 的 NodeInputs 语义一致）
NodeInputs = Mapping[str, Any]
#: 节点参数表（键为参数 key，值为 coerce 后的数值）
NodeParams = Mapping[str, Any]


# ------------------------------------------------------------------ 运行时适配器


def _extract_model(model: Any, bundle_kind: str) -> tuple:
    """拆包 bundle 并返回 solve_* 所需的前 3~4 个位置参数.

    :return:
        - ModelBundle → (mesh, materials, sections, case)
        - ConductionBundle → (mesh, materials, sections) — 双工况由调用方组合传入
    """
    if bundle_kind == "ModelBundle":
        if not isinstance(model, ModelBundle):
            raise TypeError(f"模型应为 ModelBundle，得到 {type(model).__name__}")
        return model.mesh, model.materials, model.sections, model.case
    if bundle_kind == "ConductionBundle":
        if not isinstance(model, ConductionBundle):
            raise TypeError(f"模型应为 ConductionBundle，得到 {type(model).__name__}")
        return model.mesh, model.materials, model.sections
    raise FlowchartError(f"未知 bundle 类型: {bundle_kind!r}")


def make_run_fn(spec: Any) -> Callable[[NodeInputs, NodeParams, Any], Any]:
    """由 SolverSpec 生成运行时适配函数.

    调用顺序：
    1. arg_builder（如果有）从 **可变** params 字典中 pop 掉它消费的键，返回位置参数；
    2. make_run_fn 再把 params 里**剩余的键** + extra_inputs + report 当 kwargs 传；
    3. 没有 arg_builder 时回退默认 ModelBundle 拆包，全部 params 键当 kwargs 传。

    这样既不重复（arg_builder 消费的键不会再当 kwargs），也不遗漏（没被消费的
    键如 harmonic 的 alpha/beta、transient 的 duration/n_steps 仍作为 kwargs 传）。
    """
    solve_fn = spec.solve_fn
    extra_port_names = tuple(p.name for p in spec.extra_inputs)
    arg_builder = spec.arg_builder
    renames = dict(spec.kwarg_renames)  # source_key → dest_key

    def run(inputs: NodeInputs, params: NodeParams, report: Any = None) -> Any:
        model = inputs["model"]
        # 可变字典：arg_builder 会 pop 掉它消费的键
        work_params = dict(params)
        work_inputs = dict(inputs)

        extra_kwargs: dict[str, Any] = {}
        if arg_builder is not None:
            result = arg_builder(model, work_params, work_inputs)
            # arg_builder 可返回 tuple 或 (tuple, dict)
            if len(result) == 2 and isinstance(result[1], dict):
                args, extra_kwargs = result
            else:
                args = result
        else:
            if not isinstance(model, ModelBundle):
                raise TypeError(f"模型应为 ModelBundle，得到 {type(model).__name__}")
            args = (model.mesh, model.materials, model.sections, model.case)

        # kwargs：arg_builder 生成的（优先） + 剩余 params 键（经 renames 重映射） + extra_inputs
        kwargs: dict[str, Any] = dict(extra_kwargs)
        for key, val in work_params.items():
            dest = renames.get(key, key)
            kwargs[dest] = val
        for ename in extra_port_names:
            if ename in inputs and inputs[ename] is not None:
                kwargs[ename] = inputs[ename]

        return solve_fn(*args, report=report, **kwargs)

    # 设置稳定的函数名（供 pickle 按 __qualname__ 定位）
    func_name = f"_run_{spec.type_id.replace('.', '_')}"
    run.__name__ = func_name
    run.__qualname__ = func_name
    run.__module__ = __name__
    return run


# ------------------------------------------------------------------ 全局注册表


#: type_id → 适配函数 的全局注册表（目标字符串 "..." : ... 按此查找）
RUN_FN_REGISTRY: dict[str, Callable[..., Any]] = {}

#: type_id → ModuleSpec 的全局注册表（module.py 的 SOLVER_MODULES 从此拷贝）
_SOLVER_MODULES: dict[str, ModuleSpec] = {}


def get_run_fn(type_id: str) -> Callable[..., Any]:
    """按 type_id 取已注册的适配函数；不存在抛 FlowchartError."""
    try:
        return RUN_FN_REGISTRY[type_id]
    except KeyError:
        raise FlowchartError(f"求解器 {type_id!r} 未注册适配函数") from None


# ------------------------------------------------------------------ SolverSpec → ModuleSpec


def param_meta_to_spec(meta: Any) -> ParamSpec:
    """ParamMeta → ParamSpec（flowchart 层的参数规格）."""
    kind = ParamType.INT if meta.kind == "int" else ParamType.STR if meta.kind == "str" else ParamType.FLOAT
    default = meta.default if meta.default is not None else 0
    return ParamSpec(
        key=meta.name,
        label=meta.label,
        param_type=kind,
        default=default,
        minimum=meta.minimum,
        maximum=meta.maximum,
        step=meta.step,
        unit=meta.unit,
        doc=meta.doc,
    )


def port_meta_to_spec(meta: Any) -> PortSpec:
    """PortMeta → PortSpec."""
    port_type = PortType(meta.port_type)
    return PortSpec(
        name=meta.name,
        port_type=port_type,
        label=meta.label,
        required=meta.required,
    )


def build_module_spec(solver: Any) -> ModuleSpec:
    """由 fea.solvers.SolverSpec 构建完整的 flowchart.ModuleSpec.

    核心工作：
    1. 注册适配函数到 RUN_FN_REGISTRY，生成 target 字符串；
    2. 组装输入端口（model + 额外端口）、输出端口、参数 schema。
    """
    run_fn = make_run_fn(solver)
    func_name = run_fn.__name__
    target = f"{__name__}:{func_name}"

    # 注册到全局表（供 runner 的 resolve_target + pickle 查找）
    globals()[func_name] = run_fn
    RUN_FN_REGISTRY[solver.type_id] = run_fn

    # 输入端口：固定 model + bundle 类型 + 可选端口
    bundle_port_type = PortType.MODEL if solver.bundle.value == "ModelBundle" else PortType.ET_MODEL
    inputs = [PortSpec(name="model", port_type=bundle_port_type, label="模型")]
    for extra in solver.extra_inputs:
        inputs.append(port_meta_to_spec(extra))

    # 输出端口
    outputs = [PortSpec(name="solution", port_type=PortType(solver.output_port_type), label=solver.name)]

    # 参数 schema
    params = [param_meta_to_spec(p) for p in solver.params]

    return ModuleSpec(
        type_id=solver.type_id,
        name=solver.name,
        category=ModuleCategory.ANALYSIS,
        target=target,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        params=tuple(params),
    )


def build_all_module_specs() -> dict[str, ModuleSpec]:
    """由 fea.solvers.ALL_SOLVERS 一次性构建全部 ModuleSpec 并缓存到 _SOLVER_MODULES."""
    # 延迟导入避免 flowchart 模块级循环
    from zylab.fea.solvers import ALL_SOLVERS

    if _SOLVER_MODULES:
        return dict(_SOLVER_MODULES)
    for solver in ALL_SOLVERS:
        _SOLVER_MODULES[solver.type_id] = build_module_spec(solver)
    return dict(_SOLVER_MODULES)


def extract_solver_params(solver: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """按 solver.params schema 合并默认值并校验收敛参数（供外部调用）."""
    result: dict[str, Any] = {}
    for p in solver.params:
        result[p.name] = p.default if p.default is not None else 0
    result.update(dict(values))
    return result


# ------------------------------------------------------------------ 模块初始化

build_all_module_specs()
