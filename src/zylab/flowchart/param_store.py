"""参数中心化存储：输入/输出参数统一命名空间，支持声明式输出提取.

Workbench 风格的「参数是一等公民」基础设施——输入参数按 ``"node_id.param_key"``
扁平命名空间，输出参数由声明式 :class:`~zylab.flowchart.template.OutputParam`
定义来源节点 + 提取路径 + 可选变换表达式，运行后统一解析为
``"output_name" -> value`` 可供后续设计点、扫参、DOE/优化模块消费。

与既有 :mod:`results` 的关系：``results`` 是视图层（用户看什么），
``param_store`` 是计算层（参数空间维度，供自动化探索消费）。
两者正交互不重叠——一个输出参数可以同时被 results 引用做展示、被 design_space
引用做优化目标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import FlowchartError, ParamError, TemplateError
from .expressions import ARRAY_MATH_NAMESPACE, safe_eval
from .results import resolve_path
from .template import OutputParam

__all__ = ["OutputParam", "ParameterStore"]


@dataclass
class ParameterStore:
    """统一参数存储（输入 + 输出声明 + 运行后解析结果）.

    典型使用流程::

        store = ParameterStore.from_template(template)
        outcome = run_workflow(template)
        outputs = {o.node_id: o.result for o in outcome.outcomes if o.ok}
        resolved = store.resolve_all(outputs)
        # resolved["max_displacement"] = 0.123
    """

    #: 输入参数扁平表（``"node_id.param_key" -> 值``）
    inputs: dict[str, Any] = field(default_factory=dict)
    #: 输出参数声明表（``"output_name" -> OutputParam``）
    outputs: dict[str, OutputParam] = field(default_factory=dict)
    #: 运行后解析结果（``"output_name" -> 值``），由 :meth:`resolve_all` 填充
    resolved: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 构造

    @classmethod
    def from_template(
        cls,
        template: Any,
        output_params: Mapping[str, OutputParam] | None = None,
    ) -> ParameterStore:
        """由参数化计算构造——inputs 从 Template 节点 params 扁平化，
        outputs 来自参数化计算声明或显式传入.

        :param template: :class:`~zylab.flowchart.template.Template` 或
            :class:`~zylab.flowchart.graph.WorkflowGraph` 实例
            （带 ``nodes`` 属性且节点有 ``id``/``params``）。
        :param output_params: 显式 OutputParam 声明覆盖参数化计算内建声明；
            Template 无 output_params 时可通过此参数注入。
        """
        store = cls()
        store._collect_inputs(template)
        if output_params is not None:
            store.outputs.update(output_params)
        else:
            template_outs = getattr(template, "output_params", None)
            if template_outs:
                for op in template_outs:
                    store.outputs[op.name] = op
        return store

    def _collect_inputs(self, template: Any) -> None:
        """从 Template/Graph 的节点 params 扁平化收集输入参数."""
        for node in template.nodes:
            node_id = node.id
            for key, value in node.params.items():
                self.inputs[f"{node_id}.{key}"] = value

    # ------------------------------------------------------------------ 查询

    def get_input(self, ref: str) -> Any:
        """按 ``"node_id.param_key"`` 取输入参数值；不存在抛 :class:`FlowchartError`."""
        if ref not in self.inputs:
            raise FlowchartError(f"ParameterStore 无输入参数 {ref!r}")
        return self.inputs[ref]

    def get_output_decl(self, name: str) -> OutputParam:
        """按名取输出参数声明；不存在抛 :class:`FlowchartError`."""
        if name not in self.outputs:
            raise FlowchartError(f"ParameterStore 无输出参数 {name!r}")
        return self.outputs[name]

    def get_resolved(self, name: str) -> Any:
        """取已解析的输出参数值；未解析或不存在抛 :class:`FlowchartError`."""
        if name not in self.resolved:
            raise FlowchartError(f"ParameterStore 输出参数 {name!r} 尚未解析")
        return self.resolved[name]

    # ------------------------------------------------------------------ 解析

    def resolve_all(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        """运行后统一解析全部输出参数.

        :param outputs: 节点 id -> 输出载荷（``RunOutcome.outcomes`` 里
            成功节点的 ``result``）。
        :return: 解析结果表（与 :attr:`resolved` 同引用）。
        :raises FlowchartError: source 节点无输出 / 路径不存在 /
            expr 求值失败。
        """
        resolved: dict[str, Any] = {}
        for name, decl in self.outputs.items():
            resolved[name] = self._resolve_one(decl, outputs)
        self.resolved = resolved
        return resolved

    def _resolve_one(self, decl: OutputParam, outputs: Mapping[str, Any]) -> Any:
        """解析单个输出参数：source 路径取值 + 可选 expr 变换."""
        try:
            value = resolve_path(decl.source, outputs)
        except TemplateError as exc:
            raise FlowchartError(f"输出参数 {decl.name!r} source 解析失败: {exc}") from exc
        if decl.expr:
            # 构造 expr 命名空间：
            # - ARRAY_MATH_NAMESPACE：numpy 数组版数学函数（支持逐元素）
            # - value：source 原值
            # - 输入参数下划线别名："node_id.param" → "node_id_param"
            #   （Python Attribute 白名单禁属性访问，不能在表达式里写带点键名）
            aliased_inputs = {k.replace(".", "_"): v for k, v in self.inputs.items()}
            namespace = {**ARRAY_MATH_NAMESPACE, "value": value, **self.inputs, **aliased_inputs}
            try:
                value = safe_eval(decl.expr, namespace)
            except (ParamError, NameError, ZeroDivisionError) as exc:
                raise FlowchartError(f"输出参数 {decl.name!r} 表达式求值失败: {exc}") from exc
        return value
