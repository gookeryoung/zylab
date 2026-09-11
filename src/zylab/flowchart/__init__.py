"""zylab.flowchart - 参数化计算配置化多学科分析流程图内核（Qt-free）.

模块划分：
- :mod:`zylab.flowchart.module`：模块类型系统（端口/参数 schema/内置模块表）；
- :mod:`zylab.flowchart.bundle`：MODEL 端口载荷（模型四要素）；
- :mod:`zylab.flowchart.meshing3d`：三维 HEX8 网格生成器（圆柱电阻 + V 形薄膜电阻）；
- :mod:`zylab.flowchart.nodes`：节点执行函数（源节点建模 + 七类分析节点 + compute/post 计算后处理节点）；
- :mod:`zylab.flowchart.template`：参数化计算定义与 JSON 加载/校验；
- :mod:`zylab.flowchart.registry`：参数化计算注册表（内置 + 用户目录）；
- :mod:`zylab.flowchart.builtin`：内置参数化计算表；
- :mod:`zylab.flowchart.graph`：工作流图（节点状态机 + 级联脏传播 + 拓扑执行计划）；
- :mod:`zylab.flowchart.runner`：编排执行（拓扑序驱动 ProcessExecutor，缓存命中跳过）；
- :mod:`zylab.flowchart.batch`：批处理执行（进程内拓扑序求解 + 参数覆盖/扫描 + 摘要）；
- :mod:`zylab.flowchart.project_io`：工程文件持久化（人类可读 JSON，兼容旧 HDF5）；
- :mod:`zylab.flowchart.dsl`：DSL 参数化计算（YAML 声明式参数化计算解析/校验/参数绑定）；
- :mod:`zylab.flowchart.expressions`：表达式安全求值（AST 白名单 + 受限命名空间）；
- :mod:`zylab.flowchart.results`：DSL 结果视图数据解析（curve/table/text/cloud）；
- :mod:`zylab.flowchart.report`：DSL 报告生成器（Markdown/HTML 双载体，曲线内嵌 SVG）。
"""

from __future__ import annotations

from .batch import NodeOutcome, ReportFn, RunOutcome, resolve_target, run_scan, run_workflow, summarize
from .builtin import BUILTIN_TEMPLATES
from .bundle import ConductionBundle, ModelBundle
from .errors import (
    FlowchartError,
    LinkError,
    ModuleNotFoundError_,
    ParamError,
    TemplateError,
    TemplateNotFoundError,
)
from .graph import NodeInstance, NodeState, WorkflowGraph
from .module import (
    BUILTIN_MODULES,
    ModuleCategory,
    ModuleSpec,
    ParamSpec,
    ParamType,
    PortSpec,
    PortType,
    all_modules,
    module_spec,
)
from .project_io import ProjectIOError, load_workflow, save_workflow
from .registry import TemplateRegistry
from .report import build_html, build_markdown
from .results import CloudData, CurveData, CurveSeries, TableColumn, TableData, TextData, ViewData, build_result
from .runner import NodeRunEvent, WorkflowRunner
from .template import ParamGroup, Template, TemplateNode, load_template, save_template, template_from_json

__all__ = [
    "BUILTIN_MODULES",
    "BUILTIN_TEMPLATES",
    "CloudData",
    "ConductionBundle",
    "CurveData",
    "CurveSeries",
    "FlowchartError",
    "LinkError",
    "ModelBundle",
    "ModuleCategory",
    "ModuleNotFoundError_",
    "ModuleSpec",
    "NodeInstance",
    "NodeOutcome",
    "NodeRunEvent",
    "NodeState",
    "ParamError",
    "ParamGroup",
    "ParamSpec",
    "ParamType",
    "PortSpec",
    "PortType",
    "ProjectIOError",
    "ReportFn",
    "RunOutcome",
    "TableColumn",
    "TableData",
    "Template",
    "TemplateError",
    "TemplateNode",
    "TemplateNotFoundError",
    "TemplateRegistry",
    "TextData",
    "ViewData",
    "WorkflowGraph",
    "WorkflowRunner",
    "all_modules",
    "build_html",
    "build_markdown",
    "build_result",
    "load_template",
    "load_workflow",
    "module_spec",
    "resolve_target",
    "run_scan",
    "run_workflow",
    "save_template",
    "save_workflow",
    "summarize",
    "template_from_json",
]
