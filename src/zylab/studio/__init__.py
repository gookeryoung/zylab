"""zylab.studio - 模板配置化多学科分析工作台内核（Qt-free）.

模块划分：
- :mod:`zylab.studio.module`：模块类型系统（端口/参数 schema/内置模块表）；
- :mod:`zylab.studio.bundle`：MODEL 端口载荷（模型四要素）；
- :mod:`zylab.studio.meshing3d`：三维 HEX8 网格生成器（圆柱电阻 + V 形薄膜电阻）；
- :mod:`zylab.studio.nodes`：节点执行函数（源节点建模 + 七类分析节点 + compute/post 计算后处理节点）；
- :mod:`zylab.studio.template`：模板定义与 JSON 加载/校验；
- :mod:`zylab.studio.registry`：模板注册表（内置 + 用户目录）；
- :mod:`zylab.studio.builtin`：内置模板表；
- :mod:`zylab.studio.graph`：工作流图（节点状态机 + 级联脏传播 + 拓扑执行计划）；
- :mod:`zylab.studio.runner`：编排执行（拓扑序驱动 ProcessExecutor，缓存命中跳过）；
- :mod:`zylab.studio.batch`：批处理执行（进程内拓扑序求解 + 参数覆盖/扫描 + 摘要）；
- :mod:`zylab.studio.project_io`：工程文件持久化（人类可读 JSON，兼容旧 HDF5）；
- :mod:`zylab.studio.dsl`：DSL 分析模板（YAML 声明式模板解析/校验/参数绑定）；
- :mod:`zylab.studio.expressions`：表达式安全求值（AST 白名单 + 受限命名空间）；
- :mod:`zylab.studio.results`：DSL 结果视图数据解析（curve/table/text/cloud）；
- :mod:`zylab.studio.report`：DSL 报告生成器（Markdown/HTML 双载体，曲线内嵌 SVG）。
"""

from __future__ import annotations

from .batch import NodeOutcome, ReportFn, RunOutcome, resolve_target, run_scan, run_workflow, summarize
from .builtin import BUILTIN_TEMPLATES
from .bundle import ConductionBundle, ModelBundle
from .errors import (
    LinkError,
    ModuleNotFoundError_,
    ParamError,
    StudioError,
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
from .results import CloudData, CurveData, CurveSeries, TableData, TextData, ViewData, build_result
from .runner import NodeRunEvent, WorkflowRunner
from .template import ParamGroup, Template, TemplateNode, load_template, save_template, template_from_json

__all__ = [
    "BUILTIN_MODULES",
    "BUILTIN_TEMPLATES",
    "CloudData",
    "ConductionBundle",
    "CurveData",
    "CurveSeries",
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
    "StudioError",
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
