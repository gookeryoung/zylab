"""zylab.studio - 模板配置化多学科分析工作台内核（Qt-free）.

模块划分：
- :mod:`zylab.studio.module`：模块类型系统（端口/参数 schema/内置模块表）；
- :mod:`zylab.studio.bundle`：MODEL 端口载荷（模型四要素）；
- :mod:`zylab.studio.nodes`：节点执行函数（源节点建模 + 六类分析节点）；
- :mod:`zylab.studio.template`：模板定义与 JSON 加载/校验；
- :mod:`zylab.studio.registry`：模板注册表（内置 + 用户目录）；
- :mod:`zylab.studio.builtin`：内置模板表；
- :mod:`zylab.studio.graph`：工作流图（节点状态机 + 级联脏传播 + 拓扑执行计划）；
- :mod:`zylab.studio.runner`：编排执行（拓扑序驱动 ProcessExecutor，缓存命中跳过）。
"""

from __future__ import annotations

from .builtin import BUILTIN_TEMPLATES
from .bundle import ModelBundle
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
from .registry import TemplateRegistry
from .runner import NodeRunEvent, WorkflowRunner
from .template import ParamGroup, Template, TemplateNode, load_template, save_template, template_from_json

__all__ = [
    "BUILTIN_MODULES",
    "BUILTIN_TEMPLATES",
    "LinkError",
    "ModelBundle",
    "ModuleCategory",
    "ModuleNotFoundError_",
    "ModuleSpec",
    "NodeInstance",
    "NodeRunEvent",
    "NodeState",
    "ParamError",
    "ParamGroup",
    "ParamSpec",
    "ParamType",
    "PortSpec",
    "PortType",
    "StudioError",
    "Template",
    "TemplateError",
    "TemplateNode",
    "TemplateNotFoundError",
    "TemplateRegistry",
    "WorkflowGraph",
    "WorkflowRunner",
    "all_modules",
    "load_template",
    "module_spec",
    "save_template",
    "template_from_json",
]
