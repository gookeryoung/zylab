"""zylab.flowchart 异常类型."""

from __future__ import annotations

__all__ = [
    "FlowchartError",
    "LinkError",
    "ModuleNotFoundError_",
    "ParamError",
    "TemplateError",
    "TemplateNotFoundError",
]


class FlowchartError(Exception):
    """flowchart 包异常基类."""


class ModuleNotFoundError_(FlowchartError):
    """模块类型 id 未注册（命名带下划线后缀以规避与内建 ModuleNotFoundError 混淆）."""


class ParamError(FlowchartError):
    """参数值缺失、类型不符或越界."""


class TemplateError(FlowchartError):
    """参数化计算定义非法（结构/引用/连接/环）或参数化计算文件解析失败."""


class TemplateNotFoundError(FlowchartError):
    """参数化计算 id 未注册."""


class LinkError(FlowchartError):
    """节点连接非法（端口不存在/类型不匹配/成环）."""
