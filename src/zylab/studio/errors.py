"""zylab.studio 异常类型."""

from __future__ import annotations

__all__ = [
    "LinkError",
    "ModuleNotFoundError_",
    "ParamError",
    "StudioError",
    "TemplateError",
    "TemplateNotFoundError",
]


class StudioError(Exception):
    """studio 包异常基类."""


class ModuleNotFoundError_(StudioError):
    """模块类型 id 未注册（命名带下划线后缀以规避与内建 ModuleNotFoundError 混淆）."""


class ParamError(StudioError):
    """参数值缺失、类型不符或越界."""


class TemplateError(StudioError):
    """模板定义非法（结构/引用/连接/环）或模板文件解析失败."""


class TemplateNotFoundError(StudioError):
    """模板 id 未注册."""


class LinkError(StudioError):
    """节点连接非法（端口不存在/类型不匹配/成环）."""
