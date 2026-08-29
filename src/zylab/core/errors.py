"""zylab 异常体系.

所有自研异常的公共基类为 :class:`ZylabError`，按场景细分派生类，
便于调用方按粒度捕获，也便于统一日志与错误码扩展。
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "PluginError",
    "PluginNotFoundError",
    "ProjectFileError",
    "TaskCancelledError",
    "WorkerCrashError",
    "WorkerError",
    "ZylabError",
]


class ZylabError(Exception):
    """zylab 所有自研异常的基类."""


class ConfigError(ZylabError):
    """配置错误（缺失必填项、非法取值、文件解析失败等）."""


class ProjectFileError(ZylabError):
    """工程文件错误（格式不符、版本不兼容、读写失败等）."""


class PluginError(ZylabError):
    """插件错误（注册冲突、加载失败、工厂解析失败等）."""


class PluginNotFoundError(PluginError):
    """插件未注册或无法定位."""


class WorkerError(ZylabError):
    """worker 进程内任务执行失败（含子进程回传的异常信息）."""


class WorkerCrashError(WorkerError):
    """worker 进程崩溃（非正常退出，未回传任何结果/错误事件）."""


class TaskCancelledError(WorkerError):
    """任务被用户取消."""
