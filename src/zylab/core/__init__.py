"""zylab.core - 基础设施层（日志/配置/事件/插件/执行器/工程文件，Qt-free）.

本模块采用 ``__getattr__`` 懒加载：访问 ``from zylab.core import EventBus``
时仅加载 ``.events`` 子模块，不触发 ``.project``（h5py 74ms）、
``.registry``（importlib.metadata 12ms）等重依赖链，冷启动从 ~133ms
降至 ~1ms。属性名到子模块的映射集中维护在 ``_LAZY_ATTRS`` 字典里。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查用
    from .config import (
        ENV_PREFIX,
        AppConfig,
        default_data_dir,
        load_config,
        read_toml,
    )
    from .errors import (
        ConfigError,
        PluginError,
        PluginNotFoundError,
        ProjectFileError,
        TaskCancelledError,
        WorkerCrashError,
        WorkerError,
        ZylabError,
    )
    from .events import EventBus
    from .executor import (
        EventKind,
        ProcessExecutor,
        TaskEvent,
        TaskHandle,
        TaskSpec,
        TaskStatus,
    )
    from .log import (
        LOG_FILE_NAME,
        LOG_LEVELS,
        set_debug,
        set_root_level,
        setup_logging,
    )
    from .project import PROJECT_SCHEMA_VERSION, PROJECT_SUFFIX, Project
    from .registry import (
        ENTRY_POINT_PREFIX,
        PluginKind,
        PluginRegistry,
        PluginSpec,
    )
    from .registry import (
        PluginRegistry as registry,
    )
    from .runtime_config import (
        AUTOSAVE_MAX,
        AUTOSAVE_MIN,
        DEFAULT_MAX_WORKERS,
        SOLVER_TIMEOUT_MAX,
        WORKERS_MAX,
        WORKERS_MIN,
        get_autosave_interval,
        get_max_workers,
        get_solver_timeout,
        get_workspace_history_limit,
        update_runtime_config,
    )

__all__ = [
    "AUTOSAVE_MAX",
    "AUTOSAVE_MIN",
    "DEFAULT_MAX_WORKERS",
    "ENTRY_POINT_PREFIX",
    "ENV_PREFIX",
    "LOG_FILE_NAME",
    "LOG_LEVELS",
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_SUFFIX",
    "SOLVER_TIMEOUT_MAX",
    "WORKERS_MAX",
    "WORKERS_MIN",
    "AppConfig",
    "ConfigError",
    "EventBus",
    "EventKind",
    "PluginError",
    "PluginKind",
    "PluginNotFoundError",
    "PluginRegistry",
    "PluginSpec",
    "ProcessExecutor",
    "Project",
    "ProjectFileError",
    "TaskCancelledError",
    "TaskEvent",
    "TaskHandle",
    "TaskSpec",
    "TaskStatus",
    "WorkerCrashError",
    "WorkerError",
    "ZylabError",
    "default_data_dir",
    "get_autosave_interval",
    "get_max_workers",
    "get_solver_timeout",
    "get_workspace_history_limit",
    "load_config",
    "read_toml",
    "registry",
    "set_debug",
    "set_root_level",
    "setup_logging",
    "update_runtime_config",
]


#: 懒加载映射：属性名 → (子模块相对路径, 子模块内的属性名).
#: 子模块加载一次后被 Python 缓存到 sys.modules，后续访问零开销.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # config
    "ENV_PREFIX": (".config", "ENV_PREFIX"),
    "AppConfig": (".config", "AppConfig"),
    "default_data_dir": (".config", "default_data_dir"),
    "load_config": (".config", "load_config"),
    "read_toml": (".config", "read_toml"),
    # errors
    "ConfigError": (".errors", "ConfigError"),
    "PluginError": (".errors", "PluginError"),
    "PluginNotFoundError": (".errors", "PluginNotFoundError"),
    "ProjectFileError": (".errors", "ProjectFileError"),
    "TaskCancelledError": (".errors", "TaskCancelledError"),
    "WorkerCrashError": (".errors", "WorkerCrashError"),
    "WorkerError": (".errors", "WorkerError"),
    "ZylabError": (".errors", "ZylabError"),
    # events
    "EventBus": (".events", "EventBus"),
    # executor
    "EventKind": (".executor", "EventKind"),
    "ProcessExecutor": (".executor", "ProcessExecutor"),
    "TaskEvent": (".executor", "TaskEvent"),
    "TaskHandle": (".executor", "TaskHandle"),
    "TaskSpec": (".executor", "TaskSpec"),
    "TaskStatus": (".executor", "TaskStatus"),
    # log
    "LOG_FILE_NAME": (".log", "LOG_FILE_NAME"),
    "LOG_LEVELS": (".log", "LOG_LEVELS"),
    "set_debug": (".log", "set_debug"),
    "set_root_level": (".log", "set_root_level"),
    "setup_logging": (".log", "setup_logging"),
    # project — 重依赖 h5py（74ms），仅在真正读写 .zprj 时加载
    "PROJECT_SCHEMA_VERSION": (".project", "PROJECT_SCHEMA_VERSION"),
    "PROJECT_SUFFIX": (".project", "PROJECT_SUFFIX"),
    "Project": (".project", "Project"),
    # registry — 重依赖 importlib.metadata（12ms），插件发现时才加载
    "ENTRY_POINT_PREFIX": (".registry", "ENTRY_POINT_PREFIX"),
    "PluginKind": (".registry", "PluginKind"),
    "PluginRegistry": (".registry", "PluginRegistry"),
    "PluginSpec": (".registry", "PluginSpec"),
    "registry": (".registry", "PluginRegistry"),
    # runtime_config
    "AUTOSAVE_MAX": (".runtime_config", "AUTOSAVE_MAX"),
    "AUTOSAVE_MIN": (".runtime_config", "AUTOSAVE_MIN"),
    "DEFAULT_MAX_WORKERS": (".runtime_config", "DEFAULT_MAX_WORKERS"),
    "SOLVER_TIMEOUT_MAX": (".runtime_config", "SOLVER_TIMEOUT_MAX"),
    "WORKERS_MAX": (".runtime_config", "WORKERS_MAX"),
    "WORKERS_MIN": (".runtime_config", "WORKERS_MIN"),
    "get_autosave_interval": (".runtime_config", "get_autosave_interval"),
    "get_max_workers": (".runtime_config", "get_max_workers"),
    "get_solver_timeout": (".runtime_config", "get_solver_timeout"),
    "get_workspace_history_limit": (".runtime_config", "get_workspace_history_limit"),
    "update_runtime_config": (".runtime_config", "update_runtime_config"),
}


def __getattr__(name: str) -> object:
    """懒加载 facade：首次访问时才 import 对应子模块."""
    mapping = _LAZY_ATTRS.get(name)
    if mapping is None:
        raise AttributeError(f"module 'zylab.core' has no attribute {name!r}")
    submodule_path, attr_name = mapping
    module = import_module(submodule_path, __name__)
    value = getattr(module, attr_name)
    # 缓存到模块 dict，后续直接命中，不再走 __getattr__
    globals()[name] = value
    return value
