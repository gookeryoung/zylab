"""zylab.core - 基础设施层（日志/配置/事件/插件/执行器/工程文件，Qt-free）."""

from __future__ import annotations

from .config import ENV_PREFIX, AppConfig, default_data_dir, load_config, read_toml
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
from .executor import EventKind, ProcessExecutor, TaskEvent, TaskHandle, TaskSpec, TaskStatus
from .log import LOG_FILE_NAME, LOG_LEVELS, set_debug, set_root_level, setup_logging
from .project import PROJECT_SCHEMA_VERSION, PROJECT_SUFFIX, Project
from .registry import ENTRY_POINT_PREFIX, PluginKind, PluginRegistry, PluginSpec
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
    "set_debug",
    "set_root_level",
    "setup_logging",
    "update_runtime_config",
]
