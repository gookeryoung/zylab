"""运行时可变配置（SettingsPanel 调整后即时生效，不重启）.

与 :mod:`zylab.core.config`（启动时一次性加载的不可变 AppConfig，来源
config.toml + 环境变量）分工：本模块承载 SettingsPanel 可调整的运行时
状态，每次保存后经 gui 层 ``apply_runtime`` 写回，业务代码通过本模块
getter 读取最新值。

设计约束：
- 模块级 ``_lock`` 保护多线程读写（GUI 主线程写 + 求解 worker 线程读）；
- 所有字段有类型注解和校验（非法值自动夹紧到允许范围）；
- 不做持久化（持久化由 SettingsPanel 负责）；
- 各 getter 返回防御性拷贝（dict）或只读值（int/float/str），避免外部
  意外修改内部状态。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

__all__ = [
    "AUTOSAVE_MAX",
    "AUTOSAVE_MIN",
    "DEFAULT_MAX_WORKERS",
    "SOLVER_TIMEOUT_MAX",
    "WORKERS_MAX",
    "WORKERS_MIN",
    "get_autosave_interval",
    "get_max_workers",
    "get_solver_timeout",
    "get_workspace_history_limit",
    "update_runtime_config",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 范围常量（与 SettingsPanel 保持一致）
# ---------------------------------------------------------------------------

WORKERS_MIN = 1
WORKERS_MAX = 64
SOLVER_TIMEOUT_MAX = 86400  # 24 小时上限
AUTOSAVE_MIN = 0  # 0 表示关闭
AUTOSAVE_MAX = 3600  # 1 小时上限
HISTORY_LIMIT_MIN = 1
HISTORY_LIMIT_MAX = 100

DEFAULT_MAX_WORKERS: int = max(1, (os.cpu_count() or 4) // 2)
DEFAULT_SOLVER_TIMEOUT: int = 0  # 0 表示不限制
DEFAULT_AUTOSAVE_INTERVAL: int = 60
DEFAULT_HISTORY_LIMIT: int = 10

# ---------------------------------------------------------------------------
# 模块级可变状态
# ---------------------------------------------------------------------------

_lock = threading.RLock()

_max_workers: int = DEFAULT_MAX_WORKERS
_solver_timeout_s: int = DEFAULT_SOLVER_TIMEOUT
_autosave_interval_sec: int = DEFAULT_AUTOSAVE_INTERVAL
_workspace_history_limit: int = DEFAULT_HISTORY_LIMIT


def _clamp(value: int, lo: int, hi: int) -> int:
    """把整数夹紧到 [lo, hi] 范围（非法值不抛异常，只夹紧）."""
    return max(lo, min(hi, int(value)))


# ---------------------------------------------------------------------------
# Getter
# ---------------------------------------------------------------------------


def get_max_workers() -> int:
    """求解器默认并发进程数（供 batch/bridge/optimize 等并行入口读取）."""
    with _lock:
        return _max_workers


def get_solver_timeout() -> int:
    """求解器超时秒数（0 = 不限制）."""
    with _lock:
        return _solver_timeout_s


def get_autosave_interval() -> int:
    """自动保存间隔秒数（0 = 关闭）."""
    with _lock:
        return _autosave_interval_sec


def get_workspace_history_limit() -> int:
    """工作区历史保留条数上限."""
    with _lock:
        return _workspace_history_limit


# ---------------------------------------------------------------------------
# Setter（供 gui 层 SettingsPanel 保存后调用）
# ---------------------------------------------------------------------------


def update_runtime_config(**overrides: Any) -> dict[str, int]:
    """批量更新运行时配置（仅接受已知且在范围内的字段，其余静默忽略）.

    典型调用::

        update_runtime_config(
            max_workers=4,
            autosave_interval_sec=120,
            workspace_history_limit=20,
        )

    Args:
        **overrides: 任意字段名 → 新值；未知字段忽略。

    Returns:
        实际被更新的字段 dict（key = 字段名，value = 夹紧后的新值），
        便于调用方确认哪些设置已进入运行时状态。
    """
    global _max_workers, _solver_timeout_s, _autosave_interval_sec, _workspace_history_limit  # noqa: PLW0603  模块级可变运行时状态
    updated: dict[str, int] = {}
    with _lock:
        if "max_workers" in overrides:
            v = _clamp(overrides["max_workers"], WORKERS_MIN, WORKERS_MAX)
            if v != _max_workers:
                _max_workers = v
                updated["max_workers"] = v
        if "solver_timeout_s" in overrides:
            v = _clamp(overrides["solver_timeout_s"], 0, SOLVER_TIMEOUT_MAX)
            if v != _solver_timeout_s:
                _solver_timeout_s = v
                updated["solver_timeout_s"] = v
        if "autosave_interval_sec" in overrides:
            v = _clamp(overrides["autosave_interval_sec"], AUTOSAVE_MIN, AUTOSAVE_MAX)
            if v != _autosave_interval_sec:
                _autosave_interval_sec = v
                updated["autosave_interval_sec"] = v
        if "workspace_history_limit" in overrides:
            v = _clamp(overrides["workspace_history_limit"], HISTORY_LIMIT_MIN, HISTORY_LIMIT_MAX)
            if v != _workspace_history_limit:
                _workspace_history_limit = v
                updated["workspace_history_limit"] = v
    if updated:
        logger.debug("运行时配置已更新: %s", updated)
    return updated
