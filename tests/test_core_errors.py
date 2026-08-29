"""core.errors 异常体系测试."""

from __future__ import annotations

from zylab.core.errors import (
    ConfigError,
    PluginError,
    PluginNotFoundError,
    ProjectFileError,
    TaskCancelledError,
    WorkerCrashError,
    WorkerError,
    ZylabError,
)


def test_all_exceptions_inherit_zylaberror() -> None:
    """所有自研异常均继承 ZylabError."""
    exc_classes = (
        ConfigError,
        ProjectFileError,
        PluginError,
        PluginNotFoundError,
        WorkerError,
        WorkerCrashError,
        TaskCancelledError,
    )
    for exc_cls in exc_classes:
        exc = exc_cls("test")
        assert isinstance(exc, ZylabError)


def test_exception_str() -> None:
    """异常消息应完整保留."""
    msg = "配置文件缺失"
    exc = ConfigError(msg)
    assert str(exc) == msg


def test_exception_cause_chain() -> None:
    """异常链应保留底层原因."""
    try:
        try:
            raise ValueError("底层原因")
        except ValueError as exc:
            raise ConfigError("配置错误") from exc
    except ConfigError as e:
        assert str(e) == "配置错误"
        assert isinstance(e.__cause__, ValueError)
