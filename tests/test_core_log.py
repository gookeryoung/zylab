"""core.log 日志基础设施测试."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import pytest

from zylab.core.log import LOG_FILE_NAME, LOG_LEVELS, set_debug, set_root_level, setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """每次测试后重置 logging 全局状态，防止 dictConfig 叠加."""
    yield
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})


def test_setup_logging_dev() -> None:
    """dev 模式应将 root 设为 DEBUG 并写 console handler."""
    setup_logging("dev")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) >= 1

    # 验证 logger 实际可输出到自定义 handler（dictConfig 会替换 root handlers，caplog 不适用）
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger = logging.getLogger("test.dev")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug("dev 日志测试")
    logger.removeHandler(handler)
    assert "dev 日志测试" in records


def test_setup_logging_prod(tmp_path: Path) -> None:
    """prod 模式应创建轮转文件 handler."""
    setup_logging("prod", log_dir=tmp_path)
    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) >= 2

    handler_names = [type(h).__name__ for h in root.handlers]
    assert "StreamHandler" in handler_names
    assert "RotatingFileHandler" in handler_names

    log_file = tmp_path / LOG_FILE_NAME
    assert log_file.exists()


def test_setup_logging_invalid_env() -> None:
    """非法 env 应抛 ValueError."""
    with pytest.raises(ValueError, match="env 必须是"):
        setup_logging("invalid")


def test_setup_logging_prod_without_log_dir() -> None:
    """prod 模式未提供 log_dir 应抛 ValueError."""
    with pytest.raises(ValueError, match="prod 模式必须提供 log_dir"):
        setup_logging("prod")


def test_set_debug() -> None:
    """set_debug 应动态调整某模块日志级别."""
    setup_logging("dev")
    logger = logging.getLogger("test_module")
    assert logger.level == logging.NOTSET  # 未单独设置前继承 root
    set_debug("test_module", enabled=True)
    assert logging.getLogger("test_module").level == logging.DEBUG
    set_debug("test_module", enabled=False)
    assert logging.getLogger("test_module").level == logging.INFO


def test_set_root_level_uppercase() -> None:
    """set_root_level 接受大写级别名称并立即生效."""
    setup_logging("dev")
    set_root_level("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    set_root_level("INFO")  # 还原


def test_set_root_level_case_insensitive() -> None:
    """set_root_level 对大小写不敏感."""
    setup_logging("dev")
    set_root_level("debug")  # 小写
    assert logging.getLogger().level == logging.DEBUG
    set_root_level("Critical")  # 混合大小写
    assert logging.getLogger().level == logging.CRITICAL
    set_root_level("INFO")  # 还原


def test_set_root_level_handlers_also_updated() -> None:
    """根日志器级别调整时，handler 级别也应同步更新."""
    setup_logging("dev")
    root = logging.getLogger()
    # 先设为 ERROR
    set_root_level("ERROR")
    assert root.level == logging.ERROR
    for handler in root.handlers:
        assert handler.level == logging.ERROR

    # 再设为 DEBUG
    set_root_level("DEBUG")
    assert root.level == logging.DEBUG
    for handler in root.handlers:
        assert handler.level == logging.DEBUG

    set_root_level("INFO")  # 还原


def test_set_root_level_invalid_raises() -> None:
    """非法级别应抛 ValueError."""
    setup_logging("dev")
    with pytest.raises(ValueError, match="非法日志级别"):
        set_root_level("VERBOSE")
    with pytest.raises(ValueError, match="非法日志级别"):
        set_root_level("INVALID")


def test_log_levels_constant() -> None:
    """LOG_LEVELS 常量包含全部合法级别."""
    assert set(LOG_LEVELS) == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
