"""core.log 日志基础设施测试."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import pytest

from zylab.core.log import LOG_FILE_NAME, set_debug, setup_logging


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
