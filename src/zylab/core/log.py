"""zylab 日志基础设施.

约定：
- 每模块顶部 ``logger = logging.getLogger(__name__)``，库代码禁止 ``basicConfig``。
- 日志配置统一由入口调用 :func:`setup_logging` 一次（GUI/CLI/worker 进程各自入口）。
- 消息用中文短语，参数用 ``%s`` 延迟格式化；凭证脱敏后记录。
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

__all__ = ["LOG_FILE_NAME", "set_debug", "setup_logging"]

logger = logging.getLogger(__name__)

LOG_FILE_NAME = "zylab.log"

_SIMPLE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_VERBOSE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d %(funcName)s(): %(message)s"

# 第三方库一律 WARNING，避免噪声淹没自研日志
_THIRD_PARTY: dict[str, dict[str, Any]] = {
    name: {"level": "WARNING"} for name in ("urllib3", "asyncio", "h5py", "numba", "matplotlib")
}


def _build_config(env: str, log_dir: Path | None) -> dict[str, Any]:
    """构建 dictConfig 配置（dev: 控制台 DEBUG；prod: 控制台 INFO + 轮转文件）."""
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG" if env == "dev" else "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
    }
    root_handlers = ["console"]
    if env == "prod":
        if log_dir is None:
            raise ValueError("prod 模式必须提供 log_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "verbose",
            "filename": str(log_dir / LOG_FILE_NAME),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        root_handlers.append("file")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"format": _SIMPLE_FMT}, "verbose": {"format": _VERBOSE_FMT}},
        "handlers": handlers,
        "loggers": _THIRD_PARTY,
        "root": {"level": "DEBUG" if env == "dev" else "INFO", "handlers": root_handlers},
    }


def setup_logging(env: str = "dev", log_dir: Path | None = None) -> None:
    """初始化日志系统，按环境加载配置.

    :param env: ``"dev"``（控制台 DEBUG）或 ``"prod"``（控制台 INFO + 轮转文件）。
    :param log_dir: prod 模式日志目录，自动创建；dev 模式忽略。
    :raises ValueError: env 非法或 prod 模式未提供 log_dir。
    """
    if env not in ("dev", "prod"):
        raise ValueError(f"env 必须是 'dev' 或 'prod'，得到 {env!r}")
    logging.config.dictConfig(_build_config(env, log_dir))
    logger.debug("日志系统初始化完成: env=%s", env)


def set_debug(module: str, enabled: bool = True) -> None:
    """运行时动态调整某模块日志级别（True=DEBUG，False=INFO），用于单点排查."""
    logging.getLogger(module).setLevel(logging.DEBUG if enabled else logging.INFO)
