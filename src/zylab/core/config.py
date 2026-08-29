"""zylab 配置管理.

配置层次（低 → 高优先级覆盖）：默认值 < TOML 配置文件 < ``ZYLAB_`` 前缀环境变量。

配置类用 ``@dataclass(frozen=True)``，不可变天然线程安全；凭证只走环境变量，
不进配置文件。3.8 兼容：``tomllib`` 缺失时回退 ``tomli``。
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError

__all__ = ["ENV_PREFIX", "AppConfig", "default_data_dir", "load_config", "read_toml"]

logger = logging.getLogger(__name__)

ENV_PREFIX = "ZYLAB_"

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def default_data_dir() -> Path:
    """返回平台默认数据目录（Windows: ``%APPDATA%/zylab``；其他: ``$XDG_CONFIG_HOME/zylab`` 或 ``~/.config/zylab``）."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "zylab"


@dataclass(frozen=True)
class AppConfig:
    """应用级配置（不可变）.

    :param log_level: 日志级别，取值为 DEBUG/INFO/WARNING/ERROR/CRITICAL。
    :param data_dir: 数据目录（配置、日志、缓存的根目录）。
    :param max_workers: 求解 worker 最大并发进程数。
    :param autosave_interval_s: 工程自动保存间隔（秒），0 表示关闭。
    """

    log_level: str = "INFO"
    data_dir: Path = field(default_factory=default_data_dir)
    max_workers: int = max(1, (os.cpu_count() or 4) // 2)
    autosave_interval_s: int = 300

    def __post_init__(self) -> None:
        """校验字段合法性，聚合所有错误一次性抛出."""
        errors: list[str] = []
        if self.log_level.upper() not in _LOG_LEVELS:
            errors.append(f"log_level 必须是 {_LOG_LEVELS} 之一，得到 {self.log_level!r}")
        if self.max_workers < 1:
            errors.append(f"max_workers 不能小于 1，得到 {self.max_workers}")
        if self.autosave_interval_s < 0:
            errors.append(f"autosave_interval_s 不能为负，得到 {self.autosave_interval_s}")
        if errors:
            raise ConfigError("配置校验失败:\n  - " + "\n  - ".join(errors))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """从字典构造配置；未知字段忽略，``data_dir`` 支持字符串路径."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                logger.debug("忽略未知配置项: %s", key)
                continue
            if key == "data_dir" and isinstance(value, str):
                value = Path(value)  # noqa: PLW2901
            kwargs[key] = value
        return cls(**kwargs)

    def merge(self, overrides: dict[str, Any]) -> AppConfig:
        """合并覆盖项并返回新实例（仅覆盖显式出现的已知字段，不修改原对象）."""
        if not overrides:
            return self
        normalized = type(self).from_dict(overrides)  # 规范化（如 data_dir str→Path）
        known = {f.name for f in fields(self)}
        return replace(self, **{k: getattr(normalized, k) for k in overrides if k in known})


def read_toml(path: Path) -> dict[str, Any]:
    """读取 TOML 文件（3.11+ 用标准库 tomllib，低版本回退 tomli）."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover（3.11+ 环境不覆盖回退分支）
        import tomli as tomllib  # type: ignore[missing-import]
    with path.open("rb") as f:  # tomllib 只接受二进制流
        return tomllib.load(f)


def _get_typed(key: str, target_type: type, default: Any) -> Any:
    """读取环境变量并做类型转换；缺失或非法时回退默认值."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return target_type(raw)
    except (TypeError, ValueError):
        logger.warning("环境变量 %s=%r 无法转为 %s，使用默认值 %r", key, raw, target_type.__name__, default)
        return default


# (环境变量名后缀, 字段名, 目标类型)；仅显式设置的环境变量才参与覆盖
_ENV_FIELD_MAP: tuple[tuple[str, str, type], ...] = (
    ("LOG_LEVEL", "log_level", str),
    ("DATA_DIR", "data_dir", str),
    ("MAX_WORKERS", "max_workers", int),
    ("AUTOSAVE_INTERVAL_S", "autosave_interval_s", int),
)


def load_config(path: Path | None = None) -> AppConfig:
    """按层次加载配置：默认值 < TOML 文件 < ``ZYLAB_`` 环境变量.

    环境变量仅在显式设置时参与覆盖，避免默认值回写覆盖文件配置。

    :param path: TOML 配置路径；为 None 时尝试 ``default_data_dir()/config.toml``，不存在则用默认值。
    :raises ConfigError: TOML 解析失败或字段校验失败。
    """
    data: dict[str, Any] = {}
    config_path = path if path is not None else default_data_dir() / "config.toml"
    if config_path.exists():
        try:
            data = read_toml(config_path)
        except Exception as exc:
            raise ConfigError(f"配置文件解析失败: {config_path}") from exc
        logger.debug("已加载配置文件: %s", config_path)

    defaults = AppConfig()
    env_overrides: dict[str, Any] = {}
    for env_key, field_name, target_type in _ENV_FIELD_MAP:
        full_key = f"{ENV_PREFIX}{env_key}"
        if full_key not in os.environ:
            continue
        default_value: Any = getattr(defaults, field_name)
        if isinstance(default_value, Path):
            default_value = str(default_value)
        value = _get_typed(full_key, target_type, default_value)
        if field_name == "data_dir":
            value = Path(value)
        env_overrides[field_name] = value
    return AppConfig.from_dict(data).merge(env_overrides)
