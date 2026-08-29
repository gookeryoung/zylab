"""core.config 配置管理测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.core.config import ENV_PREFIX, AppConfig, default_data_dir, load_config, read_toml
from zylab.core.errors import ConfigError


@pytest.fixture
def mock_data_dir(monkeypatch, tmp_path: Path):
    """劫持默认数据目录到 tmp_path，避免污染真实系统目录."""
    monkeypatch.setattr("zylab.core.config.default_data_dir", lambda: tmp_path)
    yield tmp_path


def test_default_config() -> None:
    """默认配置字段应合法."""
    cfg = AppConfig()
    assert cfg.log_level == "INFO"
    assert cfg.max_workers >= 1
    assert cfg.autosave_interval_s == 300


def test_config_validation_log_level() -> None:
    """非法 log_level 应抛 ConfigError."""
    with pytest.raises(ConfigError, match="log_level 必须是"):
        AppConfig(log_level="TRACE")


def test_config_validation_max_workers() -> None:
    """max_workers < 1 应抛 ConfigError."""
    with pytest.raises(ConfigError, match="max_workers 不能小于 1"):
        AppConfig(max_workers=0)


def test_config_validation_autosave() -> None:
    """autosave_interval_s 为负应抛 ConfigError."""
    with pytest.raises(ConfigError, match="autosave_interval_s 不能为负"):
        AppConfig(autosave_interval_s=-1)


def test_config_from_dict_roundtrip() -> None:
    """from_dict 应正确解析合法字段并忽略未知字段."""
    cfg = AppConfig.from_dict(
        {
            "log_level": "DEBUG",
            "data_dir": "/tmp/test_zylab",
            "unknown_key": 123,
        }
    )
    assert cfg.log_level == "DEBUG"
    assert cfg.data_dir == Path("/tmp/test_zylab")


def test_config_merge() -> None:
    """merge 应返回新实例，不修改原实例."""
    original = AppConfig(log_level="INFO")
    merged = original.merge({"log_level": "DEBUG"})
    assert original.log_level == "INFO"
    assert merged.log_level == "DEBUG"


def test_default_data_dir_returns_path() -> None:
    """default_data_dir 应返回 pathlib.Path 且指向 home 子目录."""
    d = default_data_dir()
    assert isinstance(d, Path)
    assert d.name == "zylab"


def test_read_toml(tmp_path: Path) -> None:
    """read_toml 应正确读取文件内容."""
    path = tmp_path / "test.toml"
    path.write_text('[tool]\nname = "zylab"\n', encoding="utf-8")
    data = read_toml(path)
    assert data["tool"]["name"] == "zylab"


def test_load_config_from_file(mock_data_dir: Path) -> None:
    """load_config 应能读取 TOML 配置文件且不被默认值回写覆盖."""
    config_path = mock_data_dir / "config.toml"
    mock_data_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text('log_level = "DEBUG"\nmax_workers = 8\n', encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.log_level == "DEBUG"
    assert cfg.max_workers == 8


def test_load_config_invalid_toml(tmp_path: Path) -> None:
    """TOML 解析失败应抛 ConfigError."""
    bad = tmp_path / "bad.toml"
    bad.write_text("log_level = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="配置文件解析失败"):
        load_config(bad)


def test_get_typed_bool(monkeypatch) -> None:
    """_get_typed 对 bool 类型应按枚举集合判断（私有函数，直接测试）."""
    from zylab.core.config import _get_typed

    monkeypatch.setenv("T_BOOL", "true")
    assert _get_typed("T_BOOL", bool, False) is True
    monkeypatch.setenv("T_BOOL", "0")
    assert _get_typed("T_BOOL", bool, True) is False


def test_get_typed_invalid_falls_back(monkeypatch) -> None:
    """_get_typed 对非法值应回退默认值（私有函数，直接测试）."""
    from zylab.core.config import _get_typed

    monkeypatch.setenv("T_INT", "not_a_number")
    assert _get_typed("T_INT", int, 7) == 7


def test_default_data_dir_linux(monkeypatch) -> None:
    """非 win32 平台应使用 XDG_CONFIG_HOME（默认 ~/.config）."""
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert default_data_dir() == Path("/tmp/xdg/zylab")
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert default_data_dir() == Path.home() / ".config" / "zylab"


def test_load_config_env_override(mock_data_dir: Path, monkeypatch) -> None:
    """环境变量应覆盖配置文件与默认值."""
    monkeypatch.setenv(f"{ENV_PREFIX}LOG_LEVEL", "ERROR")
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_WORKERS", "1")
    cfg = load_config()
    assert cfg.log_level == "ERROR"
    assert cfg.max_workers == 1


def test_load_config_env_data_dir(mock_data_dir: Path, monkeypatch) -> None:
    """ZYLAB_DATA_DIR 环境变量应规范化为 Path."""
    monkeypatch.setenv(f"{ENV_PREFIX}DATA_DIR", "/tmp/custom_zylab")
    cfg = load_config()
    assert cfg.data_dir == Path("/tmp/custom_zylab")
