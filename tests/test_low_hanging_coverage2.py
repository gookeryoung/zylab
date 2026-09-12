"""builtin.py + module.py plugin 错误路径补测."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zylab.flowchart import module as mod
from zylab.flowchart.errors import ModuleNotFoundError_
from zylab.flowchart.module import ModuleCategory, ModuleSpec

__all__ = []


# ================================================================== builtin


class TestBuiltinLoadErrorPaths:
    """builtin._load_assets_templates 目录缺失 + 坏文件分支."""

    def test_directory_missing_returns_empty(self, tmp_path, caplog) -> None:
        from zylab.flowchart import builtin

        result = builtin._load_assets_templates(tmp_path / "not_exists")
        assert result == ()
        assert any("目录缺失" in r.message for r in caplog.records)

    def test_bad_file_skipped(self, tmp_path, caplog) -> None:
        from zylab.flowchart import builtin

        sub = tmp_path / "structural"
        sub.mkdir()
        (sub / "bad.json").write_text("{not valid", encoding="utf-8")
        result = builtin._load_assets_templates(tmp_path)
        assert result == ()
        assert any("跳过非法" in r.message for r in caplog.records)


# ================================================================== module plugin resolution


class TestModuleSpecPluginPath:
    """module_spec 插件解析路径 (miss at 798-812)."""

    def test_plugin_returns_module_spec(self) -> None:
        fake = ModuleSpec(type_id="test.plugin", name="t", category=ModuleCategory.ANALYSIS, target="x")
        reg = MagicMock()
        reg_entry = MagicMock()
        reg_entry.name = "e"
        reg.list.return_value = [reg_entry]
        reg.resolve.return_value = fake

        with patch("zylab.core.registry.PluginRegistry", return_value=reg), patch.dict(mod._MODULES_BY_ID, clear=True):
            result = mod.module_spec("test.plugin")
            assert result.type_id == "test.plugin"

    def test_module_not_found_error(self) -> None:
        reg = MagicMock()
        reg.list.return_value = []
        with (
            patch("zylab.core.registry.PluginRegistry", return_value=reg),
            patch.dict(mod._MODULES_BY_ID, clear=True),
            pytest.raises(ModuleNotFoundError_),
        ):
            mod.module_spec("definitely.unknown")
