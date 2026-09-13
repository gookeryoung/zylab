"""settings_panel 设置面板测试（主题切换 / 日志级别 / 自动保存 + 持久化）."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zylab.gui.widgets.settings_panel import SettingsPanel


@pytest.fixture
def _mock_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 default_data_dir 指向隔离的 tmp_path."""
    monkeypatch.setattr("zylab.core.default_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.gui
class TestSettingsPanel:
    """SettingsPanel 核心行为测试."""

    def test_constructed_defaults(self, qtbot) -> None:
        """构造控件并加载默认配置."""
        panel = SettingsPanel()
        qtbot.addWidget(panel)
        # 主题下拉列表应有 3 项
        assert panel._theme_combo.count() == 3
        # autosave 初始值来自默认（60 秒）
        assert panel._autosave_spin.value() == 60

    def test_save_and_load_roundtrip(self, qtbot, _mock_data_dir: Path) -> None:
        """save() 写入磁盘 → load() 能读回相同配置."""
        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 修改控件值
        panel._autosave_spin.setValue(120)
        level_idx = panel._log_level_combo.findText("DEBUG")
        if level_idx >= 0:
            panel._log_level_combo.setCurrentIndex(level_idx)

        cfg = panel.save()
        assert cfg["autosave_interval_sec"] == 120
        assert cfg["log_level"] == "DEBUG"

        # settings.json 应落盘
        p = _mock_data_dir / "settings.json"
        assert p.is_file()

        # 新建面板读回
        panel2 = SettingsPanel()
        qtbot.addWidget(panel2)
        assert panel2._autosave_spin.value() == 120
        assert panel2._log_level_combo.currentText() == "DEBUG"

    def test_corrupted_settings_json_falls_back(self, qtbot, _mock_data_dir: Path) -> None:
        """settings.json 损坏时静默 fallback 到默认值."""
        p = _mock_data_dir / "settings.json"
        p.write_text("THIS IS NOT JSON", encoding="utf-8")

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        # 不应崩溃，autosave 保持默认 60
        assert panel._autosave_spin.value() == 60

    def test_partial_config_merges_defaults(self, qtbot, _mock_data_dir: Path) -> None:
        """只有部分字段的 settings.json 通过默认值合并（新增字段不破坏老文件）."""
        p = _mock_data_dir / "settings.json"
        p.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

        panel = SettingsPanel()
        qtbot.addWidget(panel)

        idx = panel._theme_combo.findData("dark")
        assert idx >= 0
        assert panel._theme_combo.currentIndex() == idx
        # autosave / log_level 走默认
        assert panel._autosave_spin.value() == 60
