"""settings_panel 设置面板测试（主题 / 字体 / 性能 / 行为 + 持久化）."""

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
        # Tab 数量 = 4（外观/性能/行为/高级）
        assert panel._tabs.count() == 4
        # 字体缩放默认 1.0（SpinBox * 10 = 10）
        assert panel._font_scale_spin.value() == 10

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

    # --- 新增配置项测试 ---

    def test_full_config_roundtrip(self, qtbot, _mock_data_dir: Path) -> None:
        """完整配置（外观 + 性能 + 行为）save/load roundtrip."""
        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 外观
        theme_idx = panel._theme_combo.findData("dark")
        if theme_idx >= 0:
            panel._theme_combo.setCurrentIndex(theme_idx)
        panel._font_body_combo.setCurrentText("Microsoft YaHei")
        panel._font_mono_combo.setCurrentText("Consolas")
        panel._font_scale_spin.setValue(12)  # 1.2x

        # 性能
        panel._max_workers_spin.setValue(4)
        panel._solver_timeout_spin.setValue(300)

        # 行为
        level_idx = panel._log_level_combo.findText("WARNING")
        if level_idx >= 0:
            panel._log_level_combo.setCurrentIndex(level_idx)
        panel._autosave_spin.setValue(180)
        panel._history_limit_spin.setValue(20)

        cfg = panel.save()

        # 验证写入值
        assert cfg["theme"] == "dark"
        assert cfg["font_family_body"] == "Microsoft YaHei"
        assert cfg["font_family_mono"] == "Consolas"
        assert cfg["font_scale"] == pytest.approx(1.2)
        assert cfg["max_workers"] == 4
        assert cfg["solver_timeout_s"] == 300
        assert cfg["log_level"] == "WARNING"
        assert cfg["autosave_interval_sec"] == 180
        assert cfg["workspace_history_limit"] == 20

        # 新建面板读回
        panel2 = SettingsPanel()
        qtbot.addWidget(panel2)
        assert panel2._theme_combo.currentData() == "dark"
        assert panel2._font_body_combo.currentText() == "Microsoft YaHei"
        assert panel2._font_mono_combo.currentText() == "Consolas"
        assert panel2._font_scale_spin.value() == 12
        assert panel2._max_workers_spin.value() == 4
        assert panel2._solver_timeout_spin.value() == 300
        assert panel2._log_level_combo.currentText() == "WARNING"
        assert panel2._autosave_spin.value() == 180
        assert panel2._history_limit_spin.value() == 20

    def test_reset_to_defaults(self, qtbot, _mock_data_dir: Path) -> None:
        """重置按钮把控件恢复为默认值（不持久化）."""
        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 先改值
        panel._max_workers_spin.setValue(16)
        panel._font_scale_spin.setValue(8)  # 0.8x

        # 重置
        panel._reset_to_defaults()

        # 应该回到默认
        import os

        default_workers = max(1, (os.cpu_count() or 4) // 2)
        assert panel._max_workers_spin.value() == default_workers
        assert panel._font_scale_spin.value() == 10  # 1.0

    def test_partial_new_fields_merge(self, qtbot, _mock_data_dir: Path) -> None:
        """只有老字段（无新增字段）的 settings.json，新增字段应走默认."""
        p = _mock_data_dir / "settings.json"
        # 模拟老版本 settings.json（只有 theme + autosave）
        p.write_text(
            json.dumps(
                {
                    "theme": "high_contrast",
                    "autosave_interval_sec": 300,
                    "log_level": "ERROR",
                }
            ),
            encoding="utf-8",
        )

        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 老字段读回
        theme_idx = panel._theme_combo.findData("high_contrast")
        assert panel._theme_combo.currentIndex() == theme_idx
        assert panel._autosave_spin.value() == 300
        assert panel._log_level_combo.currentText() == "ERROR"

        # 新增字段走默认
        assert panel._font_scale_spin.value() == 10  # 1.0
        assert panel._history_limit_spin.value() == 10

    def test_save_returns_complete_dict(self, qtbot, _mock_data_dir: Path) -> None:
        """save() 返回的 dict 应包含全部持久化字段."""
        panel = SettingsPanel()
        qtbot.addWidget(panel)
        cfg = panel.save()

        # 所有持久化 key 应存在
        expected_keys = {
            "theme",
            "font_family_body",
            "font_family_mono",
            "font_scale",
            "max_workers",
            "solver_timeout_s",
            "autosave_interval_sec",
            "log_level",
            "workspace_history_limit",
        }
        assert set(cfg.keys()) == expected_keys
