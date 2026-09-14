"""SettingsPanel 端到端测试：保存 → 重启 → 运行时生效全链路.

验证修复后 9 个配置字段（外观 4 个 + 非外观 5 个）在完整链路中
都能正确落盘并重启后恢复；同时覆盖 MainWindow 设置对话框保存后
运行时状态（theme/runtime_config/log_level）即时应用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from zylab.core import (
    get_autosave_interval,
    get_max_workers,
    get_solver_timeout,
    get_workspace_history_limit,
    update_runtime_config,
)


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """劫持 default_data_dir 到临时路径，避免污染真实用户目录."""
    monkeypatch.setattr("zylab.core.config.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("zylab.core.default_data_dir", lambda: tmp_path)
    monkeypatch.setattr("zylab.gui.main_window.default_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_runtime_state() -> None:
    """每个测试前把 runtime_config / theme 模块级状态恢复到默认值，避免跨测试污染."""
    from zylab.gui import theme

    update_runtime_config(
        max_workers=4,  # 独立于 cpu_count，保证断言确定性
        solver_timeout_s=0,
        autosave_interval_sec=60,
        workspace_history_limit=10,
    )
    theme.set_current_theme("light")
    theme.set_font_families(body="test-body", mono="test-mono")
    theme.set_font_scale(1.0)


# =====================================================================
# 1. SettingsPanel 修改 → save 落盘 → 新实例 load 恢复（完整 9 字段）
# =====================================================================


@pytest.mark.gui
class TestSettingsPanelSaveLoadRestart:
    """完整 9 字段 save → 重启 → load 链路."""

    def test_all_nine_fields_roundtrip(self, qtbot, isolated_data_dir: Path) -> None:
        """设置全部 9 个字段并保存，重建实例后 9 个字段全部恢复."""
        from zylab.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # --- 外观 4 字段 ---
        theme_idx = panel._theme_combo.findData("dark")
        assert theme_idx >= 0, "内置 dark 主题应存在"
        panel._theme_combo.setCurrentIndex(theme_idx)
        panel._font_body_combo.setCurrentText("Microsoft YaHei")
        panel._font_mono_combo.setCurrentText("Consolas")
        panel._font_scale_spin.setValue(1.2)  # 1.2x

        # --- 性能 2 字段 ---
        panel._max_workers_spin.setValue(8)
        panel._solver_timeout_spin.setValue(600)

        # --- 行为 3 字段 ---
        level_idx = panel._log_level_combo.findText("DEBUG")
        panel._log_level_combo.setCurrentIndex(level_idx)
        panel._autosave_spin.setValue(180)
        panel._history_limit_spin.setValue(20)

        cfg = panel.save()

        # 1. settings.json 落盘且 9 字段齐全
        p = isolated_data_dir / "settings.json"
        assert p.is_file()
        persisted = json.loads(p.read_text(encoding="utf-8"))
        assert set(persisted.keys()) == set(cfg.keys()), "落盘字段集合应等于 cfg 返回值"

        # 2. 新建 SettingsPanel 读回（模拟重启）
        panel2 = SettingsPanel()
        qtbot.addWidget(panel2)

        assert panel2._theme_combo.currentData() == "dark"
        assert panel2._font_body_combo.currentText() == "Microsoft YaHei"
        assert panel2._font_mono_combo.currentText() == "Consolas"
        assert panel2._font_scale_spin.value() == pytest.approx(1.2)
        assert panel2._max_workers_spin.value() == 8
        assert panel2._solver_timeout_spin.value() == 600
        assert panel2._log_level_combo.currentText() == "DEBUG"
        assert panel2._autosave_spin.value() == 180
        assert panel2._history_limit_spin.value() == 20

    def test_partial_save_then_restart(self, qtbot, isolated_data_dir: Path) -> None:
        """先写一个只含部分字段的 settings.json，重启后缺失字段走默认."""
        p = isolated_data_dir / "settings.json"
        p.write_text(
            json.dumps(
                {
                    "theme": "high_contrast",
                    "font_family_body": "PingFang SC",
                    "log_level": "WARNING",
                    # 缺失：font_family_mono / font_scale / max_workers /
                    # solver_timeout_s / autosave_interval_sec / workspace_history_limit
                }
            ),
            encoding="utf-8",
        )

        from zylab.gui.widgets.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 已有的字段读回
        assert panel._theme_combo.currentData() == "high_contrast"
        assert panel._font_body_combo.currentText() == "PingFang SC"
        assert panel._log_level_combo.currentText() == "WARNING"

        # 缺失的字段走默认
        assert panel._font_scale_spin.value() == 1.0  # 默认 1.0
        assert panel._autosave_spin.value() == 60
        assert panel._history_limit_spin.value() == 10

    def test_reset_to_defaults_roundtrip(self, qtbot, isolated_data_dir: Path) -> None:
        """重置按钮恢复默认 → 保存 → 重启仍然是默认."""
        from zylab.gui.widgets.settings_panel import SettingsPanel

        # 先写一份非默认的 settings.json
        p = isolated_data_dir / "settings.json"
        p.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "font_family_body": "Arial",
                    "font_family_mono": "Menlo",
                    "font_scale": 1.3,
                    "max_workers": 6,
                    "solver_timeout_s": 300,
                    "autosave_interval_sec": 120,
                    "log_level": "ERROR",
                    "workspace_history_limit": 15,
                }
            ),
            encoding="utf-8",
        )

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        # 验证先读回非默认
        assert panel._theme_combo.currentData() == "dark"

        # 点重置 → 回到默认值
        panel._reset_to_defaults()

        # 验证控件值回到默认
        # 默认主题是 light
        assert panel._theme_combo.currentData() == "light"
        # font_scale 默认 1.0 → DoubleSpinBox 值 1.0
        assert panel._font_scale_spin.value() == 1.0
        assert panel._log_level_combo.currentText() == "INFO"

        # 重置后保存
        panel.save()

        # 新建面板应读回默认（因为保存后落盘的就是默认值）
        panel2 = SettingsPanel()
        qtbot.addWidget(panel2)
        assert panel2._theme_combo.currentData() == "light"
        assert panel2._log_level_combo.currentText() == "INFO"


# =====================================================================
# 2. MainWindow 设置对话框：点击保存后运行时全部状态被即时应用
# =====================================================================


@pytest.mark.gui
class TestMainWindowSettingsDialogAppliesAtRuntime:
    """_open_settings_dialog 保存回调应把 9 字段全部应用到运行时."""

    def test_save_applies_theme_font_and_runtime_config(
        self, qtbot, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """直接构造 SettingsPanel 调 apply_settings，完整 9 字段进运行时."""
        from PySide2.QtWidgets import QApplication

        from zylab.gui import theme
        from zylab.gui.app import apply_settings, create_app
        from zylab.gui.widgets.settings_panel import SettingsPanel

        # 确保有一个 QApplication（pyqtbot 提供的也可以）
        app = QApplication.instance() or create_app()

        panel = SettingsPanel()
        qtbot.addWidget(panel)

        # 修改全部 9 个字段
        theme_idx = panel._theme_combo.findData("dark")
        panel._theme_combo.setCurrentIndex(theme_idx)
        panel._font_body_combo.setCurrentText("Microsoft YaHei")
        panel._font_mono_combo.setCurrentText("Consolas")
        panel._font_scale_spin.setValue(1.2)
        panel._max_workers_spin.setValue(12)
        panel._solver_timeout_spin.setValue(900)
        panel._autosave_spin.setValue(240)
        level_idx = panel._log_level_combo.findText("WARNING")
        panel._log_level_combo.setCurrentIndex(level_idx)
        panel._history_limit_spin.setValue(25)

        cfg = panel.save()

        # 模拟 MainWindow._open_settings_dialog 的保存回调
        apply_settings(
            app,
            theme_name=cfg.get("theme"),
            font_family_body=cfg.get("font_family_body"),
            font_family_mono=cfg.get("font_family_mono"),
            font_scale=cfg.get("font_scale"),
            log_level=cfg.get("log_level"),
            max_workers=cfg.get("max_workers"),
            solver_timeout_s=cfg.get("solver_timeout_s"),
            autosave_interval_sec=cfg.get("autosave_interval_sec"),
            workspace_history_limit=cfg.get("workspace_history_limit"),
        )

        # ① theme 模块级状态
        assert theme.current_palette().name == "dark"

        # ② 字体 & 字号
        fam = theme.current_font_families()
        assert fam["body"] == "Microsoft YaHei"
        assert fam["mono"] == "Consolas"
        assert theme.current_font_scale() == pytest.approx(1.2)

        # ③ runtime_config 全部 4 字段
        assert get_max_workers() == 12
        assert get_solver_timeout() == 900
        assert get_autosave_interval() == 240
        assert get_workspace_history_limit() == 25

        # ④ 根日志器级别
        assert logging.getLogger().level == logging.WARNING

        # ⑤ settings.json 落盘
        import json

        persisted = json.loads((isolated_data_dir / "settings.json").read_text(encoding="utf-8"))
        assert persisted["theme"] == "dark"
        assert persisted["font_family_body"] == "Microsoft YaHei"
        assert persisted["log_level"] == "WARNING"
        assert persisted["max_workers"] == 12
        assert persisted["workspace_history_limit"] == 25


# =====================================================================
# 3. main() 入口：从 settings.json 读取并应用全部 9 字段
# =====================================================================


@pytest.mark.gui
class TestMainAppStartupRestoresAllSettings:
    """gui.app.main() 应从 settings.json 读取全部 9 字段并应用到运行时."""

    def test_main_restores_all_nine_fields(self, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() 启动时读取完整 settings.json，全部 9 字段进入运行时."""
        import json

        from PySide2.QtWidgets import QApplication

        from zylab.gui import theme
        from zylab.gui.app import main as app_main

        # 写入完整 9 字段的 settings.json
        settings_json = isolated_data_dir / "settings.json"
        settings_json.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "font_family_body": "Arial",
                    "font_family_mono": "Menlo",
                    "font_scale": 1.1,
                    "max_workers": 7,
                    "solver_timeout_s": 1200,
                    "autosave_interval_sec": 300,
                    "log_level": "ERROR",
                    "workspace_history_limit": 30,
                }
            ),
            encoding="utf-8",
        )

        # 让 main() 立即退出（exec_app 被替换为直接返回）
        # 同时避免窗口真的弹出（离屏）

        monkeypatch.setattr("zylab.gui.app.exec_app", lambda app: 0)
        monkeypatch.setattr("zylab.gui.main_window.MainWindow.show", lambda self: None)
        # 避免 create_app 在已有实例时走复用分支后主题残留——先清掉旧 QApplication
        existing = QApplication.instance()
        if existing is not None:
            existing.closeAllWindows()

        rc = app_main()
        assert rc == 0

        # 断言：main() 启动后运行时全部 9 字段正确
        assert theme.current_palette().name == "dark"
        fam = theme.current_font_families()
        assert fam["body"] == "Arial"
        assert fam["mono"] == "Menlo"
        assert theme.current_font_scale() == pytest.approx(1.1)

        assert get_max_workers() == 7
        assert get_solver_timeout() == 1200
        assert get_autosave_interval() == 300
        assert get_workspace_history_limit() == 30

        assert logging.getLogger().level == logging.ERROR

    def test_main_without_settings_json_uses_defaults(
        self, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """没有 settings.json 时走默认值（不崩溃，运行时状态合法）."""
        from PySide2.QtWidgets import QApplication

        from zylab.gui import theme
        from zylab.gui.app import main as app_main

        # 确认没有 settings.json
        settings_json = isolated_data_dir / "settings.json"
        assert not settings_json.is_file()

        existing = QApplication.instance()
        if existing is not None:
            existing.closeAllWindows()

        monkeypatch.setattr("zylab.gui.app.exec_app", lambda app: 0)
        monkeypatch.setattr("zylab.gui.main_window.MainWindow.show", lambda self: None)

        rc = app_main()
        assert rc == 0

        # 默认主题是 light
        assert theme.current_palette().name == "light"
        # runtime_config 默认值合法
        assert get_max_workers() >= 1

    def test_main_with_corrupted_settings_json_falls_back(
        self, isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """settings.json 损坏时静默忽略，走默认启动."""
        from PySide2.QtWidgets import QApplication

        from zylab.gui import theme
        from zylab.gui.app import main as app_main

        (isolated_data_dir / "settings.json").write_text("NOT VALID JSON", encoding="utf-8")

        existing = QApplication.instance()
        if existing is not None:
            existing.closeAllWindows()

        monkeypatch.setattr("zylab.gui.app.exec_app", lambda app: 0)
        monkeypatch.setattr("zylab.gui.main_window.MainWindow.show", lambda self: None)

        rc = app_main()
        assert rc == 0
        # 默认主题
        assert theme.current_palette().name == "light"
