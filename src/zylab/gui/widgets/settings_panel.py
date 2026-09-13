"""设置面板：主题切换 + 日志级别 + 自动保存间隔，持久化 settings.json.

设计为 QWidget（可嵌入 Dock 或对话框），通过 ``SettingsPanel`` 暴露
读写接口，调用方（MainWindow 命令面板）负责弹窗挂载与 theme 应用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import theme
from ..qt_compat import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

__all__ = ["SettingsPanel"]

logger = logging.getLogger(__name__)

# 默认配置（首次启动或 settings.json 缺失时用）
_DEFAULTS = {
    "theme": theme.DEFAULT_THEME,  # "light"
    "log_level": "INFO",
    "autosave_interval_sec": 60,
}

# settings.json 文件名（与 gui_state.json 同目录，由 default_data_dir() 决定）
_SETTINGS_FILE = "settings.json"


class SettingsPanel(QWidget):
    """设置面板 QWidget（主题 / 日志 / 自动保存，可独立嵌入任何容器）.

    构造时 ``load()`` 自动从 settings.json 读取并填充控件；
    ``save()`` 把当前控件状态写回磁盘并返回配置 dict 供调用方应用。

    设计约束：

    - 持久化通过 ``default_data_dir() / _SETTINGS_FILE``，不存在时自动创建；
    - 配置字段改动（如新增 field）通过 ``| _DEFAULTS`` 合并兼容旧文件；
    - UI 控件值变更不直接应用——用户点击"保存"才落盘，调用方（命令面板）
      负责在保存后 apply_theme / 配置日志级别等副作用。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPanel")
        self._build_ui()
        self.load()

    # ---------------------------------------------------------------- UI 组装

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        root.setSpacing(theme.SPACING_MD)

        # -- 外观 --
        appearance = QGroupBox("外观")
        appearance_layout = QFormLayout(appearance)

        self._theme_combo = QComboBox()
        for name, pal in theme.THEMES.items():
            self._theme_combo.addItem(pal.display_name, name)
        appearance_layout.addRow("主题", self._theme_combo)

        root.addWidget(appearance)

        # -- 行为 --
        behavior = QGroupBox("行为")
        behavior_layout = QFormLayout(behavior)

        self._autosave_spin = QSpinBox()
        self._autosave_spin.setRange(0, 3600)
        self._autosave_spin.setSuffix(" 秒")
        self._autosave_spin.setSpecialValueText("关闭")
        behavior_layout.addRow("自动保存间隔", self._autosave_spin)

        self._log_level_combo = QComboBox()
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self._log_level_combo.addItem(level)
        behavior_layout.addRow("日志级别", self._log_level_combo)

        root.addWidget(behavior)

        # -- 按钮行 --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._save_btn = QPushButton("保存")
        self._cancel_btn = QPushButton("取消")
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._cancel_btn)
        self._save_btn.clicked.connect(self.save)
        self._cancel_btn.clicked.connect(self.load)  # 重载回滚

        root.addLayout(btn_row)
        root.addStretch()

    # ---------------------------------------------------------------- 持久化

    def _settings_path(self) -> Path:
        from zylab.core import default_data_dir

        return default_data_dir() / _SETTINGS_FILE

    def load(self) -> dict:
        """从 settings.json 读取并填充控件；文件缺失时用默认值.

        :returns: 合并后的配置 dict（可能为默认值）。
        """
        cfg = dict(_DEFAULTS)
        p = self._settings_path()
        if p.is_file():
            try:
                cfg.update(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                logger.warning("settings.json 损坏，使用默认: %s", exc)
        self._apply_to_ui(cfg)
        return cfg

    def save(self) -> dict:
        """把当前控件状态写回 settings.json.

        :returns: 写入的配置 dict。
        """
        cfg = self._collect_from_ui()
        p = self._settings_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("设置已保存: %s", p)
        except OSError as exc:
            logger.warning("settings.json 写入失败: %s", exc)
        return cfg

    # ---------------------------------------------------------------- 控件 ↔ 配置 转换

    def _apply_to_ui(self, cfg: dict) -> None:
        theme_name = cfg.get("theme", _DEFAULTS["theme"])
        idx = self._theme_combo.findData(theme_name)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)

        self._autosave_spin.setValue(int(cfg.get("autosave_interval_sec", 0)))

        level = cfg.get("log_level", _DEFAULTS["log_level"])
        level_idx = self._log_level_combo.findText(level)
        if level_idx >= 0:
            self._log_level_combo.setCurrentIndex(level_idx)

    def _collect_from_ui(self) -> dict:
        return {
            "theme": self._theme_combo.currentData(),
            "autosave_interval_sec": self._autosave_spin.value(),
            "log_level": self._log_level_combo.currentText(),
        }
