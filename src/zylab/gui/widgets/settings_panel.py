"""设置面板：QTabWidget 四分页（外观 / 性能 / 行为 / 高级），持久化 settings.json.

设计为 QWidget（可嵌入 Dock 或对话框），通过 ``SettingsPanel`` 暴露
读写接口，调用方（MainWindow 命令面板 / 右侧 Dock）负责挂载与副作用
（主题应用、日志级别刷新等）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .. import theme
from ..qt_compat import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = ["SettingsPanel"]

logger = logging.getLogger(__name__)

# 默认配置（首次启动或 settings.json 缺失时用）
# 字段改动通过 dict | _DEFAULTS 合并兼容旧文件
_DEFAULTS: dict = {
    # --- 外观 ---
    "theme": theme.DEFAULT_THEME,  # "light"
    "font_family_body": theme.FONT_FAMILY,  # 正文字体族
    "font_family_mono": theme.FONT_MONO,  # 等宽字体族
    "font_scale": 1.0,  # 字号缩放倍率（0.8 ~ 1.4，步长 0.1）
    # --- 性能 ---
    "max_workers": max(1, (os.cpu_count() or 4) // 2),  # 求解最大并发进程数
    "solver_timeout_s": 0,  # 求解器超时（秒），0 表示不限制
    # --- 行为 ---
    "autosave_interval_sec": 60,  # 自动保存间隔（秒），0 表示关闭
    "log_level": "INFO",  # 日志级别
    "workspace_history_limit": 10,  # 工作区历史保留条数
    # --- 高级（只读展示，不含持久化字段） ---
}

# 合法值范围约束
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_FONT_SCALE_MIN = 0.8
_FONT_SCALE_MAX = 1.4
_FONT_SCALE_STEP = 0.1
_MAX_WORKERS_MIN = 1
_MAX_WORKERS_MAX = 64
_SOLVER_TIMEOUT_MIN = 0
_SOLVER_TIMEOUT_MAX = 86400  # 24 小时上限
_AUTOSAVE_MIN = 0
_AUTOSAVE_MAX = 3600
_HISTORY_LIMIT_MIN = 1
_HISTORY_LIMIT_MAX = 100

# settings.json 文件名（与 gui_state.json 同目录，由 default_data_dir() 决定）
_SETTINGS_FILE = "settings.json"


class SettingsPanel(QWidget):
    """设置面板 QWidget（QTabWidget 分页，可独立嵌入任何容器）.

    构造时 ``load()`` 自动从 settings.json 读取并填充控件；
    ``save()`` 把当前控件状态写回磁盘并返回配置 dict 供调用方应用。

    设计约束：

    - 持久化通过 ``default_data_dir() / _SETTINGS_FILE``，不存在时自动创建；
    - 配置字段改动（如新增 field）通过 ``| _DEFAULTS`` 合并兼容旧文件；
    - UI 控件值变更不直接应用——用户点击"保存"才落盘，调用方（主窗口）
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

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_appearance_tab(), "外观")
        self._tabs.addTab(self._build_performance_tab(), "性能")
        self._tabs.addTab(self._build_behavior_tab(), "行为")
        self._tabs.addTab(self._build_advanced_tab(), "高级")
        root.addWidget(self._tabs, stretch=1)

        # -- 按钮行 --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._reset_btn = QPushButton("重置为默认")
        self._save_btn = QPushButton("保存")
        self._cancel_btn = QPushButton("取消")
        btn_row.addWidget(self._reset_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._cancel_btn)
        self._save_btn.clicked.connect(self.save)
        self._cancel_btn.clicked.connect(self.load)  # 重载回滚
        self._reset_btn.clicked.connect(self._reset_to_defaults)

        root.addLayout(btn_row)

    # ---- Tab 1: 外观 ----

    def _build_appearance_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_MD)

        # 主题
        theme_group = QGroupBox("主题")
        theme_form = QFormLayout(theme_group)

        self._theme_combo = QComboBox()
        for name, pal in theme.THEMES.items():
            self._theme_combo.addItem(pal.display_name, name)
        theme_form.addRow("主题方案", self._theme_combo)

        layout.addWidget(theme_group)

        # 字体
        font_group = QGroupBox("字体")
        font_form = QFormLayout(font_group)

        self._font_body_combo = QComboBox()
        self._font_body_combo.setEditable(True)
        for family in self._default_body_font_families():
            self._font_body_combo.addItem(family)
        font_form.addRow("正文字体族", self._font_body_combo)

        self._font_mono_combo = QComboBox()
        self._font_mono_combo.setEditable(True)
        for family in self._default_mono_font_families():
            self._font_mono_combo.addItem(family)
        font_form.addRow("等宽字体族", self._font_mono_combo)

        self._font_scale_spin = QDoubleSpinBox()
        self._font_scale_spin.setRange(_FONT_SCALE_MIN, _FONT_SCALE_MAX)
        self._font_scale_spin.setDecimals(1)
        self._font_scale_spin.setSingleStep(_FONT_SCALE_STEP)
        self._font_scale_spin.setSuffix(" 倍")
        self._font_scale_spin.setSpecialValueText(f"{_FONT_SCALE_MIN} 倍")
        font_form.addRow("字号缩放", self._font_scale_spin)

        layout.addWidget(font_group)
        layout.addStretch()

        return page

    # ---- Tab 2: 性能 ----

    def _build_performance_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_MD)

        perf_group = QGroupBox("求解性能")
        perf_form = QFormLayout(perf_group)

        self._max_workers_spin = QSpinBox()
        self._max_workers_spin.setRange(_MAX_WORKERS_MIN, _MAX_WORKERS_MAX)
        self._max_workers_spin.setSuffix(" 个进程")
        perf_form.addRow("最大并发进程", self._max_workers_spin)

        cpu_count = os.cpu_count() or 4
        hint = QLabel(f"当前系统 CPU 核心数: {cpu_count}")
        hint.setObjectName("secondaryText")
        perf_form.addRow("", hint)

        self._solver_timeout_spin = QSpinBox()
        self._solver_timeout_spin.setRange(_SOLVER_TIMEOUT_MIN, _SOLVER_TIMEOUT_MAX)
        self._solver_timeout_spin.setSuffix(" 秒")
        self._solver_timeout_spin.setSpecialValueText("不限制")
        perf_form.addRow("求解器超时", self._solver_timeout_spin)

        layout.addWidget(perf_group)
        layout.addStretch()

        return page

    # ---- Tab 3: 行为 ----

    def _build_behavior_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_MD)

        # 自动保存
        save_group = QGroupBox("自动保存")
        save_form = QFormLayout(save_group)

        self._autosave_spin = QSpinBox()
        self._autosave_spin.setRange(_AUTOSAVE_MIN, _AUTOSAVE_MAX)
        self._autosave_spin.setSuffix(" 秒")
        self._autosave_spin.setSpecialValueText("关闭")
        save_form.addRow("自动保存间隔", self._autosave_spin)

        layout.addWidget(save_group)

        # 日志
        log_group = QGroupBox("日志")
        log_form = QFormLayout(log_group)

        self._log_level_combo = QComboBox()
        for level in _LOG_LEVELS:
            self._log_level_combo.addItem(level)
        log_form.addRow("日志级别", self._log_level_combo)

        layout.addWidget(log_group)

        # 工作区
        workspace_group = QGroupBox("工作区")
        workspace_form = QFormLayout(workspace_group)

        self._history_limit_spin = QSpinBox()
        self._history_limit_spin.setRange(_HISTORY_LIMIT_MIN, _HISTORY_LIMIT_MAX)
        self._history_limit_spin.setSuffix(" 条")
        workspace_form.addRow("历史保留条数", self._history_limit_spin)

        layout.addWidget(workspace_group)
        layout.addStretch()

        return page

    # ---- Tab 4: 高级 ----

    def _build_advanced_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_MD)

        # 应用信息
        info_group = QGroupBox("应用信息")
        info_form = QFormLayout(info_group)

        from zylab import __version__

        info_form.addRow("版本", QLabel(f"v{__version__}"))

        layout.addWidget(info_group)

        # 数据目录
        dir_group = QGroupBox("数据目录")
        dir_form = QFormLayout(dir_group)

        from zylab.core import default_data_dir

        from ..qt_compat import Qt

        data_dir = default_data_dir()
        self._data_dir_label = QLabel(str(data_dir))
        self._data_dir_label.setWordWrap(True)
        # Qt5/Qt6 都支持 Qt.TextSelectableByMouse
        self._data_dir_label.setTextInteractionFlags(
            self._data_dir_label.textInteractionFlags() | Qt.TextSelectableByMouse
        )
        dir_form.addRow("配置/日志/缓存根目录", self._data_dir_label)

        hint = QLabel("设置、日志、工作区历史均保存在此目录下。")
        hint.setObjectName("secondaryText")
        hint.setWordWrap(True)
        dir_form.addRow("", hint)

        layout.addWidget(dir_group)
        layout.addStretch()

        return page

    # ---- 字体候选辅助 ----

    @staticmethod
    def _default_body_font_families() -> list[str]:
        """正文字体候选列表（回退顺序与 theme.FONT_FAMILY 保持一致）."""
        raw = theme.FONT_FAMILY
        # 从 '"PingFang SC", "Microsoft YaHei", ...' 解析字体族列表
        import re

        families = re.findall(r'"([^"]+)"', raw)
        if not families:
            families = ["Segoe UI", "Microsoft YaHei", "PingFang SC"]
        return families

    @staticmethod
    def _default_mono_font_families() -> list[str]:
        """等宽字体候选列表."""
        raw = theme.FONT_MONO
        import re

        families = re.findall(r'"([^"]+)"', raw)
        if not families:
            families = ["Consolas", "Cascadia Mono", "DejaVu Sans Mono"]
        return families

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

    def _reset_to_defaults(self) -> None:
        """把控件重置为 _DEFAULTS 值（不持久化，用户可再点击保存）."""
        self._apply_to_ui(dict(_DEFAULTS))

    # ---------------------------------------------------------------- 控件 ↔ 配置 转换

    def _apply_to_ui(self, cfg: dict) -> None:
        theme_name = cfg.get("theme", _DEFAULTS["theme"])
        idx = self._theme_combo.findData(theme_name)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)

        # 字体族：可编辑 ComboBox，找不到就 setCurrentText
        body_font = cfg.get("font_family_body", _DEFAULTS["font_family_body"])
        body_idx = self._font_body_combo.findText(body_font)
        if body_idx >= 0:
            self._font_body_combo.setCurrentIndex(body_idx)
        else:
            self._font_body_combo.setCurrentText(body_font)

        mono_font = cfg.get("font_family_mono", _DEFAULTS["font_family_mono"])
        mono_idx = self._font_mono_combo.findText(mono_font)
        if mono_idx >= 0:
            self._font_mono_combo.setCurrentIndex(mono_idx)
        else:
            self._font_mono_combo.setCurrentText(mono_font)

        scale = cfg.get("font_scale", _DEFAULTS["font_scale"])
        self._font_scale_spin.setValue(float(scale))

        self._max_workers_spin.setValue(int(cfg.get("max_workers", _DEFAULTS["max_workers"])))
        self._solver_timeout_spin.setValue(int(cfg.get("solver_timeout_s", 0)))
        self._autosave_spin.setValue(int(cfg.get("autosave_interval_sec", 0)))

        level = cfg.get("log_level", _DEFAULTS["log_level"])
        level_idx = self._log_level_combo.findText(level)
        if level_idx >= 0:
            self._log_level_combo.setCurrentIndex(level_idx)

        self._history_limit_spin.setValue(int(cfg.get("workspace_history_limit", _DEFAULTS["workspace_history_limit"])))

    def _collect_from_ui(self) -> dict:
        return {
            # --- 外观 ---
            "theme": self._theme_combo.currentData(),
            "font_family_body": self._font_body_combo.currentText().strip(),
            "font_family_mono": self._font_mono_combo.currentText().strip(),
            "font_scale": round(self._font_scale_spin.value(), 1),
            # --- 性能 ---
            "max_workers": self._max_workers_spin.value(),
            "solver_timeout_s": self._solver_timeout_spin.value(),
            # --- 行为 ---
            "autosave_interval_sec": self._autosave_spin.value(),
            "log_level": self._log_level_combo.currentText(),
            "workspace_history_limit": self._history_limit_spin.value(),
        }
