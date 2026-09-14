"""AppController - 跨页面协调控制器.

将 MainWindow 中跨页面协调逻辑（主题切换、工作区管理、运行状态、对话框）
抽离为独立 QObject。MainWindow 保留窗口骨架 + Dock 布局 + 页面构造，
通过 ``self._controller`` 访问协调能力。

与 MainWindow 的职责边界：

| 职责 | MainWindow | AppController |
|------|-----------|--------------|
| 窗口骨架 / Dock 布局 | ✅ | ❌ |
| 头部栏构建 | ✅ | ❌ |
| 三页面 StackedWidget | ✅ | ❌ |
| EventBus / ReplKernel 持有 | ✅ | ❌ |
| 工作区管理器持有 | ✅ | ❌ |
| 主题切换 + 图标刷新 | 调用转发 | ✅ 实现 |
| 工作区切换 + 持久化 | 调用转发 | ✅ 实现 |
| 运行状态 indicator | 调用转发 | ✅ 实现 |
| 设置 / 关于对话框 | 调用转发 | ✅ 实现 |
| gui_state.json 持久化 | ✅ 持有 | ❌ |
| 命令面板 / 快捷键注册 | ✅ | ❌ |
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .. import theme
from ..app import save_theme_name
from ..qt_compat import (
    Property,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QObject,
    QVBoxLayout,
    Signal,
    Slot,
)
from .theme_controller import ThemeController

if TYPE_CHECKING:
    from zylab.gui.main_window import MainWindow  # noqa: F401  仅类型提示

    from ..icons import NAV_ICON_NAMES  # noqa: F401  仅类型提示


__all__ = ["AppController"]

logger = logging.getLogger(__name__)


class AppController(QObject):
    """跨页面协调控制器.

    :param parent: MainWindow 实例（必传，Controller 通过它访问 UI 组件）
    """

    # --- Signals ---

    #: 主题切换后发射（参数：新主题名）
    theme_changed = Signal(str)

    #: 工作区切换后发射（参数：新工作区路径）
    workspace_changed = Signal(str)

    #: 运行状态变更
    run_status_changed = Signal()

    # --- Properties ---

    current_theme = Property(str, notify=theme_changed)
    workspace_path = Property(str, notify=workspace_changed)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        # parent 即为 MainWindow，类型提示在调用方已确证
        self._mw = parent  # type: ignore[assignment]
        # 主题属性代理（暴露 Palette 字段为 Q_PROPERTY，供 Widget/QML 统一取色）
        self.theme_proxy = ThemeController(self)

    # ------------------------------------------------------------------ 主题切换

    @Slot(str, bool)
    def set_theme(self, name: str, persist: bool = True) -> None:
        """应用主题并刷新全部页面与图标.

        :param name: 主题名（dark / light / custom-*）
        :param persist: True 写入 settings.json，False 仅预览
        """
        from zylab.core.config import default_data_dir

        from ..app import apply_theme
        from ..icons import nav_icon
        from ..qt_compat import Qt

        mw = self._mw
        if name != theme.current_palette().name:
            apply_theme(QApplication.instance(), name)

        # 刷新侧边栏操作按钮图标
        pal = theme.current_palette()
        mw._workspace_history_btn.setIcon(nav_icon("arrow_down", pal.nav_text))
        mw._workspace_open_btn.setIcon(nav_icon("open_file", pal.nav_text))
        mw._settings_btn.setIcon(nav_icon("settings", pal.nav_text))
        mw._help_btn.setIcon(nav_icon("question", pal.nav_text))

        # 刷新项目浏览器树节点图标
        from ..icons import NAV_ICON_NAMES

        root = mw._project_tree.topLevelItem(0)
        if root is not None:
            for idx in range(root.childCount()):
                child = root.child(idx)
                if child is None:
                    continue
                data = child.data(0, Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "page":
                    page_idx = int(data[1])
                    if 0 <= page_idx < len(NAV_ICON_NAMES):
                        child.setIcon(0, nav_icon(NAV_ICON_NAMES[page_idx], pal.text_primary))

        # 刷新页面内主题
        mw._notebook_page.refresh_theme()
        mw._flowchart_page.refresh_theme()
        mw._template_page.refresh_theme()

        # 刷新工作区标签颜色（已在 apply_theme 中由 QSS 覆盖，但文本可能需要更新）
        mw._workspace_label.setText(str(mw._workspace_manager.cwd))

        if persist:
            save_theme_name(default_data_dir(), name)
            mw.statusBar().showMessage(f"主题已切换: {theme.current_palette().display_name}")

        # 通知 ThemeController 同步新色板，所有订阅方收到 theme_proxy.theme_changed
        self.theme_proxy.refresh()
        self.theme_changed.emit(name)

    # ------------------------------------------------------------------ 工作区管理

    @Slot(str)
    def switch_workspace_to(self, target: str) -> None:
        """切换工作区（校验 + 应用 + 持久化 + 刷新 UI）.

        :param target: 目标目录路径
        """

        mw = self._mw
        wm = mw._workspace_manager

        target_path = Path(target)
        if not target_path.is_dir():
            QMessageBox.warning(mw, "切换工作区", f"目录不存在:\n{target}")
            return

        try:
            wm.set_cwd(target_path)
            wm.save()
        except OSError as exc:
            QMessageBox.warning(mw, "切换工作区", f"无法切换到该目录:\n{exc}")
            return

        # 刷新 UI
        mw._workspace_label.setText(str(target_path))
        mw._workspace_label.setToolTip(f"当前工作区：{target_path}")
        if hasattr(mw, "_status_cwd_label"):
            mw._status_cwd_label.setText(f"  📁 {target_path}")

        self._refresh_workspace_menu()
        mw.statusBar().showMessage(f"工作区：{target_path}")
        self.workspace_changed.emit(str(target_path))

    def _refresh_workspace_menu(self) -> None:
        """重建历史工作区菜单."""
        mw = self._mw
        menu = mw._workspace_history_menu
        menu.clear()
        history = mw._workspace_manager.recent_workspaces(limit=10)
        if not history:
            empty = menu.addAction("（暂无历史）")
            empty.setEnabled(False)
            return
        for path in history:
            action = menu.addAction(str(path))
            action.setData(str(path))
            action.triggered.connect(lambda _checked=False, p=str(path): self.switch_workspace_to(p))
        menu.addSeparator()
        open_action = menu.addAction("选择其他目录…")
        open_action.triggered.connect(self._on_pick_workspace)

    @Slot()
    def _on_pick_workspace(self) -> None:
        """弹出目录选择对话框."""
        mw = self._mw
        current = str(mw._workspace_manager.cwd)
        target = QFileDialog.getExistingDirectory(
            mw,
            "选择工作区目录",
            current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if target:
            self.switch_workspace_to(target)

    # ------------------------------------------------------------------ 运行状态 indicator

    @Slot(str, str)
    def set_run_status(self, state: str, detail: str = "") -> None:
        """设置右下角运行状态 indicator.

        :param state: idle / running / success / error
        :param detail: 失败时的错误详情（tooltip）
        """
        from ..icons import tinted_pixmap

        mw = self._mw
        pal = theme.current_palette()
        state_map = {
            "idle": ("question", "就绪", pal.text_secondary),
            "running": ("play", "计算中…", pal.primary),
            "success": ("check", "运行完成", pal.success_text),
            "error": ("cross", "运行失败", pal.danger_text),
        }
        icon_name, label, color = state_map.get(state, state_map["idle"])
        mw._indicator_icon.setPixmap(tinted_pixmap(icon_name, color, 16))
        mw._indicator_text.setText(label)
        mw._indicator_text.setProperty("state", state)
        mw._indicator_text.style().unpolish(mw._indicator_text)
        mw._indicator_text.style().polish(mw._indicator_text)
        if detail and state == "error":
            mw._indicator_text.setToolTip(detail[:200])
        else:
            mw._indicator_text.setToolTip("")
        self.run_status_changed.emit()

    # ------------------------------------------------------------------ 对话框

    @Slot()
    def open_about_dialog(self) -> None:
        """弹出关于对话框."""
        from zylab import __version__

        mw = self._mw
        dlg = QDialog(mw)
        dlg.setWindowTitle("关于 zylab")
        dlg.setMinimumWidth(440)
        root = QVBoxLayout(dlg)
        root.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        root.setSpacing(theme.SPACING_MD)

        brand_frame = QFrame(objectName="aboutBrand")
        brand_layout = QVBoxLayout(brand_frame)
        brand_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        brand_layout.setSpacing(theme.SPACING_SM)
        brand = QLabel("zylab", objectName="aboutAppName")
        desc = QLabel("通用科学计算仿真分析平台", objectName="aboutAppDesc")
        desc.setWordWrap(True)
        brand_layout.addWidget(brand)
        brand_layout.addWidget(desc)
        root.addWidget(brand_frame)

        info_card = QFrame(objectName="aboutCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        info_layout.setSpacing(theme.SPACING_MD)
        info_title = QLabel("应用信息", objectName="aboutCardTitle")
        info_layout.addWidget(info_title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(theme.SPACING_MD)
        grid.setVerticalSpacing(theme.SPACING_SM)
        grid.setColumnStretch(1, 1)
        for row, (k, v) in enumerate(
            (
                ("版本", f"v{__version__}"),
                ("技术栈", "PySide2/PySide6 · NumPy · SciPy · matplotlib"),
                ("求解内核", "离线 FEA 求解器"),
                ("开源许可", "MIT License"),
            )
        ):
            key_lbl = QLabel(k, objectName="aboutInfoTitle")
            val_lbl = QLabel(v, objectName="aboutInfoValue")
            val_lbl.setWordWrap(True)
            grid.addWidget(key_lbl, row, 0)
            grid.addWidget(val_lbl, row, 1)
        info_layout.addLayout(grid)
        root.addWidget(info_card)

        lic_card = QFrame(objectName="aboutCard")
        lic_layout = QVBoxLayout(lic_card)
        lic_layout.setContentsMargins(theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG, theme.SPACING_LG)
        lic_layout.setSpacing(theme.SPACING_SM)
        lic_title = QLabel("开源许可", objectName="aboutCardTitle")
        lic_body = QLabel(
            "zylab 采用 MIT License 开源发布，使用 Python 标准库与第三方开源库。\n详见项目根目录 LICENSE 文件。",
            objectName="aboutBody",
        )
        lic_body.setWordWrap(True)
        lic_layout.addWidget(lic_title)
        lic_layout.addWidget(lic_body)
        root.addWidget(lic_card)
        root.addStretch()

        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

    @Slot()
    def open_settings_dialog(self) -> None:
        """弹出设置对话框（SettingsPanel 嵌入 QDialog）."""
        from ..app import apply_settings
        from ..widgets.settings_panel import SettingsPanel

        mw = self._mw
        dlg = QDialog(mw)
        dlg.setWindowTitle("设置")
        dlg.setMinimumSize(560, 480)

        panel = SettingsPanel()
        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(panel)

        def _on_save() -> None:
            cfg = panel.save()
            apply_settings(
                QApplication.instance(),
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
            # 完整刷新：主题 + 图标 + 页面
            self.set_theme(cfg.get("theme", theme.DEFAULT_THEME), persist=False)
            mw.statusBar().showMessage("设置已保存并应用", 3000)
            dlg.accept()

        with contextlib.suppress(RuntimeError, TypeError):
            panel._save_btn.clicked.disconnect()
        panel._save_btn.clicked.connect(_on_save)
        with contextlib.suppress(RuntimeError, TypeError):
            panel._cancel_btn.clicked.disconnect()
        panel._cancel_btn.clicked.connect(dlg.reject)

        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
