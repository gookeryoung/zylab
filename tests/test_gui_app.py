"""gui.app 应用装配测试."""

from __future__ import annotations

import pytest

from zylab.gui.app import create_app, load_stylesheet
from zylab.gui.qt_compat import QApplication


@pytest.mark.gui
def test_load_stylesheet_replaces_tokens() -> None:
    """QSS 加载后不应残留令牌占位符."""
    qss = load_stylesheet()
    assert "${" not in qss
    assert "#056574" in qss  # 默认浅色主题主色


@pytest.mark.gui
def test_create_app_reuses_instance(qapp) -> None:
    """已有 QApplication 实例时应复用而非重复创建."""
    app = create_app([])
    assert app is QApplication.instance()
    assert app.styleSheet()  # 样式表已应用


@pytest.mark.gui
def test_register_fonts_loads_builtin(qapp) -> None:
    """内置 DejaVu Sans Mono 注册后应进入应用字体库（重复调用幂等）."""
    from zylab.gui.app import register_fonts
    from zylab.gui.qt_compat import QFontDatabase

    assert "DejaVu Sans Mono" in register_fonts()
    # PySide2 需实例化调用，PySide6 静态/实例均可
    assert "DejaVu Sans Mono" in QFontDatabase().families()


@pytest.mark.gui
def test_load_qt_translations_skips_non_chinese(qapp, monkeypatch) -> None:
    """非中文环境下应跳过 Qt 翻译加载（不安装新翻译器）."""
    from PySide2.QtCore import QLocale, QTranslator

    # 先记录已有翻译器（pytest-qt qapp 可能已加载）
    before = {id(t) for t in qapp.findChildren(QTranslator)}

    monkeypatch.setattr(QLocale, "system", staticmethod(lambda: QLocale(QLocale.English, QLocale.UnitedStates)))

    from zylab.gui.app import _load_qt_translations

    _load_qt_translations(qapp)

    after = {id(t) for t in qapp.findChildren(QTranslator)}
    # 非中文环境不应新增任何翻译器
    assert after - before == set()


@pytest.mark.gui
def test_register_fonts_handles_add_failure(monkeypatch) -> None:
    """字体注册失败时应返回空列表且不抛异常（覆盖 if font_id < 0 分支）."""
    from zylab.gui.app import register_fonts
    from zylab.gui.qt_compat import QFontDatabase

    def _fail_add(_path):
        return -1

    monkeypatch.setattr(QFontDatabase, "addApplicationFont", _fail_add)
    assert register_fonts() == []


class TestGuiMain:
    """gui/__main__.py 入口转发."""

    def test_main_forwards_to_app(self, monkeypatch) -> None:
        """__main__.main() 应转发到 gui.app.main()."""
        from unittest.mock import MagicMock

        mock_app_main = MagicMock(return_value=0)
        monkeypatch.setattr("zylab.gui.app.main", mock_app_main)

        from zylab.gui.__main__ import main

        assert main() == 0
        mock_app_main.assert_called_once()


class TestQtCompatWrappers:
    """qt_compat 的版本兼容包装函数."""

    def test_exec_app_pyside6_path(self) -> None:
        """exec_app 应调用 app.exec（PySide6 有 exec 属性）."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import exec_app

        app = MagicMock(spec=[])  # 用空 spec 控制 hasattr
        app.exec = MagicMock(return_value=0)
        assert exec_app(app) == 0
        app.exec.assert_called_once()

    def test_exec_app_pyside2_path(self) -> None:
        """exec_app 在无 exec 时回退到 app.exec_（PySide2 兼容分支）."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import exec_app

        app = MagicMock(spec=[])  # 空 spec → hasattr 全部 False
        app.exec_ = MagicMock(return_value=1)
        assert exec_app(app) == 1
        app.exec_.assert_called_once()

    def test_exec_menu_path(self) -> None:
        """exec_menu 应调用 menu.exec 或 menu.exec_ 并返回结果."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import exec_menu

        # PySide6 路径
        menu6 = MagicMock(spec=[])
        menu6.exec = MagicMock(return_value="selected")
        assert exec_menu(menu6, MagicMock()) == "selected"

        # PySide2 路径
        menu2 = MagicMock(spec=[])
        menu2.exec_ = MagicMock(return_value="selected")
        pos = MagicMock()
        exec_menu(menu2, pos)
        menu2.exec_.assert_called_once_with(pos)

    def test_exec_dialog_path(self) -> None:
        """exec_dialog 应调用 dialog.exec 或 dialog.exec_ 并返回 int."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import exec_dialog

        # PySide6 路径
        dlg6 = MagicMock(spec=[])
        dlg6.exec = MagicMock(return_value=1)
        assert exec_dialog(dlg6) == 1

        # PySide2 路径
        dlg2 = MagicMock(spec=[])
        dlg2.exec_ = MagicMock(return_value=0)
        assert exec_dialog(dlg2) == 0

    def test_mouse_event_pos_pyside6_path(self) -> None:
        """mouse_event_pos 在 Qt6 环境下取 event.position() 并转 QPoint."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import mouse_event_pos

        event = MagicMock(spec=[])  # 空 spec → hasattr 全部 False
        qpointf = MagicMock()
        qpointf.toPoint.return_value = MagicMock()
        event.position = MagicMock(return_value=qpointf)
        result = mouse_event_pos(event)
        assert result is qpointf.toPoint.return_value

    def test_mouse_event_pos_pyside2_path(self) -> None:
        """mouse_event_pos 在 Qt5 环境下取 event.pos() 直接返回."""
        from unittest.mock import MagicMock

        from zylab.gui.qt_compat import mouse_event_pos

        event = MagicMock(spec=[])  # 空 spec → 无 position 属性
        qpoint = MagicMock(spec=[])  # 空 spec → 无 toPoint 属性
        event.pos = MagicMock(return_value=qpoint)
        assert mouse_event_pos(event) is qpoint
