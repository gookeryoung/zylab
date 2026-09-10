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
    before = set(id(t) for t in qapp.findChildren(QTranslator))

    monkeypatch.setattr(QLocale, "system", staticmethod(lambda: QLocale(QLocale.English, QLocale.UnitedStates)))

    from zylab.gui.app import _load_qt_translations

    _load_qt_translations(qapp)

    after = set(id(t) for t in qapp.findChildren(QTranslator))
    # 非中文环境不应新增任何翻译器
    assert after - before == set()
