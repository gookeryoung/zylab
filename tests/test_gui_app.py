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
