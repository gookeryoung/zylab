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
