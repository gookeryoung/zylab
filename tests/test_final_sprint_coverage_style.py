from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# GUI: style.py —— fallback 兜底 + svg_tokens branch
# ---------------------------------------------------------------------------


class TestStyleFallbackAndSvgTokens:
    """style.load_qss_fragments 和 load_stylesheet 的 miss 分支."""

    def test_load_qss_fragments_all_missing(self) -> None:
        from pathlib import Path

        from zylab.gui import style as style_mod

        fake_dir = Path("/tmp/_zylab_nonexistent_fragments_zzz")
        with patch.object(style_mod, "FRAGMENTS_DIR", fake_dir), patch.object(Path, "is_file", return_value=False):
            text = style_mod.load_qss_fragments()
        assert text == ""

    def test_load_stylesheet_with_svg_tokens(self) -> None:
        from zylab.gui import style as style_mod

        with patch.object(
            style_mod,
            "load_qss_fragments",
            return_value="* { background: $QSS_PRIMARY; icon: $MY_ICON; }",
        ):
            svg_tokens = {"MY_ICON": "/tmp/icon.svg"}
            qss = style_mod.load_stylesheet(svg_tokens=svg_tokens)
        assert "$MY_ICON" not in qss
        assert "/tmp/icon.svg" in qss


# ---------------------------------------------------------------------------
# GUI: proxy_style.py —— pixelMetric 全分支
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestProxyStylePixelMetric:
    """ProxyStyle.pixelMetric 每个 QStyle.PM_* 枚举分支都要走一遍."""

    def test_pm_all_branches(self, qtbot) -> None:
        from PySide2.QtWidgets import QApplication, QStyle

        from zylab.gui.proxy_style import ProxyStyle
        from zylab.gui.theme import SPACING_MD, SPACING_SM, SPACING_XS

        app = QApplication.instance()
        ps = ProxyStyle(app.style())

        assert ps.pixelMetric(QStyle.PM_FocusFrameHMargin) == SPACING_XS
        assert ps.pixelMetric(QStyle.PM_FocusFrameVMargin) == SPACING_XS
        assert ps.pixelMetric(QStyle.PM_IndicatorWidth) == 16
        assert ps.pixelMetric(QStyle.PM_IndicatorHeight) == 16
        assert ps.pixelMetric(QStyle.PM_ExclusiveIndicatorWidth) == 16
        assert ps.pixelMetric(QStyle.PM_ExclusiveIndicatorHeight) == 16
        assert ps.pixelMetric(QStyle.PM_TabBarTabHSpace) == SPACING_MD
        assert ps.pixelMetric(QStyle.PM_TabBarTabVSpace) == SPACING_SM
        assert ps.pixelMetric(QStyle.PM_SmallIconSize) == 16

        from zylab.gui import theme as theme_mod

        with patch.object(theme_mod, "SPACING_MD", "32px"):
            val = ps.pixelMetric(QStyle.PM_TabBarTabHSpace)
            assert val == 32


# ---------------------------------------------------------------------------
# GUI: app.py —— set_root_level ValueError + _write_theme_svgs close_template branch
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestAppThemeSvgAndLogLevel:
    """app.py 两个 miss 分支."""

    def test_set_root_level_valueerror_caught(self, qtbot) -> None:
        from PySide2.QtWidgets import QApplication

        from zylab.gui import app as app_mod

        app = QApplication.instance()
        # apply_settings 后续所有 Qt 操作全部 patch 掉，只保留 set_root_level try/except
        with (
            patch.object(app_mod, "set_root_level", side_effect=ValueError("bad level")),
            patch.object(app, "setPalette"),
            patch.object(app, "setStyleSheet"),
            patch.object(app_mod, "_ensure_proxy_style"),
            patch.object(app_mod, "load_stylesheet", return_value=""),
            patch.object(app_mod._style_layer, "build_qpalette", return_value=MagicMock()),
        ):
            app_mod.apply_settings(app, log_level="DEBUG")

    def test_write_theme_svgs_close_template_branch(self, qtbot) -> None:
        from zylab.gui import app as app_mod
        from zylab.gui import theme

        fake_template = '<svg width="16" height="16">X</svg>'
        with patch.object(app_mod, "_load_qrc_svg", return_value=fake_template):
            tokens = app_mod._write_theme_svgs(theme.current_palette())
        assert "QSS_CLOSE_ICON" in tokens
        assert "QSS_CLOSE_ICON_HOVER" in tokens
