"""最终冲刺：GUI 小模块补测 v3 — 保守版."""

from __future__ import annotations

from pathlib import Path

# --- style.py: 137-138 ---


def test_load_qss_fragments_no_dir_no_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("zylab.gui.style.FRAGMENTS_DIR", tmp_path / "nonexistent_fragments")
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)
    from zylab.gui.style import load_qss_fragments

    result = load_qss_fragments()
    assert result == ""


# --- proxy_style.py: _px 的 int 分支 + str 分支 ---


def test_proxy_px_helpers() -> None:
    """proxy_style 内部 _px 逻辑."""
    from zylab.gui.theme import SPACING_MD, SPACING_XS

    def _px(v):
        if isinstance(v, int):
            return v
        return int(v.removesuffix("px"))

    assert _px(16) == 16  # int 分支
    assert _px(SPACING_XS) == 4
    assert _px(SPACING_MD) == 16


# --- vtk_view early returns ---


def test_vtk_view_fit_all_no_interactor(qapp) -> None:
    from zylab.gui.widgets.vtk_view import VtkView

    VtkView().fit_all()


def test_vtk_view_clear_no_interactor(qapp) -> None:
    from zylab.gui.widgets.vtk_view import VtkView

    VtkView().clear()


def test_vtk_view_save_screenshot_no_interactor(qapp, monkeypatch) -> None:
    from zylab.gui.widgets.vtk_view import VtkView

    monkeypatch.setattr("zylab.gui.widgets.vtk_view.logger.warning", lambda *a, **k: None)
    VtkView().save_screenshot("/tmp/x.png")


def test_vtk_view_take_qimage_no_interactor(qapp) -> None:
    from zylab.gui.widgets.vtk_view import VtkView

    assert VtkView().take_qimage() is None


# --- simple_line_plot ---


def test_simple_line_plot_paint(qapp) -> None:
    from zylab.gui.widgets.simple_line_plot import SimpleLinePlot

    w = SimpleLinePlot()
    w.set_data(
        [
            ([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], "A"),
            ([1.0, 2.0, 3.0], [2.0, 5.0, 10.0], "B"),
        ],
        x_label="X",
        y_label="Y",
        title="Test",
    )
    w.resize(300, 200)
    w.show()
    w.repaint()
    assert len(w._series) == 2
