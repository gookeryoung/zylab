"""gui.widgets.docs_panel 说明卡测试：渲染/缺图占位/相对路径解析."""

from __future__ import annotations

from pathlib import Path

import pytest

from zylab.flowchart.dsl import load_dsl
from zylab.gui.widgets.docs_panel import DocsPanel, resolve_docs_image

_YAML = """
meta: {id: t.docs, name: 说明参数化计算}
params:
  输入:
    items:
      n: {value: 2, min: 0, max: 10}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "n + 1"}
docs:
  intro:
    text: |
      ### 标题

      正文**加粗**说明。
    image: pic.svg
"""

_NO_DOCS_YAML = """
meta: {id: t.plain, name: 无说明参数化计算}
params:
  输入:
    items:
      n: {value: 2, min: 0, max: 10}
pipeline:
  - id: calc
    type: compute.expr
    params: {expr: "n + 1"}
"""

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="#888"/></svg>'


@pytest.mark.gui
def test_docs_panel_without_docs_hidden(qtbot, tmp_path: Path) -> None:
    """无 docs 声明：整卡隐藏不占布局."""
    path = tmp_path / "plain.yaml"
    path.write_text(_NO_DOCS_YAML, encoding="utf-8")
    panel = DocsPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_template(load_dsl(path))
    assert panel.isHidden()


@pytest.mark.gui
def test_docs_panel_text_and_missing_image_hint(qtbot, tmp_path: Path) -> None:
    """markdown 正文渲染；声明图片缺失时占位提示（作者路径错误可发现）."""
    path = tmp_path / "docs.yaml"
    path.write_text(_YAML, encoding="utf-8")
    panel = DocsPanel()
    qtbot.addWidget(panel)
    panel.resize(400, 600)
    panel.show()
    panel.set_template(load_dsl(path))
    assert not panel.isHidden()
    assert "标题" in panel._text_browser.toPlainText()
    assert panel._text_browser.isVisible()
    assert not panel._image_label.isVisible()
    assert "示意图缺失" in panel._image_hint.text()


@pytest.mark.gui
def test_docs_panel_image_render(qtbot, tmp_path: Path) -> None:
    """图片文件存在时等比渲染；正文始终可见（说明页已独立 tab，无需折叠）."""
    path = tmp_path / "docs.yaml"
    path.write_text(_YAML, encoding="utf-8")
    (tmp_path / "pic.svg").write_text(_SVG, encoding="utf-8")
    panel = DocsPanel()
    qtbot.addWidget(panel)
    panel.resize(400, 600)
    panel.show()
    panel.set_template(load_dsl(path))
    assert panel._pixmap is not None and not panel._pixmap.isNull()
    assert panel._image_label.isVisible()
    assert not panel._image_hint.isVisible()
    assert panel._body.isVisible()  # 正文始终展开


def test_resolve_docs_image(tmp_path: Path) -> None:
    """绝对路径直用；相对路径按模板来源目录解析；空声明或无基准返回 None."""
    assert resolve_docs_image("", "/any/t.yaml") is None
    assert resolve_docs_image("pic.svg", "") is None
    source = tmp_path / "t.yaml"
    source.write_text("{}", encoding="utf-8")
    assert resolve_docs_image("pic.svg", str(source)) == (tmp_path / "pic.svg").resolve()
    absolute = tmp_path / "abs.svg"
    assert resolve_docs_image(str(absolute), str(source)) == absolute


@pytest.mark.gui
def test_refresh_theme_when_text_visible(qtbot, tmp_path: Path) -> None:
    """refresh_theme 在文本可见时重刷 QSS 配色."""
    path = tmp_path / "docs.yaml"
    path.write_text(_YAML, encoding="utf-8")
    panel = DocsPanel()
    qtbot.addWidget(panel)
    panel.resize(400, 600)
    panel.show()
    panel.set_template(load_dsl(path))
    assert panel._text_browser.isVisible()
    panel.refresh_theme()  # 仅验证不抛异常，内部重刷 QSS
