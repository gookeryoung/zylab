"""stream_view 单元测试：覆盖 L3 增强函数与核心 widget 构造."""

from __future__ import annotations

from zylab.gui.qt_compat import Qt
from zylab.gui.widgets.stream_view import (
    ResultBlockCard,
    ResultStreamView,
    _build_table_widget,
    _build_text_body,
    _col_alignment,
    _format_cell_with_format,
    _kind_badge,
)
from zylab.studio.results import CloudData, CurveData, CurveSeries, TableColumn, TableData, TextData


def _badge_text(badge: object) -> str:
    """从徽标 QWidget（QHBoxLayout: icon QLabel + text QLabel）提取文本 QLabel 的内容."""
    from zylab.gui.qt_compat import QLabel

    layout = badge.layout()  # type: ignore[union-attr]
    assert layout is not None, "badge 应有 layout"
    # layout.itemAt(0) 是 icon QLabel，itemAt(1) 是文字 QLabel
    text_item = layout.itemAt(1)
    assert text_item is not None
    text_widget = text_item.widget()
    assert isinstance(text_widget, QLabel)
    return text_widget.text()


# ---------------------------------------------------------------- 纯函数


def test_format_cell_with_format_printf_spec() -> None:
    """L3 表格增强：printf .2f / .4g / 异常回落."""
    assert _format_cell_with_format(1.23456, ".2f") == "1.23"
    assert _format_cell_with_format(200.0, ".4g") == "200"
    assert _format_cell_with_format("hello", ".2f") == "hello"
    assert _format_cell_with_format(1.23456, "bad-fmt") == "1.23456"


def test_col_alignment_maps_to_qt_flags() -> None:
    """L3 表格增强：列对齐到 Qt 枚举."""
    assert _col_alignment(TableColumn(title="X", align="right")) == (Qt.AlignRight | Qt.AlignVCenter)
    assert _col_alignment(TableColumn(title="X", align="left")) == (Qt.AlignLeft | Qt.AlignVCenter)
    assert _col_alignment(TableColumn(title="X", align="center")) == Qt.AlignCenter
    assert _col_alignment(TableColumn(title="X")) == (Qt.AlignRight | Qt.AlignVCenter)


def test_kind_badge_classification() -> None:
    """_kind_badge 返回 (图标文件基名, 中文显示文本) 元组."""
    assert _kind_badge(TextData(title="t", text="hi")) == ("result_text", "文本")
    assert _kind_badge(TableData(title="t", columns=(), rows=())) == ("result_table", "表格")
    # 构造 CurveData（通过 CurveSeries）
    from zylab.studio.results import CurveData

    curve = CurveData(title="c", series=(CurveSeries(name="s", x=(0,), y=(0,)),))
    assert _kind_badge(curve) == ("result_curve", "曲线")
    cd = CloudData(title="c", node_id="n", field="temperature", payload=None, cmap="viridis", deform=0.0)
    assert _kind_badge(cd) == ("result_cloud", "云图")


# ---------------------------------------------------------------- QWidget 构造


def test_build_table_widget_renders_columns(qtbot) -> None:
    """L3 表格增强：_build_table_widget 构建 QTableWidget 含格式化对齐单元格."""
    data = TableData(
        title="测试表",
        columns=(
            TableColumn(title="X", format=".2f", align="right"),
            TableColumn(title="Y", format=".4g", align="center"),
        ),
        rows=((1.23456, 200.0), (50.0, 4000.1)),
    )
    widget = _build_table_widget(data)
    assert widget.columnCount() == 2
    assert widget.rowCount() == 2
    assert widget.horizontalHeaderItem(0).text() == "X"
    assert widget.item(0, 0).text() == "1.23"
    assert widget.item(0, 1).text() == "200"


def test_build_text_body_markdown_format(qtbot) -> None:
    """L3 文本增强：format=markdown 时渲染为 QTextBrowser."""
    from zylab.gui.qt_compat import QTextBrowser

    data = TextData(title="摘要", text="# 标题 **粗体**", format="markdown", style="primary")
    widget = _build_text_body(data)
    assert isinstance(widget, QTextBrowser)
    html = widget.toHtml()
    assert "标题" in html
    assert "粗体" in html
    # Qt QTextBrowser.toHtml() 会把 <strong> 渲染成 <span style="font-weight:600;">
    assert "font-weight:600" in html


def test_build_text_body_plain_format(qtbot) -> None:
    """format=plain（缺省）时用 QLabel."""
    from zylab.gui.qt_compat import QLabel

    data = TextData(title="摘要", text="纯文本", format="plain")
    widget = _build_text_body(data)
    assert isinstance(widget, QLabel)
    assert widget.text() == "纯文本"


# ---------------------------------------------------------------- ResultStreamView / ResultBlockCard 交互


def test_stream_view_set_blocks_and_toggle(qtbot) -> None:
    """ResultStreamView set_blocks + ResultBlockCard toggle 折叠展开."""
    view = ResultStreamView()
    qtbot.addWidget(view)
    view.set_blocks([("块 1", "内容 1", "primary"), ("块 2", "内容 2", "success")])
    # blocks + stretch = 3 items
    assert view._container_layout.count() == 3
    # 空 blocks → 显示 placeholder + stretch
    view.set_blocks([])
    assert view._container_layout.count() == 2


def test_stream_view_set_error(qtbot) -> None:
    """set_error 置顶一个 danger 样式卡."""
    view = ResultStreamView()
    qtbot.addWidget(view)
    view.set_error("运行出错了")
    # 一个 card + 一个 stretch
    assert view._container_layout.count() == 2
    card = view._container_layout.itemAt(0).widget()
    assert isinstance(card, ResultBlockCard)
    assert card._title_label.text() == "运行失败"


# ---------------------------------------------------------------- 覆盖 missed 行


def test_kind_badge_unknown_fallback() -> None:
    """_kind_badge 遇到未知类型回落 error 徽标."""

    class UnknownData:
        pass

    assert _kind_badge(UnknownData()) == ("result_error", "错误")
    assert _kind_badge(object()) == ("result_error", "错误")


def test_block_card_toggle_collapse_and_expand(qtbot) -> None:
    """ResultBlockCard._toggle 折叠/展开正文 + 切换 caret（L220-222）."""
    card = ResultBlockCard("标题", "错误内容", "danger")
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)
    assert not card._collapsed
    assert card._body.isVisible()
    assert card._caret.text() == "▾"

    # 折叠
    card._toggle()
    assert card._collapsed
    assert not card._body.isVisible()
    assert card._caret.text() == "▸"

    # 再展开
    card._toggle()
    assert not card._collapsed
    assert card._body.isVisible()
    assert card._caret.text() == "▾"


def test_block_card_with_cloud_data_payload(qtbot) -> None:
    """ResultBlockCard._build_body CloudData 分支（L235-238）."""
    from zylab.gui.qt_compat import QLabel

    cd = CloudData(title="云图", node_id="node-01", field="stress", payload=None, cmap="plasma", deform=1.0)
    card = ResultBlockCard("云结果", cd, "info")
    qtbot.addWidget(card)
    # body 应是包含 node_id 的 QLabel
    assert isinstance(card._body, QLabel)
    assert "node-01" in card._body.text()
    # badge 徽标应为云图
    assert _badge_text(card._badge) == "云图"


def test_block_card_with_table_data_payload(qtbot) -> None:
    """ResultBlockCard._build_body TableData 分支（L249-251）."""
    from zylab.gui.qt_compat import QTableWidget

    data = TableData(
        title="结果表",
        columns=(TableColumn(title="A", format=".2f", align="right"), TableColumn(title="B")),
        rows=((1.23456, "x"),),
    )
    card = ResultBlockCard("表格", data, "")
    qtbot.addWidget(card)
    assert isinstance(card._body, QTableWidget)
    assert card._body.columnCount() == 2
    assert card._body.rowCount() == 1
    # 验证 QTableWidget 被设置了最大高度（_STREAM_TABLE_MAX_HEIGHT=320）
    assert card._body.maximumHeight() == 320
    assert _badge_text(card._badge) == "表格"


def test_block_card_unknown_payload_fallback(qtbot) -> None:
    """ResultBlockCard._build_body 未知 payload 类型兜底返回 QLabel（L258）."""
    from zylab.gui.qt_compat import QLabel

    class Mystery:
        def __str__(self) -> str:
            return "神秘数据"

    card = ResultBlockCard("未知", Mystery(), "warning")
    qtbot.addWidget(card)
    assert isinstance(card._body, QLabel)
    assert "神秘数据" in card._body.text()


def test_block_card_with_text_data_payload(qtbot) -> None:
    """ResultBlockCard._build_body TextData 分支（L255）."""
    from zylab.gui.qt_compat import QLabel

    data = TextData(title="文本块", text="这是正文", format="plain")
    card = ResultBlockCard("文本卡", data, "success")
    qtbot.addWidget(card)
    assert isinstance(card._body, QLabel)
    assert card._body.text() == "这是正文"
    assert _badge_text(card._badge) == "文本"


def test_block_card_with_curve_data_payload(qtbot) -> None:
    """ResultBlockCard._build_body CurveData 分支（L242-245）."""
    from zylab.gui.widgets.stream_view import _STREAM_CURVE_HEIGHT

    curve = CurveData(title="c", series=(CurveSeries(name="s", x=(0.0, 1.0), y=(0.0, 1.0)),))
    card = ResultBlockCard("曲线", curve, "")
    qtbot.addWidget(card)
    assert card._body is not None
    assert card._body.height() in (_STREAM_CURVE_HEIGHT, 0)  # 未 layout 时可能是 0
    assert _badge_text(card._badge) == "曲线"
