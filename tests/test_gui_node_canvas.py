"""gui.widgets.node_canvas 工作流画布测试：装配、布局、状态刷新、交互信号."""

from __future__ import annotations

import pytest

from zylab.gui import qt_compat
from zylab.gui.qt_compat import QContextMenuEvent, QEvent, QMouseEvent, QPointF, Qt
from zylab.gui.widgets.node_canvas import NodeCanvasWidget
from zylab.studio import Template, WorkflowGraph


def _dbl_click_event(pos, widget=None) -> QMouseEvent:
    """构造双击鼠标事件（Qt6 用 local+global+device 新签名，Qt5 旧签名保留）."""
    if qt_compat.QT_API == "pyside6":
        from PySide6.QtGui import QPointingDevice

        global_pos = widget.mapToGlobal(pos) if widget is not None else QPointF(pos)
        return QMouseEvent(
            QEvent.MouseButtonDblClick,
            QPointF(pos),
            QPointF(global_pos),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
            QPointingDevice.primaryPointingDevice(),
        )
    return QMouseEvent(QEvent.MouseButtonDblClick, QPointF(pos), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


_COMBO = {
    "id": "t.combo",
    "name": "组合",
    "nodes": [
        {"id": "model", "type": "example.cantilever_q4", "params": {"nx": 4, "ny": 2}},
        {"id": "static", "type": "analysis.static", "inputs": {"model": "model.model"}},
        {"id": "modal", "type": "analysis.modal", "inputs": {"model": "model.model"}},
    ],
}


def _combo_graph() -> WorkflowGraph:
    """构造三层组合图（model 第 0 层，static/modal 第 1 层并列）."""
    return WorkflowGraph(Template.from_dict(_COMBO))


def _canvas_with_graph(qtbot) -> tuple[NodeCanvasWidget, WorkflowGraph]:
    """构建已装配组合图的画布."""
    canvas = NodeCanvasWidget()
    qtbot.addWidget(canvas)
    canvas.resize(800, 600)
    graph = _combo_graph()
    canvas.set_graph(graph)
    canvas.show()
    return canvas, graph


@pytest.mark.gui
def test_canvas_builds_cards_and_edges(qtbot) -> None:
    """组合图装配：3 卡片 + 2 连接线；分层布局 model 在上、static/modal 同层并列."""
    canvas, _graph = _canvas_with_graph(qtbot)
    items = canvas._scene.items()
    cards = [item for item in items if hasattr(item, "node_id")]
    assert len(cards) == 3
    assert len(items) - len(cards) == 2  # 两条连接线
    model_rect = canvas.card_rect("model")
    static_rect = canvas.card_rect("static")
    modal_rect = canvas.card_rect("modal")
    assert model_rect is not None and static_rect is not None and modal_rect is not None
    assert model_rect.top() < static_rect.top()  # model 在上层
    assert static_rect.top() == modal_rect.top()  # 同层
    assert static_rect.left() != modal_rect.left()  # 并列


@pytest.mark.gui
def test_state_refresh_and_spinner(qtbot) -> None:
    """状态迁移驱动徽标与旋转动画定时器启停."""
    canvas, graph = _canvas_with_graph(qtbot)
    assert not canvas.spinner_active()
    graph.mark_running("model")
    canvas.refresh_states()
    assert canvas.spinner_active()
    graph.mark_result("model", result=object(), elapsed=0.25)
    canvas.refresh_states()
    assert not canvas.spinner_active()


@pytest.mark.gui
def test_spinner_advances_angle(qtbot) -> None:
    """定时器推进运行中卡片旋转角."""
    canvas, graph = _canvas_with_graph(qtbot)
    graph.mark_running("static")
    canvas.refresh_states()
    canvas._advance_spinner()
    card = canvas._cards["static"]
    assert card.spin_angle == 12
    canvas.refresh_states()


@pytest.mark.gui
def test_detail_text_variants(qtbot) -> None:
    """摘要文字：就绪/耗时/失败三态."""
    canvas, graph = _canvas_with_graph(qtbot)
    graph.mark_result("model", result=object(), elapsed=0.5)
    graph.mark_failed("static", "SolverError: 矩阵奇异\n第二行")
    canvas.refresh_states()
    assert canvas._detail_of(graph.node("model")) == "已完成 · 0.50s"
    assert canvas._detail_of(graph.node("static")) == "失败 · SolverError: 矩阵奇异"
    assert canvas._detail_of(graph.node("modal")) == "待运行"


@pytest.mark.gui
def test_click_selects_node(qtbot) -> None:
    """单击卡片选中并发 node_clicked 信号."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[str] = []
    canvas.node_clicked.connect(received.append)
    center = canvas.mapFromScene(canvas.card_rect("model").center())
    qtbot.mouseClick(canvas.viewport(), Qt.LeftButton, pos=center)
    assert received == ["model"]
    assert canvas.selected_node_id == "model"
    assert canvas._cards["model"].selected


@pytest.mark.gui
def test_double_click_emits(qtbot) -> None:
    """双击卡片发 node_double_clicked 信号."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[str] = []
    canvas.node_double_clicked.connect(received.append)
    center = canvas.mapFromScene(canvas.card_rect("modal").center())
    event = _dbl_click_event(center, canvas)
    canvas.mouseDoubleClickEvent(event)
    assert received == ["modal"]


@pytest.mark.gui
def test_context_menu_emits(qtbot) -> None:
    """右键卡片发 node_context_menu 信号并选中."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[tuple[str, object]] = []
    canvas.node_context_menu.connect(lambda nid, pos: received.append((nid, pos)))
    pos = canvas.mapFromScene(canvas.card_rect("static").center())
    global_pos = canvas.viewport().mapToGlobal(pos)
    event = QContextMenuEvent(QContextMenuEvent.Mouse, pos, global_pos)
    canvas.contextMenuEvent(event)
    assert len(received) == 1
    assert received[0][0] == "static"
    assert canvas.selected_node_id == "static"


@pytest.mark.gui
def test_click_empty_keeps_selection(qtbot) -> None:
    """点击空白区域不改变选中."""
    canvas, _graph = _canvas_with_graph(qtbot)
    canvas.select_node("model")
    qtbot.mouseClick(canvas.viewport(), Qt.LeftButton, pos=canvas.rect().bottomRight())
    assert canvas.selected_node_id == "model"


@pytest.mark.gui
def test_card_rect_unknown_returns_none(qtbot) -> None:
    """未知节点 id 返回 None."""
    canvas, _graph = _canvas_with_graph(qtbot)
    assert canvas.card_rect("ghost") is None


@pytest.mark.gui
def test_refresh_without_graph_is_noop(qtbot) -> None:
    """未装配图时刷新为空操作."""
    canvas = NodeCanvasWidget()
    qtbot.addWidget(canvas)
    canvas.refresh_states()  # 不抛异常
    canvas._advance_spinner()
    assert not canvas.spinner_active()


@pytest.mark.gui
def test_paint_all_states(qtbot) -> None:
    """抓取视口像素触发全部状态分支的绘制（RUNNING 旋转弧/已完成耗时/失败）."""
    canvas, graph = _canvas_with_graph(qtbot)
    graph.mark_result("model", result=object(), elapsed=0.5)
    graph.mark_running("static")
    graph.mark_failed("modal", "SolverError: 矩阵奇异")
    canvas.refresh_states()
    pixmap = canvas.grab()  # 强制离屏重绘
    assert not pixmap.isNull()
    graph.mark_reset("static")
    canvas.refresh_states()
    assert not canvas.spinner_active()


@pytest.mark.gui
def test_set_graph_rebuilds_scene(qtbot) -> None:
    """重复装配：旧卡片清除，新图生效."""
    canvas, _graph = _canvas_with_graph(qtbot)
    assert canvas.card_rect("modal") is not None
    single = WorkflowGraph(
        Template.from_dict(
            {
                "id": "t.single",
                "name": "单链",
                "nodes": [
                    {"id": "model", "type": "example.truss2_two_bar"},
                    {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
                ],
            }
        )
    )
    canvas.set_graph(single)
    assert canvas.card_rect("modal") is None
    assert canvas.card_rect("solve") is not None
    assert canvas.selected_node_id == ""
