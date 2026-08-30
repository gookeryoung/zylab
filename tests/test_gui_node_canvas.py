"""gui.widgets.node_canvas 工作流画布测试：装配、布局、状态刷新、交互信号."""

from __future__ import annotations

import pytest

from zylab.gui import qt_compat
from zylab.gui.qt_compat import QContextMenuEvent, QEvent, QKeyEvent, QMouseEvent, QPointF, Qt
from zylab.gui.widgets.node_canvas import _ARROW_LEN, NodeCanvasWidget, _NodeCard
from zylab.studio import NodeState, Template, WorkflowGraph


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
def test_node_card_fitted_font(qtbot) -> None:
    """长节点名自适应：短名保持原字号，超长名逐级缩小或省略号截断."""
    from zylab.gui.qt_compat import QFont, QFontMetrics

    base = QFont()
    base.setPointSize(12)
    font, text = _NodeCard._fitted_font(base, "求解", 1000.0)
    assert text == "求解"
    assert font.pointSize() == 12  # 短名不缩字号
    font2, text2 = _NodeCard._fitted_font(base, "通用加热板电热耦合分析超长环节名称测试", 40.0)
    assert font2.pointSize() <= 12
    assert QFontMetrics(font2).horizontalAdvance(text2) <= 40.0  # 最终必不超宽


@pytest.mark.gui
def test_node_card_paint_with_pill(qtbot) -> None:
    """单元渲染冒烟：长名 + 圆角状态标签（READY/UP_TO_DATE pill）离屏绘制无异常."""
    from zylab.gui.qt_compat import QPainter, QPixmap

    canvas, _graph = _canvas_with_graph(qtbot)
    canvas._cards["model"].refresh(NodeState.UP_TO_DATE, "已完成 · 0.5s")
    canvas._cards["static"].refresh(NodeState.READY, "待运行 · 静力分析")
    pixmap = QPixmap(600, 400)
    painter = QPainter(pixmap)
    canvas.render(painter)  # 触发全部图元 paint（含 pill 与自适应字号路径）
    painter.end()
    assert not pixmap.isNull()


@pytest.mark.gui
def test_canvas_builds_cards_and_edges(qtbot) -> None:
    """组合图装配：3 单元 + 2 连接线 + 1 组合框；横向分层布局 model 在左、static/modal 同列并列."""
    canvas, _graph = _canvas_with_graph(qtbot)
    items = canvas._scene.items()
    cards = [item for item in items if hasattr(item, "node_id")]
    assert len(cards) == 3
    assert len(items) - len(cards) == 3  # 两条连接线 + 一个组合框
    model_rect = canvas.card_rect("model")
    static_rect = canvas.card_rect("static")
    modal_rect = canvas.card_rect("modal")
    assert model_rect is not None and static_rect is not None and modal_rect is not None
    assert model_rect.left() < static_rect.left()  # model 在左（0 层）
    assert static_rect.left() == modal_rect.left()  # 同层同列
    assert static_rect.top() != modal_rect.top()  # 并列纵向堆叠
    # 组合框覆盖全部单元
    frame = canvas.frame_rect()
    assert frame is not None and frame.contains(model_rect) and frame.contains(modal_rect)


@pytest.mark.gui
def test_edge_geometry_arrow_separate(qtbot) -> None:
    """连线几何：折线止于箭头底部（不穿过）、箭头为独立闭合三角、拐角圆角."""
    canvas, _graph = _canvas_with_graph(qtbot)
    for edge, dst_id in canvas._edges:
        assert dst_id in ("static", "modal")
        line_rect = edge._line.boundingRect()
        arrow_rect = edge._arrow.boundingRect()
        # 箭头右端即目标单元左边（贴合），折线右端止于箭头底部（不重叠穿越）
        assert arrow_rect.right() <= line_rect.right() + _ARROW_LEN + 1.0
        assert canvas.card_rect(dst_id).left() - arrow_rect.right() < 1.0  # 箭头贴目标
    # 跨行边为肘形折线（多段 + 圆角曲线元素），同行边为两段直线
    element_counts = [edge._line.elementCount() for edge, _dst in canvas._edges]
    assert min(element_counts) == 2  # model->static 同高直线
    assert max(element_counts) > 3  # model->modal 肘形圆角
    # 箭头路径闭合且面积非零（实心三角）
    arrow = canvas._edges[0][0]._arrow
    assert arrow.elementCount() == 4  # 三点 + closeSubpath
    assert arrow.boundingRect().width() == pytest.approx(_ARROW_LEN)


@pytest.mark.gui
def test_edge_flow_animation(qtbot) -> None:
    """连线流动动画：下游运行中时进入流动态，相位随定时器推进；结束回落."""
    canvas, graph = _canvas_with_graph(qtbot)
    model_to_static = next(e for e, dst in canvas._edges if dst == "static")
    assert not model_to_static.flowing
    graph.mark_running("static")
    canvas.refresh_states()
    assert model_to_static.flowing
    phase0 = model_to_static._phase
    canvas._advance_spinner()
    assert model_to_static._phase > phase0  # 相位推进（虚线流动）
    graph.mark_result("static", result=object(), elapsed=0.1)
    canvas.refresh_states()
    assert not model_to_static.flowing


@pytest.mark.gui
def test_state_badge_mapping(qtbot) -> None:
    """检查徽标映射：问号=待连接、红叉=失败、绿对勾=就绪/已完成."""
    from zylab.gui.widgets.node_canvas import _state_badge

    assert _state_badge(NodeState.UNFULFILLED)[0] == "question"
    assert _state_badge(NodeState.FAILED)[0] == "cross"
    assert _state_badge(NodeState.READY)[0] == "check"
    assert _state_badge(NodeState.UP_TO_DATE)[0] == "check"


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
    """重复装配：旧单元清除，新图生效."""
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


@pytest.mark.gui
def test_select_all_programmatic_and_signal(qtbot) -> None:
    """程序化全选：全部单元高亮 + 标题栏高亮 + 发 all_selected 信号."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[bool] = []
    canvas.all_selected.connect(lambda: received.append(True))
    canvas.select_all()
    assert received == [True]
    assert canvas.selected_all
    assert canvas.selected_node_id == ""
    assert all(card.selected for card in canvas._cards.values())
    assert canvas._frame.selected_all
    # 单选节点后退出全选态
    canvas.select_node("model")
    assert not canvas.selected_all
    assert canvas._cards["model"].selected
    assert not canvas._cards["static"].selected


@pytest.mark.gui
def test_ctrl_a_selects_all(qtbot) -> None:
    """Ctrl+A 触发全选."""
    canvas, _graph = _canvas_with_graph(qtbot)
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.ControlModifier)
    canvas.keyPressEvent(event)
    assert canvas.selected_all


@pytest.mark.gui
def test_click_header_selects_all(qtbot) -> None:
    """点击组合框标题栏触发全选（发信号）."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[bool] = []
    canvas.all_selected.connect(lambda: received.append(True))
    header_center = QPointF(canvas._frame.header_rect().center())
    pos = canvas.mapFromScene(header_center)
    qtbot.mouseClick(canvas.viewport(), Qt.LeftButton, pos=pos)
    assert received == [True]
    assert canvas.selected_all


@pytest.mark.gui
def test_background_context_menu_emits(qtbot) -> None:
    """右键空白区域发 background_context_menu 信号（全局坐标）."""
    canvas, _graph = _canvas_with_graph(qtbot)
    received: list[object] = []
    canvas.background_context_menu.connect(received.append)
    # 场景右下角空白（组合框外）
    pos = canvas.mapFromScene(canvas.sceneRect().bottomRight() - QPointF(2.0, 2.0))
    global_pos = canvas.viewport().mapToGlobal(pos)
    event = QContextMenuEvent(QContextMenuEvent.Mouse, pos, global_pos)
    canvas.contextMenuEvent(event)
    assert len(received) == 1
