"""工作流画布：GitHub Actions 风格竖向流式节点图（QGraphicsView）.

- 节点卡片按层竖向排列（层 = 最长上游链深度），同层分支水平居中分列；
- 卡片显示模块名 + 状态徽标（色点/旋转动画）+ 耗时/错误摘要；
- 连接线为上游卡片底边中点到下游卡片顶边中点的肘形折线；
- 运行中节点由 QTimer 驱动旋转动画（角度递增重绘）。
"""

from __future__ import annotations

from zylab.studio import NodeInstance, NodeState, WorkflowGraph

from .. import theme
from ..qt_compat import (
    QBrush,
    QColor,
    QFont,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QPainter,
    QPainterPath,
    QPen,
    QRectF,
    Qt,
    QTimer,
    Signal,
    mouse_event_pos,
)

__all__ = ["NodeCanvasWidget"]

_CARD_W = 240.0
_CARD_H = 68.0
_GAP_X = 48.0
_GAP_Y = 64.0
_MARGIN = 32.0
_SPIN_STEP_DEG = 12  # 每帧旋转角度（33ms 定时器 ≈ 360°/s）

_STATE_LABELS = {
    NodeState.UNFULFILLED: "待连接",
    NodeState.READY: "待运行",
    NodeState.RUNNING: "运行中",
    NodeState.UP_TO_DATE: "已完成",
    NodeState.FAILED: "失败",
}


def _state_color(state: NodeState) -> QColor:
    """状态徽标颜色（绘制时取当前主题，主题切换即生效）."""
    pal = theme.current_palette()
    color = {
        NodeState.UNFULFILLED: pal.text_disabled,
        NodeState.READY: pal.text_secondary,
        NodeState.RUNNING: pal.primary,
        NodeState.UP_TO_DATE: pal.success_text,
        NodeState.FAILED: pal.danger_text,
    }[state]
    return QColor(color)


def _layers(graph: WorkflowGraph) -> dict[str, int]:
    """按最长上游链深度分层（源节点为 0 层）."""
    memo: dict[str, int] = {}

    def layer(node_id: str) -> int:
        if node_id not in memo:
            upstream = graph.upstream_ids(node_id)
            memo[node_id] = 0 if not upstream else 1 + max(layer(uid) for uid in upstream)
        return memo[node_id]

    for node in graph.nodes():
        layer(node.id)
    return memo


class _NodeCard(QGraphicsItem):
    """节点卡片图元（手绘：圆角矩形 + 状态徽标 + 双行文字）."""

    def __init__(self, node_id: str, name: str, rect: QRectF) -> None:
        """初始化卡片（rect 为场景坐标矩形）."""
        super().__init__()
        self.node_id = node_id
        self._name = name
        self._rect = rect
        self._state = NodeState.READY
        self._detail = ""
        self.selected = False
        self.spin_angle = 0
        self.setPos(rect.topLeft())

    def refresh(self, state: NodeState, detail: str) -> None:
        """更新状态与摘要文字并重绘."""
        self._state = state
        self._detail = detail
        self.update()

    def boundingRect(self) -> QRectF:  # Qt 命名约定
        """卡片矩形（本地坐标）."""
        return QRectF(0.0, 0.0, self._rect.width(), self._rect.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # Qt 命名约定
        """绘制卡片."""
        del option, widget
        pal = theme.current_palette()
        rect = self.boundingRect()

        # 卡片体（选中态主色描边）
        painter.setPen(QPen(QColor(pal.primary if self.selected else pal.border), 2 if self.selected else 1))
        painter.setBrush(QBrush(QColor(pal.bg_muted)))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        # 状态徽标（RUNNING 为旋转弧 + 中心点，其余为实心圆）
        cx, cy = rect.left() + 20.0, rect.center().y()
        color = _state_color(self._state)
        if self._state is NodeState.RUNNING:
            painter.setPen(QPen(color, 2.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(QRectF(cx - 7.0, cy - 7.0, 14.0, 14.0), self.spin_angle * 16, 270 * 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(cx - 2.5, cy - 2.5, 5.0, 5.0))
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(cx - 5.0, cy - 5.0, 10.0, 10.0))

        # 文字：模块名 + 状态摘要
        name_font = QFont(painter.font())
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QColor(pal.text_primary))
        painter.drawText(QRectF(rect.left() + 36, rect.top() + 8, rect.width() - 44, 22), Qt.AlignVCenter, self._name)
        detail_font = QFont(painter.font())
        detail_font.setBold(False)
        painter.setFont(detail_font)
        painter.setPen(color)
        painter.drawText(
            QRectF(rect.left() + 36, rect.top() + 36, rect.width() - 44, 22), Qt.AlignVCenter, self._detail
        )


class NodeCanvasWidget(QGraphicsView):
    """竖向流式工作流画布."""

    #: 单击节点（节点 id）
    node_clicked = Signal(str)
    #: 双击节点（节点 id）
    node_double_clicked = Signal(str)
    #: 右键节点（节点 id, 全局坐标 QPoint）
    node_context_menu = Signal(str, object)

    def __init__(self, parent=None) -> None:
        """初始化空画布（场景/视图配置 + 旋转动画定时器）."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setDragMode(QGraphicsView.NoDrag)
        self._cards: dict[str, _NodeCard] = {}
        self._graph: WorkflowGraph | None = None
        self._selected_id = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_spinner)
        self.refresh_theme()

    # ------------------------------------------------------------------ 图装配

    def set_graph(self, graph: WorkflowGraph) -> None:
        """按工作流图重建场景（卡片 + 连接线）并刷新状态."""
        self._timer.stop()
        self._scene.clear()
        self._cards.clear()
        self._selected_id = ""
        self._graph = graph

        layers = _layers(graph)
        by_layer: dict[int, list[str]] = {}
        for node in graph.nodes():
            by_layer.setdefault(layers[node.id], []).append(node.id)

        positions: dict[str, tuple[float, float]] = {}
        for lv, ids in by_layer.items():
            for col, node_id in enumerate(ids):
                x = (col - (len(ids) - 1) / 2.0) * (_CARD_W + _GAP_X) - _CARD_W / 2.0
                positions[node_id] = (x, lv * (_CARD_H + _GAP_Y))

        # 连接线先建（置于卡片下层）
        for node in graph.nodes():
            for ref in node.inputs.values():
                self._add_edge(positions[ref.partition(".")[0]], positions[node.id])

        for node in graph.nodes():
            x, y = positions[node.id]
            card = _NodeCard(node.id, node.name, QRectF(x, y, _CARD_W, _CARD_H))
            self._scene.addItem(card)
            self._cards[node.id] = card

        n_layers = len(by_layer)
        scene_h = _MARGIN * 2 + n_layers * _CARD_H + max(0, n_layers - 1) * _GAP_Y
        max_cols = max(len(ids) for ids in by_layer.values())
        scene_w = _MARGIN * 2 + max_cols * _CARD_W + max(0, max_cols - 1) * _GAP_X
        self._scene.setSceneRect(-scene_w / 2.0, -_MARGIN, scene_w, scene_h)
        self.refresh_states()

    def _add_edge(self, src_xy: tuple[float, float], dst_xy: tuple[float, float]) -> None:
        """添加肘形连接线（垂直-水平-垂直，上游底边中点 -> 下游顶边中点）."""
        path = QPainterPath()
        x1 = src_xy[0] + _CARD_W / 2.0
        y1 = src_xy[1] + _CARD_H
        x2 = dst_xy[0] + _CARD_W / 2.0
        y2 = dst_xy[1]
        mid_y = (y1 + y2) / 2.0
        path.moveTo(x1, y1)
        path.lineTo(x1, mid_y)
        path.lineTo(x2, mid_y)
        path.lineTo(x2, y2)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor(theme.current_palette().border), 1.5))
        self._scene.addItem(item)

    # ------------------------------------------------------------------ 状态刷新

    def refresh_states(self) -> None:
        """从图同步全部卡片状态；有运行中节点时启动旋转动画."""
        if self._graph is None:
            return
        any_running = False
        for node in self._graph.nodes():
            card = self._cards.get(node.id)
            if card is None:
                continue
            card.refresh(node.state, self._detail_of(node))
            any_running = any_running or node.state is NodeState.RUNNING
        if any_running and not self._timer.isActive():
            self._timer.start()
        elif not any_running and self._timer.isActive():
            self._timer.stop()

    @staticmethod
    def _detail_of(node: NodeInstance) -> str:
        """状态摘要文字（状态名 + 耗时 / 失败原因首行截断）."""
        label = _STATE_LABELS[node.state]
        if node.state is NodeState.UP_TO_DATE and node.elapsed > 0.0:
            return f"{label} · {node.elapsed:.2f}s"
        if node.state is NodeState.FAILED and node.error:
            return f"{label} · {node.error.splitlines()[0][:24]}"
        return label

    def _advance_spinner(self) -> None:
        """推进运行中卡片的旋转角并重绘."""
        if self._graph is None:
            self._timer.stop()
            return
        for node in self._graph.nodes():
            if node.state is NodeState.RUNNING:
                card = self._cards.get(node.id)
                if card is not None:
                    card.spin_angle = (card.spin_angle + _SPIN_STEP_DEG) % 360
                    card.update()

    def refresh_theme(self) -> None:
        """主题切换后重刷背景与卡片."""
        self.setBackgroundBrush(QBrush(QColor(theme.current_palette().bg_app)))
        if self._graph is not None:
            self.refresh_states()

    # ------------------------------------------------------------------ 交互

    @property
    def selected_node_id(self) -> str:
        """当前选中节点 id（未选中为空串）."""
        return self._selected_id

    def select_node(self, node_id: str) -> None:
        """程序化选中节点（不发信号）."""
        self._selected_id = node_id
        for nid, card in self._cards.items():
            card.selected = nid == node_id
            card.update()

    def _card_at(self, pos) -> _NodeCard | None:
        """取视图坐标下的卡片."""
        item = self.itemAt(pos)
        return item if isinstance(item, _NodeCard) else None

    def mousePressEvent(self, event) -> None:  # Qt 命名约定
        """单击选中节点."""
        card = self._card_at(event.pos())
        if card is not None:
            self.select_node(card.node_id)
            self.node_clicked.emit(card.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # Qt 命名约定
        """双击节点."""
        card = self._card_at(mouse_event_pos(event))
        if card is not None:
            self.node_double_clicked.emit(card.node_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:  # Qt 命名约定
        """右键节点弹出上下文菜单（由页面实现菜单内容）."""
        card = self._card_at(event.pos())
        if card is not None:
            self.select_node(card.node_id)
            self.node_context_menu.emit(card.node_id, event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)

    def card_rect(self, node_id: str) -> QRectF | None:
        """节点卡片的场景矩形（测试与定位用）."""
        card = self._cards.get(node_id)
        return card.sceneBoundingRect() if card is not None else None

    def spinner_active(self) -> bool:
        """旋转动画定时器是否运行中（测试用）."""
        return self._timer.isActive()
