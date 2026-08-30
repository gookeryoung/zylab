"""工作流画布：ANSYS Workbench 风格系统图（QGraphicsView）.

- 整个模板呈现为一个「系统组合框」：外框 + 模板名标题栏（点击全选）；
- 每个环节为框内单元（圆角矩形：名称 + 状态摘要 + 检查徽标），按层从左到右
  排列（层 = 最长上游链深度），同层分支纵向堆叠；
- 检查徽标（单元右侧）：问号 = 输入未连接（数据未提供）、红叉 = 检查失败
  （运行出错）、绿对勾 = 参数已就绪/已完成，运行中为旋转动画；
- 连接线为单元右边中点到下游单元左边中点的肘形折线（带箭头）；
- 交互：单击选中单元、双击运行、右键菜单（单元/空白）、Ctrl+A 或点击
  标题栏全选（参数面板显示全部参数）。
"""

from __future__ import annotations

from zylab.studio import NodeInstance, NodeState, WorkflowGraph

from .. import theme
from ..icons import tinted_pixmap
from ..qt_compat import (
    QBrush,
    QColor,
    QFont,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
    mouse_event_pos,
)

__all__ = ["NodeCanvasWidget"]

_CELL_W = 190.0
_CELL_H = 64.0
_GAP_X = 56.0
_GAP_Y = 28.0
_MARGIN = 24.0  # 组合框内边距
_HEADER_H = 34.0  # 标题栏高度
_FRAME_PAD = 12.0  # 组合框外框与单元区间距
_BADGE = 20  # 检查徽标边长（像素）
_SPIN_STEP_DEG = 12  # 每帧旋转角度（33ms 定时器 ≈ 360°/s）

_STATE_LABELS = {
    NodeState.UNFULFILLED: "待连接",
    NodeState.READY: "待运行",
    NodeState.RUNNING: "运行中",
    NodeState.UP_TO_DATE: "已完成",
    NodeState.FAILED: "失败",
}


def _state_badge(state: NodeState) -> tuple[str, str]:
    """检查徽标（图标基名, 主题语义色）.

    问号 = 输入未连接（上游数据未提供）；红叉 = 检查失败（运行出错）；
    绿对勾 = 参数已提供且通过检查（待运行/已完成）；运行中不显示静态徽标。
    """
    pal = theme.current_palette()
    if state is NodeState.UNFULFILLED:
        return "question", pal.warning_text
    if state is NodeState.FAILED:
        return "cross", pal.danger_text
    return "check", pal.success_text


def _state_color(state: NodeState) -> QColor:
    """状态文字颜色（摘要行，主题切换即生效）."""
    pal = theme.current_palette()
    color = {
        NodeState.UNFULFILLED: pal.warning_text,
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


class _SystemFrame(QGraphicsItem):
    """系统组合框图元：外框 + 模板名标题栏（点击标题栏触发全选）."""

    def __init__(self, title: str, rect: QRectF) -> None:
        """初始化组合框（rect 为场景坐标矩形）."""
        super().__init__()
        self.title = title
        self._rect = rect
        self.selected_all = False
        self.setPos(rect.topLeft())
        self.setZValue(-1.0)  # 置于单元与连线之下

    def header_rect(self) -> QRectF:
        """标题栏矩形（本地坐标）."""
        return QRectF(0.0, 0.0, self._rect.width(), _HEADER_H)

    def boundingRect(self) -> QRectF:  # Qt 命名约定
        """组合框矩形（本地坐标）."""
        return QRectF(0.0, 0.0, self._rect.width(), self._rect.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # Qt 命名约定
        """绘制组合框：标题栏（全选态主色高亮）+ 外框."""
        del option, widget
        pal = theme.current_palette()
        rect = self.boundingRect()
        # 外框
        painter.setPen(
            QPen(QColor(pal.border_strong if self.selected_all else pal.border), 2 if self.selected_all else 1)
        )
        painter.setBrush(QBrush(QColor(pal.bg_app)))
        painter.drawRoundedRect(rect, 10.0, 10.0)
        # 标题栏
        header = self.header_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(pal.primary if self.selected_all else pal.bg_muted)))
        painter.drawRoundedRect(header, 10.0, 10.0)
        painter.setClipRect(header.intersected(rect))  # 圆角仅保留顶部
        painter.drawRect(header)
        painter.setClipping(False)
        # 标题文字
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(pal.text_on_primary if self.selected_all else pal.text_primary))
        painter.drawText(header.adjusted(12.0, 0.0, -12.0, 0.0), Qt.AlignVCenter | Qt.AlignLeft, self.title)


class _NodeCard(QGraphicsItem):
    """环节单元图元（手绘：圆角矩形 + 检查徽标 + 双行文字）."""

    def __init__(self, node_id: str, name: str, rect: QRectF) -> None:
        """初始化单元（rect 为场景坐标矩形）."""
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
        """单元矩形（本地坐标）."""
        return QRectF(0.0, 0.0, self._rect.width(), self._rect.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # Qt 命名约定
        """绘制单元."""
        del option, widget
        pal = theme.current_palette()
        rect = self.boundingRect()

        # 单元体（选中态主色描边）
        painter.setPen(QPen(QColor(pal.primary if self.selected else pal.border), 2 if self.selected else 1))
        painter.setBrush(QBrush(QColor(pal.bg_muted)))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        # 文字区（徽标右侧）
        text_rect = QRectF(rect.left() + 10, rect.top() + 6, rect.width() - _BADGE - 24, 22)
        name_font = QFont(painter.font())
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QColor(pal.text_primary))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._name)
        detail_font = QFont(painter.font())
        detail_font.setBold(False)
        painter.setFont(detail_font)
        color = _state_color(self._state)
        painter.setPen(color)
        painter.drawText(
            QRectF(text_rect.left(), rect.top() + 32, text_rect.width(), 22),
            Qt.AlignVCenter | Qt.AlignLeft,
            self._detail,
        )

        # 检查徽标（右侧居中）：问号/红叉/绿对勾，运行中为旋转弧
        bx = rect.right() - _BADGE - 10
        by = rect.center().y() - _BADGE / 2
        if self._state is NodeState.RUNNING:
            cx, cy = bx + _BADGE / 2, by + _BADGE / 2
            painter.setPen(QPen(color, 2.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(QRectF(cx - 8.0, cy - 8.0, 16.0, 16.0), self.spin_angle * 16, 270 * 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(cx - 2.5, cy - 2.5, 5.0, 5.0))
        else:
            icon_name, tint = _state_badge(self._state)
            pixmap = tinted_pixmap(icon_name, tint, _BADGE)
            if not pixmap.isNull():
                painter.drawPixmap(int(bx), int(by), pixmap)


class NodeCanvasWidget(QGraphicsView):
    """Workbench 风格系统画布."""

    #: 单击单元（节点 id）
    node_clicked = Signal(str)
    #: 双击单元（节点 id）
    node_double_clicked = Signal(str)
    #: 右键单元（节点 id, 全局坐标 QPoint）
    node_context_menu = Signal(str, object)
    #: 全选（Ctrl+A 或点击组合框标题栏）—— 参数面板显示全部参数
    all_selected = Signal()
    #: 右键空白区域（全局坐标 QPoint）—— 页面据此弹出「运行全部」等菜单
    background_context_menu = Signal(object)

    def __init__(self, parent=None) -> None:
        """初始化空画布（场景/视图配置 + 旋转动画定时器）."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setDragMode(QGraphicsView.NoDrag)
        self._cards: dict[str, _NodeCard] = {}
        self._frame: _SystemFrame | None = None
        self._graph: WorkflowGraph | None = None
        self._selected_id = ""
        self._selected_all = False
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_spinner)
        self.refresh_theme()

    # ------------------------------------------------------------------ 图装配

    def set_graph(self, graph: WorkflowGraph) -> None:
        """按工作流图重建场景（组合框 + 单元 + 连接线）并刷新状态."""
        self._timer.stop()
        self._scene.clear()
        self._cards.clear()
        self._frame = None
        self._selected_id = ""
        self._selected_all = False
        self._graph = graph

        layers = _layers(graph)
        by_layer: dict[int, list[str]] = {}
        for node in graph.nodes():
            by_layer.setdefault(layers[node.id], []).append(node.id)

        # 单元坐标：层 -> 列（x），同层纵向堆叠（y）
        positions: dict[str, tuple[float, float]] = {}
        for lv in sorted(by_layer):
            ids = by_layer[lv]
            for row, node_id in enumerate(ids):
                x = _FRAME_PAD + lv * (_CELL_W + _GAP_X)
                y = _HEADER_H + _FRAME_PAD + row * (_CELL_H + _GAP_Y)
                positions[node_id] = (x, y)

        # 组合框整体尺寸
        n_layers = max(1, len(by_layer))
        max_rows = max(1, *(len(ids) for ids in by_layer.values()))
        frame_w = _FRAME_PAD * 2 + n_layers * _CELL_W + (n_layers - 1) * _GAP_X
        frame_h = _HEADER_H + _FRAME_PAD * 2 + max_rows * _CELL_H + (max_rows - 1) * _GAP_Y
        self._frame = _SystemFrame(graph.template.name, QRectF(0.0, 0.0, frame_w, frame_h))
        self._scene.addItem(self._frame)

        # 连接线（置于单元下层）
        for node in graph.nodes():
            for ref in node.inputs.values():
                self._add_edge(positions[ref.partition(".")[0]], positions[node.id])

        for node in graph.nodes():
            x, y = positions[node.id]
            card = _NodeCard(node.id, node.name, QRectF(x, y, _CELL_W, _CELL_H))
            self._scene.addItem(card)
            self._cards[node.id] = card

        scene_rect = QRectF(-_MARGIN, -_MARGIN, frame_w + _MARGIN * 2, frame_h + _MARGIN * 2)
        self._scene.setSceneRect(scene_rect)
        self.refresh_states()

    def _add_edge(self, src_xy: tuple[float, float], dst_xy: tuple[float, float]) -> None:
        """添加肘形连接线（水平-垂直-水平，源右边中点 -> 目标左边中点，带箭头）."""
        x1 = src_xy[0] + _CELL_W
        y1 = src_xy[1] + _CELL_H / 2.0
        x2 = dst_xy[0]
        y2 = dst_xy[1] + _CELL_H / 2.0
        path = QPainterPath()
        path.moveTo(x1, y1)
        if abs(y2 - y1) < 0.5:
            path.lineTo(x2 - 8, y1)
        else:
            mid_x = (x1 + x2) / 2.0
            path.lineTo(mid_x, y1)
            path.lineTo(mid_x, y2)
            path.lineTo(x2 - 8, y2)
        # 箭头小三角
        path.lineTo(x2 - 8, y2 - 4)
        path.lineTo(x2, y2)
        path.lineTo(x2 - 8, y2 + 4)
        path.closeSubpath()
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor(theme.current_palette().border_strong), 1.5))
        item.setBrush(QBrush(QColor(theme.current_palette().border_strong)))
        self._scene.addItem(item)

    # ------------------------------------------------------------------ 状态刷新

    def refresh_states(self) -> None:
        """从图同步全部单元状态；有运行中节点时启动旋转动画."""
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
            return f"{label} · {node.error.splitlines()[0][:20]}"
        return label

    def _advance_spinner(self) -> None:
        """推进运行中单元的旋转角并重绘."""
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
        """主题切换后重刷背景与单元（连线颜色随主题，整体重建）."""
        self.setBackgroundBrush(QBrush(QColor(theme.current_palette().bg_app)))
        if self._graph is not None:
            # 连接线颜色随主题，整体重建最简单可靠；保留选择态
            graph = self._graph
            selected, selected_all = self._selected_id, self._selected_all
            self.set_graph(graph)
            if selected_all:
                self._selected_all = True
                self._apply_selection()
            elif selected:
                self.select_node(selected)

    # ------------------------------------------------------------------ 交互

    @property
    def selected_node_id(self) -> str:
        """当前选中单元 id（未选中或全选为空串）."""
        return self._selected_id

    @property
    def selected_all(self) -> bool:
        """是否处于全选态."""
        return self._selected_all

    def select_node(self, node_id: str) -> None:
        """程序化选中单个单元（不发信号）."""
        self._selected_id = node_id
        self._selected_all = False
        self._apply_selection()

    def select_all(self) -> None:
        """全选全部单元（标题栏高亮 + 发 all_selected 信号）."""
        self._selected_id = ""
        self._selected_all = True
        self._apply_selection()
        self.all_selected.emit()

    def _apply_selection(self) -> None:
        """按当前选择态刷新单元/组合框高亮."""
        for nid, card in self._cards.items():
            card.selected = self._selected_all or nid == self._selected_id
            card.update()
        if self._frame is not None:
            self._frame.selected_all = self._selected_all
            self._frame.update()

    def _card_at(self, pos) -> _NodeCard | None:
        """取视图坐标下的单元."""
        item = self.itemAt(pos)
        return item if isinstance(item, _NodeCard) else None

    def mousePressEvent(self, event) -> None:  # Qt 命名约定
        """单击：单元选中；组合框标题栏触发全选."""
        pos = mouse_event_pos(event)
        card = self._card_at(pos)
        if card is not None:
            self.select_node(card.node_id)
            self.node_clicked.emit(card.node_id)
            super().mousePressEvent(event)
            return
        item = self.itemAt(pos)
        if isinstance(item, _SystemFrame) and self._frame is not None:
            # 命中组合框（空白区域落在标题栏）→ 全选
            local = self.mapToScene(pos) - QPointF(self._frame.x(), self._frame.y())
            if self._frame.header_rect().contains(local):
                self.select_all()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # Qt 命名约定
        """双击单元."""
        card = self._card_at(mouse_event_pos(event))
        if card is not None:
            self.node_double_clicked.emit(card.node_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:  # Qt 命名约定
        """右键：单元弹出节点菜单，空白区域发 background_context_menu."""
        card = self._card_at(event.pos())
        if card is not None:
            self.select_node(card.node_id)
            self.node_context_menu.emit(card.node_id, event.globalPos())
            event.accept()
            return
        self.background_context_menu.emit(event.globalPos())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # Qt 命名约定
        """Ctrl+A 全选单元."""
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier and self._cards:
            self.select_all()
            event.accept()
            return
        super().keyPressEvent(event)

    def card_rect(self, node_id: str) -> QRectF | None:
        """节点单元的场景矩形（测试与定位用）."""
        card = self._cards.get(node_id)
        return card.sceneBoundingRect() if card is not None else None

    def frame_rect(self) -> QRectF | None:
        """组合框的场景矩形（测试与定位用）."""
        return self._frame.sceneBoundingRect() if self._frame is not None else None

    def spinner_active(self) -> bool:
        """旋转动画定时器是否运行中（测试用）."""
        return self._timer.isActive()
