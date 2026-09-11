"""参数化计算说明卡：图文引导面板（markdown 文本 + 示意图，可折叠）.

设计语言对齐 :class:`~zylab.gui.widgets.stream_view.ResultBlockCard`：
默认透明背景、hover 微背景提示、主色左边框视觉锚点，与 Jupyter 式结果流
卡片风格统一，避免侧边栏风格碎片化。

布局（TemplatePage 左侧栏顶部）：

- 卡片容器 ``#docsCard``（透明 + 主色左边框 + hover 微背景）；
- 标题行：问号图标 ``#docsIcon`` + 加粗标题 ``#docsTitle`` + 折叠箭头
  ``#docsCaret``（整行可点击，折叠/展开正文）；
- 正文：markdown 文本（QTextBrowser 只读、高度随内容自适应、上限内滚动）
  + 示意图（圆角边框卡片式展示，等比缩放至卡片宽度）。

``DslDocs.image`` 声明非空但文件缺失时显示次级占位提示（帮助模板作者
发现路径错误）；模板无 ``docs`` 声明或内容全空时整卡隐藏。文本渲染复用
报告管线的 :func:`~zylab.studio.richtext.markdown_to_html`（受控子集，
与结果流 markdown 块渲染语言统一）。
"""

from __future__ import annotations

from pathlib import Path

from zylab.studio.dsl import DslTemplate
from zylab.studio.richtext import markdown_to_html

from .. import theme
from ..icons import tinted_pixmap
from ..qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPixmap,
    Qt,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    Signal,
)

__all__ = ["DocsPanel", "resolve_docs_image"]

#: 说明正文高度上限（px，超出内部滚动，避免长说明挤占参数区）
_DOCS_TEXT_MAX_HEIGHT = 240

#: 示意图高度上限（px，等比缩放）
_DOCS_IMAGE_MAX_HEIGHT = 180

#: 标题栏图标尺寸（正方形，像素）
_HEADER_ICON_SIZE = 16


def resolve_docs_image(image: str, source: str) -> Path | None:
    """解析 ``docs.intro.image`` 声明到文件路径.

    绝对路径直接采用；相对路径以模板来源文件（``source``）所在目录为
    基准解析；路径为空或基准缺失（非文件加载构造的模板）返回 None。
    """
    if not image:
        return None
    candidate = Path(image)
    if candidate.is_absolute():
        return candidate
    if not source:
        return None
    return Path(source).resolve().parent / candidate


class _DocsHeader(QFrame):
    """说明卡标题行（图标 + 加粗标题 + 折叠箭头，整行可点击折叠）."""

    toggled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化标题行：问号图标 + 「说明」粗体 + 右侧折叠箭头."""
        super().__init__(parent, objectName="docsHeader")
        self.setCursor(Qt.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_XS)

        # 问号图标（主题色着色，hover 时由 QSS 联动）
        self._icon = QLabel(objectName="docsIcon")
        self._icon.setFixedSize(_HEADER_ICON_SIZE, _HEADER_ICON_SIZE)
        layout.addWidget(self._icon)

        # 标题（粗体，QSS docsTitle 控制字号/字重）
        self._title = QLabel("说明", objectName="docsTitle")
        layout.addWidget(self._title)

        # 拉伸 + 折叠箭头（jupyter 式 ▾/▸）
        layout.addStretch()
        self._caret = QLabel("▾", objectName="docsCaret")
        layout.addWidget(self._caret)

        self._refresh_icon()

    def _refresh_icon(self) -> None:
        """按当前主题刷新图标着色."""
        palette = theme.current_palette()
        pix = tinted_pixmap("question", palette.primary, _HEADER_ICON_SIZE)
        if not pix.isNull():
            self._icon.setPixmap(pix)

    def set_collapsed(self, collapsed: bool) -> None:
        """切换折叠箭头方向（▾ 展开 / ▸ 折叠）."""
        self._caret.setText("▸" if collapsed else "▾")

    def mousePressEvent(self, event) -> None:
        """左键点击整行发射折叠切换信号."""
        if event.button() == Qt.LeftButton:
            self.toggled.emit()
        super().mousePressEvent(event)


class DocsPanel(QWidget):
    """图文说明卡（标题行折叠 + markdown 正文 + 示意图）.

    设计语言：透明卡片 + 主色左边框 + hover 微背景，与 ResultBlockCard
    风格统一。详见模块文档。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化空卡（无内容时整体隐藏，set_template 后按声明呈现）."""
        super().__init__(parent)
        self._collapsed = False
        self._pixmap: QPixmap | None = None
        self._image_declared = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 卡片容器（QSS docsCard 控制整体外观）----
        card = QFrame(objectName="docsCard")
        root.addWidget(card)
        layout = QVBoxLayout(card)
        # 内边距：左右留白与 ResultBlockCard 对齐，上下稍紧凑
        layout.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_MD)
        layout.setSpacing(theme.SPACING_SM)

        # ---- 标题行 ----
        self._header = _DocsHeader()
        self._header.toggled.connect(self._toggle)
        layout.addWidget(self._header)

        # ---- 正文区 ----
        self._body = QWidget()
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(theme.SPACING_SM)

        # markdown 文本（QTextBrowser，只读可复制）
        self._text_browser = QTextBrowser(objectName="docsText")
        self._text_browser.setOpenExternalLinks(False)
        self._text_browser.setFrameShape(QFrame.NoFrame)
        self._text_browser.setVisible(False)
        body.addWidget(self._text_browser)

        # 示意图（圆角卡片式边框，QSS docsImage 控制）
        self._image_label = QLabel(objectName="docsImage")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setVisible(False)
        body.addWidget(self._image_label)

        # 图片缺失占位提示
        self._image_hint = QLabel("", objectName="secondaryText")
        self._image_hint.setWordWrap(True)
        self._image_hint.setVisible(False)
        body.addWidget(self._image_hint)

        layout.addWidget(self._body)

        self.setVisible(False)

    # ------------------------------------------------------------------ 数据

    def set_template(self, template: DslTemplate | None) -> None:
        """按模板 ``docs`` 声明重建说明卡（无内容整卡隐藏）."""
        docs = template.docs if template is not None else None
        text = (docs.text if docs is not None else "").strip()
        self._image_declared = (docs.image if docs is not None else "").strip()
        image_path = resolve_docs_image(self._image_declared, template.source if template is not None else "")

        # 文本：markdown 渲染（plain 文本经转换亦安全）
        if text:
            palette = theme.current_palette()
            self._text_browser.setHtml(markdown_to_html(text))
            self._text_browser.setStyleSheet(f"border: none; background: transparent; color: {palette.text_primary};")
            self._text_browser.setVisible(True)
        else:
            self._text_browser.clear()
            self._text_browser.setVisible(False)

        # 示意图：文件存在渲染；声明非空但缺失时占位提示
        self._pixmap = None
        if image_path is not None and image_path.is_file():
            pixmap = QPixmap(str(image_path))
            self._pixmap = pixmap if not pixmap.isNull() else None
        self._image_label.setVisible(False)
        self._image_hint.setVisible(False)
        if self._pixmap is not None:
            self._image_label.setVisible(True)
        elif self._image_declared:
            self._image_hint.setText(f"示意图缺失: {self._image_declared}")
            self._image_hint.setVisible(True)

        has_content = bool(text) or self._pixmap is not None or bool(self._image_declared)
        self.setVisible(has_content)
        self._collapsed = False
        self._body.setVisible(True)
        self._header.set_collapsed(False)
        self._sync_sizes()

    def refresh_theme(self) -> None:
        """主题切换后重刷图标着色与正文配色."""
        self._header._refresh_icon()
        if self._text_browser.isVisible():
            palette = theme.current_palette()
            self._text_browser.setStyleSheet(f"border: none; background: transparent; color: {palette.text_primary};")

    # ------------------------------------------------------------------ 内部

    def _toggle(self) -> None:
        """折叠/展开正文（无内容时不响应）."""
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._header.set_collapsed(self._collapsed)

    def _sync_sizes(self) -> None:
        """按当前宽度同步正文高度与示意图缩放（resize/set_template 后调用）."""
        if not self.isVisible():
            return
        # 文本高度：文档实际高度，上限内滚动
        if self._text_browser.isVisible():
            doc = self._text_browser.document()
            doc.setTextWidth(max(self._text_browser.viewport().width(), 16))
            height = min(int(doc.size().height()) + 4, _DOCS_TEXT_MAX_HEIGHT)
            self._text_browser.setFixedHeight(max(height, 20))
        # 示意图：等比缩放至卡内容宽（预留卡内边距 + 图片边框）
        if self._pixmap is not None:
            avail = max(self.width() - 2 * theme.SPACING_MD - 12, 32)
            scaled = self._pixmap.scaled(avail, _DOCS_IMAGE_MAX_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        """宽度变化后重算文本高度与示意图缩放."""
        super().resizeEvent(event)
        self._sync_sizes()
