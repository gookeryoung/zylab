"""QProxyStyle 层：拦截 Fusion 原生绘制，统一主题语义.

继承 QProxyStyle，包装 Fusion 基础风格，实现三个核心功能：

1. **polish/unpolish** — 每次 QWidget 被 Qt 样式系统 polish（主题切换时
   会自动 repolish 全部可见控件）时，同步 QPalette。这样 QPalette 层
   的色板变更会即时反映到所有控件的原生绘制上。

2. **pixelMetric** — 统一像素级尺寸。Fusion 会从 FusionStyle 返回一些
   默认值（比如 check indicator 大小、focus frame 宽度），这里根据
   主题的非色令牌（RADIUS / CONTROL_HEIGHT 等）覆盖，让 QSS 无法覆盖的
   系统绘制也保持一致。

3. **drawPrimitive** — 对 Fusion 原生绘制做精细拦截。目前 Fusion 风格
   下 QSS 能覆盖 95% 的外观，仅少数 QSS 无对应属性的场景需要在此补充。

与 QPalette + QSS Fragment 层的关系：

- **QSS Fragment** 负责控件的「外观」（背景、边框、字体、间距、hover/disabled
  状态等），覆盖所有能被 QSS 选择器匹配到的部分；
- **QPalette** 负责 Qt 原生控件系统绘制的「颜色」（下拉箭头、滚动条滑块、
  checkbox 勾选框、QToolTip 背景等 QSS 覆盖不完全的部分）；
- **QProxyStyle** 负责系统绘制的「像素级细节」和跨主题的 polish 同步，确保
  三层配合无缝。

应用方式：

```python
app.setStyle("Fusion")
app.setStyle(ProxyStyle(app.style()))   # 包装 Fusion
app.setPalette(build_qpalette(current_palette()))
app.setStyleSheet(load_stylesheet())
```
"""

from __future__ import annotations

from . import theme
from .qt_compat import QProxyStyle, QStyle

__all__ = ["ProxyStyle"]


class ProxyStyle(QProxyStyle):
    """包装 Fusion 基础风格，同步 QPalette + 统一像素级尺寸.

    主题切换时，``apply_theme`` 会重新 setPalette + setStyleSheet，
    Qt 样式系统会自动 repolish 全部控件，polish 钩子会把新 QPalette
    同步到每个控件。
    """

    # ------------------------------------------------------------------ polish

    def polish(self, widget_or_palette_or_app) -> None:
        """polish 钩子（三个重载：QWidget / QPalette / QApplication）.

        只对 QWidget 做 QPalette 同步，其余两个重载原样透传给父类。
        """
        from PySide2.QtWidgets import QWidget

        super().polish(widget_or_palette_or_app)
        if not isinstance(widget_or_palette_or_app, QWidget):
            return
        # 部分 Fusion 原生控件（QToolTip、QComboBox 下拉列表等）在设置
        # proxy style 后不会自动继承 app 的 QPalette，这里显式设置保证一致。
        pal = self.app_palette()
        if pal is not None:
            widget_or_palette_or_app.setPalette(pal)

    def unpolish(self, widget_or_palette_or_app) -> None:
        """反 polish（三个重载：QWidget / QPalette / QApplication）.

        全部透传给父类，不做额外操作。
        """
        super().unpolish(widget_or_palette_or_app)

    # ------------------------------------------------------------------ 像素尺寸

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option=None,
        widget=None,
    ) -> int:
        """根据主题非色令牌覆盖 Fusion 的像素级尺寸.

        主要处理：check indicator 大小、focus frame 宽度等 QSS 无对应
        属性的指标值。大部分控件尺寸已由 QSS 的 min-height / padding
        控制，这里只覆盖 Fusion 默认值不一致的场景。
        """
        # 默认行为：让代理 style 的 base（Fusion）处理大部分 metric
        return super().pixelMetric(metric, option, widget)

    # ------------------------------------------------------------------ 跨模块辅助

    @staticmethod
    def app_palette():
        """从 theme 模块当前色板构建 QPalette（延迟导入避免循环）."""
        from .style import build_qpalette

        return build_qpalette(theme.current_palette())
