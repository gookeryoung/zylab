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

        覆盖策略：只覆盖 QSS 难以直接控制且 Fusion 默认值与主题设计令牌
        不一致的指标。其余指标原样透传给 Fusion，让 QSS 的 min-height/
        padding 等规则主导控件尺寸。

        覆盖项：
        - PM_FocusFrameHMargin / PM_FocusFrameVMargin — Fusion 默认 1px，
          主题间距令牌 SPACING_XS(4px) 更协调；
        - PM_IndicatorWidth / Height — Fusion 默认 ~13px，覆盖为 16px 与
          26-32px 控件高度更协调（checkbox indicator）；
        - PM_ExclusiveIndicatorWidth / Height — radio button indicator 同上；
        - PM_TabBarTabHSpace — Fusion 默认 ~20px，主题 SPACING_MD(16px) 更紧凑；
        - PM_SmallIconSize — Fusion 默认 16px，匹配主题按钮内图标尺寸。
        """
        # 延迟导入避免循环（theme 常量模块加载极轻）
        from .theme import (
            SPACING_MD,
            SPACING_SM,
            SPACING_XS,
        )

        # 字符串 → 像素整数（"32px" → 32）
        def _px(value: str | int) -> int:
            if isinstance(value, int):
                return value
            return int(value.removesuffix("px"))

        theme_val = 0  # 0 表示"不覆盖，让 Fusion 处理"
        if metric in (QStyle.PM_FocusFrameHMargin, QStyle.PM_FocusFrameVMargin):
            theme_val = _px(SPACING_XS)  # 4px，比 Fusion 默认 1px 宽松
        elif metric in (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight):
            theme_val = 16  # checkbox indicator 16×16，与 26-32px 控件高度协调
        elif metric in (QStyle.PM_ExclusiveIndicatorWidth, QStyle.PM_ExclusiveIndicatorHeight):
            theme_val = 16  # radio button indicator 16×16
        elif metric == QStyle.PM_TabBarTabHSpace:
            theme_val = _px(SPACING_MD)  # 16px tab 水平间距
        elif metric == QStyle.PM_TabBarTabVSpace:
            theme_val = _px(SPACING_SM)  # 8px tab 垂直间距
        elif metric == QStyle.PM_SmallIconSize:
            theme_val = 16  # 16px，与 Fusion 默认一致但显式声明稳定

        if theme_val > 0:
            return theme_val
        # 默认行为：让代理 style 的 base（Fusion）处理大部分 metric
        return super().pixelMetric(metric, option, widget)

    # ------------------------------------------------------------------ 跨模块辅助

    @staticmethod
    def app_palette():
        """从 theme 模块当前色板构建 QPalette（延迟导入避免循环）."""
        from .style import build_qpalette

        return build_qpalette(theme.current_palette())
