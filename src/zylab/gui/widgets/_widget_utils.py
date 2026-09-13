"""Qt 控件工具函数：布局清理、spin 框创建等跨控件共用的轻量辅助.

目的：消除各 widget 中重复的 ``while layout.count(): takeAt + deleteLater``
循环，以及参数表单中相似的 spin 框构建逻辑。
"""

from __future__ import annotations

from ..qt_compat import QLayout

__all__ = ["clear_layout"]


def clear_layout(layout: QLayout) -> None:
    """清空布局中的全部子控件并计划销毁（deleteLater）.

    适用于表单/分块容器在 set_data/set_graph 等重建方法前的旧控件清理。
    与直接 ``clear()`` 不同：QLayout.clear() 只移出不销毁 widget，
    这里调用 ``deleteLater`` 让 Qt 在事件循环中安全释放。

    Args:
        layout: 待清空的布局（QVBoxLayout/QHBoxLayout/QFormLayout 等）。
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
