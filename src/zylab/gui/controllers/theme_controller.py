"""ThemeController - QML 友好的主题属性代理.

将 :class:`zylab.gui.theme.Palette` 的全部色令牌暴露为 Q_PROPERTY，
供纯 Widget 场景（Python）与 QML 场景（数据驱动绑定）统一取色。
主题切换后调用 :meth:`refresh` 同步内部引用并发射 ``theme_changed`` 信号，
所有属性的 ``notify`` 均挂到同一信号，订阅方收到一次信号即可全量刷新。

与直接调用 :func:`theme.current_palette` 相比，本控制器的优势是：
- 可作为 QML ``contextProperty`` 直接绑定（`color: theme.bg_app`）；
- Python 端订阅方只需连接 ``theme_changed`` 一个信号；
- 类型安全（每个属性都是显式 ``str``）。
"""

from __future__ import annotations

from dataclasses import fields

from .. import theme as _theme
from ..qt_compat import Property, QObject, Signal

__all__ = ["ThemeController"]


class ThemeController(QObject):
    """主题属性代理（QObject 子类，可被 QML 注册）.

    :param parent: 父 QObject（通常为 AppController 或 MainWindow）
    """

    #: 主题切换后发射，所有属性的 notify 均连到此信号
    theme_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pal = _theme.current_palette()

    # -------------------------------------------------------------- 刷新入口

    def refresh(self) -> None:
        """从 ``theme.current_palette()`` 拉取最新色板并发射信号."""
        self._pal = _theme.current_palette()
        self.theme_changed.emit()

    # -------------------------------------------------------------- 动态属性注入


# --- 自动注入：遍历 Palette dataclass 字段，生成 Q_PROPERTY ---
# 每个字段对应一个 Property(str, getter, notify=theme_changed)
for _f in fields(_theme.Palette):
    _name = _f.name

    # 构造绑定 self 的 getter（闭包捕获 _name，Property 框架会注入 self）
    def _make_getter(field_name: str):
        def _getter(self: ThemeController) -> str:
            return getattr(self._pal, field_name)

        return _getter

    setattr(ThemeController, _name, Property(str, _make_getter(_name), notify=ThemeController.theme_changed))
del _f, _name, _make_getter
