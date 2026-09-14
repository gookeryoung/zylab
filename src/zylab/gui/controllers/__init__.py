"""gui.controllers - 跨页面协调控制器层（QObject 子类，Signal/Slot 通信）.

Controller 模式将 MainWindow 中的跨页面协调逻辑（主题切换、工作区管理、
运行状态、设置对话框等）抽离为独立 QObject，MainWindow 保留窗口骨架
与 Dock 布局，页面通过 Signal/Slot 与 Controller 通信。

职责拆分原则：

- **MainWindow**：窗口骨架、Dock 布局、头部栏、StackedWidget 三页面
- **AppController**（本模块）：跨页面协调、主题/工作区/运行时状态、对话框
- **页面（NotebookPage/FlowchartPage/TemplatePage）**：页面自身的 UI 与业务逻辑

与 transucer-calc 的差异：zylab 核心业务（FEA 求解、流程图编排、DSL 参数化）
复杂度远高于 transducer-calc，Controller 层保持**薄协调**，不下沉业务逻辑。
"""

from __future__ import annotations

from .app_controller import AppController
from .theme_controller import ThemeController

__all__ = ["AppController", "ThemeController"]
