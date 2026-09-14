"""zylab.gui.pages - 主窗口内容页（笔记本/流程图/参数化计算应用）.

采用 ``__getattr__`` 懒加载，避免 main_window 导入时触发
flowchart_page → widgets 全链（含 pyqtgraph 210ms）。
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CellEditor",
    "CellWidget",
    "FlowchartPage",
    "NotebookPage",
    "TemplatePage",
    "VarTableModel",
]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FlowchartPage": (".flowchart_page", "FlowchartPage"),
    "CellEditor": (".notebook_page", "CellEditor"),
    "CellWidget": (".notebook_page", "CellWidget"),
    "NotebookPage": (".notebook_page", "NotebookPage"),
    "VarTableModel": (".notebook_page", "VarTableModel"),
    "TemplatePage": (".template_page", "TemplatePage"),
}


def __getattr__(name: str) -> object:
    """懒加载 facade：首次访问时才 import 对应子模块."""
    mapping = _LAZY_ATTRS.get(name)
    if mapping is None:
        raise AttributeError(f"module 'zylab.gui.pages' has no attribute {name!r}")
    submodule_path, attr_name = mapping
    module = import_module(submodule_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
