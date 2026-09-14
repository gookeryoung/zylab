"""zylab.gui.widgets - 可复用控件（工作流画布/参数表单/结果视图/统一绘图）.

采用 ``__getattr__`` 懒加载，避免 main_window 导入时触发
dsl_result_view → pyqtgraph（210ms）整条链。仅当真正访问
``ZyPlotWidget``/``ResultView`` 等重控件时才加载对应子模块。
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DslParamForm",
    "DslResultView",
    "NodeCanvasWidget",
    "ParamForm",
    "PlotMenuConfig",
    "ResultView",
    "TrialRecordEdit",
    "ZyPlotWidget",
    "apply_plot_context_menu",
]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DslParamForm": (".dsl_param_form", "DslParamForm"),
    "DslResultView": (".dsl_result_view", "DslResultView"),
    "NodeCanvasWidget": (".node_canvas", "NodeCanvasWidget"),
    "ParamForm": (".param_form", "ParamForm"),
    "PlotMenuConfig": (".plot_widget", "PlotMenuConfig"),
    "ResultView": (".result_view", "ResultView"),
    "TrialRecordEdit": (".trial_record_edit", "TrialRecordEdit"),
    "ZyPlotWidget": (".plot_widget", "ZyPlotWidget"),
    "apply_plot_context_menu": (".plot_widget", "apply_plot_context_menu"),
}


def __getattr__(name: str) -> object:
    """懒加载 facade：首次访问时才 import 对应子模块."""
    mapping = _LAZY_ATTRS.get(name)
    if mapping is None:
        raise AttributeError(f"module 'zylab.gui.widgets' has no attribute {name!r}")
    submodule_path, attr_name = mapping
    module = import_module(submodule_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
