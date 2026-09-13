"""zylab.gui.widgets - 可复用控件（工作流画布/参数表单/结果视图/统一绘图）."""

from __future__ import annotations

from .dsl_param_form import DslParamForm
from .dsl_result_view import DslResultView
from .node_canvas import NodeCanvasWidget
from .param_form import ParamForm
from .plot_widget import PlotMenuConfig, ZyPlotWidget, apply_plot_context_menu
from .result_view import ResultView
from .trial_record_edit import TrialRecordEdit

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
