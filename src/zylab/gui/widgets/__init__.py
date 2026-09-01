"""zylab.gui.widgets - 可复用控件（工作流画布/参数表单/结果视图）."""

from __future__ import annotations

from .dsl_param_form import DslParamForm
from .dsl_result_view import DslResultView
from .node_canvas import NodeCanvasWidget
from .param_form import ParamForm
from .result_view import ResultView

__all__ = ["DslParamForm", "DslResultView", "NodeCanvasWidget", "ParamForm", "ResultView"]
