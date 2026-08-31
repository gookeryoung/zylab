"""zylab.sci - 科学计算门面（数组/工作区/绘图，Qt-free）.

re-export NumPy 常用符号供 REPL 命名空间与脚本直接使用，保持 MATLAB 式习惯
（``zeros``/``ones``/``linspace``/``sin`` 等直接可用，同时提供 ``np`` 别名）。
"""

from __future__ import annotations

import numpy as np
from numpy import (
    arange,
    array,
    cos,
    diag,
    e,
    exp,
    eye,
    inf,
    linspace,
    log,
    log10,
    nan,
    ones,
    pi,
    sin,
    sqrt,
    tan,
    zeros,
)

from .notebook import (
    ErrorOutput,
    Notebook,
    NotebookCell,
    NotebookError,
    PlotOutput,
    PlotSeries,
    ResultOutput,
    StreamOutput,
    load_notebook,
    new_cell,
    save_notebook,
)
from .plotting import TOPIC_PLOT_REQUESTED, PlotRequest, make_plot_function
from .workspace import VarInfo, format_whos, whos

__all__ = [
    "TOPIC_PLOT_REQUESTED",
    "ErrorOutput",
    "Notebook",
    "NotebookCell",
    "NotebookError",
    "PlotOutput",
    "PlotRequest",
    "PlotSeries",
    "ResultOutput",
    "StreamOutput",
    "VarInfo",
    "arange",
    "array",
    "cos",
    "diag",
    "e",
    "exp",
    "eye",
    "format_whos",
    "inf",
    "linspace",
    "load_notebook",
    "log",
    "log10",
    "make_plot_function",
    "nan",
    "new_cell",
    "np",
    "ones",
    "pi",
    "save_notebook",
    "sin",
    "sqrt",
    "tan",
    "whos",
    "zeros",
]
