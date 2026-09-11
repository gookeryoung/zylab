"""doe — 通用参数空间采样基础设施.

本包为 **多变量** 参数化计算提供设计空间探索能力（DOE，Design of Experiments）。
与 :mod:`~zylab.reliability`（针对伯努利响应的感度试验）和
:mod:`~zylab.flowchart`（流程调度）正交互补——三者分别负责「采样策略」
「可靠性评估」「任务编排」。

核心入口：

- :class:`DesignVariable`：连续/离散变量声明；
- :class:`DesignSpace`：变量集合 + 便捷采样 + ParameterStore 集成；
- :func:`sample` / :func:`sample_unit`：四种采样器（全因子/LHC/Sobol/随机）。
"""

from __future__ import annotations

from .design_space import DesignSpace
from .errors import DoeError
from .sampling import SamplingMethod, sample, sample_unit
from .variable import DesignVariable

__all__ = [
    "DesignSpace",
    "DesignVariable",
    "DoeError",
    "SamplingMethod",
    "sample",
    "sample_unit",
]
