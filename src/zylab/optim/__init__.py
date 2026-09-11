"""optim — 代理模型、Sobol 感度分析、Pareto 多目标与基于 DOE 样本的代理优化.

本包位于 **采样（:mod:`~zylab.doe`）与评估（:mod:`~zylab.reliability`）**
之间的链路：DOE 采样批量跑完 workflow → 训练
:class:`~zylab.optim.surrogate.Surrogate` →
:func:`~zylab.optim.optimize.optimize` 在代理上找最优 → 可选真函数验证；
或直接对真实函数做 :func:`~zylab.optim.sensitivity.sobol_analysis`
量化各参数贡献（Phase 5 感度分析层）；多目标场景则用
:func:`~zylab.optim.pareto.pareto_ranks` /
:func:`~zylab.optim.pareto.pareto_summary` 提取 Pareto 前沿.

核心入口：

- :class:`Surrogate` / :class:`GprSurrogate` / :class:`RbfSurrogate`：两种代理；
- :func:`optimize` / :func:`optimize_direct`：单目标优化；
- :func:`sobol_analysis` / :class:`SobolIndices`：Saltelli 方差分解感度分析；
- :func:`pareto_ranks` / :func:`pareto_summary` / :func:`crowding_distance`：多目标 Pareto 分析；
- :func:`scalarize`：多目标标量化辅助.

设计原则：**计算层**——不依赖 GUI，不直接操控 workflow，
只负责「给定样本造代理、给定代理找最优、给定函数算感度、给定多目标样本划 Pareto 前沿」的纯函数语义。
"""

from __future__ import annotations

from .errors import OptimError, SurrogateError
from .optimize import Optimizer, OptimResult, optimize, optimize_direct
from .pareto import (
    ParetoSummary,
    crowding_distance,
    pareto_front,
    pareto_ranks,
    pareto_summary,
    scalarize,
)
from .sensitivity import (
    SaltelliSample,
    SobolIndices,
    build_saltelli_sample,
    sobol_analysis,
    sobol_indices,
)
from .surrogate import GprSurrogate, RbfSurrogate, Surrogate

__all__ = [
    "GprSurrogate",
    "OptimError",
    "OptimResult",
    "Optimizer",
    "ParetoSummary",
    "RbfSurrogate",
    "SaltelliSample",
    "SobolIndices",
    "Surrogate",
    "SurrogateError",
    "build_saltelli_sample",
    "crowding_distance",
    "optimize",
    "optimize_direct",
    "pareto_front",
    "pareto_ranks",
    "pareto_summary",
    "scalarize",
    "sobol_analysis",
    "sobol_indices",
]
