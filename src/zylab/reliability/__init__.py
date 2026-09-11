"""可靠性计算：感度试验数理统计方法 + FORM/SORM 极限状态分析.

- 六个标准方法的试验设计与数据分析（GJB/Z 377A 框架）：方法101 兰利法、
  方法102 OSTR 法、方法103 升降法、方法104 D-优化法、方法201 概率单位法、
  方法202 完全步进法；
- **FORM / SORM**：HLRF 迭代 + Breitung 曲率修正，处理正态/对数正态/Gumbel/
  Weibull/指数/均匀等独立随机变量的极限状态函数求解.
"""

from __future__ import annotations

from .analysis import (
    ResponsePoints,
    SensitivityEstimate,
    SensitivityTestResult,
    analyze_updown_records,
    dixon_mood,
    karber,
    mle_estimate,
    parse_trial_records,
    response_points,
    run_sensitivity_test,
)
from .errors import ReliabilityError
from .form import (
    Distribution,
    FORMResult,
    RandomVariable,
    SORMResult,
    form_analysis,
    sorm_analysis,
)
from .methods import METHOD_LABELS, METHOD_NAMES
from .model import MODEL_NAMES, response_prob
from .updown import DixonMoodDetail, dixon_mood_core, gh_factors

__all__ = [
    "METHOD_LABELS",
    "METHOD_NAMES",
    "MODEL_NAMES",
    "Distribution",
    "DixonMoodDetail",
    "FORMResult",
    "RandomVariable",
    "ReliabilityError",
    "ResponsePoints",
    "SORMResult",
    "SensitivityEstimate",
    "SensitivityTestResult",
    "analyze_updown_records",
    "dixon_mood",
    "dixon_mood_core",
    "form_analysis",
    "gh_factors",
    "karber",
    "mle_estimate",
    "parse_trial_records",
    "response_points",
    "response_prob",
    "run_sensitivity_test",
    "sorm_analysis",
]
