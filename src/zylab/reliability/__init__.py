"""可靠性计算：感度试验数理统计方法（GJB/Z 377A 框架）.

六个标准方法的试验设计与数据分析：方法101 兰利法、方法102 OSTR 法、
方法103 升降法、方法104 D-优化法、方法201 概率单位法、方法202 完全
步进法。试验设计见 :mod:`.methods`，统计分析（MLE/Dixon-Mood/Kärber
与响应点估计）见 :mod:`.analysis`，升降法核心与 G/H 系数标定见
:mod:`.updown`，响应模型见 :mod:`.model`。
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
from .methods import METHOD_LABELS, METHOD_NAMES
from .model import MODEL_NAMES, response_prob
from .updown import DixonMoodDetail, dixon_mood_core, gh_factors

__all__ = [
    "METHOD_LABELS",
    "METHOD_NAMES",
    "MODEL_NAMES",
    "DixonMoodDetail",
    "ReliabilityError",
    "ResponsePoints",
    "SensitivityEstimate",
    "SensitivityTestResult",
    "analyze_updown_records",
    "dixon_mood",
    "dixon_mood_core",
    "gh_factors",
    "karber",
    "mle_estimate",
    "parse_trial_records",
    "response_points",
    "response_prob",
    "run_sensitivity_test",
]
