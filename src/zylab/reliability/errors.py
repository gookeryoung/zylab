"""可靠性计算异常体系."""

from __future__ import annotations

__all__ = ["ReliabilityError"]


class ReliabilityError(Exception):
    """可靠性计算错误基类（非法方法名/参数配置等）."""
