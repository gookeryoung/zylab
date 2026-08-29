"""fea 异常体系.

继承 :class:`ZylabError` 公共基类，按 FEA 场景细分（网格/单元/求解），
便于调用方按粒度捕获。
"""

from __future__ import annotations

from zylab.core.errors import ZylabError

__all__ = [
    "ElementError",
    "MeshError",
    "SolverError",
]


class MeshError(ZylabError):
    """网格错误（连接表越界、维数不匹配、单元数不符等）."""


class ElementError(ZylabError):
    """单元错误（退化几何、雅可比非正、材料参数非法等）."""


class SolverError(ZylabError):
    """求解错误（刚度矩阵奇异、约束不足等）."""
