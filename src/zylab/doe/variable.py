"""设计变量声明：连续变量与离散变量统一建模.

每个 :class:`DesignVariable` 对应一个待探索的输入维度，支持两种
声明方式：

- **连续变量**：``(lower, upper)`` 有界区间，采样在区间内均匀/按分布取；
- **离散变量**：``levels`` 显式枚举的有限取值集合，采样从中取一。

与 :mod:`~zylab.flowchart.param_store` 的关系：设计变量的 ``name``
与 :attr:`~zylab.flowchart.param_store.ParameterStore.inputs` 的扁平键
``"node_id.param_key"`` 对齐后，采样矩阵的每一行即一组完整输入参数值。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .errors import DoeError

__all__ = ["DesignVariable"]


@dataclass(frozen=True)
class DesignVariable:
    """设计变量声明（不可变 dataclass）.

    :param name: 变量名——与 ``ParameterStore.inputs`` 的键（如
        ``"node_id.param_key"``）对齐，采样值可直接展开到 workflow 输入。
    :param lower: 连续变量下界（离散变量忽略）。
    :param upper: 连续变量上界（须 > ``lower``）。
    :param levels: 离散变量的显式取值集合（list/tuple/ndarray）。
        非空时忽略 ``lower/upper``，连续采样器会在 levels 网格上取值。
    :param label: 中文标签（可选，报告/界面展示用）。
    """

    name: str
    lower: float = 0.0
    upper: float = 1.0
    levels: tuple[float, ...] = field(default_factory=tuple)
    label: str = ""

    # ------------------------------------------------------------------ 构造

    @classmethod
    def continuous(cls, name: str, lower: float, upper: float, label: str = "") -> DesignVariable:
        """快速构造连续设计变量."""
        if upper <= lower:
            raise DoeError(f"连续变量 {name!r} 上界须严格大于下界，得到 lower={lower}, upper={upper}")
        return cls(name=name, lower=float(lower), upper=float(upper), label=label)

    @classmethod
    def discrete(cls, name: str, levels: Sequence[float], label: str = "") -> DesignVariable:
        """快速构造离散设计变量."""
        seq = tuple(float(v) for v in levels)
        if len(seq) < 2:
            raise DoeError(f"离散变量 {name!r} 至少需 2 个水平，得到 {len(seq)}")
        return cls(name=name, levels=seq, label=label)

    # ------------------------------------------------------------------ 查询

    @property
    def is_discrete(self) -> bool:
        """是否为离散变量（``levels`` 非空）."""
        return len(self.levels) > 0

    @property
    def is_continuous(self) -> bool:
        """是否为连续变量."""
        return not self.is_discrete

    def validate(self) -> None:
        """自校验：连续变量边界合法 / 离散变量水平数 ≥ 2."""
        if self.is_continuous:
            if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
                raise DoeError(f"连续变量 {self.name!r} 边界须为有限数")
            if self.upper <= self.lower:
                raise DoeError(f"连续变量 {self.name!r} 上界须严格大于下界")
        else:
            if len(self.levels) < 2:
                raise DoeError(f"离散变量 {self.name!r} 至少需 2 个水平")
            if not all(np.isfinite(v) for v in self.levels):
                raise DoeError(f"离散变量 {self.name!r} 水平值须为有限数")

    # ------------------------------------------------------------------ 变换

    def normalize(self, value: float | np.ndarray) -> float | np.ndarray:
        """将实际值归一到 [0, 1]（unit hypercube 坐标）.

        - 连续：``(value - lower) / (upper - lower)``；
        - 离散：映射到水平索引对应的 [0, 1] 位置（``i / (n_levels - 1)``）。
        """
        if self.is_discrete:
            arr = np.asarray(value, dtype=float)
            idx = np.array([self.levels.index(float(v)) if float(v) in self.levels else np.nan for v in arr.flat])
            result = idx.reshape(arr.shape)
            n = len(self.levels) - 1
            return result / n if n > 0 else np.zeros_like(result)
        lo, hi = self.lower, self.upper
        return (np.asarray(value, dtype=float) - lo) / (hi - lo)

    def denormalize(self, unit: float | np.ndarray) -> float | np.ndarray:
        """将 [0, 1] 单位坐标还原到实际值（与 :meth:`normalize` 互逆）."""
        u = np.asarray(unit, dtype=float)
        if self.is_discrete:
            n = len(self.levels)
            idx = np.clip(np.round(u * (n - 1)).astype(int), 0, n - 1)
            return np.asarray(self.levels)[idx]
        lo, hi = self.lower, self.upper
        return u * (hi - lo) + lo
