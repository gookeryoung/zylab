"""设计空间容器：变量集合 + 便捷采样接口.

相比 :func:`~zylab.doe.sampling.sample` 每次都传变量列表，
:class:`DesignSpace` 将变量集合固化，提供 ``space.sample(method, n)``
形式的短路径调用，以及与 :mod:`~zylab.flowchart.param_store` 对接的
:meth:`to_input_rows` 便捷转换（采样矩阵 → 每组完整输入参数字典）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .errors import DoeError
from .sampling import SamplingMethod
from .sampling import sample as _sample
from .sampling import sample_unit as _sample_unit
from .variable import DesignVariable

__all__ = ["DesignSpace"]


@dataclass(frozen=True)
class DesignSpace:
    """设计空间：一组有顺序的 :class:`DesignVariable` 构成的探索维度."""

    variables: tuple[DesignVariable, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ 构造

    @classmethod
    def from_variables(cls, variables: Sequence[DesignVariable]) -> DesignSpace:
        """由变量序列构造（自动去重 + 校验）."""
        var_list = list(variables)
        if not var_list:
            raise DoeError("DesignSpace 至少需 1 个设计变量")
        seen_names: set[str] = set()
        for v in var_list:
            if v.name in seen_names:
                raise DoeError(f"设计变量名重复: {v.name!r}")
            seen_names.add(v.name)
            v.validate()
        return cls(variables=tuple(var_list))

    # ------------------------------------------------------------------ 查询

    @property
    def n_vars(self) -> int:
        """变量数."""
        return len(self.variables)

    def __getitem__(self, idx: int) -> DesignVariable:
        return self.variables[idx]

    def __len__(self) -> int:
        return len(self.variables)

    # ------------------------------------------------------------------ 采样

    def sample(
        self,
        method: SamplingMethod | str,
        n_samples: int | None = None,
        *,
        seed: int = 42,
        n_per_dim: int = 5,
    ) -> np.ndarray:
        """对本设计空间采样，返回实际值矩阵."""
        return _sample(self.variables, method, n_samples, seed=seed, n_per_dim=n_per_dim)

    def sample_unit(
        self,
        method: SamplingMethod | str,
        n_samples: int | None = None,
        *,
        seed: int = 42,
        n_per_dim: int = 5,
    ) -> np.ndarray:
        """对本设计空间在单位超立方体采样."""
        return _sample_unit(self.variables, method, n_samples, seed=seed, n_per_dim=n_per_dim)

    # ------------------------------------------------------------------ ParameterStore 集成

    def to_input_rows(self, X: np.ndarray) -> list[dict[str, Any]]:
        """将采样矩阵展开为输入参数字典序列.

        输出可直接喂给 :mod:`~zylab.flowchart.batch` 的批量运行入口——
        每行 ``{变量名: 值}``，与 :class:`~zylab.flowchart.param_store.ParameterStore.inputs`
        的扁平键（``"node_id.param_key"``）对齐后，即可替换 Template 节点
        的默认参数做探索。

        :param X: shape ``(n_samples, n_vars)`` 的实际值采样矩阵（通常来自
            :meth:`sample`）。
        :return: 长度为 ``n_samples`` 的字典列表，每个字典的 key 是
            :attr:`DesignVariable.name`，value 是对应采样值（离散变量
            取原始 ``levels`` 值，连续变量取区间内的连续值）。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_vars:
            raise DoeError(f"采样矩阵 shape 须为 (n, {self.n_vars})，得到 {X.shape}")
        rows: list[dict[str, Any]] = []
        for row in X:
            d: dict[str, Any] = {}
            for i, v in enumerate(self.variables):
                d[v.name] = float(row[i])
            rows.append(d)
        return rows
