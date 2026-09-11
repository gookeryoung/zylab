"""dsl.py 剩余 miss 补测 + report.py 简单路径."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from zylab.flowchart import dsl, report
from zylab.flowchart.errors import ParamError, TemplateError

__all__ = []


class TestResolveDerivedCacheHit:
    """_resolve_derived 缓存命中 (miss 592) + 循环依赖 (miss 593-595)."""

    def test_cache_hit_short_circuits(self) -> None:
        """resolved 缓存中已有 → 直接返回."""
        from zylab.flowchart.dsl import DslParam

        derived = {"x": DslParam(value=None, expr="1+1")}
        result = dsl._resolve_derived("x", derived, {}, {"x": 42}, ())
        assert result == 42

    def test_cycle_dependency_raises(self) -> None:
        """派生参数循环依赖 → ParamError."""
        from zylab.flowchart.dsl import DslParam

        # x 依赖 y, y 依赖 x → 循环
        derived = {
            "x": DslParam(value=None, expr="y + 1"),
            "y": DslParam(value=None, expr="x + 1"),
        }
        with pytest.raises(ParamError, match="循环依赖"):
            dsl._resolve_derived("x", derived, {}, {}, ())


class TestExpectList:
    """_expect_list 类型校验 (miss 622)."""

    def test_not_list_raises(self) -> None:
        with pytest.raises(TemplateError, match="应为列表"):
            dsl._expect_list({"not": "a list"}, "key", "测试字段")


class TestReportAutoField:
    """report._auto_field / _last_frame 纯 Python 分支."""

    def test_auto_field_no_attributes_raises(self) -> None:
        """载荷不含任何可渲染属性 → TypeError."""
        empty = MagicMock()
        for attr in ("temperatures", "displacements", "voltages", "element_results", "mode_shapes"):
            setattr(empty, attr, None)
        with pytest.raises(TypeError, match="不含可渲染"):
            report._auto_field(empty)

    def test_last_frame_2d_extracts_last_row(self) -> None:
        """_last_frame 2D 数组 (n_frames > n_nodes, n_nodes cols) 取末帧."""
        result = report._last_frame(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 3 frames x 2 nodes
            n_nodes=2,
        )
        assert result.tolist() == [5.0, 6.0]

    def test_last_frame_1d_passthrough(self) -> None:
        """1D 数组 (n_nodes,) 不变."""
        arr = np.array([10.0, 20.0, 30.0])
        result = report._last_frame(arr, n_nodes=3)
        assert result.tolist() == [10.0, 20.0, 30.0]
