"""sci.workspace 工作区检视测试."""

from __future__ import annotations

import numpy as np

from zylab.sci.workspace import VarInfo, format_whos, whos


def test_whos_ndarray() -> None:
    """ndarray 应提取形状/dtype/字节数."""
    infos = whos({"a": np.zeros((3, 4))})
    assert len(infos) == 1
    info = infos[0]
    assert info.name == "a"
    assert info.type_name == "ndarray"
    assert info.shape == "3x4"
    assert info.dtype == "float64"
    assert info.nbytes == 96


def test_whos_skips_private_modules_and_commands() -> None:
    """下划线开头、模块与内建命令（whos/plot/run）不应出现在列表."""
    import sys

    def _noop() -> None:
        pass

    ns = {"_hidden": 1, "sys": sys, "whos": _noop, "plot": _noop, "run": _noop, "x": 5}
    assert [i.name for i in whos(ns)] == ["x"]


def test_whos_scalar_and_sequence() -> None:
    """标量无形状；序列标注 len；集合/字典字节数为 0."""
    infos = {i.name: i for i in whos({"n": 42, "lst": [1, 2, 3], "d": {"k": 1}, "s": "hello"})}
    assert infos["n"].type_name == "int"
    assert infos["n"].shape == ""
    assert infos["lst"].shape == "len=3"
    assert infos["lst"].nbytes > 0
    assert infos["d"].nbytes == 0
    assert infos["s"].nbytes > 0


def test_whos_scalar_ndarray_shape() -> None:
    """0 维 ndarray 形状显示为标量."""
    info = whos({"s": np.array(1.5)})[0]
    assert info.shape == "标量"


def test_whos_sorted_by_name() -> None:
    """结果按名称排序."""
    assert [i.name for i in whos({"b": 1, "a": 2})] == ["a", "b"]


def test_preview_truncated() -> None:
    """长值预览应截断到 60 字符."""
    info = whos({"big": np.arange(1000)})[0]
    assert len(info.preview) <= 60


def test_format_whos_empty() -> None:
    """空工作区返回提示."""
    assert format_whos([]) == "工作区为空"


def test_format_whos_table() -> None:
    """表格应包含表头与行数据."""
    infos = [VarInfo(name="x", type_name="int", shape="", dtype="", nbytes=28, preview="1")]
    text = format_whos(infos)
    assert "名称" in text
    assert "x" in text
    assert "int" in text


def test_safe_sizeof_failure() -> None:
    """__sizeof__ 抛异常的对象应返回 0（私有函数故障注入）."""
    from zylab.sci.workspace import _safe_sizeof

    class Bad:
        """__sizeof__ 总是抛 TypeError 的测试桩."""

        def __sizeof__(self) -> int:
            raise TypeError("no size")

    assert _safe_sizeof(Bad()) == 0
