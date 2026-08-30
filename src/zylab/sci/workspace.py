"""zylab.sci 工作区变量检视（MATLAB 式 whos）.

从命名空间字典提取变量的结构化描述（名称/类型/形状/字节数/预览），
供控制台 whos 命令与 GUI 变量浏览器共用（GUI 显示层不重复实现格式化逻辑）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping

import numpy as np

__all__ = ["VarInfo", "whos"]

# 值预览的最大长度
_PREVIEW_MAXLEN = 60


@dataclass(frozen=True)
class VarInfo:
    """工作区单个变量的描述.

    :param name: 变量名。
    :param type_name: 类型名（如 ``ndarray``/``int``/``function``）。
    :param shape: 形状描述（ndarray 为 ``3x4``，序列为 ``len=5``，标量为空串）。
    :param dtype: 元素类型（ndarray 的 dtype，其他为空串）。
    :param nbytes: 占用字节数（ndarray 精确值，其他为估计值）。
    :param preview: 值的短预览（截断到 60 字符）。
    :param builtin: 是否为系统内置符号（NumPy 符号/np 模块/whos 等命令），
        GUI 变量浏览器据此用次级色区分用户变量。
    """

    name: str
    type_name: str
    shape: str
    dtype: str
    nbytes: int
    preview: str
    builtin: bool = False


def _describe(name: str, value: Any, builtin: bool = False) -> VarInfo:
    """构造单个变量的 VarInfo."""
    type_name = type(value).__name__
    shape = ""
    dtype = ""
    nbytes = 0
    if isinstance(value, np.ndarray):
        shape = "x".join(str(d) for d in value.shape) if value.shape else "标量"
        dtype = str(value.dtype)
        nbytes = int(value.nbytes)
    elif isinstance(value, (list, tuple, dict, set, frozenset)):
        shape = f"len={len(value)}"
        nbytes = (
            sum(_safe_sizeof(v) for v in getattr(value, "__iter__", lambda: ())())
            if isinstance(value, (list, tuple))
            else 0
        )
    else:
        nbytes = _safe_sizeof(value)
    preview = repr(value)
    if len(preview) > _PREVIEW_MAXLEN:
        preview = preview[: _PREVIEW_MAXLEN - 1] + "…"
    return VarInfo(
        name=name, type_name=type_name, shape=shape, dtype=dtype, nbytes=nbytes, preview=preview, builtin=builtin
    )


def _safe_sizeof(obj: Any) -> int:
    """安全获取对象字节数，失败返回 0."""
    try:
        import sys

        return sys.getsizeof(obj)
    except (TypeError, AttributeError):
        return 0


def whos(namespace: Mapping[str, Any], builtin_names: Collection[str] = ()) -> list[VarInfo]:
    """列出命名空间中的变量（跳过 ``_`` 开头项），按名称排序.

    内置符号（NumPy 符号、np 模块、whos/plot/run 等命令）不剔除，而是标记
    ``builtin=True`` —— GUI 变量浏览器据此用次级色区分用户变量。

    :param namespace: 命名空间映射（如 ``ReplKernel.namespace``）。
    :param builtin_names: 内置符号名集合（如 ``ReplKernel.builtin_names``）。
    :returns: VarInfo 列表。
    """
    infos = [
        _describe(name, value, builtin=name in builtin_names)
        for name, value in namespace.items()
        if not name.startswith("_")
    ]
    return sorted(infos, key=_var_name)


def _var_name(info: VarInfo) -> str:
    """提取变量名（sorted key，替代 lambda 以保持类型标注完整）."""
    return info.name


def format_whos(infos: list[VarInfo]) -> str:
    """将 VarInfo 列表格式化为等宽表格文本（MATLAB whos 风格）.

    :param infos: :func:`whos` 的返回。
    :returns: 表格字符串；空工作区返回提示行。
    """
    if not infos:
        return "工作区为空"
    headers = ("名称", "类型", "形状", "元素类型", "字节数")
    rows = [(i.name, i.type_name, i.shape, i.dtype, str(i.nbytes)) for i in infos]
    widths = [max(len(h), *(len(r[c]) for r in rows)) for c, h in enumerate(headers)]
    header_line = "  ".join(h.ljust(widths[c]) for c, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(r[c].ljust(widths[c]) for c in range(len(headers))) for r in rows]
    return "\n".join([header_line, sep, *body])
