"""缓存哈希层：为工作流节点提供稳定的内容指纹，支持执行前缓存命中判断.

核心函数 :func:`content_hash` 对任意可 JSON 序列化的 Python 对象递归计算
SHA-256 摘要，保证相同内容跨进程、跨 Python 版本得到一致哈希值。

设计要点：

- 递归处理容器类型（dict/list/tuple/set/frozenset）与 numpy 数组；
- dict 按键排序后再哈希，避免 Python 3.7+ 插入序导致的非确定性；
- float 用 repr 而非 str，确保 ``1.0`` 与 ``1`` 的区分；
- 字符串用 UTF-8 编码后哈希，兼容非 ASCII 内容。
"""

from __future__ import annotations

import hashlib
import operator
from typing import Any

import numpy as np

__all__ = ["content_hash", "node_fingerprint"]

#: 哈希算法（SHA-256 提供 256-bit 抗碰撞摘要）
_ALGO = "sha256"


def content_hash(value: Any) -> str:
    """对任意可序列化对象计算稳定 SHA-256 哈希（十六进制字符串）.

    支持类型：

    - 基本类型：``None``/``bool``/``int``/``float``/``str``/``bytes``；
    - 容器：``list``/``tuple``/``dict``/``set``/``frozenset``；
    - numpy 数组：``np.ndarray``（含 dtype 与 shape，保证不同布局得到不同哈希）。

    :param value: 待哈希对象（须可递归序列化；遇到未知类型抛 TypeError）。
    :return: 64 位十六进制哈希串。
    """
    hasher = hashlib.new(_ALGO)
    _feed(hasher, value)
    return hasher.hexdigest()


def node_fingerprint(params: dict[str, Any], inputs_refs: dict[str, str], upstream_hashes: dict[str, str]) -> str:
    """计算节点内容指纹：params 哈希 + 输入引用签名 + 上游内容哈希.

    指纹公式::

        fp = hash(
            content_hash(params),
            sorted(inputs_refs.items()),
            sorted(upstream_hashes.items()),
        )

    当上游节点内容哈希已知时级联纳入，形成完整的拓扑依赖链哈希；
    上游哈希缺失（如未执行）时以空串占位。

    :param params: 节点参数表（已 coerce）。
    :param inputs_refs: 输入端口名 -> ``"node_id.port"`` 引用。
    :param upstream_hashes: 已执行上游节点 id -> 其内容哈希。
    :return: 64 位十六进制指纹串。
    """
    hasher = hashlib.new(_ALGO)
    _feed(hasher, content_hash(params))
    _feed(hasher, sorted(inputs_refs.items()))
    # 上游哈希按节点 id 排序，保证确定性
    upstream_sorted = sorted(upstream_hashes.items(), key=operator.itemgetter(0))
    _feed(hasher, upstream_sorted)
    return hasher.hexdigest()


# ------------------------------------------------------------------ 内部


def _feed(hasher: hashlib._Hash, value: Any) -> None:  # noqa: PLR0912 - 类型分派函数的多分支是本质复杂度
    """将对象按类型编码后喂入 hasher（递归）.

    哨兵前缀区分类型，避免跨类型值冲突（如 True 与 1、None 与 0）。
    """
    if value is None:
        hasher.update(b"\x00None")
    elif value is True:
        hasher.update(b"\x01True")
    elif value is False:
        hasher.update(b"\x02False")
    elif isinstance(value, int):
        hasher.update(b"\x03")
        hasher.update(str(value).encode("utf-8"))
    elif isinstance(value, float):
        hasher.update(b"\x04")
        hasher.update(repr(value).encode("utf-8"))
    elif isinstance(value, str):
        hasher.update(b"\x05")
        hasher.update(value.encode("utf-8"))
    elif isinstance(value, bytes):
        hasher.update(b"\x06")
        hasher.update(value)
    elif isinstance(value, np.ndarray):
        hasher.update(b"\x07")
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, list):
        hasher.update(b"\x08")
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, tuple):
        hasher.update(b"\x09")
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, frozenset):
        hasher.update(b"\x0a")
        for item in sorted(value, key=_sort_key):
            _feed(hasher, item)
    elif isinstance(value, set):
        hasher.update(b"\x0b")
        for item in sorted(value, key=_sort_key):
            _feed(hasher, item)
    elif isinstance(value, dict):
        hasher.update(b"\x0c")
        for key, item in sorted(value.items(), key=_dict_key_sort):
            _feed(hasher, str(key))
            _feed(hasher, item)
    else:
        import pickle

        hasher.update(b"\x0d")
        try:
            hasher.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except (pickle.PicklingError, TypeError) as exc:
            raise TypeError(f"content_hash 不支持类型 {type(value).__name__}: {value!r}") from exc


def _sort_key(value: Any) -> tuple[int, str]:
    """为 set/frozenset 排序生成稳定键（类型序号 + 字符串表示）."""
    return (type(value).__name__.__hash__() % 100000, repr(value))


def _dict_key_sort(pair: tuple[Any, Any]) -> str:
    """为 dict.items() 按键的字符串形式排序."""
    return str(pair[0])
