"""flowchart.cache 内容哈希层测试：content_hash 稳定性、类型覆盖、node_fingerprint 级联."""

from __future__ import annotations

import numpy as np

from zylab.flowchart.cache import content_hash, node_fingerprint

__all__ = []


class TestContentHashStability:
    """哈希稳定性与确定性."""

    def test_same_value_same_hash(self) -> None:
        """同值多次哈希一致."""
        assert content_hash(42) == content_hash(42)
        assert content_hash("hello") == content_hash("hello")
        assert content_hash([1, 2, 3]) == content_hash([1, 2, 3])

    def test_64_char_hex(self) -> None:
        """SHA-256 输出 64 位十六进制串."""
        h = content_hash(None)
        assert len(h) == 64
        int(h, 16)  # 能解析为 16 进制

    def test_different_values_different_hash(self) -> None:
        """不同值产生不同哈希."""
        assert content_hash(42) != content_hash(43)
        assert content_hash("hello") != content_hash("world")
        assert content_hash([1, 2, 3]) != content_hash([1, 2, 4])

    def test_cross_type_distinction(self) -> None:
        """True/1、False/0、None/0 等跨类型值产生不同哈希."""
        assert content_hash(True) != content_hash(1)
        assert content_hash(False) != content_hash(0)
        assert content_hash(None) != content_hash(0)
        assert content_hash("1") != content_hash(1)

    def test_dict_order_independence(self) -> None:
        """dict 键排序后哈希：插入序不影响."""
        a = {"a": 1, "b": 2}
        b = {"b": 2, "a": 1}
        assert content_hash(a) == content_hash(b)

    def test_float_repr_precision(self) -> None:
        """float 用 repr 区分 1.0 与 1."""
        assert content_hash(1.0) != content_hash(1)
        assert content_hash(3.14159) == content_hash(3.14159)


class TestContentHashTypes:
    """类型覆盖."""

    def test_basic_types(self) -> None:
        """None/bool/int/float/str/bytes."""
        content_hash(None)
        content_hash(True)
        content_hash(False)
        content_hash(0)
        content_hash(-100)
        content_hash(3.14)
        content_hash("中文内容")
        content_hash(b"\x00\x01\x02")

    def test_containers(self) -> None:
        """list/tuple/set/frozenset/dict."""
        content_hash([1, 2, 3])
        content_hash((1, 2, 3))
        content_hash({1, 2, 3})
        content_hash(frozenset({1, 2, 3}))
        content_hash({"a": 1, "b": 2})

    def test_nested_containers(self) -> None:
        """嵌套容器递归哈希."""
        content_hash({"a": [1, {"b": (2, 3)}], "c": {4, 5}})

    def test_numpy_array(self) -> None:
        """numpy 数组（含 dtype/shape 区分）."""
        arr = np.array([1, 2, 3], dtype=np.float64)
        h = content_hash(arr)
        assert len(h) == 64
        # 不同 dtype -> 不同哈希
        arr_int = arr.astype(np.int64)
        assert content_hash(arr) != content_hash(arr_int)
        # 不同 shape -> 不同哈希
        arr_2d = arr.reshape(3, 1)
        assert content_hash(arr) != content_hash(arr_2d)

    def test_empty_containers(self) -> None:
        """空容器有确定性哈希（不同空容器哈希不同）."""
        assert content_hash([]) != content_hash(())
        assert content_hash([]) != content_hash({})
        assert content_hash(()) != content_hash(frozenset())


class TestNodeFingerprint:
    """节点指纹级联语义."""

    def test_params_deterministic(self) -> None:
        """相同 params + 引用 + 上游哈希 => 相同指纹."""
        fp1 = node_fingerprint({"a": 1}, {"m": "src.model"}, {"src": "abc123"})
        fp2 = node_fingerprint({"a": 1}, {"m": "src.model"}, {"src": "abc123"})
        assert fp1 == fp2

    def test_params_change_changes_fingerprint(self) -> None:
        """参数变更 => 指纹变更."""
        fp1 = node_fingerprint({"a": 1}, {}, {})
        fp2 = node_fingerprint({"a": 2}, {}, {})
        assert fp1 != fp2

    def test_upstream_hash_propagates(self) -> None:
        """上游哈希变更 => 下游指纹变更（级联语义）."""
        fp1 = node_fingerprint({"a": 1}, {"m": "src.model"}, {"src": "old_hash"})
        fp2 = node_fingerprint({"a": 1}, {"m": "src.model"}, {"src": "new_hash"})
        assert fp1 != fp2

    def test_upstream_missing_hash(self) -> None:
        """上游哈希缺失（空 dict）时指纹与有值不同."""
        fp_no = node_fingerprint({"a": 1}, {"m": "src.model"}, {})
        fp_yes = node_fingerprint({"a": 1}, {"m": "src.model"}, {"src": "hash1"})
        assert fp_no != fp_yes

    def test_input_ref_change(self) -> None:
        """输入引用变更 => 指纹变更."""
        fp1 = node_fingerprint({}, {"m": "src.model"}, {"src": "h1"})
        fp2 = node_fingerprint({}, {"m": "other.model"}, {"other": "h1"})
        assert fp1 != fp2

    def test_upstream_sorted_deterministic(self) -> None:
        """多上游时按 id 排序，保证确定性."""
        fp_a = node_fingerprint({}, {}, {"b": "hb", "a": "ha"})
        fp_b = node_fingerprint({}, {}, {"a": "ha", "b": "hb"})
        assert fp_a == fp_b
