"""端口类型系统 ports.py 测试."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from zylab.flowchart import LinkError
from zylab.flowchart.module import PortSpec, PortType
from zylab.flowchart.ports import PortTypeRegistry, assert_can_connect, can_connect, compat_check, compatible


class TestPortTypeRegistry:
    """PortTypeRegistry 开放注册表."""

    def test_list_types_builtin(self) -> None:
        """默认注册表里包含全部内置枚举值."""
        reg = PortTypeRegistry()
        types = reg.list_types()
        assert len(types) == len(PortType)

    def test_register_builtin(self) -> None:
        """注册已存在于枚举中的端口类型直接返回."""
        reg = PortTypeRegistry()
        result = reg.register("model")
        assert result is PortType.MODEL

    def test_register_unknown_raises(self) -> None:
        """尝试注册不存在于枚举的新类型抛 ValueError（v2 首版不支持）."""
        reg = PortTypeRegistry()
        with pytest.raises(ValueError, match="暂不支持运行时追加"):
            reg.register("ghost_port_type")

    def test_register_extra_hit(self) -> None:
        """_extra 已有时命中直接返回（entry point 扩展预留路径）."""
        reg = PortTypeRegistry()
        # 直接注入 _extra（模拟未来 entry point 或手动 register 追加）
        extra_pt = MagicMock(spec=PortType)
        reg._extra["custom"] = extra_pt
        result = reg.register("custom")
        assert result is extra_pt


class TestCompatCheck:
    """端口类型兼容判断."""

    def test_same_type_always_compat(self) -> None:
        """同名类型始终兼容."""
        assert compat_check(PortType.MODEL, PortType.MODEL)
        assert compat_check(PortType.STATIC, PortType.STATIC)

    def test_any_as_dst_allows_all(self) -> None:
        """目标是 ANY → 任何源类型都可连."""
        assert compat_check(PortType.MODEL, PortType.ANY)
        assert compat_check(PortType.STATIC, PortType.ANY)

    def test_any_as_src_allows_all(self) -> None:
        """源是 ANY → 任何目标类型都可连."""
        assert compat_check(PortType.ANY, PortType.MODEL)
        assert compat_check(PortType.ANY, PortType.STATIC)

    def test_incompatible_denied(self) -> None:
        """类型不兼容时拒绝."""
        assert not compat_check(PortType.MODEL, PortType.STATIC)
        assert not compat_check(PortType.STATIC, PortType.MODEL)


class TestCompatible:
    """compatible（端口规格版）."""

    def test_compatible_same_port_type(self) -> None:
        src = PortSpec("out", PortType.MODEL)
        dst = PortSpec("in", PortType.MODEL)
        assert compatible(src, dst)

    def test_incompatible(self) -> None:
        src = PortSpec("out", PortType.MODEL)
        dst = PortSpec("in", PortType.STATIC)
        assert not compatible(src, dst)


def _mock_graph(nodes: dict[str, dict]) -> MagicMock:
    """构造一个最小 mock graph.

    nodes 格式：{node_id: {"outputs": {"port_name": PortType}, "inputs": {"port_name": PortType}}}
    """
    graph = MagicMock()

    def _make_node(spec: dict) -> MagicMock:
        node = MagicMock()
        node.spec = MagicMock()

        def _out_port(name: str) -> PortSpec:
            if name in spec.get("outputs", {}):
                return PortSpec(name, spec["outputs"][name])
            raise KeyError(name)

        def _in_port(name: str) -> PortSpec:
            if name in spec.get("inputs", {}):
                return PortSpec(name, spec["inputs"][name])
            raise KeyError(name)

        node.spec.output_port.side_effect = _out_port
        node.spec.input_port.side_effect = _in_port
        return node

    graph.node.side_effect = lambda nid: (
        _make_node(nodes[nid]) if nid in nodes else (_ for _ in ()).throw(KeyError(nid))
    )
    graph.ancestors.return_value = set()  # 默认无上游，环检测永远通过
    return graph


class TestCanConnect:
    """can_connect 连接校验（MagicMock graph）."""

    @pytest.fixture()
    def graph(self) -> MagicMock:
        return _mock_graph(
            {
                "src": {"outputs": {"model": PortType.MODEL}},
                "mid": {
                    "inputs": {"model": PortType.MODEL},
                    "outputs": {"model": PortType.MODEL},
                },
                "dst": {"inputs": {"model": PortType.MODEL}},
            }
        )

    def test_self_connection_denied(self, graph: MagicMock) -> None:
        """自连接直接拒绝."""
        ok, reason = can_connect(graph, "src", "model", "src", "model")
        assert not ok
        assert "自连接" in reason

    def test_unknown_dst_node(self, graph: MagicMock) -> None:
        """目标节点不存在."""
        ok, reason = can_connect(graph, "src", "model", "ghost", "model")
        assert not ok
        assert "目标节点不存在" in reason

    def test_unknown_src_node(self, graph: MagicMock) -> None:
        """源节点不存在."""
        ok, reason = can_connect(graph, "ghost", "model", "dst", "model")
        assert not ok
        assert "源节点不存在" in reason

    def test_unknown_src_port(self, graph: MagicMock) -> None:
        """源端口不存在."""
        ok, reason = can_connect(graph, "src", "ghost_port", "dst", "model")
        assert not ok
        assert "源端口不存在" in reason

    def test_unknown_dst_port(self, graph: MagicMock) -> None:
        """目标端口不存在."""
        ok, reason = can_connect(graph, "src", "model", "dst", "ghost_port")
        assert not ok
        assert "目标端口不存在" in reason

    def test_incompatible_types(self) -> None:
        """端口类型不兼容."""
        g = _mock_graph(
            {
                "src": {"outputs": {"model": PortType.MODEL}},
                "dst": {"inputs": {"model": PortType.STATIC}},  # 类型不兼容
            }
        )
        ok, reason = can_connect(g, "src", "model", "dst", "model")
        assert not ok
        assert "端口类型不兼容" in reason

    def test_valid_connection(self, graph: MagicMock) -> None:
        """合法连接：MODEL → MODEL."""
        ok, reason = can_connect(graph, "src", "model", "mid", "model")
        assert ok
        assert reason == ""

    def test_cycle_detection(self) -> None:
        """环检测：dst 已是 src 的递归上游."""
        g = _mock_graph(
            {
                "src": {"outputs": {"model": PortType.MODEL}},
                "dst": {"inputs": {"model": PortType.MODEL}},
            }
        )
        g.ancestors.return_value = {"dst"}  # dst 是 src 的上游
        ok, reason = can_connect(g, "src", "model", "dst", "model")
        assert not ok
        assert "环" in reason


class TestAssertCanConnect:
    """assert_can_connect 抛 LinkError."""

    def test_raises_on_false(self) -> None:
        g = _mock_graph(
            {
                "s": {"outputs": {"model": PortType.MODEL}},
            }
        )
        with pytest.raises(LinkError, match="自连接"):
            assert_can_connect(g, "s", "model", "s", "model")
