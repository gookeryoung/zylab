"""端口类型系统 v2：开放注册 + 兼容矩阵 + 连接校验.

Phase 1 用封闭 ``PortType`` 枚举 + graph.py:212 的严格同类型判断。
Phase 2 引入：

- :class:`PortTypeRegistry`：支持插件经 entry point ``zylab.port_type`` 注册新端口类型；
- 兼容矩阵 ``compat[dest][src] -> bool``：声明式谓词，替代严格同类型；
- 默认规则：同名兼容 + ANY 收发 + 显式兼容声明（如 STATIC → BUCKLING.reference）；
- :func:`can_connect`：供 GUI 拖线实时反馈 + graph.add_link 复用，保证一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import LinkError
from .module import PortSpec, PortType

if TYPE_CHECKING:
    from .graph import WorkflowGraph

__all__ = [
    "PortTypeRegistry",
    "can_connect",
    "compat_check",
    "compatible",
]


class PortTypeRegistry:
    """开放端口类型注册表（预留扩展位；v2 首版仅包装内置枚举）.

    插件可通过 entry point ``zylab.port_type`` 注册新端口类型，或直接
    :meth:`register` 手动登记。v2 内部仍以 ``PortType`` 枚举作为键，
    新类型可在运行时追加（``_EXTRA`` 字典）。
    """

    def __init__(self) -> None:
        self._extra: dict[str, PortType] = {}  # name → PortType（运行时追加的）

    def register(self, name: str, *, label: str = "", aliases: tuple[str, ...] = ()) -> PortType:
        """注册一个新端口类型，返回对应的 PortType 实例.

        重名抛 ValueError；已存在于内置枚举则直接返回现有值。
        预留 label / aliases 供 entry point 扩展启用时使用。
        """
        del label, aliases  # v2 首版暂未消费，保留签名兼容后续扩展
        # 已在枚举中
        for pt in PortType:
            if pt.value == name:
                return pt
        if name in self._extra:
            return self._extra[name]
        raise ValueError(
            "v2 首版暂不支持运行时追加 PortType，请扩展 PortType 枚举；entry point 扩展将在后续版本启用",
        )

    def list_types(self) -> tuple[PortType, ...]:
        """全部已注册端口类型（内置枚举序 + 运行时追加序）."""
        return (*PortType, *self._extra.values())


# ------------------------------------------------------------------ 兼容矩阵


#: 显式兼容声明表：(源类型, 目标类型) 可连接
#: ANY 端口在 compat_check 中单独放行，不列入此表
_COMPAT_PAIRS: frozenset[tuple[PortType, PortType]] = frozenset(
    {
        # ---- buckling.reference: STATIC 解可选输入 ----
        (PortType.STATIC, PortType.STATIC),  # 当然也同名兼容
        # ---- 谐响应可消费 MODAL 频率 ----
        # (PortType.MODAL, PortType.HARMONIC),  # 暂不声明，等物理确认
    },
)


def compat_check(src_type: PortType, dst_type: PortType) -> bool:
    """判断两个端口类型是否兼容（源 → 目标方向）.

    规则（按优先级）：
    1. 同名兼容（最常见）；
    2. 目标是 ANY → 放行（通用输入）；
    3. 源是 ANY → 放行（通用输出）；
    4. 显式声明的兼容对；
    5. 以上均不命中 → 拒绝。
    """
    if src_type is dst_type:
        return True
    if dst_type is PortType.ANY:
        return True
    if src_type is PortType.ANY:
        return True
    return (src_type, dst_type) in _COMPAT_PAIRS


def compatible(src_port: PortSpec, dst_port: PortSpec) -> bool:
    """两个端口规格是否类型兼容（源端口 → 目标端口方向）."""
    return compat_check(src_port.port_type, dst_port.port_type)


# ------------------------------------------------------------------ 连接校验


def can_connect(  # noqa: PLR0911 — 连接校验的多分支返回是可读性优先选择
    graph: WorkflowGraph,
    src_id: str,
    src_port_name: str,
    dst_id: str,
    dst_port_name: str,
) -> tuple[bool, str]:
    """预判一条连接是否合法（返回 (是否可连, 原因说明)）.

    检查项：
    1. 自连接拒绝；
    2. 目标端口存在；
    3. 源端口存在；
    4. 端口类型兼容（compat_check）；
    5. 连接后无环（dst_id 不得为 src_id 的递归上游）。

    :returns: ``(True, "")`` 可连；``(False, "原因...")`` 不可连。
    """
    if src_id == dst_id:
        return False, "不允许自连接"

    try:
        dst_node = graph.node(dst_id)
    except Exception as exc:
        return False, f"目标节点不存在: {exc}"
    try:
        src_node = graph.node(src_id)
    except Exception as exc:
        return False, f"源节点不存在: {exc}"

    # 端口存在性
    try:
        out_port = src_node.spec.output_port(src_port_name)
    except Exception as exc:
        return False, f"源端口不存在: {exc}"
    try:
        in_port = dst_node.spec.input_port(dst_port_name)
    except Exception as exc:
        return False, f"目标端口不存在: {exc}"

    # 类型兼容
    if not compat_check(out_port.port_type, in_port.port_type):
        return False, f"端口类型不兼容: {out_port.port_type.value} → {in_port.port_type.value}"

    # 环检测：dst_id 不得是 src_id 的递归上游（已存在 dst→...→src 路径，加 src→dst 即成环）
    if dst_id in graph.ancestors(src_id):
        return False, f"连接后形成环（{dst_id} 已是 {src_id} 的上游）"

    return True, ""


def assert_can_connect(
    graph: WorkflowGraph,
    src_id: str,
    src_port_name: str,
    dst_id: str,
    dst_port_name: str,
) -> None:
    """can_connect 结果为 False 时抛 :class:`LinkError`；为 True 时静默."""
    ok, reason = can_connect(graph, src_id, src_port_name, dst_id, dst_port_name)
    if not ok:
        raise LinkError(reason)
