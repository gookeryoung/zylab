"""zylab.sci 工作区管理（MATLAB 式 cwd + whos）.

- :class:`WorkspaceManager`：当前工作目录管理器，负责持久化上一次关闭前的
  工作区路径到 ``workspace.json``，启动时自动恢复；切换路径经 EventBus
  广播 ``workspace.changed`` 事件，供内核和 UI 订阅。
- :func:`whos` / :func:`format_whos`：从命名空间提取变量的结构化描述，
  MATLAB 风格 ``whos`` 表格输出，供 REPL 命令与 GUI 变量浏览器共用。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zylab.core import EventBus, default_data_dir

__all__ = ["CURRENT_WORKSPACE_FILE", "TOPIC_WORKSPACE_CHANGED", "VarInfo", "WorkspaceManager", "whos"]

logger = logging.getLogger(__name__)

#: 值预览的最大长度
_PREVIEW_MAXLEN = 60

#: 工作区持久化文件名（放在 default_data_dir 下）
CURRENT_WORKSPACE_FILE = "workspace.json"

#: 工作区变更事件主题
TOPIC_WORKSPACE_CHANGED = "workspace.changed"

#: 历史工作区保留的最大条数
_MAX_HISTORY = 10


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


@dataclass(frozen=True)
class WorkspaceInfo:
    """WorkspaceManager 状态快照（TOPIC_WORKSPACE_CHANGED 事件载荷）.

    :param path: 当前工作区绝对路径。
    :param prev_path: 切换前的路径（首次恢复时为 None）。
    :param source: 触发来源（"init" / "set" / "restore"）。
    """

    path: Path
    prev_path: Path | None
    source: str


class WorkspaceManager:
    """MATLAB 风格当前工作目录管理器（Qt-free）.

    职责：
    1. 维护 ``cwd`` 属性（类型为 :class:`pathlib.Path`），默认为 ``Path.cwd()``。
    2. 切换路径经 :meth:`set_workspace` 校验（必须存在且为目录），
       成功后同时更新 ``os.chdir()`` 使 ``Path.cwd()`` 一致。
    3. 切换成功后将新路径前置到 ``history``（去重、限制最近 10 条），
       下次启动可经 :meth:`recent_workspaces` 取出供 UI 下拉展示。
    4. 关闭前调用 :meth:`save` 持久化当前路径与历史到
       ``default_data_dir()/workspace.json``，下次启动 :meth:`load` 自动恢复。
    5. 切换后经 :class:`EventBus` 广播 ``TOPIC_WORKSPACE_CHANGED`` 事件，
       内核（注入 namespace["cwd"]）与 UI（状态栏显示）各取所需。

    持久化 JSON 结构::

        {"path": "F:/projects/my-work", "history": ["F:/prev1", "F:/prev2"]}

    历史数组长度上限 :data:`_MAX_HISTORY`（10），当前路径不出现在历史中。

    用法::

        bus = EventBus()
        wm = WorkspaceManager(bus)
        wm.load()                       # 恢复上次关闭前的路径 + 历史
        wm.set_workspace(Path.home())   # 切换工作区（自动并入历史）
        wm.recent_workspaces()          # 最近 10 个（不含当前）
        wm.save()                       # 关闭前持久化
    """

    def __init__(self, bus: EventBus | None = None, data_dir: Path | None = None) -> None:
        """初始化管理器（默认 cwd = Path.cwd()，不触发 EventBus 事件）.

        :param bus: 事件总线（可选）。
        :param data_dir: 持久化目录；默认 ``default_data_dir()``。
        """
        self.bus = bus or EventBus()
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self._cwd = Path.cwd().resolve()
        self._history: list[Path] = []  # 最近切换过的路径（不含当前），首项为最新

    @property
    def cwd(self) -> Path:
        """当前工作区绝对路径（只读）."""
        return self._cwd

    def recent_workspaces(self, limit: int = 10) -> list[Path]:
        """返回最近切换过的工作区路径列表（不含当前）。

        :param limit: 最多返回的条数（默认 10）。
        :returns: 最近工作区路径列表，按时间倒序排列。
        """
        return list(self._history[:limit])

    # ------------------------------------------------------------ 切换与持久化

    def set_workspace(self, path: str | Path) -> WorkspaceInfo:
        """切换工作区（校验 + 应用 + 更新历史 + 广播事件）.

        路径必须存在且为目录；非法路径直接拒绝（不抛异常，返回带错误标记
        的 WorkspaceInfo，调用方可据此提示 UI）。校验通过后同时更新
        ``os.chdir()`` 使全局 cwd 与管理器一致，并将旧 cwd 并入历史。

        :param path: 目标路径（字符串或 Path）。
        :returns: 工作区状态快照。
        :raises OSError: ``os.chdir()`` 失败时抛出（极罕见）。
        """
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            # 路径不存在或不是目录：静默拒绝，保持当前 cwd 不变
            logger.warning("拒绝无效工作区路径: %s", target)
            return WorkspaceInfo(path=self._cwd, prev_path=self._cwd, source="invalid")
        if target == self._cwd:
            return WorkspaceInfo(path=self._cwd, prev_path=None, source="same")
        prev = self._cwd
        self._cwd = target
        os.chdir(target)
        self._update_history(prev)
        info = WorkspaceInfo(path=target, prev_path=prev, source="set")
        self.bus.publish(TOPIC_WORKSPACE_CHANGED, info)
        logger.info("工作区已切换: %s", target)
        return info

    def _update_history(self, prev_cwd: Path) -> None:
        """将旧 cwd 并入历史（去重并限制数量，当前路径不出现在历史中）."""
        # 过滤掉与当前 cwd 或新加入的旧 cwd 重复的条目
        filtered = [p for p in self._history if p not in (prev_cwd, self._cwd)]
        self._history = [prev_cwd, *filtered][:_MAX_HISTORY]

    def save(self) -> Path | None:
        """持久化当前工作区路径与历史（关闭前调用）.

        写入 ``default_data_dir()/workspace.json``（包含 ``{"path": "...", "history": [...]}``）。
        目录不存在时自动创建。

        :returns: 写入路径；持久化失败返回 None（仅记日志不抛）。
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        target = self.data_dir / CURRENT_WORKSPACE_FILE
        try:
            # 历史持久化前过滤掉当前路径，避免重复；数量上限 _MAX_HISTORY
            history_strs = [str(p) for p in self._history if p != self._cwd][:_MAX_HISTORY]
            payload = json.dumps(
                {"path": str(self._cwd), "history": history_strs},
                ensure_ascii=False,
                indent=2,
            )
            target.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("工作区路径持久化失败: %s", exc)
            return None
        logger.debug("工作区路径已持久化: %s -> %s", self._cwd, target)
        return target

    def load(self) -> WorkspaceInfo | None:
        """从持久化文件恢复工作区路径与历史（启动时调用）.

        文件缺失、JSON 非法或路径已不存在时静默跳过，保持初始化 cwd。
        历史数组中非字符串条目与不存在的路径会被自动剔除。

        :returns: 成功恢复时返回 WorkspaceInfo；无文件或非法时返回 None。
        """
        target = self.data_dir / CURRENT_WORKSPACE_FILE
        if not target.is_file():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("工作区持久化文件读取失败: %s", exc)
            return None
        # 先尝试恢复历史（即使当前路径恢复失败，历史仍可在后续有效切换时合并）
        raw_history = data.get("history")
        if isinstance(raw_history, list):
            history: list[Path] = []
            seen: set[str] = set()
            for item in raw_history:
                if not isinstance(item, str) or not item.strip():
                    continue
                try:
                    p = Path(item).expanduser().resolve()
                except (OSError, ValueError):
                    continue
                key = str(p).lower()
                if key in seen or not p.is_dir():
                    continue
                seen.add(key)
                history.append(p)
            self._history = history[:_MAX_HISTORY]
        # 再恢复当前路径
        raw_path = data.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        info = self.set_workspace(raw_path)  # 复用切换逻辑（校验 + os.chdir + 广播 + 并入历史）
        # 持久化路径已不存在时 set_workspace 返回 source="invalid"，须静默跳过
        if info.source == "invalid":
            return None
        return info
