"""zylab.console 命令历史（内存环形缓冲 + JSON 原子持久化）."""

from __future__ import annotations

import json
import logging
from pathlib import Path

__all__ = ["CommandHistory"]

logger = logging.getLogger(__name__)


class CommandHistory:
    """命令历史.

    - ``add`` 跳过空命令与连续重复；
    - ``previous``/``next`` 浏览历史，首次 ``previous`` 会暂存当前输入，浏览到底后 ``next`` 还原；
    - 持久化为 JSON 数组，原子写入（临时文件 + replace）。
    """

    def __init__(self, path: Path | None = None, maxsize: int = 1000) -> None:
        """初始化历史；``path`` 非空时启用持久化."""
        self._path = path
        self._maxsize = maxsize
        self._entries: list[str] = []
        self._cursor: int | None = None  # None 表示未在浏览
        self._stashed = ""  # 浏览前暂存的当前输入

    @property
    def entries(self) -> list[str]:
        """全部历史条目（旧 → 新）."""
        return list(self._entries)

    def add(self, command: str) -> None:
        """追加命令；空串与连续重复不入库；浏览游标复位."""
        command = command.rstrip()
        if not command or (self._entries and self._entries[-1] == command):
            self._cursor = None
            return
        self._entries.append(command)
        if len(self._entries) > self._maxsize:
            del self._entries[: len(self._entries) - self._maxsize]
        self._cursor = None

    def previous(self, current: str = "") -> str | None:
        """上翻一条历史；已在最新条目时返回 None."""
        if not self._entries:
            return None
        if self._cursor is None:
            self._stashed = current
            self._cursor = len(self._entries) - 1
        elif self._cursor > 0:
            self._cursor -= 1
        else:
            return None
        return self._entries[self._cursor]

    def next(self) -> str | None:
        """下翻一条历史；越过最新条目时还原暂存输入并复位游标."""
        if self._cursor is None:
            return None
        if self._cursor < len(self._entries) - 1:
            self._cursor += 1
            return self._entries[self._cursor]
        self._cursor = None
        return self._stashed

    def load(self) -> None:
        """从文件加载历史；文件缺失或损坏时静默忽略."""
        if self._path is None or not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("命令历史文件损坏，忽略: %s", self._path)
            return
        if isinstance(data, list):
            self._entries = [str(item) for item in data][-self._maxsize :]
        logger.debug("命令历史已加载: %d 条", len(self._entries))

    def save(self) -> None:
        """原子保存历史到文件（未配置路径时跳过）."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(self._entries, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            logger.warning("命令历史保存失败: %s", self._path, exc_info=True)
            tmp.unlink(missing_ok=True)
