"""zylab 事件总线.

进程内轻量 pub/sub：core/sci/fea 等 Qt-free 层通过它解耦通信，
GUI 层可将主题桥接为 Qt 信号。回调异常仅记录日志，不中断发布流程。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

__all__ = ["EventBus"]

logger = logging.getLogger(__name__)

Callback = Callable[[Any], None]


class EventBus:
    """线程安全的事件总线.

    用法::

        bus = EventBus()
        bus.subscribe("solver.progress", lambda p: print(p))
        bus.publish("solver.progress", {"value": 0.5})
    """

    def __init__(self) -> None:
        """初始化空总线."""
        self._subscribers: dict[str, list[Callback]] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Callback) -> None:
        """订阅主题；同一回调重复订阅只生效一次."""
        with self._lock:
            callbacks = self._subscribers.setdefault(topic, [])
            if callback not in callbacks:
                callbacks.append(callback)

    def unsubscribe(self, topic: str, callback: Callback) -> None:
        """取消订阅；主题或回调不存在时静默忽略."""
        with self._lock:
            callbacks = self._subscribers.get(topic)
            if not callbacks:
                return
            try:
                callbacks.remove(callback)
            except ValueError:
                return
            if not callbacks:
                del self._subscribers[topic]

    def publish(self, topic: str, payload: Any = None) -> None:
        """发布事件，同步依次调用订阅回调；单个回调异常仅记录不影响其他订阅者."""
        with self._lock:
            callbacks = list(self._subscribers.get(topic, ()))
        for callback in callbacks:
            try:
                callback(payload)
            except Exception:
                logger.warning("事件回调执行失败: topic=%s callback=%r", topic, callback, exc_info=True)

    def subscriber_count(self, topic: str) -> int:
        """返回某主题的订阅者数量（用于诊断与测试）."""
        with self._lock:
            return len(self._subscribers.get(topic, ()))
