"""core.events 事件总线测试."""

from __future__ import annotations

import threading

from zylab.core.events import EventBus


def test_subscribe_and_publish() -> None:
    """订阅后发布应正确触发回调."""
    bus = EventBus()
    results = []

    def cb(payload):
        results.append(payload)

    bus.subscribe("test.topic", cb)
    bus.publish("test.topic", 42)
    assert results == [42]


def test_unsubscribe() -> None:
    """取消订阅后不应再收到事件."""
    bus = EventBus()
    results = []

    def cb(payload):
        results.append(payload)

    bus.subscribe("test.topic", cb)
    bus.publish("test.topic", 1)
    bus.unsubscribe("test.topic", cb)
    bus.publish("test.topic", 2)
    assert results == [1]


def test_unsubscribe_nonexistent() -> None:
    """对未订阅的主题/回调调用 unsubscribe 应静默忽略."""
    bus = EventBus()

    def noop(_):
        pass

    bus.unsubscribe("none", noop)


def test_multiple_subscribers() -> None:
    """同一主题多个订阅者应全部被调用."""
    bus = EventBus()
    a, b = [], []

    def cb_a(p):
        a.append(p)

    def cb_b(p):
        b.append(p)

    bus.subscribe("multi", cb_a)
    bus.subscribe("multi", cb_b)
    bus.publish("multi", "x")
    assert a == ["x"] and b == ["x"]


def test_callback_exception_isolated() -> None:
    """单个回调异常不应阻断其余订阅者."""
    bus = EventBus()
    results = []

    def ok1(_):
        results.append("ok")

    def boom(_):
        raise RuntimeError("boom")

    def ok2(_):
        results.append("ok2")

    bus.subscribe("fail", ok1)
    bus.subscribe("fail", boom)
    bus.subscribe("fail", ok2)
    bus.publish("fail", None)
    assert results == ["ok", "ok2"]


def test_duplicate_subscribe_ignored() -> None:
    """同一回调重复订阅只生效一次."""
    bus = EventBus()
    results = []

    def cb(p):
        results.append(p)

    bus.subscribe("dup", cb)
    bus.subscribe("dup", cb)
    bus.publish("dup", 1)
    assert results == [1]


def test_thread_safety_publish() -> None:
    """多线程并发发布应无竞态（至少不会崩溃）."""
    bus = EventBus()
    results = []
    lock = threading.Lock()

    def cb(payload):
        with lock:
            results.append(payload)

    bus.subscribe("thread", cb)
    threads = [threading.Thread(target=bus.publish, args=("thread", i)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 100


def test_subscriber_count() -> None:
    """subscriber_count 应返回正确订阅者数."""
    bus = EventBus()

    def noop(_):
        pass

    def noop2(_):
        pass

    assert bus.subscriber_count("empty") == 0
    bus.subscribe("cnt", noop)
    bus.subscribe("cnt", noop2)
    assert bus.subscriber_count("cnt") == 2


def test_unsubscribe_callback_not_in_topic() -> None:
    """主题存在但回调不在订阅列表时，unsubscribe 应静默忽略."""
    bus = EventBus()

    def cb1(_):
        pass

    def cb2(_):
        pass

    bus.subscribe("topic", cb1)
    bus.unsubscribe("topic", cb2)  # cb2 未订阅，不抛异常
    assert bus.subscriber_count("topic") == 1
