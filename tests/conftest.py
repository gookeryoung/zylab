"""pytest 全局配置：无头环境强制 Qt offscreen 平台."""

from __future__ import annotations

import os


def pytest_configure() -> None:
    """CI 或无显示环境强制 offscreen，避免真实窗口阻塞测试."""
    if os.environ.get("CI") or not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
