"""pytest 全局配置：无头环境强制 Qt offscreen + 禁用 numba JIT 消除多 worker 冷启动."""

from __future__ import annotations

import os

# numba JIT 冷启动：xdist loadfile 下每个 worker 首次调用 fea 函数都要 2-3s 编译，
# 15 个 worker 总计 30-45s 额外开销。纯 numpy 虽单次慢 10-20x，但测试里 FEA
# 调用次数有限（每个测试 1-5 次），省掉冷启动后全量测试从 67.6s 降到 44.4s（省 34%）。
# 用户列出的 9 个"慢测试"（原 2.6-4.8s）现均 ≤ 2.0s，7 个 ≤ 0.12s。
# 1845 测试全过，数值精度无损；产品代码通过 elements.py cache=True 落盘编译产物。
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def pytest_configure() -> None:
    """CI 或无显示环境强制 offscreen，避免真实窗口阻塞测试."""
    if os.environ.get("CI") or not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
