"""zylab GUI 入口（PySide2/PySide6 双兼容）."""

from __future__ import annotations

import sys

try:  # PySide2 支持 Python 3.6-3.10；PySide6 支持 3.8+
    from PySide2.QtWidgets import QApplication, QLabel  # type: ignore[missing-import]
except ImportError:
    from PySide6.QtWidgets import QApplication, QLabel

__all__ = ["main"]


def main() -> int:  # pragma: no cover
    """启动 GUI 应用（事件循环阻塞，需图形环境手动测试）."""
    app = QApplication(sys.argv)
    label = QLabel("zylab 已就绪")
    label.setWindowTitle("zylab")
    label.resize(400, 200)
    label.show()
    # PySide2 用 exec_()，PySide6 推荐 exec()（exec_ 已弃用）
    run = app.exec if hasattr(app, "exec") else app.exec_
    return run()


if __name__ == "__main__":
    sys.exit(main())
