"""zylab GUI 应用装配（QApplication 工厂、样式加载、入口）."""

from __future__ import annotations

import sys
from pathlib import Path
from string import Template

from . import theme
from .qt_compat import QApplication, exec_app

__all__ = ["create_app", "load_stylesheet"]


def load_stylesheet() -> str:
    """加载 QSS 并替换设计令牌占位符."""
    qss_path = Path(__file__).parent / "style.qss"
    return Template(qss_path.read_text(encoding="utf-8")).substitute(theme.QSS_TOKENS)


def create_app(argv: list[str] | None = None) -> QApplication:
    """创建 QApplication（应用全局样式表与 Fusion 风格，跨平台观感一致）；已有实例则复用."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet())
    return app


def main() -> int:  # pragma: no cover（事件循环阻塞，需图形环境手动测试）
    """启动 GUI 应用."""
    from zylab.core.log import setup_logging

    setup_logging("dev")
    app = create_app()
    from .main_window import MainWindow  # 惰性导入，加速 --help 等非 GUI 路径

    window = MainWindow()
    window.show()
    return exec_app(app)
