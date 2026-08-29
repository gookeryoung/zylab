"""zylab GUI 应用装配（QApplication 工厂、主题加载、样式加载、入口）."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from string import Template

from . import theme
from .qt_compat import QApplication, exec_app

__all__ = [
    "apply_theme",
    "create_app",
    "load_stylesheet",
    "load_theme_name",
    "main",
    "save_theme_name",
]

logger = logging.getLogger(__name__)

_THEME_FILE = "theme.txt"


def load_stylesheet(palette: theme.Palette | None = None) -> str:
    """加载 QSS 并替换当前主题的设计令牌占位符."""
    pal = palette if palette is not None else theme.current_palette()
    qss_path = Path(__file__).parent / "style.qss"
    return Template(qss_path.read_text(encoding="utf-8")).substitute(theme.qss_tokens(pal))


def apply_theme(app: QApplication, name: str) -> theme.Palette:
    """运行时切换主题：更新当前色板并重刷全局样式表.

    Args:
        app: QApplication 实例。
        name: 主题标识（light/dark/high_contrast）。

    Returns:
        生效的色板。

    Raises:
        ValueError: 主题名不存在时抛出（样式表保持不变）。
    """
    pal = theme.palette(name)  # 未知名先抛错，不动当前状态
    theme.set_current_theme(name)
    app.setStyleSheet(load_stylesheet(pal))
    logger.debug("主题已切换: %s", name)
    return pal


def create_app(argv: list[str] | None = None, theme_name: str = theme.DEFAULT_THEME) -> QApplication:
    """创建 QApplication（Fusion 风格 + 指定主题样式表）；已有实例则复用并重刷主题."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    apply_theme(app, theme_name)
    return app


def load_theme_name(data_dir: Path) -> str:
    """读取持久化的主题名；缺失或非法时回退默认主题."""
    path = data_dir / _THEME_FILE
    if not path.exists():
        return theme.DEFAULT_THEME
    try:
        name = path.read_text(encoding="utf-8").strip()
        theme.palette(name)  # 校验合法
        return name
    except (OSError, ValueError):
        logger.warning("主题文件损坏，回退默认主题: %s", path)
        return theme.DEFAULT_THEME


def save_theme_name(data_dir: Path, name: str) -> None:
    """持久化主题名（失败仅告警，不影响切换）."""
    theme.palette(name)  # 校验合法
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / _THEME_FILE).write_text(name, encoding="utf-8")


def main() -> int:  # pragma: no cover（事件循环阻塞，需图形环境手动测试）
    """启动 GUI 应用."""
    from zylab.core.log import setup_logging

    setup_logging("dev")
    from zylab.core.config import default_data_dir

    app = create_app(theme_name=load_theme_name(default_data_dir()))
    from .main_window import MainWindow  # 惰性导入，加速 --help 等非 GUI 路径

    window = MainWindow()
    window.show()
    return exec_app(app)
