"""zylab GUI 应用装配（QApplication 工厂、主题加载、样式加载、入口）."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
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

# 箭头三角形 SVG 模板（QSS image 引用；Qt QSS 不支持 border 画三角，
# 须用位图/矢量资源），颜色由主题令牌注入
_ARROW_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><path d="{path}" fill="{color}"/></svg>'
_ARROW_UP_PATH = "M0 6 L5 0 L10 6 Z"
_ARROW_DOWN_PATH = "M0 0 L10 0 L5 6 Z"


def _write_arrow_svgs(pal: theme.Palette) -> dict[str, str]:
    """按主题色生成上/下箭头 SVG 到临时缓存目录，返回 QSS 令牌映射.

    文件名含进程号 + 主题名：xdist 并行测试/多进程场景下各进程
    写各自文件，避免并发重写同一文件导致读到半截 SVG（Qt 解析
    失败即箭头不渲染）。同进程切换主题后清理本进程旧主题文件。
    """
    cache = Path(tempfile.gettempdir()) / "zylab-icons"
    cache.mkdir(parents=True, exist_ok=True)
    color = pal.text_secondary
    tag = f"{os.getpid()}-{pal.name}"
    stale: list[Path] = []
    tokens: dict[str, str] = {}
    for name, path_data in (("arrow-up", _ARROW_UP_PATH), ("arrow-down", _ARROW_DOWN_PATH)):
        target = cache / f"{name}-{tag}.svg"
        target.write_text(_ARROW_SVG.format(path=path_data, color=color), encoding="utf-8")
        tokens[f"QSS_{name.upper().replace('-', '_')}"] = target.as_posix()
        stale.extend(p for p in cache.glob(f"{name}-{os.getpid()}-*.svg") if p != target)
    for path in stale:  # 清理本进程旧主题残留（失败无害，忽略）
        with contextlib.suppress(OSError):
            path.unlink()
    return tokens


def load_stylesheet(palette: theme.Palette | None = None) -> str:
    """加载 QSS 并替换当前主题的设计令牌占位符（含箭头 SVG 资源路径）."""
    pal = palette if palette is not None else theme.current_palette()
    qss_path = Path(__file__).parent / "style.qss"
    tokens = {**theme.qss_tokens(pal), **_write_arrow_svgs(pal)}
    return Template(qss_path.read_text(encoding="utf-8")).substitute(tokens)


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
