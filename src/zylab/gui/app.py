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
from .qt_compat import QApplication, QFontDatabase, QLibraryInfo, QLocale, QTranslator, exec_app

__all__ = [
    "apply_theme",
    "create_app",
    "load_stylesheet",
    "load_theme_name",
    "main",
    "register_fonts",
    "register_user_themes",
    "save_theme_name",
]

logger = logging.getLogger(__name__)

_THEME_FILE = "theme.txt"

#: 内置字体目录（随包分发，当前为 DejaVu Sans Mono 等宽件）
_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def register_fonts() -> list[str]:
    """注册 assets/fonts 下内置字体到应用字体库，返回已生效的字体家族名.

    重复调用无害（Qt 对同一字体文件幂等）；注册失败（文件缺失/损坏）仅告警，
    界面回退系统等宽字体，不中断启动。
    """
    loaded: list[str] = []
    for path in sorted(_FONTS_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            logger.warning("内置字体注册失败: %s", path.name)
            continue
        loaded.extend(QFontDatabase.applicationFontFamilies(font_id))
    logger.debug("内置字体已注册: %s", loaded or "无")
    return loaded


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
    """创建 QApplication（Fusion 风格 + 指定主题样式表 + Qt 标准对话框本地化）.

    加载系统 locale 对应的 Qt 翻译文件（如 ``qtbase_zh_CN.qm``），
    使 QMessageBox/QFileDialog/QInputDialog 等标准对话框的按钮文字
    （Save/Discard/Cancel 等）显示为本地化语言。
    """
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    _load_qt_translations(app)
    register_fonts()
    apply_theme(app, theme_name)
    return app


def _load_qt_translations(app: QApplication) -> None:
    """加载 Qt 标准翻译文件，使 QMessageBox/QFileDialog 等按钮本地化.

    优先加载项目 ``assets/translations/`` 下的内置翻译文件（兜底 PySide2
    Windows 发行包漏掉的 ``qtbase_zh_CN.qm``），再 fallback 到 PySide2
    自带翻译目录。仅中文环境生效。
    """
    locale = QLocale.system()
    if locale.language() != QLocale.Chinese:
        return  # 非中文环境跳过，Qt 自带英文无需翻译

    # 项目内置翻译目录（随包分发，兜底 PySide2 发行包缺失）
    bundled = Path(__file__).resolve().parent.parent / "assets" / "translations"

    # Qt6 用 .path()，Qt5 用 .location()
    path_getter = getattr(QLibraryInfo, "path", getattr(QLibraryInfo, "location", None))
    sys_transl = path_getter(QLibraryInfo.TranslationsPath) if path_getter else None

    # 候选搜索路径：项目内置 → 系统 PySide2 自带
    search_dirs: list[Path] = []
    if bundled.is_dir():
        search_dirs.append(bundled)
    if sys_transl and Path(sys_transl).is_dir():
        search_dirs.append(Path(sys_transl))

    for fname in ("qtbase", "qt"):
        # 用 locale.name() 得到 "zh_CN"，尝试精确匹配
        translator = QTranslator(app)
        loaded = translator.load(locale, fname, "_", str(bundled))
        if not loaded and sys_transl:
            loaded = translator.load(locale, fname, "_", sys_transl)
        if loaded:
            app.installTranslator(translator)
            logger.debug("Qt 翻译已加载: %s", fname)


def register_user_themes(data_dir: Path) -> list[str]:
    """注册用户主题扩展目录（数据目录 ``themes/``，未来配置扩展点）.

    用户可放置部分字段 JSON 微调内置主题，或新增完整主题；
    目录不存在时为无操作。

    Args:
        data_dir: 应用数据目录。

    Returns:
        本次新引入的主题名列表。
    """
    from .theme import register_theme_dir

    directory = data_dir / "themes"
    if not directory.is_dir():
        return []
    added = register_theme_dir(directory)
    logger.debug("用户主题已注册: %s", added or "无")
    return added


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
    register_user_themes(default_data_dir())
    from .main_window import MainWindow  # 惰性导入，加速 --help 等非 GUI 路径

    window = MainWindow()
    window.show()
    return exec_app(app)
