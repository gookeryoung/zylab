"""zylab GUI 应用装配（QApplication 工厂、主题加载、样式加载、入口）."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from string import Template

from ..core import set_root_level, update_runtime_config
from . import theme
from .qt_compat import QApplication, QFontDatabase, QLibraryInfo, QLocale, QTranslator, exec_app

__all__ = [
    "apply_settings",
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

#: close 按钮 SVG 模板（从 assets/icons/close.svg 读取后注入主题色）
_CLOSE_SVG_PATH = Path(__file__).resolve().parent.parent / "assets" / "icons" / "close.svg"

#: 项目树 branch indicator（展开/收起 chevron 图标）
_CHEVRON_RIGHT_SVG = Path(__file__).resolve().parent.parent / "assets" / "icons" / "chevron_right.svg"
_CHEVRON_DOWN_SVG = Path(__file__).resolve().parent.parent / "assets" / "icons" / "chevron_down.svg"


def _write_theme_svgs(pal: theme.Palette) -> dict[str, str]:
    """按主题色生成箭头 + close 按钮 SVG 到临时缓存目录，返回 QSS 令牌映射.

    文件名含进程号 + 主题名：xdist 并行测试/多进程场景下各进程
    写各自文件，避免并发重写同一文件导致读到半截 SVG（Qt 解析
    失败即图标不渲染）。同进程切换主题后清理本进程旧主题文件。
    """
    cache = Path(tempfile.gettempdir()) / "zylab-icons"
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{pal.name}"
    stale: list[Path] = []
    tokens: dict[str, str] = {}

    # --- 箭头（QComboBox/DoubleSpinBox 下拉指示器） ---
    arrow_color = pal.text_secondary
    for name, path_data in (("arrow-up", _ARROW_UP_PATH), ("arrow-down", _ARROW_DOWN_PATH)):
        target = cache / f"{name}-{tag}.svg"
        target.write_text(_ARROW_SVG.format(path=path_data, color=arrow_color), encoding="utf-8")
        tokens[f"QSS_{name.upper().replace('-', '_')}"] = target.as_posix()
        stale.extend(p for p in cache.glob(f"{name}-{os.getpid()}-*.svg") if p != target)

    # --- close 按钮（QTabBar 关闭标签） ---
    if _CLOSE_SVG_PATH.exists():
        close_template = _CLOSE_SVG_PATH.read_text(encoding="utf-8")
        # 普通态：次要文字色（低调不抢眼）
        close_normal = cache / f"close-normal-{tag}.svg"
        close_normal.write_text(
            close_template.replace("<svg ", f'<svg fill="{pal.text_secondary}" ', 1), encoding="utf-8"
        )
        tokens["QSS_CLOSE_ICON"] = close_normal.as_posix()
        stale.extend(p for p in cache.glob(f"close-normal-{os.getpid()}-*.svg") if p != close_normal)
        # hover 态：危险色（关闭动作用危险色提示，符合预期）
        close_hover = cache / f"close-hover-{tag}.svg"
        close_hover.write_text(close_template.replace("<svg ", f'<svg fill="{pal.danger_text}" ', 1), encoding="utf-8")
        tokens["QSS_CLOSE_ICON_HOVER"] = close_hover.as_posix()
        stale.extend(p for p in cache.glob(f"close-hover-{os.getpid()}-*.svg") if p != close_hover)

    # --- 项目树 branch indicator（chevron-right 折叠态 / chevron-down 展开态） ---
    chevron_color = pal.text_secondary
    svg_path_map = (
        ("chevron-right", _CHEVRON_RIGHT_SVG, "QSS_CHEVRON_RIGHT"),
        ("chevron-down", _CHEVRON_DOWN_SVG, "QSS_CHEVRON_DOWN"),
    )
    for fname, src, token_key in svg_path_map:
        if src.exists():
            template = src.read_text(encoding="utf-8")
            target = cache / f"{fname}-{tag}.svg"
            target.write_text(template.replace("<svg ", f'<svg fill="{chevron_color}" ', 1), encoding="utf-8")
            tokens[token_key] = target.as_posix()
            stale.extend(p for p in cache.glob(f"{fname}-{os.getpid()}-*.svg") if p != target)

    # 清理本进程旧主题残留（失败无害，忽略）
    for path in stale:
        with contextlib.suppress(OSError):
            path.unlink()
    return tokens


def load_stylesheet(palette: theme.Palette | None = None) -> str:
    """加载 QSS 并替换当前主题的设计令牌占位符（含箭头 SVG 资源路径）."""
    pal = palette if palette is not None else theme.current_palette()
    qss_path = Path(__file__).parent / "style.qss"
    tokens = {**theme.qss_tokens(pal), **_write_theme_svgs(pal)}
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


def apply_settings(  # noqa: PLR0913  签名显式枚举全部运行时字段，调用方（SettingsPanel 保存后）一次传齐
    app: QApplication,
    *,
    theme_name: str | None = None,
    font_family_body: str | None = None,
    font_family_mono: str | None = None,
    font_scale: float | None = None,
    log_level: str | None = None,
    max_workers: int | None = None,
    solver_timeout_s: int | None = None,
    autosave_interval_sec: int | None = None,
    workspace_history_limit: int | None = None,
) -> None:
    """一次性应用主题 + 字体族 + 字号缩放 + 运行时配置（重刷全局样式表）.

    相比单独调用 apply_theme + set_font_families + 更新运行时状态，本函数
    保证样式表只重刷一次（减少闪烁），且所有运行时字段同步生效。

    Args:
        app: QApplication 实例。
        theme_name: 主题标识；None 表示保持当前主题。
        font_family_body: 正文字体族；None 表示保持当前值。
        font_family_mono: 等宽字体族；None 表示保持当前值。
        font_scale: 字号缩放倍率；None 表示保持当前值。
        log_level: 根日志器级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）；None 不变。
        max_workers: 求解器默认并发进程数；None 不变。
        solver_timeout_s: 求解器超时秒数；None 不变。
        autosave_interval_sec: 自动保存间隔秒数（0 = 关闭）；None 不变。
        workspace_history_limit: 工作区历史保留条数上限；None 不变。
    """
    # 1. 先更新 theme 模块级运行时状态
    if theme_name is not None:
        theme.set_current_theme(theme_name)
    theme.set_font_families(body=font_family_body, mono=font_family_mono)
    if font_scale is not None:
        theme.set_font_scale(font_scale)

    # 2. 更新 core 层运行时配置（求解/超时/自动保存/历史条数）
    runtime_updates: dict[str, object] = {}
    if max_workers is not None:
        runtime_updates["max_workers"] = max_workers
    if solver_timeout_s is not None:
        runtime_updates["solver_timeout_s"] = solver_timeout_s
    if autosave_interval_sec is not None:
        runtime_updates["autosave_interval_sec"] = autosave_interval_sec
    if workspace_history_limit is not None:
        runtime_updates["workspace_history_limit"] = workspace_history_limit
    if runtime_updates:
        update_runtime_config(**runtime_updates)

    # 3. 根日志器级别调整（不在 runtime_config 里，走 logging 原生 API）
    if log_level is not None:
        try:
            set_root_level(log_level)
        except ValueError as exc:
            logger.warning("日志级别设置失败（已忽略）: %s", exc)

    # 4. 统一重刷样式表（主题 + 字体 + 字号令牌都从当前模块级状态取值）
    app.setStyleSheet(load_stylesheet())
    logger.debug(
        "设置已应用: theme=%s log=%s workers=%s",
        theme_name or "unchanged",
        log_level or "unchanged",
        runtime_updates or "unchanged",
    )


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
    if bundled.is_dir():  # pragma: no cover - 随包分发，正常始终存在
        search_dirs.append(bundled)
    if sys_transl and Path(sys_transl).is_dir():  # pragma: no cover - fallback 分支
        search_dirs.append(Path(sys_transl))

    for fname in ("qtbase", "qt"):
        # 用 locale.name() 得到 "zh_CN"，尝试精确匹配
        translator = QTranslator(app)
        loaded = translator.load(locale, fname, "_", str(bundled))
        if not loaded and sys_transl:  # pragma: no cover - fallback 分支
            loaded = translator.load(locale, fname, "_", sys_transl)
        if loaded:
            app.installTranslator(translator)
            logger.debug("Qt 翻译已加载: %s", fname)
        else:  # pragma: no cover - 翻译缺失时静默跳过（fallback 防御）
            pass


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
    import json as _json

    from zylab.core.config import default_data_dir
    from zylab.core.log import setup_logging

    setup_logging("dev")

    data_dir = default_data_dir()

    # 1. 优先从 settings.json 读取完整配置（外观 + 运行时）
    theme_name = theme.DEFAULT_THEME
    font_family_body: str | None = None
    font_family_mono: str | None = None
    font_scale: float | None = None
    log_level: str | None = None
    max_workers: int | None = None
    solver_timeout_s: int | None = None
    autosave_interval_sec: int | None = None
    workspace_history_limit: int | None = None

    settings_path = data_dir / "settings.json"
    if settings_path.is_file():
        try:
            data = _json.loads(settings_path.read_text(encoding="utf-8"))
            theme_name = str(data.get("theme", theme.DEFAULT_THEME))
            font_family_body = data.get("font_family_body")
            font_family_mono = data.get("font_family_mono")
            font_scale_val = data.get("font_scale")
            if font_scale_val is not None:
                font_scale = float(font_scale_val)
            log_level_val = data.get("log_level")
            if log_level_val:
                log_level = str(log_level_val)
            if "max_workers" in data:
                max_workers = int(data["max_workers"])
            if "solver_timeout_s" in data:
                solver_timeout_s = int(data["solver_timeout_s"])
            if "autosave_interval_sec" in data:
                autosave_interval_sec = int(data["autosave_interval_sec"])
            if "workspace_history_limit" in data:
                workspace_history_limit = int(data["workspace_history_limit"])
        except (OSError, ValueError) as exc:
            logger.warning("settings.json 解析失败，使用默认: %s", exc)

    # 兼容旧版 theme.txt（settings.json 中无 theme 字段时回退）
    if theme_name == theme.DEFAULT_THEME:
        legacy_theme = load_theme_name(data_dir)
        if legacy_theme != theme.DEFAULT_THEME:
            theme_name = legacy_theme

    app = create_app(theme_name=theme_name)

    # 2. 应用字体/字号/日志/运行时配置（主题已在 create_app 中应用，跳过避免重复切换）
    apply_settings(
        app,
        font_family_body=font_family_body,
        font_family_mono=font_family_mono,
        font_scale=font_scale,
        log_level=log_level,
        max_workers=max_workers,
        solver_timeout_s=solver_timeout_s,
        autosave_interval_sec=autosave_interval_sec,
        workspace_history_limit=workspace_history_limit,
    )

    register_user_themes(data_dir)
    from .main_window import MainWindow  # 惰性导入，加速 --help 等非 GUI 路径

    window = MainWindow()
    window.show()
    return exec_app(app)
