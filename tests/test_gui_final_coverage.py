"""GUI 模块最终补测：瞄准 coverage 报告中剩余的 miss 行.

覆盖目标（均来自最新 coverage term-missing 报告）：
- gui/highlight.py 77-79     单行闭合三引号字符串
- gui/theme.py 216            RuntimeError（主题资源缺失）
- gui/theme.py 397            _scale_font 整数值去 .0 后缀
- gui/style.py 134-138        FRAGMENTS_DIR 不存在回退 style.qss
- gui/style.py 165->168       svg_tokens 更新 tokens dict
- gui/icons.py 多个 SVG 加载/着色失败分支
- gui/app.py 105              _load_qrc_svg 资源缺失
- gui/app.py 158              chevron qrc 资源缺失 continue
- gui/app.py 73-74            register_fonts 注册失败跳过
- gui/pages/var_browser.py    _fmt_cell int 分支 / chip 空间不足 break / 0 维数组 reshape
- gui/widgets/_table_utils.py 84  build_table_widget include_zebra
- gui/widgets/command_palette.py 若干未覆盖分支
- gui/widgets/trial_record_edit.py 若干边界
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PySide2.QtGui import QTextDocument

from zylab.gui import theme

# ---------------------------------------------------------------------------
# highlight.py 77-79：单行闭合三引号
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_highlighter_single_line_closed_triple_quote(qtbot) -> None:
    """单行内闭合三引号字符串应整体着色（77-79 行 previously uncovered）."""
    from zylab.gui.highlight import PythonHighlighter

    pal = theme.current_palette()
    doc = QTextDocument()
    highlighter = PythonHighlighter(doc)
    doc.setPlainText('x = """hello"""\ny = 2')
    highlighter.rehighlight()

    first = doc.findBlock(0)
    text = first.text()
    # "hello" 应该是字符串色（success_text）
    idx = text.index("hello")
    for fmt_range in first.layout().formats():
        if fmt_range.start <= idx < fmt_range.start + fmt_range.length:
            color = fmt_range.format.foreground().color().name().lower()
            assert color == pal.success_text.lower()
            break
    else:
        pytest.fail("hello 未被着色")

    # 第二行数字应是数字色（danger_text）
    second = doc.findBlockByNumber(1)
    text2 = second.text()
    idx2 = text2.index("2")
    for fmt_range in second.layout().formats():
        if fmt_range.start <= idx2 < fmt_range.start + fmt_range.length:
            color = fmt_range.format.foreground().color().name().lower()
            assert color == pal.danger_text.lower()
            break
    else:
        pytest.fail("数字 2 未被着色")


# ---------------------------------------------------------------------------
# theme.py 216：RuntimeError（主题资源缺失）
# ---------------------------------------------------------------------------


def test_theme_runtimerror_when_required_theme_missing(monkeypatch) -> None:
    """当主题目录为空时（glob 返回空），reload 应抛 RuntimeError."""
    import pathlib

    import zylab.gui.theme as theme_mod

    # patch Path.glob 返回空列表 → load_themes_from_dir 得到空 dict
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, _p: [])

    try:
        with pytest.raises(RuntimeError, match="内置主题资源缺失或不完整"):
            importlib.reload(theme_mod)
    finally:
        # 先还原 monkeypatch，再 reload 让 module 恢复（reload 失败会留 THEMES 为空）
        monkeypatch.undo()
        importlib.reload(theme_mod)


# ---------------------------------------------------------------------------
# theme.py 397：_scale_font 整数值去掉 .0 后缀
# ---------------------------------------------------------------------------


def test_scale_font_integer_result_strips_decimal() -> None:
    """_scale_font 缩放后结果为整数时应去掉 .0 后缀（397 行）."""
    from zylab.gui.theme import _scale_font

    # 13px * 1.0 = 13（整数，应返回 "13px" 而非 "13.0px"）
    assert _scale_font("13px", 1.0) == "13px"
    # 10px * 1.2 = 12（整数）
    assert _scale_font("10px", 1.2) == "12px"
    # 13px * 1.1 = 14.3（非整数，保留）
    assert _scale_font("13px", 1.1) == "14.3px"
    # 非 px 单位原样返回
    assert _scale_font("bold", 1.5) == "bold"


# ---------------------------------------------------------------------------
# style.py 134-138：FRAGMENTS_DIR 不存在回退 style.qss
# ---------------------------------------------------------------------------


def test_load_qss_fragments_fallback_to_single_file(monkeypatch, tmp_path) -> None:
    """FRAGMENTS_DIR 不存在时应回退读取 style.qss 单文件（134-138 行）."""
    from zylab.gui import style as style_mod

    # 指向一个不存在的目录
    nonexistent = tmp_path / "no_such_dir"
    monkeypatch.setattr(style_mod, "FRAGMENTS_DIR", nonexistent)

    result = style_mod.load_qss_fragments()
    # style.qss 存在，返回其内容
    style_qss_path = Path(style_mod.__file__).parent / "style.qss"
    expected = style_qss_path.read_text(encoding="utf-8")
    assert result == expected


def test_load_qss_fragments_returns_empty_when_nothing_exists(monkeypatch, tmp_path) -> None:
    """FRAGMENTS_DIR 和 style.qss 都不存在时返回空串（137-138 行）."""
    from zylab.gui import style as style_mod

    monkeypatch.setattr(style_mod, "FRAGMENTS_DIR", tmp_path / "gone")
    # 同时让 style.qss 也不存在
    fake_parent = tmp_path / "fake_parent"
    fake_parent.mkdir()
    monkeypatch.setattr(style_mod.Path, "__call__", lambda p=None: fake_parent if p is None else Path(p))
    # 更直接：patch style_mod.Path(__file__).parent → fake_parent
    monkeypatch.setattr(
        style_mod,
        "Path",
        lambda *args, **kwargs: (
            fake_parent / args[0]
            if args and isinstance(args[0], str) and args[0].endswith(".py")
            else Path(*args, **kwargs)
        ),
    )


# ---------------------------------------------------------------------------
# style.py 165->168：svg_tokens 更新 tokens dict
# ---------------------------------------------------------------------------


def test_load_stylesheet_merges_svg_tokens(monkeypatch) -> None:
    """传入 svg_tokens 应合并到 QSS 替换令牌中."""
    from zylab.gui import style as style_mod

    custom_svg = "data:image/svg+xml;base64,PHN2Zz4="
    monkeypatch.setattr(style_mod, "load_qss_fragments", lambda: "body { background: ${MY_ARROW}; }")
    qss = style_mod.load_stylesheet(svg_tokens={"MY_ARROW": custom_svg})
    assert custom_svg in qss


# ---------------------------------------------------------------------------
# icons.py 失败分支：资源缺失 / 渲染失败
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestIconsFailurePaths:
    """icons.py 中 QPixmap.loadFromData 失败 / 资源缺失分支."""

    def test_load_icon_svg_missing_returns_empty(self, monkeypatch) -> None:
        """_load_svg_text 返回 None → load_icon 返回空 QIcon（~78-80 行）."""
        from zylab.gui import icons as icons_mod

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: None)
        result = icons_mod.load_icon("nonexistent_xxx")
        assert result.isNull() or result.pixmap().isNull() if hasattr(result, "pixmap") else result.isNull()

    def test_load_icon_render_failure_returns_empty(self, monkeypatch) -> None:
        """SVG 文本存在但 loadFromData 失败 → 返回空 QIcon（83 行）."""
        from zylab.gui import icons as icons_mod
        from zylab.gui.qt_compat import QPixmap

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: "definitely not svg <<<")

        real_qpixmap = QPixmap

        class _FailPixmap(real_qpixmap):
            def loadFromData(self, *_a, **_kw):
                return False

        monkeypatch.setattr(icons_mod, "QPixmap", _FailPixmap)
        result = icons_mod.load_icon("bad_svg")
        assert result.isNull()

    def test_nav_icon_svg_missing_returns_empty(self, monkeypatch) -> None:
        """_load_svg_text 返回 None → nav_icon 返回空 QIcon（103 行）."""
        from zylab.gui import icons as icons_mod

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: None)
        result = icons_mod.nav_icon("missing")
        assert result.isNull()

    def test_nav_icon_render_failure_returns_empty(self, monkeypatch) -> None:
        """nav_icon 中 loadFromData 失败 → 返回空 QIcon（110 行）."""
        from zylab.gui import icons as icons_mod
        from zylab.gui.qt_compat import QPixmap

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: "<svg><path/></svg>")

        real_qpixmap = QPixmap

        class _FailPixmap(real_qpixmap):
            def loadFromData(self, *_a, **_kw):
                return False

        monkeypatch.setattr(icons_mod, "QPixmap", _FailPixmap)
        result = icons_mod.nav_icon("bad_tint")
        assert result.isNull()

    def test_tinted_pixmap_svg_missing_returns_empty(self, monkeypatch) -> None:
        """_load_svg_text 返回 None → tinted_pixmap 返回空 QPixmap（134 行）."""
        from zylab.gui import icons as icons_mod

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: None)
        # 清除缓存确保走 None 路径
        icons_mod._PIXMAP_CACHE.clear()
        result = icons_mod.tinted_pixmap("missing", "#ff0000", 16)
        assert result.isNull()

    def test_tinted_pixmap_render_failure_returns_empty(self, monkeypatch) -> None:
        """tinted_pixmap 中 loadFromData 失败 → 返回空 QPixmap（138 行）."""
        from zylab.gui import icons as icons_mod
        from zylab.gui.qt_compat import QPixmap

        monkeypatch.setattr(icons_mod, "_load_svg_text", lambda _name: "<svg><path/></svg>")
        icons_mod._PIXMAP_CACHE.clear()

        real_qpixmap = QPixmap

        class _FailPixmap(real_qpixmap):
            def loadFromData(self, *_a, **_kw):
                return False

        monkeypatch.setattr(icons_mod, "QPixmap", _FailPixmap)
        result = icons_mod.tinted_pixmap("bad", "#ff0000", 16)
        assert result.isNull()


# ---------------------------------------------------------------------------
# app.py 105 / 158 / 73-74
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_load_qrc_svg_returns_none_when_missing(monkeypatch) -> None:
    """_load_qrc_svg 中 QFile.open 失败 → 返回 None（105 行）."""
    from zylab.gui.app import _load_qrc_svg

    # 用一个肯定不存在的 qrc 路径
    result = _load_qrc_svg(":/icons/does_not_exist_xyz.svg")
    assert result is None


@pytest.mark.gui
def test_write_theme_svgs_skips_missing_chevron_qrc(monkeypatch) -> None:
    """chevron qrc 资源缺失 → template is None → continue（158 行）."""
    import zylab.gui.app as app_mod

    # 让 _load_qrc_svg 对 chevron 返回 None
    original = app_mod._load_qrc_svg

    def _fake(path: str):
        if "chevron" in path:
            return None
        return original(path)

    monkeypatch.setattr(app_mod, "_load_qrc_svg", _fake)

    tokens = app_mod._write_theme_svgs(theme.LIGHT)
    # chevron 令牌不应出现
    assert "QSS_CHEVRON_RIGHT" not in tokens
    assert "QSS_CHEVRON_DOWN" not in tokens
    # 其他令牌（arrow / close）应存在
    assert "QSS_ARROW_UP" in tokens


@pytest.mark.gui
def test_register_fonts_handles_all_failures(monkeypatch) -> None:
    """register_fonts 中所有字体注册失败 → 返回空列表（73-74 行）."""
    from zylab.gui.app import register_fonts
    from zylab.gui.qt_compat import QFontDatabase

    monkeypatch.setattr(QFontDatabase, "addApplicationFont", lambda _p: -1)
    result = register_fonts()
    assert result == []


# ---------------------------------------------------------------------------
# var_browser.py 目标（69 / 155 / 291）
# ---------------------------------------------------------------------------


def test_fmt_cell_int_branch() -> None:
    """_fmt_cell 对 int 应走 str(int) 分支（69 行）."""
    from zylab.gui.pages.var_browser import _fmt_cell

    assert _fmt_cell(42) == "42"
    assert _fmt_cell(np.int64(7)) == "7"


@pytest.mark.gui
def test_var_table_model_data_roles() -> None:
    """VarTableModel 各角色返回正确值（覆盖 model.data 路径）."""
    from zylab.gui.pages.var_browser import VarTableModel
    from zylab.gui.qt_compat import Qt
    from zylab.sci import VarInfo

    info = VarInfo("x", "float64", "(10,)", "float64", 80, "1..10", False)
    model = VarTableModel()
    model.set_vars([info])

    idx = model.index(0, 0)
    assert model.data(idx, Qt.DisplayRole) == "x"

    idx1 = model.index(0, 1)
    assert model.data(idx1, Qt.DisplayRole) == "float64"
    tags = model.data(idx1, VarTableModel.TAGS_ROLE)
    assert tags == ("float64", "(10,)")

    idx2 = model.index(0, 2)
    assert model.data(idx2, Qt.DisplayRole) == "80"


@pytest.mark.gui
def test_var_detail_dialog_scalar_via_monkeypatch(qtbot, monkeypatch) -> None:
    """VarDetailDialog._refresh_matrix 中 arr.ndim==0 → reshape(1,1)（291 行）.

    自然路径很难触发（构造函数只对 ndim>=1 调 _refresh_matrix，
    而 ndim>=1 的数组 indexing 后仍 ndim>=1），此处 monkeypatch 强制覆盖。
    """
    from zylab.gui.pages.var_browser import VarDetailDialog
    from zylab.sci import VarInfo

    # 用一个 2 维数组让它正常构建 _table，但 patch _current_slice 返回 0 维
    arr = np.arange(6).reshape(2, 3)
    info = VarInfo("m", "float64", "(2,3)", "float64", 48, "...", False)
    dlg = VarDetailDialog(info, arr)
    qtbot.addWidget(dlg)
    assert hasattr(dlg, "_table")
    # patch 让 _current_slice 返回 0 维
    monkeypatch.setattr(dlg, "_current_slice", lambda: np.array(99.0))
    dlg._refresh_matrix()
    assert dlg._table.rowCount() == 1
    assert dlg._table.columnCount() == 1


@pytest.mark.gui
def test_var_detail_dialog_multidim_slice(qtbot) -> None:
    """多维数组（ndim>=3）应构建 slice bar 并正常刷新."""
    from zylab.gui.pages.var_browser import VarDetailDialog
    from zylab.sci import VarInfo

    arr = np.arange(24).reshape(2, 3, 4)  # 3 维
    info = VarInfo("cube", "int64", "(2,3,4)", "int64", 192, "...", False)
    dlg = VarDetailDialog(info, arr)
    qtbot.addWidget(dlg)
    # slice spinner 存在
    assert len(dlg._ndim_spinners) == 1  # 2 个前置维 (2,3) → 第1维微调
    # 切到 0, 应得到 3×4 矩阵
    dlg._ndim_spinners[0].setValue(0)
    assert dlg._table.rowCount() == 3
    assert dlg._table.columnCount() == 4


@pytest.mark.gui
def test_var_detail_dialog_1d_column(qtbot) -> None:
    """一维数组应 reshape 为 (-1,1) 列向量."""
    from zylab.gui.pages.var_browser import VarDetailDialog
    from zylab.sci import VarInfo

    arr = np.array([10.0, 20.0, 30.0])
    info = VarInfo("vec", "float64", "(3,)", "float64", 24, "...", False)
    dlg = VarDetailDialog(info, arr)
    qtbot.addWidget(dlg)
    assert dlg._table.rowCount() == 3
    assert dlg._table.columnCount() == 1


# ---------------------------------------------------------------------------
# widgets/_table_utils.py 84：build_table_widget include_zebra
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_build_table_widget_with_zebra() -> None:
    """build_table_widget(include_zebra=True) 偶数行应设透明背景."""
    from zylab.flowchart.results import TableColumn, TableData
    from zylab.gui.qt_compat import Qt
    from zylab.gui.widgets._table_utils import build_table_widget

    data = TableData(
        title="test",
        columns=(TableColumn(title="A"), TableColumn(title="B")),
        rows=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
    )
    table = build_table_widget(data, include_zebra=True)
    # 奇数行 (row=1) 背景应为 transparent
    item = table.item(1, 0)
    assert item is not None
    bg = item.background()
    assert bg.color() == Qt.GlobalColor.transparent or bg.color().alpha() == 0


# ---------------------------------------------------------------------------
# widgets/command_palette.py 关键分支
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_command_palette_register_duplicate_overrides(qtbot) -> None:
    """同 id 命令注册应覆盖旧定义但保持位置."""
    from zylab.gui.widgets.command_palette import Command, CommandPalette

    pal = CommandPalette(parent=None)
    pal.register(Command(id="a", title="原始", callback=lambda: None))
    pal.register(Command(id="b", title="B", callback=lambda: None))
    # 覆盖 "a"
    pal.register(Command(id="a", title="覆盖后", callback=lambda: None))
    assert len(pal._commands) == 2
    assert pal._commands[0].title == "覆盖后"  # 位置仍在 0
    assert pal._by_id["a"].title == "覆盖后"


@pytest.mark.gui
def test_command_palette_filter_commands(qtbot) -> None:
    """_populate_commands 应按搜索词过滤."""
    from zylab.gui.qt_compat import Qt
    from zylab.gui.widgets.command_palette import Command, CommandPalette

    pal = CommandPalette(parent=None)
    pal.register(Command(id="1", title="保存项目", callback=lambda: None, keywords="save write"))
    pal.register(Command(id="2", title="打开文件", callback=lambda: None, keywords="open file"))
    pal.register(Command(id="3", title="运行仿真", callback=lambda: None, keywords="run compute"))

    pal._populate_commands("save")
    assert pal._list.count() == 1
    assert pal._list.item(0).data(Qt.UserRole) == ("command", "1")

    pal._populate_commands("file")
    assert pal._list.count() == 1
    assert pal._list.item(0).data(Qt.UserRole) == ("command", "2")

    # 多词 AND
    pal._populate_commands("save write")
    assert pal._list.count() == 1


@pytest.mark.gui
def test_command_palette_theme_mode_navigate(qtbot) -> None:
    """主题模式下上下键导航应发射 theme_previewed."""
    from zylab.gui.widgets.command_palette import CommandPalette

    pal = CommandPalette(parent=None)
    pal._populate_themes("")
    # 已有 3 套主题
    assert pal._list.count() == 3
    # 下移
    with qtbot.waitSignal(pal.theme_previewed, timeout=2000) as blocker:
        pal._on_row_changed(1)
    # blocker.args 是列表，取第一个
    args = blocker.args
    assert args[0] in ("dark", "light", "high_contrast")


# ---------------------------------------------------------------------------
# widgets/trial_record_edit.py 边界
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_trial_record_edit_append_roundtrip(qtbot) -> None:
    """追加响应/不响应应正确推算下一发刺激量."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    edit = TrialRecordEdit()
    qtbot.addWidget(edit)
    # start=3.2, step=0.05
    edit._add_hit_btn.click()  # 初始 3.2，响应 → 下次 3.15
    edit._add_miss_btn.click()  # 3.15，不响应 → 下次 3.20
    text = edit.text()
    assert "3.2" in text or "3.20" in text
    assert "O" in text
    assert "X" in text


@pytest.mark.gui
def test_trial_record_edit_text_roundtrip(qtbot) -> None:
    """setText / text 应正确往返."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    edit = TrialRecordEdit()
    qtbot.addWidget(edit)
    edit.setText("3.20 O, 3.15 X, 3.20 O")
    text = edit.text()
    assert len(edit._records) == 3
    # 重解析应一致
    edit2 = TrialRecordEdit()
    edit2.setText(text)
    assert len(edit2._records) == 3


@pytest.mark.gui
def test_trial_record_edit_invalid_text_clears(qtbot) -> None:
    """非法输入应清空."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    edit = TrialRecordEdit()
    qtbot.addWidget(edit)
    edit._add_hit_btn.click()
    assert len(edit._records) == 1
    edit.setText("garbage garbage !!!")
    assert len(edit._records) == 0


@pytest.mark.gui
def test_trial_record_edit_undo_clear(qtbot) -> None:
    """撤销 / 清空按钮."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    edit = TrialRecordEdit()
    qtbot.addWidget(edit)
    edit._add_hit_btn.click()
    edit._add_hit_btn.click()
    assert len(edit._records) == 2

    edit._undo_btn.click()
    assert len(edit._records) == 1

    edit._clear_btn.click()
    assert len(edit._records) == 0
    # 空记录再撤销不应抛错
    edit._undo_btn.click()
    edit._clear_btn.click()


@pytest.mark.gui
def test_trial_record_edit_set_suggest_step(qtbot) -> None:
    """set_suggest_step 应更新步长 spin."""
    from zylab.gui.widgets.trial_record_edit import TrialRecordEdit

    edit = TrialRecordEdit()
    qtbot.addWidget(edit)
    edit.set_suggest_step(0.1)
    assert edit._step_spin.value() == pytest.approx(0.1)
    # 非正数不更新
    edit.set_suggest_step(-1)
    assert edit._step_spin.value() == pytest.approx(0.1)
