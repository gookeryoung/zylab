"""sci.plotting 绘图事件与 matplotlib rcParams 默认配置测试."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from zylab.core import EventBus
from zylab.sci import (
    CURVE_PALETTE,
    TOPIC_PLOT_REQUESTED,
    PlotRequest,
    apply_matplotlib_defaults,
    fig_to_png_bytes,
    make_plot_function,
    save_figure,
)


def test_plot_publishes_event() -> None:
    """plot 应发布 PlotRequest 事件并携带全部参数."""
    bus = EventBus()
    received: list[PlotRequest] = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    plot = make_plot_function(bus)
    plot([1, 2, 3], [4, 5, 6], title="标题", xlabel="x轴", ylabel="y轴", clear=True, label="曲线A")
    assert len(received) == 1
    req = received[0]
    assert list(req.x) == [1, 2, 3]
    assert list(req.y) == [4, 5, 6]
    assert req.title == "标题"
    assert req.xlabel == "x轴"
    assert req.ylabel == "y轴"
    assert req.clear is True
    assert req.extra == {"label": "曲线A"}


def test_plot_y_only_generates_x() -> None:
    """单参数调用时 x 自动生成 0..n-1."""
    bus = EventBus()
    received: list[PlotRequest] = []
    bus.subscribe(TOPIC_PLOT_REQUESTED, received.append)
    make_plot_function(bus)([10, 20, 30])
    assert list(received[0].x) == [0, 1, 2]
    assert list(received[0].y) == [10, 20, 30]


def test_plot_without_subscriber_is_noop() -> None:
    """无订阅者时 plot 静默通过（CLI/worker 场景）."""
    make_plot_function(EventBus())([1.0, 2.0])  # 不抛异常


# ---------------------------------------------------------------------------
# matplotlib rcParams 全局默认配置


@pytest.fixture(autouse=True)
def _reset_mpl_rc() -> None:
    """每个测试前恢复 matplotlib rcParams 默认值，测试后原样还原."""
    plt.rcParams.update(plt.rcParamsDefault)
    yield
    plt.rcParams.update(plt.rcParamsDefault)


def test_apply_defaults_sets_grid_on() -> None:
    """应用默认后 axes.grid 应开启（matplotlib 默认 False）."""
    assert plt.rcParams["axes.grid"] is False  # 前置条件确认
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.grid"] is True


def test_apply_defaults_sets_linewidth() -> None:
    """应用默认后 lines.linewidth 应为 2.0（科学计算更醒目）."""
    apply_matplotlib_defaults()
    assert plt.rcParams["lines.linewidth"] == 2.0


def test_apply_defaults_sets_dpi() -> None:
    """应用默认后 figure.dpi 应为 120."""
    apply_matplotlib_defaults()
    assert plt.rcParams["figure.dpi"] == 120.0


def test_apply_defaults_sets_unicode_minus_false() -> None:
    """应用默认后 axes.unicode_minus 应为 False（避免中文环境负号变方块）."""
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.unicode_minus"] is False


def test_apply_defaults_injects_ns_vars() -> None:
    """传入命名空间时应注入 available_fonts 与 cn_font_candidates."""
    ns: dict[str, object] = {}
    apply_matplotlib_defaults(ns)
    assert "available_fonts" in ns
    assert "cn_font_candidates" in ns
    assert isinstance(ns["available_fonts"], frozenset)
    assert isinstance(ns["cn_font_candidates"], list)


def test_apply_defaults_idempotent() -> None:
    """幂等性：重复调用不产生额外副作用."""
    apply_matplotlib_defaults()
    rc_after_first = dict(plt.rcParams)
    apply_matplotlib_defaults()
    for key in ("axes.grid", "lines.linewidth", "font.size", "figure.dpi"):
        assert plt.rcParams[key] == rc_after_first[key]


def test_apply_defaults_returns_fonts() -> None:
    """返回值应为系统已安装字体集合（matplotlib 可用时）."""
    result = apply_matplotlib_defaults()
    assert result is not None
    assert isinstance(result, frozenset)
    assert len(result) > 0


def test_apply_defaults_without_cn_fonts_skips_merge() -> None:
    """系统无中文字体时跳过 font.sans-serif 合并（if selected_cn → False 分支）."""
    from unittest.mock import patch

    import matplotlib.font_manager as fm

    with patch.object(fm.fontManager, "ttflist", new=[]):
        # 无任何字体 → selected_cn 为空 → 跳过合并分支
        apply_matplotlib_defaults(ns={})


def test_apply_defaults_with_overlapping_fonts_dedupes() -> None:
    """默认 sans-serif 与中文字体列表有重叠时走去重分支（if name not in seen → False）."""
    from unittest.mock import patch

    import seaborn as sns

    def _fake_set_theme(*args, **kwargs):
        # 跳过真实 set_theme，保留 rcParams 中已设置的 font.sans-serif
        return None

    # 在 seaborn.set_theme 前注入重叠字体，同时 mock 掉 set_theme 防止重置
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    with patch.object(sns, "set_theme", side_effect=_fake_set_theme):
        apply_matplotlib_defaults(ns={})
    assert "Microsoft YaHei" in plt.rcParams["font.sans-serif"]
    # Microsoft YaHei 应该只出现一次（去重生效）
    assert plt.rcParams["font.sans-serif"].count("Microsoft YaHei") == 1


def test_apply_defaults_palette_in_prop_cycle() -> None:
    """CURVE_PALETTE 应进入 axes.prop_cycle（seaborn set_theme palette 参数）."""
    from matplotlib.colors import to_hex

    apply_matplotlib_defaults()
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # seaborn 把 palette 注入 prop_cycle，可能是 RGB tuple 或 hex
    assert to_hex(cycle_colors[0]).upper() == CURVE_PALETTE[0].upper()


def test_apply_defaults_unicode_minus_preserved_after_seaborn() -> None:
    """seaborn set_theme 会重置 unicode_minus，_MPL_RC_OVERRIDES 须再次覆盖为 False."""
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.unicode_minus"] is False


def test_apply_defaults_grid_visuals_from_seaborn() -> None:
    """seaborn whitegrid 接管网格视觉（alpha/color/linewidth 由主题决定），
    这里只校验网格已开启且有可见 alpha。"""
    apply_matplotlib_defaults()
    assert plt.rcParams["axes.grid"] is True
    # seaborn whitegrid 使用非零 alpha
    assert plt.rcParams["grid.alpha"] > 0.0


# ---------------------------------------------------------------------------
# plot_band / plot_reg 统计快捷函数


@pytest.fixture(autouse=True)
def _mpl_agg_backend() -> None:
    """plot_band / plot_reg 内部会切换 matplotlib 后端，这里重置。"""
    import matplotlib

    matplotlib.use("Agg")


def test_plot_band_multi_trial_sd_ci() -> None:
    """多组数据 + sd 置信带：返回均值线 + fill_between 对象."""
    from zylab.sci.plotting import plot_band

    rng = __import__("numpy").random.default_rng(42)
    ys = rng.normal(loc=0, scale=1, size=(5, 20))
    fig, ax = plt.subplots()
    mean_line, fill, _upper = plot_band(ys, ax=ax)
    assert mean_line is not None
    assert fill is not None  # multi-trial 模式有 fill_between
    plt.close(fig)


def test_plot_band_multi_trial_95_ci() -> None:
    """多组数据 + 95% 置信区间：懒加载 scipy.stats."""
    from zylab.sci.plotting import plot_band

    rng = __import__("numpy").random.default_rng(42)
    ys = rng.normal(loc=0, scale=1, size=(10, 15))
    fig, ax = plt.subplots()
    mean_line, fill, _upper = plot_band(ys, ci="95", ax=ax)
    assert mean_line is not None
    assert fill is not None
    plt.close(fig)


def test_plot_band_single_trial_degenerates_to_line() -> None:
    """单组数据（一维数组）退化为普通折线，无 fill_between."""
    from zylab.sci.plotting import plot_band

    fig, ax = plt.subplots()
    mean_line, fill, _upper = plot_band(__import__("numpy").sin(__import__("numpy").linspace(0, 6, 20)), ax=ax)
    assert mean_line is not None
    assert fill is None  # single-trial 模式无带
    plt.close(fig)


def test_plot_band_single_trial_with_explicit_x() -> None:
    """单组数据传入 x 参数（覆盖 x_arr = np.asarray(x) 分支）."""
    import numpy as np

    from zylab.sci.plotting import plot_band

    fig, ax = plt.subplots()
    mean_line, fill, _upper = plot_band(np.array([1.0, 2.0, 3.0]), x=np.array([0.0, 1.0, 2.0]), ax=ax)
    assert mean_line is not None
    assert fill is None
    plt.close(fig)


def test_plot_band_x_length_mismatch_raises() -> None:
    """多组数据的 x 长度须等于 ys 列数."""
    from zylab.sci.plotting import plot_band

    rng = __import__("numpy").random.default_rng(42)
    ys = rng.normal(size=(3, 20))
    fig, ax = plt.subplots()
    try:
        plot_band(ys, x=__import__("numpy").arange(15), ax=ax)
        pytest.fail("应抛 ValueError")
    except ValueError as exc:
        assert "长度" in str(exc)
    finally:
        plt.close(fig)


def test_plot_band_invalid_ci_raises() -> None:
    """ci 参数只接受 'sd' 或 '95'."""
    from zylab.sci.plotting import plot_band

    rng = __import__("numpy").random.default_rng(42)
    ys = rng.normal(size=(3, 10))
    fig, ax = plt.subplots()
    try:
        plot_band(ys, ci="bad", ax=ax)
        pytest.fail("应抛 ValueError")
    except ValueError as exc:
        assert "ci" in str(exc).lower()
    finally:
        plt.close(fig)


def test_plot_reg_scatter_with_regression_line() -> None:
    """plot_reg 调 seaborn.regplot 返回 Axes 对象."""
    from zylab.sci.plotting import plot_reg

    rng = __import__("numpy").random.default_rng(42)
    np = __import__("numpy")
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + rng.normal(size=50)
    fig, ax = plt.subplots()
    result = plot_reg(x, y, order=1, ci=95, ax=ax)
    assert result is ax  # seaborn 返回传入的 ax
    plt.close(fig)


def test_save_figure_png(tmp_path) -> None:
    """save_figure 正常导出 PNG 文件."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    out = tmp_path / "out.png"
    result = save_figure(fig, out)
    plt.close(fig)

    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 100  # PNG 至少几百字节


def test_save_figure_pdf(tmp_path) -> None:
    """save_figure 导出 PDF 文件（按后缀自动识别格式）."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    out = tmp_path / "report.pdf"
    result = save_figure(fig, out)
    plt.close(fig)

    assert result.suffix == ".pdf"
    assert out.is_file()
    assert out.stat().st_size > 100  # PDF 文件至少几百字节


def test_save_figure_non_figure_raises(tmp_path) -> None:
    """save_figure 非 Figure 对象抛 TypeError."""
    with pytest.raises(TypeError, match="matplotlib Figure"):
        save_figure("not a figure", tmp_path / "out.png")


def test_fig_to_png_bytes_returns_png_magic() -> None:
    """fig_to_png_bytes 返回以 PNG magic bytes 开头的非空 bytes."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    data = fig_to_png_bytes(fig)
    plt.close(fig)

    assert isinstance(data, bytes)
    assert len(data) > 100
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_fig_to_png_bytes_non_figure_raises() -> None:
    """fig_to_png_bytes 非 Figure 对象抛 TypeError."""
    with pytest.raises(TypeError, match="matplotlib Figure"):
        fig_to_png_bytes("not a figure")
