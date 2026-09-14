"""GUI perf 模块单元测试（纯 Python，不依赖 Qt）."""

from __future__ import annotations

import logging
import time

import pytest

from zylab.gui.perf import PerfReport, PerfStats, render_startup_summary, timed

# ---------------------------------------------------------------------------
# PerfStats
# ---------------------------------------------------------------------------


def test_perf_stats_is_frozen() -> None:
    """PerfStats 是 frozen dataclass，不可变。"""
    import dataclasses

    stats = PerfStats(name="测试", elapsed=0.123)
    assert stats.name == "测试"
    assert stats.elapsed == 0.123
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.name = "其他"  # type: ignore[misc]


def test_perf_stats_defaults() -> None:
    """PerfStats 无默认值（必须提供全部字段）."""
    with pytest.raises(TypeError):
        PerfStats()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PerfReport
# ---------------------------------------------------------------------------


def test_perf_report_init_defaults() -> None:
    """PerfReport 默认值：空 stages、enabled=False."""
    report = PerfReport()
    assert report.stages == []
    assert report.enabled is False


def test_perf_report_add() -> None:
    """add() 累积 PerfStats 到 stages."""
    report = PerfReport()
    s1 = PerfStats(name="阶段1", elapsed=0.01)
    s2 = PerfStats(name="阶段2", elapsed=0.02)
    report.add(s1)
    report.add(s2)
    assert report.stages == [s1, s2]


def test_perf_report_total() -> None:
    """total 属性正确累加全部阶段耗时."""
    report = PerfReport()
    assert report.total == 0.0  # 空 report
    report.add(PerfStats(name="a", elapsed=1.0))
    report.add(PerfStats(name="b", elapsed=2.5))
    report.add(PerfStats(name="c", elapsed=0.5))
    assert report.total == pytest.approx(4.0, rel=1e-9)


# ---------------------------------------------------------------------------
# timed
# ---------------------------------------------------------------------------


def test_timed_context_manager_measures_elapsed() -> None:
    """timed 正确测量代码块耗时并登记到 report."""
    report = PerfReport()
    with timed("测试阶段", report=report, level=logging.WARNING):  # WARNING 避免 DEBUG 干扰
        time.sleep(0.01)
    assert len(report.stages) == 1
    assert report.stages[0].name == "测试阶段"
    assert report.stages[0].elapsed >= 0.005  # 至少 5ms（sleep 有误差）


def test_timed_nested_stages() -> None:
    """timed 可嵌套，父子阶段各自独立计时."""
    report = PerfReport()
    with timed("外层", report=report, level=logging.WARNING):
        time.sleep(0.005)
        with timed("内层", report=report, level=logging.WARNING):
            time.sleep(0.005)
    assert len(report.stages) == 2
    outer = next(s for s in report.stages if s.name == "外层")
    inner = next(s for s in report.stages if s.name == "内层")
    assert outer.elapsed >= inner.elapsed  # 外层应 >= 内层


def test_timed_exception_also_measured() -> None:
    """timed 在代码块抛异常时仍登记耗时."""
    report = PerfReport()
    with pytest.raises(ValueError), timed("异常阶段", report=report, level=logging.WARNING):
        time.sleep(0.005)
        raise ValueError("boom")
    assert len(report.stages) == 1
    assert report.stages[0].name == "异常阶段"
    assert report.stages[0].elapsed >= 0.005


# ---------------------------------------------------------------------------
# render_startup_summary
# ---------------------------------------------------------------------------


def test_render_disabled_report_is_noop(capsys: pytest.CaptureFixture[str]) -> None:
    """enabled=False 时 render_startup_summary 不输出任何内容."""
    report = PerfReport(enabled=False)
    report.add(PerfStats(name="a", elapsed=1.0))
    render_startup_summary(report)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_render_empty_report_is_noop(capsys: pytest.CaptureFixture[str]) -> None:
    """enabled=True 但 stages 为空时，render 也是 no-op."""
    report = PerfReport(enabled=True)
    render_startup_summary(report)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_render_summary_without_rich() -> None:
    """rich 不可用时回退纯文本渲染（用 capsys 捕获 logger 输出）."""
    # 临时移除 rich 模块（模拟未安装场景）
    import builtins
    from io import StringIO

    original_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("rich"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    builtins.__import__ = _blocked_import  # type: ignore[assignment]
    try:
        report = PerfReport(enabled=True)
        report.add(PerfStats(name="阶段A", elapsed=0.1))
        report.add(PerfStats(name="阶段B", elapsed=0.2))

        # 捕获 logger.info 输出
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("zylab.gui.perf")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            render_startup_summary(report)
            output = buf.getvalue()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]

    assert "启动性能汇总" in output
    assert "阶段A" in output
    assert "阶段B" in output
    assert "总计" in output


def test_render_summary_with_rich(capsys: pytest.CaptureFixture[str]) -> None:
    """rich 可用时渲染 rich 表格输出到 stdout."""
    import rich  # noqa: F401  断言 rich 在 venv 里可用

    report = PerfReport(enabled=True)
    report.add(PerfStats(name="阶段A", elapsed=0.1))
    report.add(PerfStats(name="阶段B", elapsed=0.2))

    render_startup_summary(report)

    captured = capsys.readouterr()
    # rich 表格含中文表头和阶段名
    assert "阶段A" in captured.out
    assert "阶段B" in captured.out
    assert "总计" in captured.out
