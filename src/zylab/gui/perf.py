"""GUI 启动性能测量：分阶段计时 + 单张汇总表。

仅在启用 ``ZYLAB_PERF=1`` 环境变量时输出，发布版默认关闭、零开销。
逐阶段细节降为 DEBUG（``-vv`` 可见），外层块退出后渲染单张 rich 汇总表
（列：阶段 / 耗时 / 占比），一眼识别瓶颈。

公共 API：

- :class:`PerfReport`：分阶段耗时收集器
- :class:`PerfStats`：单阶段耗时（不可变）
- :func:`timed`：上下文管理器工厂，登记到 :class:`PerfReport`
- :func:`render_startup_summary`：渲染单张 rich 汇总表（perf 未启用时 no-op）

使用方式::

    report = PerfReport(enabled=bool(os.environ.get("Zylab_PERF")))
    with timed("启动流程", report=report, level=logging.DEBUG):
        with timed("构造 QApplication", report=report, level=logging.DEBUG):
            app = create_app()
        with timed("加载 MainWindow", report=report, level=logging.DEBUG):
            window = MainWindow()
    render_startup_summary(report)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

__all__ = ["PerfReport", "PerfStats", "render_startup_summary", "timed"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class PerfStats:
    """单阶段耗时统计。

    :ivar name: 阶段名称
    :ivar elapsed: 耗时（秒，毫秒级精度）
    """

    name: str
    elapsed: float


@dataclass
class PerfReport:
    """分阶段耗时收集器。

    使用 ``with timed("阶段名", report=report):`` 登记各阶段耗时，
    退出后由 :func:`render_startup_summary` 渲染单张汇总表。
    """

    stages: list[PerfStats] = field(default_factory=list)
    enabled: bool = False

    def add(self, stats: PerfStats) -> None:
        """登记单阶段耗时。"""
        self.stages.append(stats)

    @property
    def total(self) -> float:
        """总耗时（秒）。"""
        return sum(s.elapsed for s in self.stages)


@contextmanager
def timed(
    name: str,
    report: PerfReport,
    level: int = logging.DEBUG,
) -> Iterator[None]:
    """上下文管理器：测量代码块耗时并登记到 :class:`PerfReport`。

    :param name: 阶段名称
    :param report: 目标 PerfReport
    :param level: 日志级别（默认 DEBUG，避免刷屏）
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        report.add(PerfStats(name=name, elapsed=elapsed))
        logger.log(level, "%s 耗时 %.3fs", name, elapsed)


def render_startup_summary(report: PerfReport) -> None:
    """渲染单张启动汇总表。

    未启用 perf 时（``report.enabled`` 为 False）即刻 return，零开销。
    启用时优先用 rich 渲染表格，rich 不可用回退纯文本。

    :param report: 启动阶段 PerfReport
    """
    if not report.enabled or not report.stages:
        return
    total = report.total or 1.0  # 避免除零
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover
        _render_text_summary(report, total)
        return

    table = Table(title="启动性能汇总", show_header=True, header_style="bold cyan")
    table.add_column("阶段", style="cyan", no_wrap=True)
    table.add_column("耗时(s)", justify="right", style="magenta")
    table.add_column("占比", justify="right", style="green")
    for stage in report.stages:
        table.add_row(stage.name, f"{stage.elapsed:.3f}", f"{stage.elapsed / total * 100:.1f}%")
    table.add_row("[bold]总计[/bold]", f"[bold]{total:.3f}[/bold]", "100.0%")
    Console().print(table)


def _render_text_summary(report: PerfReport, total: float) -> None:
    """rich 不可用时的纯文本回退。"""
    lines = ["启动性能汇总："]
    for stage in report.stages:
        lines.append(f"  {stage.name:<24} {stage.elapsed:>8.3f}s  {stage.elapsed / total * 100:>5.1f}%")
    lines.append(f"  {'总计':<24} {total:>8.3f}s  100.0%")
    logger.info("\n".join(lines))
