"""zylab 命令行入口：无子命令启动 GUI；``run``/``templates`` 子命令提供无界面求解.

子命令：
- ``zylab run <模板id|模板.json|工程.zprj>``：进程内运行工作流并打印结果摘要，
  ``-p 节点.参数=值`` 覆盖参数（可多次），``--scan 节点.参数=取值`` 参数化扫描；
- ``zylab templates``：列出可用模板（内置 + 用户目录 + entry points 插件）。

退出码：0 成功 / 2 参数或目标非法 / 1 求解失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from zylab.core.project import Project, ProjectFileError
from zylab.studio.batch import run_scan, run_workflow, summarize
from zylab.studio.errors import StudioError
from zylab.studio.registry import TemplateRegistry
from zylab.studio.template import Template, load_template

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（子命令：run / templates；无子命令默认 GUI）."""
    parser = argparse.ArgumentParser(prog="zylab", description="zylab 通用科学计算与有限元分析平台")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="无界面运行模板或工程并打印结果摘要")
    run.add_argument("target", help="模板 id、模板 JSON 文件或 .zprj 工程文件路径")
    run.add_argument("-p", "--param", action="append", default=[], metavar="节点.参数=值", help="参数覆盖（可多次）")
    run.add_argument(
        "--scan",
        metavar="节点.参数=取值",
        help="参数化扫描；取值为逗号分隔列表（v1,v2,…）或起点:终点:点数（含端点线性插值）",
    )

    sub.add_parser("templates", help="列出可用模板")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口：按参数分发子命令；无子命令启动 GUI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.target, args.param, args.scan)
    if args.command == "templates":
        return _cmd_templates()
    return _launch_gui()


def _cmd_run(target: str, params: list[str], scan: str | None) -> int:
    """``run`` 子命令：加载目标 → 覆盖/扫描 → 进程内求解 → 打印摘要."""
    try:
        template = _load_target(target)
        overrides = _parse_params(params)
    except (ValueError, StudioError, ProjectFileError) as exc:
        print(f"目标加载失败: {exc}", file=sys.stderr)
        return 2
    if scan is not None:
        try:
            ref, values = _parse_scan(scan)
            runs = run_scan(template, ref, values, _report_progress)
        except ValueError as exc:
            print(f"扫描参数非法: {exc}", file=sys.stderr)
            return 2
        failed = False
        for value, outcome in zip(values, runs):
            print(f"== 扫描值 {value:g} ==")
            print(summarize(outcome))
            failed = failed or not outcome.succeeded
        return 1 if failed else 0
    outcome = run_workflow(template, overrides, _report_progress)
    print(summarize(outcome))
    return 0 if outcome.succeeded else 1


def _cmd_templates() -> int:
    """``templates`` 子命令：列出注册表全部模板."""
    for template in _registry().list():
        description = f" —— {template.description}" if template.description else ""
        print(f"{template.id}\t{template.name}{description}")
    return 0


def _launch_gui() -> int:  # pragma: no cover
    """无子命令时启动 GUI（惰性导入，避免 CLI 进程加载 Qt）."""
    from zylab.gui.app import main as gui_main

    return gui_main()


def _registry() -> TemplateRegistry:
    """构造模板注册表：内置 + 用户目录 + entry points 插件."""
    from zylab.core.config import default_data_dir

    registry = TemplateRegistry.with_builtin()
    registry.load_dir(default_data_dir() / "templates")
    registry.load_entry_points()
    return registry


def _load_target(target: str) -> Template:
    """解析运行目标：文件路径（.json 模板 / .zprj 工程）或注册表模板 id."""
    path = Path(target)
    if path.exists():
        if path.suffix == ".zprj":
            with Project.open(path) as proj:
                data = proj.read_json("model", "workflow")
            return Template.from_dict(data)
        return load_template(path)
    return _registry().get(target)


def _parse_params(raw: list[str]) -> dict[str, dict[str, float | int]]:
    """解析 ``节点.参数=值`` 覆盖列表为节点参数覆盖表."""
    overrides: dict[str, dict[str, float | int]] = {}
    for item in raw:
        ref, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"参数 {item!r} 应为 '节点.参数=值' 格式")
        node_id, dot, key = ref.partition(".")
        if not dot or not node_id or not key:
            raise ValueError(f"参数 {item!r} 应为 '节点.参数=值' 格式")
        overrides.setdefault(node_id, {})[key] = _parse_value(raw_value)
    return overrides


def _parse_value(raw: str) -> float | int:
    """解析参数值字符串（整值转 int，否则 float）."""
    value = float(raw)
    return int(value) if value.is_integer() else value


def _parse_scan(raw: str) -> tuple[str, tuple[float, ...]]:
    """解析 ``节点.参数=取值``：取值支持逗号列表或 ``起点:终点:点数`` 线性插值."""
    ref, sep, values_raw = raw.partition("=")
    if not sep or "." not in ref:
        raise ValueError(f"扫描 {raw!r} 应为 '节点.参数=取值' 格式")
    if ":" in values_raw:
        start_raw, stop_raw, count_raw = values_raw.split(":")
        start, stop, count = float(start_raw), float(stop_raw), int(count_raw)
        if count < 2:
            raise ValueError(f"扫描点数须 >= 2，得到 {count}")
        step = (stop - start) / (count - 1)
        return ref, tuple(start + i * step for i in range(count))
    return ref, tuple(float(v) for v in values_raw.split(",") if v)


def _report_progress(progress: float, message: str) -> None:
    """进度回调：单行打印到 stderr（不污染 stdout 的结果摘要流）."""
    print(f"\r[{progress:5.0%}] {message}", end="", file=sys.stderr, flush=True)
