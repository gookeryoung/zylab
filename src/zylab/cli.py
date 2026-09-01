"""zylab 命令行入口：无子命令启动 GUI；``run``/``templates`` 子命令提供无界面求解.

子命令：
- ``zylab run <模板id|模板文件|工程.zprj>``：进程内运行工作流并打印结果摘要；
  目标为 DSL 模板（``*.yaml``/``*.yml`` 或 DSL 模板 id）时 ``-p 参数名=值``
  覆盖 DSL 参数，``--report 路径`` 导出 Markdown/HTML 报告；
  经典模板 ``-p 节点.参数=值`` 覆盖参数（可多次），``--scan 节点.参数=取值``
  参数化扫描；
- ``zylab templates``：列出可用模板（内置 + 用户目录 + entry points 插件）。

退出码：0 成功 / 2 参数或目标非法 / 1 求解失败。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Sequence

from zylab.core.project import Project, ProjectFileError
from zylab.studio.batch import RunOutcome, run_scan, run_workflow, summarize
from zylab.studio.dsl import DslTemplate, load_dsl
from zylab.studio.errors import StudioError
from zylab.studio.registry import TemplateRegistry
from zylab.studio.report import build_html, build_markdown
from zylab.studio.template import Template, load_template

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（子命令：run / templates；无子命令默认 GUI）."""
    parser = argparse.ArgumentParser(prog="zylab", description="zylab 通用科学计算与有限元分析平台")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="无界面运行模板或工程并打印结果摘要")
    run.add_argument("target", help="模板 id、模板文件（JSON 经典 / YAML DSL）或 .zprj 工程文件路径")
    run.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        metavar="参数=值",
        help="参数覆盖（可多次）；DSL 模板为 '参数名=值'，经典模板为 '节点.参数=值'",
    )
    run.add_argument(
        "--scan",
        metavar="节点.参数=取值",
        help="参数化扫描（仅经典模板）；取值为逗号分隔列表（v1,v2,…）或起点:终点:点数（含端点线性插值）",
    )
    run.add_argument(
        "--export",
        metavar="目录",
        help="结果 CSV 导出目录（每个结果节点一个文件；扫描模式按值命名）",
    )
    run.add_argument(
        "--report",
        metavar="路径",
        help="DSL 模板报告导出路径（.md/.html 按后缀选择载体）",
    )

    sub.add_parser("templates", help="列出可用模板")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口：按参数分发子命令；无子命令启动 GUI."""
    _safe_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.target, args.param, args.scan, args.export, args.report)
    if args.command == "templates":
        return _cmd_templates()
    return _launch_gui()


def _cmd_run(target: str, params: list[str], scan: str | None, export: str | None, report: str | None) -> int:
    """``run`` 子命令：加载目标 → 覆盖/扫描 → 进程内求解 → 打印摘要."""
    try:
        template = _load_target(target)
    except (ValueError, StudioError, ProjectFileError) as exc:
        print(f"目标加载失败: {exc}", file=sys.stderr)
        return 2
    if isinstance(template, DslTemplate):
        return _run_dsl(template, params, report)
    if report is not None:
        print("--report 仅支持 DSL 模板（*.yaml / *.yml），已忽略", file=sys.stderr)
    try:
        overrides = _parse_params(params)
    except ValueError as exc:
        print(f"参数非法: {exc}", file=sys.stderr)
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
            if export is not None:
                _export_outcomes(outcome, export, f"@{value:g}")
        return 1 if failed else 0
    outcome = run_workflow(template, overrides, _report_progress)
    print(summarize(outcome))
    if export is not None:
        _export_outcomes(outcome, export)
    return 0 if outcome.succeeded else 1


def _run_dsl(template: DslTemplate, params: list[str], report: str | None) -> int:
    """DSL 模板无头运行：参数覆盖 → 绑定执行 → 摘要 + 可选报告导出."""
    try:
        values = _parse_dsl_params(params)
        executable = template.bind_params(template.evaluate(values))
    except (ValueError, StudioError) as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    outcome = run_workflow(executable, {}, _report_progress)
    print(summarize(outcome))
    if not outcome.succeeded:
        return 1
    if report is None:
        return 0
    outputs = {o.node_id: o.result for o in outcome.outcomes if o.result is not None}
    try:
        _write_report(template, outputs, report, template.evaluate(values))
    except OSError as exc:
        print(f"报告导出失败: {exc}", file=sys.stderr)
        return 1
    return 0


def _write_report(template: DslTemplate, outputs: dict[str, object], path: str, values: dict[str, object]) -> None:
    """按路径后缀导出 DSL 报告（.html 为内嵌图网页，其余 Markdown）."""
    target = Path(path)
    if target.suffix.lower() in (".html", ".htm"):
        text = build_html(template, outputs, values)
    else:
        text = build_markdown(template, outputs, values)
    target.write_text(text, encoding="utf-8")
    print(f"报告已导出: {target}")


def _parse_dsl_params(raw: list[str]) -> dict[str, float | int | str]:
    """解析 DSL 参数覆盖 ``参数名=值``（数值转 float/int，其余按文本）."""
    values: dict[str, float | int | str] = {}
    for item in raw:
        name, sep, raw_value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"参数 {item!r} 应为 '参数名=值' 格式")
        try:
            values[name] = _parse_value(raw_value)
        except ValueError:
            values[name] = raw_value  # 文本参数保持字符串
    return values


def _export_outcomes(outcome: RunOutcome, directory: str, suffix: str = "") -> None:
    """把运行结果按节点导出 CSV（失败节点跳过并告警）."""
    from zylab.fea import export_csv

    out_dir = Path(directory)
    for o in outcome.outcomes:
        if o.result is None:
            continue
        try:
            path = export_csv(o.result, out_dir / f"{o.node_id}{suffix}.csv")
        except ValueError:
            continue  # 模型等非解类型不导出
        except OSError as exc:
            print(f"导出失败: {o.node_id}: {exc}", file=sys.stderr)
            continue
        print(f"已导出: {path}", file=sys.stderr)


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
    """解析运行目标：文件路径（.zprj 工程 / .yaml DSL / .json 模板）或注册表模板 id."""
    path = Path(target)
    if path.exists():
        if path.suffix == ".zprj":
            with Project.open(path) as proj:
                data = proj.read_json("model", "workflow")
            return Template.from_dict(data)
        if path.suffix.lower() in (".yaml", ".yml"):
            return load_dsl(path)
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


def _safe_console_encoding() -> None:
    """输出流兜底为 errors=replace（中文 Windows GBK 控制台遇 '²' 等字符不再崩溃）."""
    with contextlib.suppress(OSError, ValueError):
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")


if __name__ == "__main__":  # python -m zylab.cli / fspack run_module 入口
    raise SystemExit(main())
