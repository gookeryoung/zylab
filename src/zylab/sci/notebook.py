"""笔记本文档模型与 ``.znbk`` 持久化（Qt-free）.

参照 Jupyter 模式：单元（cell）= 代码 + 输出列表；输出涵盖 stdout/stderr
流文本、表达式结果 repr、错误回溯与绘图数值快照。绘图保存数值数组
（不存位图），重开笔记本可离线重绘。变量数值本身不持久化——恢复
工作区靠重算单元（与 jupyter 语义一致）。

JSON 结构（``format: zylab.notebook.v1``）::

    {
      "format": "zylab.notebook.v1",
      "metadata": {"created": "...", "modified": "...", "kernel": {...}},
      "cells": [
        {"id": "hex32", "source": "...", "execution_count": 3,
         "outputs": [
           {"kind": "stream", "name": "stdout", "text": "..."},
           {"kind": "result", "repr_text": "...", "type_name": "ndarray", "shape": "(100,)"},
           {"kind": "error", "ename": "NameError", "traceback_text": "..."},
           {"kind": "plot", "title": "...", "xlabel": "...", "ylabel": "...",
            "series": [{"x": [...], "y": [...], "label": "..."}]}
         ]}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CellOutput",
    "ErrorOutput",
    "Notebook",
    "NotebookCell",
    "NotebookError",
    "PlotOutput",
    "PlotSeries",
    "ResultOutput",
    "StreamOutput",
    "load_notebook",
    "new_cell",
    "save_notebook",
]

logger = logging.getLogger(__name__)

#: 笔记本文件 schema 标识（顶层 ``format`` 字段）
_NOTEBOOK_FORMAT = "zylab.notebook.v1"


class NotebookError(Exception):
    """笔记本文件读取/解析失败."""


@dataclass(frozen=True)
class StreamOutput:
    """流输出（stdout/stderr 文本）."""

    name: str  # "stdout" / "stderr"
    text: str


@dataclass(frozen=True)
class ResultOutput:
    """表达式结果输出（repr 文本 + 类型/形状摘要）."""

    repr_text: str
    type_name: str
    shape: str


@dataclass(frozen=True)
class ErrorOutput:
    """执行错误输出（异常类型名 + 完整回溯文本）."""

    ename: str
    traceback_text: str


@dataclass(frozen=True)
class PlotSeries:
    """绘图曲线的数值快照（坐标转为 Python 数值列表以便 JSON 序列化）."""

    x: list[float]
    y: list[float]
    label: str = ""


@dataclass(frozen=True)
class PlotOutput:
    """绘图输出：同一单元执行内的全部曲线请求合并为一个图（多 series 单图）."""

    title: str
    xlabel: str
    ylabel: str
    series: list[PlotSeries] = field(default_factory=list)


#: 单元输出联合类型（kind 判别经 isinstance）
CellOutput = StreamOutput | ResultOutput | ErrorOutput | PlotOutput


@dataclass
class NotebookCell:
    """笔记本单元：代码源文本 + 执行序号 + 输出列表."""

    id: str
    source: str = ""
    #: 执行序号（jupyter 风格：未执行为 None）
    execution_count: int | None = None
    outputs: list[CellOutput] = field(default_factory=list)

    def clear_outputs(self) -> None:
        """清空输出并重置执行序号（单元编辑后待重算状态）."""
        self.outputs.clear()
        self.execution_count = None


@dataclass
class Notebook:
    """笔记本文档：单元列表 + 元数据."""

    cells: list[NotebookCell] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def new_cell(source: str = "") -> NotebookCell:
    """创建新单元（随机 id，未执行状态）."""
    return NotebookCell(id=uuid.uuid4().hex, source=source)


def save_notebook(path: Path, notebook: Notebook) -> Path:
    """将笔记本保存为人类可读 ``.znbk`` JSON（原子写）.

    :param path: 目标路径（建议 ``.znbk`` 后缀）。
    :param notebook: 笔记本文档。
    :return: 保存路径。
    :raises NotebookError: 序列化或写入失败时抛出。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        payload = json.dumps(_notebook_to_dict(notebook), ensure_ascii=False, indent=2)
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise NotebookError(f"笔记本写入失败: {path}") from exc
    logger.info("笔记本已保存: %s", path)
    return path


def load_notebook(path: Path) -> Notebook:
    """打开 ``.znbk`` 笔记本并解析为文档.

    :param path: 笔记本文件路径。
    :return: 笔记本文档。
    :raises NotebookError: 文件不存在、格式不识别或结构非法时抛出。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NotebookError(f"笔记本文件不存在或无法读取: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NotebookError(f"笔记本 JSON 解析失败: {path.name}") from exc
    if not isinstance(data, dict) or data.get("format") != _NOTEBOOK_FORMAT:
        raise NotebookError(f"笔记本格式不识别（缺 format 字段或版本不符）: {path.name}")
    return _notebook_from_dict(data)


def _notebook_to_dict(notebook: Notebook) -> dict[str, Any]:
    """文档序列化为 JSON 兼容 dict（modified 时间戳刷新为当前）."""
    meta = dict(notebook.metadata)
    meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    meta["modified"] = datetime.now().isoformat(timespec="seconds")
    return {
        "format": _NOTEBOOK_FORMAT,
        "metadata": meta,
        "cells": [_cell_to_dict(cell) for cell in notebook.cells],
    }


def _cell_to_dict(cell: NotebookCell) -> dict[str, Any]:
    """单元序列化（输出列表按 kind 分发）."""
    return {
        "id": cell.id,
        "source": cell.source,
        "execution_count": cell.execution_count,
        "outputs": [_output_to_dict(out) for out in cell.outputs],
    }


def _output_to_dict(out: CellOutput) -> dict[str, Any]:
    """输出对象序列化为带 ``kind`` 判别字段的 dict."""
    if isinstance(out, StreamOutput):
        return {"kind": "stream", "name": out.name, "text": out.text}
    if isinstance(out, ResultOutput):
        return {"kind": "result", "repr_text": out.repr_text, "type_name": out.type_name, "shape": out.shape}
    if isinstance(out, ErrorOutput):
        return {"kind": "error", "ename": out.ename, "traceback_text": out.traceback_text}
    return {
        "kind": "plot",
        "title": out.title,
        "xlabel": out.xlabel,
        "ylabel": out.ylabel,
        "series": [{"x": s.x, "y": s.y, "label": s.label} for s in out.series],
    }


def _notebook_from_dict(data: dict[str, Any]) -> Notebook:
    """dict 反序列化为文档（结构非法抛 :class:`NotebookError`）."""
    cells_raw = data.get("cells", [])
    if not isinstance(cells_raw, list):
        raise NotebookError("cells 字段须为列表")
    cells: list[NotebookCell] = []
    for index, item in enumerate(cells_raw):
        if not isinstance(item, dict):
            raise NotebookError(f"第 {index} 个单元须为对象")
        cells.append(_cell_from_dict(item, index))
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise NotebookError("metadata 字段须为对象")
    return Notebook(cells=cells, metadata=metadata)


def _cell_from_dict(item: dict[str, Any], index: int) -> NotebookCell:
    """单元反序列化（id/source 缺失时给默认值，输出 kind 未知时报错）."""
    cell_id = item.get("id")
    if not isinstance(cell_id, str) or not cell_id:
        cell_id = uuid.uuid4().hex  # 旧文件缺 id：补随机 id（仅展示用，无引用关系）
    source = item.get("source", "")
    if not isinstance(source, str):
        raise NotebookError(f"第 {index} 个单元 source 须为字符串")
    count = item.get("execution_count")
    if count is not None and not isinstance(count, int):
        raise NotebookError(f"第 {index} 个单元 execution_count 须为整数或 null")
    outputs_raw = item.get("outputs", [])
    if not isinstance(outputs_raw, list):
        raise NotebookError(f"第 {index} 个单元 outputs 须为列表")
    outputs = [_output_from_dict(out, index, i) for i, out in enumerate(outputs_raw)]
    return NotebookCell(id=cell_id, source=source, execution_count=count, outputs=outputs)


def _output_from_dict(out: Any, cell_index: int, out_index: int) -> CellOutput:
    """输出 dict 反序列化（kind 分发；未知 kind / 必需字段缺失时报错）."""
    where = f"第 {cell_index} 个单元第 {out_index} 个输出"
    if not isinstance(out, dict):
        raise NotebookError(f"{where}须为对象")
    kind = out.get("kind")
    try:
        if kind == "stream":
            return StreamOutput(name=out["name"], text=out["text"])
        if kind == "result":
            return ResultOutput(repr_text=out["repr_text"], type_name=out["type_name"], shape=out["shape"])
        if kind == "error":
            return ErrorOutput(ename=out["ename"], traceback_text=out["traceback_text"])
        if kind == "plot":
            return PlotOutput(
                title=out.get("title", ""),
                xlabel=out.get("xlabel", ""),
                ylabel=out.get("ylabel", ""),
                series=[_series_from_dict(s) for s in out.get("series", [])],
            )
    except KeyError as exc:
        raise NotebookError(f"{where}缺必需字段: {exc.args[0]}") from exc
    raise NotebookError(f"{where}kind 未知: {kind!r}")


def _series_from_dict(item: Any) -> PlotSeries:
    """曲线 dict 反序列化（坐标数组经 NumPy 校验并转数值列表）."""
    if not isinstance(item, dict):
        raise NotebookError("绘图 series 项须为对象")
    try:
        x = np.asarray(item["x"], dtype=float).tolist()
        y = np.asarray(item["y"], dtype=float).tolist()
    except (KeyError, TypeError, ValueError) as exc:
        raise NotebookError(f"绘图 series 非法: {exc}") from exc
    if len(x) != len(y):
        raise NotebookError("绘图 series x/y 长度不一致")
    label = item.get("label", "")
    if not isinstance(label, str):
        raise NotebookError("绘图 series label 须为字符串")
    return PlotSeries(x=x, y=y, label=label)
