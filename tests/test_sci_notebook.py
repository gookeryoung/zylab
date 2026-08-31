"""notebook.py 数据模型与 .znbk 持久化测试."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zylab.sci import (
    ErrorOutput,
    Notebook,
    NotebookCell,
    NotebookError,
    PlotOutput,
    PlotSeries,
    ResultOutput,
    StreamOutput,
    load_notebook,
    new_cell,
    save_notebook,
)


def _sample_cell(source: str = "x = linspace(0, pi, 4)", count: int | None = 1) -> NotebookCell:
    """构造含四类输出的样例单元."""
    cell = new_cell(source)
    cell.execution_count = count
    cell.outputs = [
        StreamOutput(name="stdout", text="你好\n"),
        ResultOutput(
            repr_text="array([0.        , 1.04719755, 2.0943951 , 3.14159265])", type_name="ndarray", shape="(4,)"
        ),
        PlotOutput(
            title="正弦",
            xlabel="t",
            ylabel="v",
            series=[PlotSeries(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0], label="sin")],
        ),
        ErrorOutput(ename="NameError", traceback_text="NameError: name 'y' is not defined"),
    ]
    return cell


class TestModel:
    """数据模型基础行为."""

    def test_new_cell_defaults(self) -> None:
        """新单元：唯一 id、未执行、无输出."""
        cell_a, cell_b = new_cell(), new_cell("x = 1")
        assert cell_a.id != cell_b.id
        assert cell_a.source == ""
        assert cell_a.execution_count is None
        assert cell_a.outputs == []

    def test_clear_outputs(self) -> None:
        """清空输出同时重置执行序号（编辑后待重算）."""
        cell = _sample_cell()
        cell.clear_outputs()
        assert cell.outputs == []
        assert cell.execution_count is None
        assert cell.source  # 源码保留


class TestRoundTrip:
    """保存/加载往返."""

    def test_round_trip_all_output_kinds(self, tmp_path: Path) -> None:
        """四类输出完整往返：字段逐一相等."""
        nb = Notebook(cells=[_sample_cell(), new_cell("plot(x, sin(x))")], metadata={"kernel": {"python": "3.10"}})
        path = save_notebook(tmp_path / "demo.znbk", nb)
        loaded = load_notebook(path)
        assert len(loaded.cells) == 2
        src, dst = nb.cells[0], loaded.cells[0]
        assert dst.id == src.id
        assert dst.source == src.source
        assert dst.execution_count == 1
        assert dst.outputs[0] == StreamOutput(name="stdout", text="你好\n")
        assert dst.outputs[1].repr_text.startswith("array([0.")
        assert dst.outputs[2].series[0] == PlotSeries(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0], label="sin")
        assert dst.outputs[2].title == "正弦"
        assert dst.outputs[3].ename == "NameError"
        assert loaded.cells[1].execution_count is None  # 未执行单元
        assert loaded.metadata["kernel"] == {"python": "3.10"}

    def test_file_is_human_readable_json(self, tmp_path: Path) -> None:
        """产物为缩进 JSON：可读、含 format 标识与中文原文（未转义）."""
        path = save_notebook(tmp_path / "demo.znbk", Notebook(cells=[_sample_cell()]))
        text = path.read_text(encoding="utf-8")
        assert json.loads(text)["format"] == "zylab.notebook.v1"
        assert "正弦" in text  # ensure_ascii=False
        assert '"kind": "plot"' in text

    def test_metadata_timestamps(self, tmp_path: Path) -> None:
        """保存刷新 modified、created 缺省补齐且往返保留原值."""
        nb = Notebook(cells=[new_cell()])
        loaded = load_notebook(save_notebook(tmp_path / "t.znbk", nb))
        assert loaded.metadata["created"]
        assert loaded.metadata["modified"]
        created = loaded.metadata["created"]
        again = load_notebook(save_notebook(tmp_path / "t2.znbk", loaded))
        assert again.metadata["created"] == created  # 已有 created 保留不覆盖
        assert again.metadata["modified"] >= loaded.metadata["modified"]  # modified 每次保存刷新

    def test_series_coordinates_normalize_to_float_list(self, tmp_path: Path) -> None:
        """series 坐标经反序列化归一为 float 列表（int 输入亦可）."""
        cell = new_cell()
        cell.outputs = [PlotOutput(title="", xlabel="", ylabel="", series=[PlotSeries(x=[1, 2, 3], y=[4, 5, 6])])]
        loaded = load_notebook(save_notebook(tmp_path / "t.znbk", Notebook(cells=[cell])))
        series = loaded.cells[0].outputs[0].series[0]
        assert series.x == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in series.y)


class TestLoadErrors:
    """加载容错与非法输入."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """文件不存在报错."""
        with pytest.raises(NotebookError, match="不存在"):
            load_notebook(tmp_path / "nope.znbk")

    def test_bad_json_and_format(self, tmp_path: Path) -> None:
        """JSON 非法 / format 缺失均报错."""
        bad = tmp_path / "bad.znbk"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(NotebookError, match="JSON 解析失败"):
            load_notebook(bad)
        bad.write_text(json.dumps({"cells": []}), encoding="utf-8")
        with pytest.raises(NotebookError, match="格式不识别"):
            load_notebook(bad)

    def test_unknown_output_kind(self, tmp_path: Path) -> None:
        """未知输出 kind 报错（含定位信息）."""
        payload = {
            "format": "zylab.notebook.v1",
            "cells": [{"id": "a", "source": "x", "execution_count": None, "outputs": [{"kind": "magic"}]}],
        }
        path = tmp_path / "k.znbk"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(NotebookError, match=r"kind 未知.*magic"):
            load_notebook(path)

    def test_missing_required_output_field(self, tmp_path: Path) -> None:
        """输出缺必需字段（如 stream 缺 text）报错."""
        payload = {
            "format": "zylab.notebook.v1",
            "cells": [{"id": "a", "source": "x", "outputs": [{"kind": "stream", "name": "stdout"}]}],
        }
        path = tmp_path / "m.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match="缺必需字段"):
            load_notebook(path)

    def test_series_length_mismatch(self, tmp_path: Path) -> None:
        """绘图 series x/y 长度不一致报错."""
        payload = {
            "format": "zylab.notebook.v1",
            "cells": [
                {
                    "id": "a",
                    "source": "plot",
                    "outputs": [{"kind": "plot", "title": "", "series": [{"x": [1, 2], "y": [1], "label": ""}]}],
                }
            ],
        }
        path = tmp_path / "s.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match="长度不一致"):
            load_notebook(path)

    def test_cell_defaults_for_legacy(self, tmp_path: Path) -> None:
        """缺 id/outputs/execution_count 的旧式单元给默认值（id 补随机）."""
        payload = {"format": "zylab.notebook.v1", "cells": [{"source": "x = 1"}]}
        path = tmp_path / "l.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_notebook(path)
        assert loaded.cells[0].source == "x = 1"
        assert loaded.cells[0].id
        assert loaded.cells[0].outputs == []
        assert loaded.cells[0].execution_count is None

    def test_cells_not_list_raises(self, tmp_path: Path) -> None:
        """cells 非列表报错."""
        payload = {"format": "zylab.notebook.v1", "cells": "nope"}
        path = tmp_path / "c.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match="cells 字段须为列表"):
            load_notebook(path)

    @pytest.mark.parametrize(
        ("cells", "match"),
        [
            ([123], "单元须为对象"),
            ([{"id": "a", "source": 1}], "source 须为字符串"),
            ([{"id": "a", "source": "x", "execution_count": "1"}], "execution_count 须为整数"),
            ([{"id": "a", "source": "x", "outputs": "nope"}], "outputs 须为列表"),
            ([{"id": "a", "source": "x", "outputs": [1]}], "须为对象"),
        ],
    )
    def test_cell_structure_errors(self, tmp_path: Path, cells: list, match: str) -> None:
        """单元结构非法（类型不符）报错."""
        payload = {"format": "zylab.notebook.v1", "cells": cells}
        path = tmp_path / "b.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match=match):
            load_notebook(path)

    def test_metadata_not_dict_raises(self, tmp_path: Path) -> None:
        """metadata 非对象报错."""
        payload = {"format": "zylab.notebook.v1", "cells": [], "metadata": 3}
        path = tmp_path / "md.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match="metadata 字段须为对象"):
            load_notebook(path)

    @pytest.mark.parametrize(
        ("series", "match"),
        [
            ([1], "series 项须为对象"),
            ([{"x": [1], "y": [1], "label": 2}], "label 须为字符串"),
            ([{"x": ["a"], "y": [1]}], "series 非法"),
            ([{"y": [1]}], "series 非法"),
        ],
    )
    def test_series_structure_errors(self, tmp_path: Path, series: list, match: str) -> None:
        """绘图 series 结构非法报错."""
        payload = {
            "format": "zylab.notebook.v1",
            "cells": [{"id": "a", "source": "p", "outputs": [{"kind": "plot", "series": series}]}],
        }
        path = tmp_path / "se.znbk"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(NotebookError, match=match):
            load_notebook(path)

    def test_save_unserializable_metadata_raises(self, tmp_path: Path) -> None:
        """metadata 含不可序列化值时保存报 NotebookError."""
        notebook = Notebook(cells=[new_cell()], metadata={"bad": object()})
        with pytest.raises(NotebookError, match="写入失败"):
            save_notebook(tmp_path / "u.znbk", notebook)


class TestSave:
    """保存行为."""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """父目录不存在时自动创建."""
        path = save_notebook(tmp_path / "sub" / "dir" / "t.znbk", Notebook())
        assert path.is_file()

    def test_atomic_no_tmp_residue(self, tmp_path: Path) -> None:
        """保存成功后无 .tmp 残留文件."""
        save_notebook(tmp_path / "t.znbk", Notebook(cells=[new_cell()]))
        assert list(tmp_path.glob("*.tmp")) == []
