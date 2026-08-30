"""zylab.cli 命令行测试：run/templates 子命令、参数解析、目标加载与退出码."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zylab.cli import _parse_params, _parse_scan, _parse_value, main
from zylab.core.project import Project
from zylab.studio import TemplateRegistry, save_template

__all__ = []


@pytest.fixture(autouse=True)
def _quiet_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """静音进度回调（避免测试输出被 stderr 进度行刷屏）."""
    monkeypatch.setattr("zylab.cli._report_progress", lambda _p, _m: None)


class TestRunCommand:
    """``zylab run`` 子命令."""

    def test_run_builtin_template(self, capsys: pytest.CaptureFixture[str]) -> None:
        """按模板 id 运行内置模板：退出码 0，摘要含节点行."""
        code = main(["run", "structural.cantilever_static"])
        out = capsys.readouterr().out
        assert code == 0
        assert "[model]" in out
        assert "[solve]" in out

    def test_run_with_param_override(self, capsys: pytest.CaptureFixture[str]) -> None:
        """-p 覆盖参数后运行成功."""
        code = main(["run", "structural.cantilever_static", "-p", "model.nx=8", "-p", "model.ny=4"])
        assert code == 0
        assert "静力" in capsys.readouterr().out

    def test_run_unknown_target_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """未知模板 id 退出码 2 并输出错误."""
        code = main(["run", "no.such_template"])
        assert code == 2
        assert "目标加载失败" in capsys.readouterr().err

    def test_run_bad_param_format_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """非法参数格式退出码 2."""
        code = main(["run", "structural.cantilever_static", "-p", "model.nx"])
        assert code == 2
        assert "应为" in capsys.readouterr().err

    def test_run_failure_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """求解失败（无压缩轴力屈曲）退出码 1，摘要含失败信息."""
        code = main(["run", "structural.column_buckling", "-p", "model.tip_load=0"])
        assert code == 1
        assert "失败" in capsys.readouterr().out

    def test_run_scan(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--scan 逗号列表扫描：逐值输出摘要，退出码 0."""
        code = main(["run", "structural.cantilever_static", "--scan", "model.tip_load=10,20"])
        out = capsys.readouterr().out
        assert code == 0
        assert "扫描值 10" in out
        assert "扫描值 20" in out

    def test_run_scan_bad_format_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--scan 非法格式退出码 2."""
        code = main(["run", "structural.cantilever_static", "--scan", "tip_load=1,2"])
        assert code == 2
        assert "扫描参数非法" in capsys.readouterr().err

    def test_run_scan_bad_node_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--scan 引用不存在节点：目标/引用非法退出码 2."""
        code = main(["run", "structural.cantilever_static", "--scan", "ghost.tip_load=1,2"])
        assert code == 2
        assert "扫描参数非法" in capsys.readouterr().err

    def test_run_export(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--export 目录：结果节点导出 CSV，模型等非解节点跳过."""
        out_dir = tmp_path / "exports"
        code = main(["run", "structural.cantilever_static", "--export", str(out_dir)])
        assert code == 0
        names = {p.name for p in out_dir.glob("*.csv")}
        assert "solve.csv" in names
        assert "model.csv" not in names  # 模型节点非解类型不导出
        assert "已导出" in capsys.readouterr().err

    def test_run_scan_export_suffix(self, tmp_path: Path) -> None:
        """--scan + --export：文件按扫描值后缀命名."""
        out_dir = tmp_path / "scan"
        code = main(["run", "structural.cantilever_static", "--scan", "model.tip_load=10,20", "--export", str(out_dir)])
        assert code == 0
        names = {p.name for p in out_dir.glob("*.csv")}
        assert "solve@10.csv" in names
        assert "solve@20.csv" in names

    def test_run_json_template_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """按 .json 模板文件路径运行."""
        template = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        path = save_template(template, tmp_path / "t.json")
        code = main(["run", str(path)])
        assert code == 0
        assert "静力" in capsys.readouterr().out

    def test_run_zprj_project_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """按 .zprj 工程文件路径运行（内嵌模板自包含）."""
        template = TemplateRegistry.with_builtin().get("structural.cantilever_static")
        path = tmp_path / "demo.zprj"
        with Project.create(path, name=template.name) as proj:
            proj.write_json("model", "workflow", template.to_dict())
        code = main(["run", str(path)])
        assert code == 0
        assert "静力" in capsys.readouterr().out


class TestTemplatesCommand:
    """``zylab templates`` 子命令."""

    def test_lists_builtin_templates(self, capsys: pytest.CaptureFixture[str]) -> None:
        """列出内置模板：id + 名称，退出码 0."""
        code = main(["templates"])
        out = capsys.readouterr().out
        assert code == 0
        assert "structural.cantilever_static" in out
        assert "悬臂梁静力分析" in out


class TestParsers:
    """CLI 参数解析单元."""

    def test_parse_params(self) -> None:
        """参数覆盖列表解析为嵌套表（整值收敛 int）."""
        assert _parse_params(["model.nx=8", "model.tip_load=10.5", "solve.n_modes=6"]) == {
            "model": {"nx": 8, "tip_load": 10.5},
            "solve": {"n_modes": 6},
        }

    @pytest.mark.parametrize("bad", ["model.nx", "nx=8", "=8", "model.=8"])
    def test_parse_params_bad_format(self, bad: str) -> None:
        """非法格式抛 ValueError."""
        with pytest.raises(ValueError, match="格式"):
            _parse_params([bad])

    def test_parse_value(self) -> None:
        """整值转 int，否则 float."""
        assert _parse_value("3") == 3
        assert isinstance(_parse_value("3"), int)
        assert _parse_value("3.5") == 3.5

    def test_parse_scan_list(self) -> None:
        """逗号列表解析."""
        assert _parse_scan("model.tip_load=1, 2,4") == ("model.tip_load", (1.0, 2.0, 4.0))

    def test_parse_scan_linspace(self) -> None:
        """起点:终点:点数线性插值（含端点）."""
        ref, values = _parse_scan("model.tip_load=0:10:5")
        assert ref == "model.tip_load"
        assert values == (0.0, 2.5, 5.0, 7.5, 10.0)

    @pytest.mark.parametrize("bad", ["tip_load=1,2", "model.tip_load", "a.b=1:2:1"])
    def test_parse_scan_bad(self, bad: str) -> None:
        """非法扫描格式抛 ValueError."""
        with pytest.raises(ValueError):
            _parse_scan(bad)


def test_no_command_launches_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    """无子命令时转发 GUI 入口（惰性导入，mock 避免启动事件循环）."""
    from types import ModuleType

    calls: list[str] = []

    def fake_gui() -> int:
        calls.append("launched")
        return 0

    fake = ModuleType("zylab.gui.app")
    fake.main = fake_gui
    monkeypatch.setitem(sys.modules, "zylab.gui.app", fake)
    assert main([]) == 0
    assert calls == ["launched"]
