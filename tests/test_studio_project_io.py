"""studio.project_io 工程持久化测试：JSON 保存/回读、旧 HDF5 兼容与失败路径."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zylab.core.project import Project
from zylab.studio import BUILTIN_TEMPLATES, ProjectIOError, Template, load_workflow, save_workflow

__all__ = []


def _sample() -> Template:
    """取一个内置模板作样本."""
    return next(t for t in BUILTIN_TEMPLATES if t.id == "structural.truss_nonlinear")


class TestSaveLoad:
    """JSON 工程保存与回读."""

    def test_roundtrip_keeps_template(self, tmp_path: Path) -> None:
        """保存后回读：模板 id/节点/参数完整还原."""
        path = tmp_path / "case.zprj"
        template = _sample()
        save_workflow(path, template)
        loaded = load_workflow(path)
        assert loaded.id == template.id
        assert loaded.to_dict() == template.to_dict()

    def test_output_is_human_readable_json(self, tmp_path: Path) -> None:
        """产物为缩进 JSON 文本（可阅读、含中文原样）."""
        path = tmp_path / "case.zprj"
        template = _sample()
        save_workflow(path, template)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert data["format"] == "zylab.workflow.v1"
        assert data["template"]["id"] == template.id
        assert "两杆" in text  # 中文未被转义
        assert "\n  " in text  # 缩进可读

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """保存自动创建父目录并返回路径."""
        path = tmp_path / "deep" / "nested" / "case.zprj"
        assert save_workflow(path, _sample()) == path
        assert path.exists()

    def test_save_atomic_cleanup_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """写入失败时清理临时文件并抛 ProjectIOError."""
        path = tmp_path / "case.zprj"

        def _boom(self: Path, target: object) -> None:
            raise OSError("磁盘故障")

        monkeypatch.setattr(Path, "replace", _boom)
        with pytest.raises(ProjectIOError, match="写入失败"):
            save_workflow(path, _sample())
        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_load_legacy_hdf5_project(self, tmp_path: Path) -> None:
        """旧版 HDF5 容器工程按魔数识别并回读."""
        template = _sample()
        with Project.create(tmp_path / "legacy.zprj", name=template.name) as proj:
            proj.write_json("model", "workflow", template.to_dict())
        loaded = load_workflow(tmp_path / "legacy.zprj")
        assert loaded.id == template.id
        assert loaded.to_dict() == template.to_dict()


class TestLoadErrors:
    """回读失败路径."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """文件不存在."""
        with pytest.raises(ProjectIOError, match="不存在"):
            load_workflow(tmp_path / "ghost.zprj")

    def test_bad_json(self, tmp_path: Path) -> None:
        """非 JSON 文本."""
        bad = tmp_path / "bad.zprj"
        bad.write_text("not a workflow", encoding="utf-8")
        with pytest.raises(ProjectIOError, match="JSON 解析失败"):
            load_workflow(bad)

    def test_unknown_format(self, tmp_path: Path) -> None:
        """缺 format 字段或版本不符."""
        bad = tmp_path / "fmt.zprj"
        bad.write_text(json.dumps({"template": {}}), encoding="utf-8")
        with pytest.raises(ProjectIOError, match="格式不识别"):
            load_workflow(bad)

    def test_missing_template_field(self, tmp_path: Path) -> None:
        """缺 template 字段."""
        bad = tmp_path / "noTpl.zprj"
        bad.write_text(json.dumps({"format": "zylab.workflow.v1"}), encoding="utf-8")
        with pytest.raises(ProjectIOError, match="template 字段"):
            load_workflow(bad)

    def test_invalid_template_payload(self, tmp_path: Path) -> None:
        """template 载荷非法（缺必填字段）."""
        bad = tmp_path / "badTpl.zprj"
        payload = json.dumps({"format": "zylab.workflow.v1", "template": {"id": "x"}})
        bad.write_text(payload, encoding="utf-8")
        with pytest.raises(ProjectIOError, match="内嵌模板非法"):
            load_workflow(bad)

    def test_legacy_hdf5_corrupt_template(self, tmp_path: Path) -> None:
        """旧版工程内嵌模板非法."""
        with Project.create(tmp_path / "legacy.zprj", name="坏工程") as proj:
            proj.write_json("model", "workflow", {"id": "x"})
        with pytest.raises(ProjectIOError, match="内嵌模板非法"):
            load_workflow(tmp_path / "legacy.zprj")
