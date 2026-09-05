"""W2：画布布局算法 + 内置模板条目."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="web 线需 Django（Python 3.10+）")

from zylab.studio import BUILTIN_TEMPLATES  # noqa: E402
from zylab.web.services import (  # noqa: E402
    ProjectCRUDError,
    ProjectService,
    builtin_template_entries,
    compute_canvas_layout,
)


class TestBuiltinTemplateEntries:
    def test_non_empty(self):
        assert len(builtin_template_entries()) > 0

    def test_sorted_by_discipline_then_id(self):
        entries = builtin_template_entries()
        pairs = [(e.discipline, e.template_id) for e in entries]
        assert pairs == sorted(pairs)

    def test_each_entry_matches_template(self):
        by_id = {t.id: t for t in BUILTIN_TEMPLATES}
        for e in builtin_template_entries():
            assert by_id[e.template_id].name == e.name
            assert by_id[e.template_id].discipline == e.discipline
            assert len(by_id[e.template_id].nodes) == e.node_count


class TestCanvasLayout:
    def test_single_node_template(self):
        """单节点模板 → 单列单行（用真实内置模板的首个节点）."""
        tpl = BUILTIN_TEMPLATES[0]
        # 截取第一个节点构造最小模板
        first_node = tpl.nodes[0]
        from zylab.studio.template import Template, TemplateNode

        mini = Template(
            id="mini",
            name="mini",
            nodes=(TemplateNode(id=first_node.id, type_id=first_node.type_id),),
        )
        layout = compute_canvas_layout(mini)
        assert layout.cols == 1
        assert layout.rows == 1
        assert len(layout.nodes) == 1
        assert len(layout.edges) == 0
        node = layout.nodes[0]
        assert node.col == 0
        assert node.row == 0
        assert node.state in ("ready", "up_to_date", "running", "unfulfilled", "failed")

    def test_two_col_layout(self):
        """source → analysis → post 三节点：3 列."""
        # 取一个真实内置模板，验证分层数等于最大 depth + 1
        for t in BUILTIN_TEMPLATES:
            layout = compute_canvas_layout(t)
            cols = {n.col for n in layout.nodes}
            assert max(cols) + 1 == layout.cols
            # 每个节点的 col <= 上游节点的 col + 1
            node_col = {n.node_id: n.col for n in layout.nodes}
            for edge in layout.edges:
                assert node_col[edge.dst_node] > node_col[edge.src_node], (
                    f"{edge.src_node}(col={node_col[edge.src_node]}) → {edge.dst_node}(col={node_col[edge.dst_node]})"
                )

    def test_edges_match_node_inputs(self):
        """连线数 = 全部节点 inputs 条目数."""
        for t in BUILTIN_TEMPLATES:
            layout = compute_canvas_layout(t)
            total_inputs = sum(len(n.inputs) for n in layout.nodes)
            assert len(layout.edges) == total_inputs


class TestProjectService:
    def test_crud_flow(self, tmp_path):
        service = ProjectService(tmp_path)
        tpl = BUILTIN_TEMPLATES[0]

        # 创建
        path = service.create_from_template("demo", tpl)
        assert path.exists()
        assert path.name == "demo.zprj"

        # 列表
        paths = service.list_projects()
        assert len(paths) == 1

        # 打开
        reloaded_path, reloaded = service.open_project("demo")
        assert reloaded_path == path
        assert reloaded.id == tpl.id

        # 保存（覆盖）
        service.save_project("demo", tpl)
        assert path.exists()

        # 删除
        service.delete_project("demo")
        assert not path.exists()
        assert service.list_projects() == ()

    def test_create_duplicate_fails(self, tmp_path):
        service = ProjectService(tmp_path)
        tpl = BUILTIN_TEMPLATES[0]
        service.create_from_template("dup", tpl)
        with pytest.raises(ProjectCRUDError, match="已存在"):
            service.create_from_template("dup", tpl)

    def test_invalid_name_rejected(self, tmp_path):
        service = ProjectService(tmp_path)
        tpl = BUILTIN_TEMPLATES[0]
        with pytest.raises(ProjectCRUDError, match="不能为空"):
            service.create_from_template("", tpl)
        with pytest.raises(ProjectCRUDError, match="非法"):
            service.create_from_template("../escape", tpl)
        with pytest.raises(ProjectCRUDError, match="非法"):
            service.create_from_template("a/b", tpl)

    def test_delete_nonexistent_silent(self, tmp_path):
        service = ProjectService(tmp_path)
        # 不存在的文件，不应抛错
        service.delete_project("no_such_file")

    def test_open_missing_raises(self, tmp_path):
        service = ProjectService(tmp_path)
        with pytest.raises(ProjectCRUDError):
            service.open_project("nope")

    def test_ensure_dir_creates(self, tmp_path):
        service = ProjectService(tmp_path / "nested" / "deep")
        result = service.ensure_dir()
        assert result.is_dir()
