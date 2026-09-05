"""W2：项目列表 + 新建 + 编辑器画布 + API 视图."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="web 线需 Django（Python 3.10+）")


def test_root_redirects_to_project_list(client):
    """根路径重定向到项目列表."""
    r = client.get("/")
    assert r.status_code == 302
    assert r.url.endswith("/projects/")


def test_project_list_empty(client, settings, tmp_path):
    """空目录时项目列表显示空态."""
    settings.PROJECTS_DIR = tmp_path
    r = client.get("/projects/")
    assert r.status_code == 200
    assert "暂无工程" in r.content.decode("utf-8")


def test_project_list_with_projects(client, settings, tmp_path):
    """列出 .zprj 工程，按时间降序."""
    import os

    old = tmp_path / "old.zprj"
    old.write_text(
        '{"format":"zylab.workflow.v1","template":{"id":"t","name":"","nodes":[]}}',
        encoding="utf-8",
    )
    os.utime(old, (1_000_000, 1_000_000))
    new = tmp_path / "new.zprj"
    new.write_text(
        '{"format":"zylab.workflow.v1","template":{"id":"t","name":"","nodes":[]}}',
        encoding="utf-8",
    )
    os.utime(new, (5_000_000, 5_000_000))

    settings.PROJECTS_DIR = tmp_path
    r = client.get("/projects/")
    text = r.content.decode("utf-8")
    # 用项目链接 /projects/<name>/ 匹配（排除 /projects/new/ 新建页链接干扰）
    import re

    links = re.findall(r'href="/projects/(old|new)/"', text)
    assert "old" in links
    assert "new" in links
    assert links.index("new") < links.index("old"), f"期望 new 在 old 前，实际顺序: {links}"


def test_project_list_create_from_template(client, settings, tmp_path):
    """POST action=create：从内置模板新建工程."""
    from zylab.studio import BUILTIN_TEMPLATES

    settings.PROJECTS_DIR = tmp_path
    tpl = BUILTIN_TEMPLATES[0]
    r = client.post(
        "/projects/",
        {"action": "create", "name": "myproj", "template_id": tpl.id},
        follow=True,
    )
    assert r.status_code == 200
    # 工程目录下应有文件
    files = list(tmp_path.glob("*.zprj"))
    assert len(files) == 1
    assert files[0].stem == "myproj"
    assert "已创建" in r.content.decode("utf-8")


def test_project_list_create_empty_name_rejected(client, settings, tmp_path):
    """空工程名被拒绝."""
    from zylab.studio import BUILTIN_TEMPLATES

    settings.PROJECTS_DIR = tmp_path
    tpl = BUILTIN_TEMPLATES[0]
    r = client.post(
        "/projects/",
        {"action": "create", "name": "", "template_id": tpl.id},
    )
    assert r.status_code == 200  # 渲染带错误的列表页
    assert "请输入工程名" in r.content.decode("utf-8")


def test_project_list_delete(client, settings, tmp_path):
    """POST action=delete：删除工程."""
    (tmp_path / "todelete.zprj").write_text(
        '{"format":"zylab.workflow.v1","template":{"id":"t","name":"","nodes":[]}}',
        encoding="utf-8",
    )
    settings.PROJECTS_DIR = tmp_path
    r = client.post(
        "/projects/",
        {"action": "delete", "name": "todelete"},
        follow=True,
    )
    assert r.status_code == 200
    assert not (tmp_path / "todelete.zprj").exists()
    assert "已删除" in r.content.decode("utf-8")


def test_project_new_get(client):
    """新建页面 GET 渲染."""
    r = client.get("/projects/new/")
    assert r.status_code == 200
    assert "新建工程" in r.content.decode("utf-8")


def test_project_new_post_success(client, settings, tmp_path):
    """新建页面 POST 成功 → 跳编辑器."""
    from zylab.studio import BUILTIN_TEMPLATES

    settings.PROJECTS_DIR = tmp_path
    tpl = BUILTIN_TEMPLATES[0]
    r = client.post(
        "/projects/new/",
        {"name": "p2", "template_id": tpl.id},
    )
    assert r.status_code == 302
    assert r.url.endswith("/projects/p2/")
    assert (tmp_path / "p2.zprj").exists()


def test_project_new_post_bad_name(client, settings, tmp_path):
    """非法工程名 → 400 + 错误提示."""
    from zylab.studio import BUILTIN_TEMPLATES

    settings.PROJECTS_DIR = tmp_path
    tpl = BUILTIN_TEMPLATES[0]
    r = client.post(
        "/projects/new/",
        {"name": "../bad", "template_id": tpl.id},
    )
    assert r.status_code == 400
    assert "非法" in r.content.decode("utf-8")


def test_project_editor_404(client, settings, tmp_path):
    """不存在的工程 → 404."""
    settings.PROJECTS_DIR = tmp_path
    r = client.get("/projects/no_such/")
    assert r.status_code == 404


def test_project_editor_renders_canvas(client, settings, tmp_path):
    """编辑器画布渲染：含节点、连线信息."""
    from zylab.studio import BUILTIN_TEMPLATES
    from zylab.web.services import ProjectService

    settings.PROJECTS_DIR = tmp_path
    service = ProjectService(tmp_path)
    tpl = BUILTIN_TEMPLATES[0]
    service.create_from_template("canvas_test", tpl)

    r = client.get("/projects/canvas_test/")
    assert r.status_code == 200
    text = r.content.decode("utf-8")
    assert tpl.name in text
    assert "节点数" in text
    assert "分层" in text


def test_project_delete_post(client, settings, tmp_path):
    """独立删除端点（POST）."""
    (tmp_path / "d2.zprj").write_text(
        '{"format":"zylab.workflow.v1","template":{"id":"t","name":"","nodes":[]}}',
        encoding="utf-8",
    )
    settings.PROJECTS_DIR = tmp_path
    r = client.post("/projects/d2/delete/")
    assert r.status_code == 302
    assert not (tmp_path / "d2.zprj").exists()


def test_api_builtin_templates_json(client):
    """内置模板列表 JSON API."""
    r = client.get("/api/templates/builtin/")
    assert r.status_code == 200
    data = r.json()
    assert "templates" in data
    assert len(data["templates"]) > 0
    first = data["templates"][0]
    assert {"template_id", "name", "discipline", "description", "node_count"}.issubset(first)
