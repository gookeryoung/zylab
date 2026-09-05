"""zylab Web 视图：项目 CRUD + 画布编辑器 + 模板 API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import Sequence

from django.conf import settings
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from .services import (
    ProjectCRUDError,
    ProjectService,
    builtin_template_entries,
    compute_canvas_layout,
)

__all__ = [
    "ProjectSummary",
    "api_builtin_templates",
    "home",
    "project_delete",
    "project_editor",
    "project_list",
    "project_new",
]


@dataclass(frozen=True)
class ProjectSummary:
    """工程列表条目."""

    name: str
    path: Path
    modified: datetime
    size_kb: float


# --------------------------------------------------------------------------- 首页/项目列表


def home(request: HttpRequest) -> HttpResponse:
    """根路径重定向到项目列表."""
    return redirect(reverse("project_list"))


def _service() -> ProjectService:
    assert isinstance(settings.PROJECTS_DIR, Path)
    return ProjectService(settings.PROJECTS_DIR)


def _post_str(request: HttpRequest, key: str) -> str:
    """安全取 POST 表单值（收窄 Django QueryDict 的 Union[str,list[str],None]）."""
    raw = request.POST.get(key)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return str(raw or "")


@require_http_methods(["GET", "POST"])
def project_list(request: HttpRequest) -> HttpResponse:
    """项目列表（首页）.

    - GET: 渲染工程文件表 + 新建/删除按钮。
    - POST action=create: 从内置模板创建新工程。
    - POST action=delete: 删除工程。
    """
    service = _service()
    msg: str | None = None

    if request.method == "POST":
        action = _post_str(request, "action")
        if action == "create":
            name = _post_str(request, "name").strip()
            template_id = _post_str(request, "template_id")
            try:
                _create_from_builtin(service, name, template_id)
                msg = f"工程「{name}」已创建"
            except ProjectCRUDError as exc:
                msg = str(exc)
        elif action == "delete":
            name = _post_str(request, "name").strip()
            service.delete_project(name)
            msg = f"工程「{name}」已删除"

    projects = _summarize(service.list_projects())
    return render(
        request,
        "zylab/project_list.html",
        {
            "projects": projects,
            "builtins": builtin_template_entries(),
            "projects_dir": str(service.dir),
            "msg": msg,
        },
    )


# --------------------------------------------------------------------------- 新建 + 编辑器


@require_http_methods(["GET", "POST"])
def project_new(request: HttpRequest) -> HttpResponse:
    """新建工程（表单：工程名 + 内置模板）."""
    if request.method == "POST":
        name = _post_str(request, "name").strip()
        template_id = _post_str(request, "template_id")
        service = _service()
        try:
            _create_from_builtin(service, name, template_id)
            return redirect(reverse("project_editor", args=[name]))
        except ProjectCRUDError as exc:
            return render(
                request,
                "zylab/project_new.html",
                {"builtins": builtin_template_entries(), "error": str(exc), "name": name},
                status=400,
            )
    return render(
        request,
        "zylab/project_new.html",
        {"builtins": builtin_template_entries()},
    )


def _create_from_builtin(service: ProjectService, name: str, template_id: str) -> Path:
    """从内置模板创建工程（内部：service 层 + studio 内置模板表）."""
    if not name:
        raise ProjectCRUDError("请输入工程名")
    from zylab.studio import BUILTIN_TEMPLATES

    template = next((t for t in BUILTIN_TEMPLATES if t.id == template_id), None)
    if template is None:
        raise ProjectCRUDError(f"未找到模板: {template_id!r}")
    return service.create_from_template(name, template)


@require_http_methods(["GET"])
def project_editor(request: HttpRequest, name: str) -> HttpResponse:
    """画布编辑器（W2 首版：静态渲染画布 + 节点详情；运行/参数编辑放 W3）."""
    service = _service()
    try:
        _path, template = service.open_project(name)
    except ProjectCRUDError as exc:
        raise Http404(str(exc)) from exc

    layout = compute_canvas_layout(template)
    return render(
        request,
        "zylab/project_editor.html",
        {
            "name": name,
            "template": template,
            "layout": layout,
        },
    )


@require_http_methods(["POST"])
def project_delete(request: HttpRequest, name: str) -> HttpResponse:
    """删除工程（POST 提交，避免 GET 副作用）."""
    _service().delete_project(name)
    return redirect(reverse("project_list"))


# --------------------------------------------------------------------------- API


def api_builtin_templates(request: HttpRequest) -> JsonResponse:
    """内置模板列表 JSON（供前端动态渲染新建表单）."""
    entries = builtin_template_entries()
    payload = [
        {
            "template_id": e.template_id,
            "name": e.name,
            "discipline": e.discipline,
            "description": e.description,
            "node_count": e.node_count,
        }
        for e in entries
    ]
    return JsonResponse({"templates": payload})


# --------------------------------------------------------------------------- 辅助


def _summarize(paths: Sequence[Path]) -> list[ProjectSummary]:
    """工程文件 → 列表条目."""
    rows: list[ProjectSummary] = []
    for path in paths:
        stat = path.stat()
        rows.append(
            ProjectSummary(
                name=path.stem,
                path=path,
                modified=datetime.fromtimestamp(stat.st_mtime),
                size_kb=stat.st_size / 1024,
            )
        )
    rows.sort(key=attrgetter("modified"), reverse=True)
    return rows
