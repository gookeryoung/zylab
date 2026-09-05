"""URL 路由（W2：项目 CRUD + 画布编辑器）."""

from __future__ import annotations

from django.urls import path

from . import views

__all__ = ["urlpatterns"]

urlpatterns = [
    path("", views.home, name="home"),  # 重定向到项目列表
    path("projects/", views.project_list, name="project_list"),
    path("projects/new/", views.project_new, name="project_new"),
    path("projects/<str:name>/", views.project_editor, name="project_editor"),
    path("projects/<str:name>/delete/", views.project_delete, name="project_delete"),
    path("api/templates/builtin/", views.api_builtin_templates, name="api_builtin_templates"),
]
