"""URL 路由（W1：首页项目列表）."""

from __future__ import annotations

from django.urls import path

from . import views

__all__ = ["urlpatterns"]

urlpatterns = [
    path("", views.home, name="home"),
]
