"""zylab Web 应用配置（Django app）."""

from __future__ import annotations

from django.apps import AppConfig

__all__ = ["WebConfig"]


class WebConfig(AppConfig):
    """zylab Web 页面应用（模板随包经 APP_DIRS 发现；无模型，首版无数据库）."""

    name = "zylab.web"
    verbose_name = "zylab Web"
