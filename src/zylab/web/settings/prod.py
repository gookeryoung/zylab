"""生产设置（gunicorn）：密钥与主机表必须经环境变量显式提供.

- ``ZYLAB_WEB_SECRET_KEY``：会话密钥（必填）
- ``ZYLAB_WEB_ALLOWED_HOSTS``：允许的主机表，逗号分隔（必填）
"""

from __future__ import annotations

import os

from django.core.exceptions import ImproperlyConfigured

from .base import *

DEBUG = False

_secret = os.environ.get("ZYLAB_WEB_SECRET_KEY")
if not _secret:
    raise ImproperlyConfigured("生产环境必须设置 ZYLAB_WEB_SECRET_KEY 环境变量")
SECRET_KEY = _secret

ALLOWED_HOSTS = [h.strip() for h in os.environ.get("ZYLAB_WEB_ALLOWED_HOSTS", "").split(",") if h.strip()]
if not ALLOWED_HOSTS:
    raise ImproperlyConfigured("生产环境必须设置 ZYLAB_WEB_ALLOWED_HOSTS 环境变量（逗号分隔）")
