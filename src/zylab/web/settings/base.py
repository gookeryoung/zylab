"""zylab Web 共享基础设置（dev/prod 继承；环境变量前缀 ``ZYLAB_WEB_``）.

首版为单用户内网部署，无认证/会话/数据库；W2 起按需演进。
"""

from __future__ import annotations

import os
from pathlib import Path

from zylab.core.config import load_config

#: Web 包根目录（zylab/web）
BASE_DIR = Path(__file__).resolve().parent.parent

#: 会话密钥：环境变量优先，缺省为开发占位值（生产环境由 prod 设置强制校验）
SECRET_KEY = os.environ.get("ZYLAB_WEB_SECRET_KEY", "dev-only-insecure-key")

DEBUG = False

INSTALLED_APPS = [
    "zylab.web",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "zylab.web.urls"

#: 模板目录（工程目录外无全局模板，app 模板经 APP_DIRS 发现）
_TEMPLATE_DIRS: list[str] = []
_CONTEXT_PROCESSORS: list[str] = []

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": _TEMPLATE_DIRS,
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": _CONTEXT_PROCESSORS},
    },
]

WSGI_APPLICATION = "zylab.web.wsgi.application"

USE_TZ = True
TIME_ZONE = "Asia/Shanghai"

#: 静态文件 URL 前缀（W6 引入 whitenoise 托管）
STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

#: 工程（.zprj）列表目录：ZYLAB_WEB_PROJECTS_DIR 显式指定，
#: 否则沿用应用配置层次（config.toml / ZYLAB_DATA_DIR）的 data_dir/projects
PROJECTS_DIR = Path(os.environ.get("ZYLAB_WEB_PROJECTS_DIR") or (load_config().data_dir / "projects"))
