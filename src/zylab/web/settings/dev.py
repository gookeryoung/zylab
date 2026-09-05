"""开发设置（runserver / pytest）：DEBUG 开启、宽松主机表."""

from __future__ import annotations

from .base import *

DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "testserver"]
