"""WSGI 入口（生产：gunicorn zylab.web.wsgi:application）."""

from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "zylab.web.settings.prod")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
