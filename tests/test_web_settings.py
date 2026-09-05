"""zylab Web 设置模块测试（base/dev/prod/wsgi）."""

from __future__ import annotations

import importlib
import sys

import pytest

django = pytest.importorskip("django", reason="web 线需 Django（Python 3.10+）")

from django.core.exceptions import ImproperlyConfigured  # noqa: E402

import zylab.web.settings.base as base_module  # noqa: E402

_PROD = "zylab.web.settings.prod"


def _fresh_prod():
    """（重新）加载 prod 设置模块，使其按当前环境变量求值."""
    if _PROD in sys.modules:
        return importlib.reload(sys.modules[_PROD])
    return importlib.import_module(_PROD)


@pytest.fixture(autouse=True)
def _restore_settings_modules():
    """每个用例后重载 base/prod，撤销用例内的 reload 副作用."""
    yield
    importlib.reload(base_module)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZYLAB_WEB_SECRET_KEY", "x")
        mp.setenv("ZYLAB_WEB_ALLOWED_HOSTS", "x")
        _fresh_prod()


def test_dev_settings_active():
    """pytest 环境挂载 dev 设置；DEBUG 由 Django 测试框架强制关闭（既有行为）."""
    from django.conf import settings

    from zylab.web.settings import dev

    assert dev.DEBUG is True
    assert settings.DEBUG is False
    assert "zylab.web" in settings.INSTALLED_APPS
    assert settings.ROOT_URLCONF == "zylab.web.urls"


def test_base_projects_dir_env_override(monkeypatch, tmp_path):
    """ZYLAB_WEB_PROJECTS_DIR 显式指定时覆盖默认位置."""
    monkeypatch.setenv("ZYLAB_WEB_PROJECTS_DIR", str(tmp_path))
    module = importlib.reload(base_module)
    assert tmp_path == module.PROJECTS_DIR


def test_base_projects_dir_default(monkeypatch):
    """未显式指定时沿用应用配置层次的 data_dir/projects."""
    monkeypatch.delenv("ZYLAB_WEB_PROJECTS_DIR", raising=False)
    module = importlib.reload(base_module)
    from zylab.core.config import load_config

    assert load_config().data_dir / "projects" == module.PROJECTS_DIR


def test_prod_requires_secret_key(monkeypatch):
    """生产设置缺 ZYLAB_WEB_SECRET_KEY 时拒绝启动."""
    monkeypatch.setenv("ZYLAB_WEB_ALLOWED_HOSTS", "example.com")
    monkeypatch.delenv("ZYLAB_WEB_SECRET_KEY", raising=False)
    with pytest.raises(ImproperlyConfigured, match="SECRET_KEY"):
        _fresh_prod()


def test_prod_requires_allowed_hosts(monkeypatch):
    """生产设置缺 ZYLAB_WEB_ALLOWED_HOSTS 时拒绝启动."""
    monkeypatch.setenv("ZYLAB_WEB_SECRET_KEY", "k")
    monkeypatch.delenv("ZYLAB_WEB_ALLOWED_HOSTS", raising=False)
    with pytest.raises(ImproperlyConfigured, match="ALLOWED_HOSTS"):
        _fresh_prod()


def test_prod_reads_env(monkeypatch):
    """生产设置从环境变量读取密钥与主机表（逗号分隔、去空白）."""
    monkeypatch.setenv("ZYLAB_WEB_SECRET_KEY", "secret")
    monkeypatch.setenv("ZYLAB_WEB_ALLOWED_HOSTS", "a.example.com, b.example.com")
    module = _fresh_prod()
    assert module.SECRET_KEY == "secret"
    assert module.ALLOWED_HOSTS == ["a.example.com", "b.example.com"]
    assert module.DEBUG is False


def test_wsgi_application():
    """WSGI 入口暴露可调用 application（gunicorn 挂载点）."""
    from zylab.web import wsgi

    assert callable(wsgi.application)
