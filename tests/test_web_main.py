"""zylab Web 启动入口测试（zylab-web 管理命令）."""

from __future__ import annotations

import sys

import pytest

from zylab.web.__main__ import main

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="web 线需 Django（Python 3.10+）")


def test_main_runs_check_command(capsys):
    """以 dev 设置执行 Django 系统检查并返回 0."""
    assert main(["zylab-web", "check"]) == 0
    assert "no issues" in capsys.readouterr().out.lower()


def test_main_version_guard(monkeypatch, capsys):
    """Python < 3.10 时给出友好提示并返回 1（不触碰 Django）."""
    monkeypatch.setattr(sys, "version_info", (3, 9, 0, "final", 0))
    assert main(["zylab-web", "check"]) == 1
    assert "Python 3.10" in capsys.readouterr().out
