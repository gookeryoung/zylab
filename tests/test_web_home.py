"""zylab Web 首页视图测试（工程列表）."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="web 线需 Django（Python 3.10+）")

from zylab.web.views import load_project_summaries  # noqa: E402


def test_load_summaries_missing_dir(tmp_path):
    """目录不存在时返回空列表."""
    assert load_project_summaries(tmp_path / "nope") == []


def test_load_summaries_sorted_by_mtime(tmp_path):
    """按修改时间降序排列，忽略非 .zprj 文件."""
    import os

    old = tmp_path / "alpha.zprj"
    old.write_text("{}", encoding="utf-8")
    os.utime(old, (1_000_000, 1_000_000))
    new = tmp_path / "beta.zprj"
    new.write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    summaries = load_project_summaries(tmp_path)

    assert [s.name for s in summaries] == ["beta", "alpha"]
    assert all(s.path.suffix == ".zprj" for s in summaries)
    assert summaries[0].size_kb > 0


def test_home_empty_state(client, settings, tmp_path):
    """无工程时渲染空态提示."""
    settings.PROJECTS_DIR = tmp_path
    response = client.get("/")
    assert response.status_code == 200
    assert "暂无工程" in response.content.decode("utf-8")


def test_home_lists_projects(client, settings, tmp_path):
    """首页按时间降序列出工程并显示名称与大小."""
    import os

    old = tmp_path / "alpha.zprj"
    old.write_text("{}", encoding="utf-8")
    os.utime(old, (1_000_000, 1_000_000))
    (tmp_path / "beta.zprj").write_text("{}", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")
    settings.PROJECTS_DIR = tmp_path

    response = client.get("/")

    assert response.status_code == 200
    text = response.content.decode("utf-8")
    assert "beta" in text
    assert "alpha" in text
    assert "ignored" not in text
    assert "KB" in text
    assert text.index("beta") < text.index("alpha")
