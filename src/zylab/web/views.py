"""首页视图：工程（.zprj）列表."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from operator import attrgetter
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

__all__ = ["ProjectSummary", "home", "load_project_summaries"]

#: 工程文件后缀
_PROJECT_SUFFIX = ".zprj"


@dataclass(frozen=True)
class ProjectSummary:
    """工程摘要（首页列表条目）."""

    name: str
    path: Path
    modified: datetime
    size_kb: float


def load_project_summaries(projects_dir: Path) -> list[ProjectSummary]:
    """扫描目录下的 ``.zprj`` 工程，按修改时间降序返回摘要列表.

    :param projects_dir: 工程目录（不存在时返回空列表）。
    """
    if not projects_dir.is_dir():
        return []
    summaries: list[ProjectSummary] = []
    for path in projects_dir.glob(f"*{_PROJECT_SUFFIX}"):
        stat = path.stat()
        summaries.append(
            ProjectSummary(
                name=path.stem,
                path=path,
                modified=datetime.fromtimestamp(stat.st_mtime),
                size_kb=stat.st_size / 1024,
            )
        )
    summaries.sort(key=attrgetter("modified"), reverse=True)
    return summaries


def home(request: HttpRequest) -> HttpResponse:
    """首页：列出最近的工程文件."""
    projects_dir = settings.PROJECTS_DIR
    assert isinstance(projects_dir, Path)  # 类型收窄：base 设置恒为 Path
    projects = load_project_summaries(projects_dir)
    return render(request, "zylab/home.html", {"projects": projects})
