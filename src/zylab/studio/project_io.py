"""工程文件持久化（Qt-free）：workflow 模板以人类可读 JSON 存 ``.zprj``.

v1 工程内嵌当前模板（含图内参数），数据量 KB 级 —— JSON 文本可直接
阅读、diff 与版本管理，性能与二进制容器无差别（HDF5 当初是为将来的
大规模结果数组预留，当前工程并无此类负载）。旧版 HDF5 容器工程在
打开时按文件魔数自动识别并回读。将来内嵌大数组时再演进为
「JSON 骨架 + 附属二进制数组」的分层格式，可读与性能兼得。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from zylab.core.errors import ProjectFileError
from zylab.core.project import Project

from .errors import StudioError, TemplateError
from .template import Template, template_from_json

__all__ = ["ProjectIOError", "load_workflow", "save_workflow"]

logger = logging.getLogger(__name__)

#: HDF5 容器文件魔数（旧版 .zprj 二进制工程判别依据）
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

#: 工程文件 schema 标识（顶层 "format" 字段）
_WORKFLOW_FORMAT = "zylab.workflow.v1"


class ProjectIOError(StudioError):
    """工程文件读取/解析失败."""


def save_workflow(path: Path, template: Template) -> Path:
    """将 workflow 模板保存为人类可读 JSON 工程（原子写）.

    :param path: 目标路径（建议 ``.zprj`` 后缀）。
    :param template: 含当前参数的模板。
    :return: 保存路径。
    :raises ProjectIOError: 写入失败时抛出。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"format": _WORKFLOW_FORMAT, "template": template.to_dict()},
        ensure_ascii=False,
        indent=2,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ProjectIOError(f"工程文件写入失败: {path}") from exc
    logger.info("工程已保存（JSON）: %s", path)
    return path


def load_workflow(path: Path) -> Template:
    """打开工程并解析内嵌 workflow 模板.

    JSON 工程（现行）与 HDF5 容器工程（旧版）按文件魔数自动判别。

    :param path: 工程文件路径。
    :return: 内嵌模板。
    :raises ProjectIOError: 文件不存在、格式非法或解析失败时抛出。
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            header = handle.read(len(_HDF5_MAGIC))
    except OSError as exc:
        raise ProjectIOError(f"工程文件不存在或无法读取: {path}") from exc
    if header == _HDF5_MAGIC:
        return _load_legacy_hdf5(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProjectIOError(f"工程文件读取失败: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProjectIOError(f"工程文件 JSON 解析失败: {path.name}") from exc
    if not isinstance(data, dict) or data.get("format") != _WORKFLOW_FORMAT:
        raise ProjectIOError(f"工程文件格式不识别（缺 format 字段或版本不符）: {path.name}")
    try:
        template_data = data["template"]
    except KeyError as exc:
        raise ProjectIOError(f"工程文件缺 template 字段: {path.name}") from exc
    try:
        return template_from_json(json.dumps(template_data, ensure_ascii=False))
    except TemplateError as exc:
        raise ProjectIOError(f"工程内嵌模板非法: {exc}") from exc


def _load_legacy_hdf5(path: Path) -> Template:
    """旧版 HDF5 容器工程回读（model/workflow JSON 数据集）."""
    try:
        with Project.open(path) as proj:
            data = proj.read_json("model", "workflow")
    except ProjectFileError as exc:
        raise ProjectIOError(f"旧版工程文件打开失败: {exc}") from exc
    try:
        return Template.from_dict(data)
    except TemplateError as exc:
        raise ProjectIOError(f"工程内嵌模板非法: {exc}") from exc
