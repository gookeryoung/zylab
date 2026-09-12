"""zylab 工程文件（HDF5 单文件容器，schema v0）.

组结构约定::

    /meta      属性: schema_version / name / created_at / app_version
    /model     模型数据（网格、材料、边界条件，由 FEA 模块写入）
    /results   结果场量（位移、应力等数组）
    /settings  设置项（JSON 文本数据集）

创建操作为原子写（临时文件 + ``replace``），中断不会损坏已有文件；
打开后的增量写入非原子，后续版本将引入事务化保存策略。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .errors import ProjectFileError

__all__ = ["PROJECT_SCHEMA_VERSION", "PROJECT_SUFFIX", "Project"]

logger = logging.getLogger(__name__)

PROJECT_SCHEMA_VERSION = "0.1.0"
PROJECT_SUFFIX = ".zprj"

_RESERVED_GROUPS = ("meta", "model", "results", "settings")


class Project:
    """工程文件句柄，支持上下文管理::

    with Project.create(Path("demo.zprj"), name="演示") as proj:
        proj.write_array("results", "displacement", np.zeros(3))
    with Project.open(Path("demo.zprj")) as proj:
        arr = proj.read_array("results", "displacement")
    """

    def __init__(self, path: Path, h5: h5py.File) -> None:
        """初始化句柄（由 :meth:`create`/:meth:`open` 内部创建）."""
        self._path = path
        self._h5 = h5

    @property
    def path(self) -> Path:
        """工程文件路径."""
        return self._path

    @property
    def meta(self) -> dict[str, Any]:
        """工程元信息（schema_version/name/created_at/app_version）."""
        return dict(self._h5["meta"].attrs)

    @classmethod
    def create(cls, path: Path, *, name: str = "", app_version: str = "") -> Project:
        """原子创建新工程文件（已存在则覆盖）.

        :param path: 目标路径，建议 ``.zprj`` 后缀。
        :param name: 工程名称。
        :param app_version: 创建方应用版本。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with h5py.File(tmp, "w") as h5:
                meta = h5.create_group("meta")
                meta.attrs["schema_version"] = PROJECT_SCHEMA_VERSION
                meta.attrs["name"] = name
                meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
                meta.attrs["app_version"] = app_version
                for group in _RESERVED_GROUPS[1:]:
                    h5.create_group(group)
            tmp.replace(path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ProjectFileError(f"工程文件创建失败: {path}") from exc
        logger.info("工程文件已创建: %s", path)
        return cls.open(path, mode="a")

    @classmethod
    def open(cls, path: Path, *, mode: str = "r") -> Project:
        """打开工程文件并校验 schema 版本.

        :param mode: h5py 打开模式（``r`` 只读 / ``a`` 读写）。
        :raises ProjectFileError: 文件不存在、格式非法或 schema 主版本不兼容。
        """
        path = Path(path)
        if not path.exists():
            raise ProjectFileError(f"工程文件不存在: {path}")
        try:
            h5 = h5py.File(path, mode)
        except OSError as exc:
            raise ProjectFileError(f"工程文件无法打开（非 HDF5 或已损坏）: {path}") from exc
        try:
            version = str(h5["meta"].attrs["schema_version"])
        except KeyError as exc:
            h5.close()
            raise ProjectFileError(f"工程文件缺少 meta.schema_version: {path}") from exc
        if version.split(".", maxsplit=1)[0] != PROJECT_SCHEMA_VERSION.split(".", maxsplit=1)[0]:
            h5.close()
            raise ProjectFileError(f"工程文件 schema 版本不兼容: 文件 {version}，当前支持 {PROJECT_SCHEMA_VERSION}")
        return cls(path, h5)

    def write_array(self, group: str, name: str, data: Any, *, attrs: dict[str, Any] | None = None) -> None:
        """写入数组数据集（同名覆盖）；``attrs`` 附加为该数据集的属性."""
        grp = self._require_group(group)
        if name in grp:
            del grp[name]
        try:
            dataset = grp.create_dataset(name, data=np.asarray(data))
        except (TypeError, ValueError) as exc:
            raise ProjectFileError(f"数组写入失败: {group}/{name}") from exc
        for key, value in (attrs or {}).items():
            dataset.attrs[key] = value
        logger.debug("数组已写入: %s/%s shape=%s", group, name, dataset.shape)

    def read_array(self, group: str, name: str) -> np.ndarray:
        """读取数组数据集.

        :raises ProjectFileError: 数据集不存在。
        """
        dataset = self._require_dataset(group, name)
        return np.asarray(dataset)

    def write_json(self, group: str, name: str, obj: Any) -> None:
        """将对象 JSON 序列化后写入文本数据集（同名覆盖）."""
        grp = self._require_group(group)
        if name in grp:
            del grp[name]
        payload = json.dumps(obj, ensure_ascii=False)
        grp.create_dataset(name, data=payload)

    def read_json(self, group: str, name: str) -> Any:
        """读取文本数据集并 JSON 反序列化.

        :raises ProjectFileError: 数据集不存在或 JSON 解析失败。
        """
        dataset = self._require_dataset(group, name)
        raw = dataset[()]
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectFileError(f"JSON 数据集解析失败: {group}/{name}") from exc

    def list_names(self, group: str) -> list[str]:
        """列出组内成员名称（排序）."""
        grp = self._require_group(group)
        return sorted(grp.keys())

    def close(self) -> None:
        """关闭文件（幂等）."""
        if self._h5:
            self._h5.close()
            logger.debug("工程文件已关闭: %s", self._path)

    def __enter__(self) -> Project:
        """进入上下文."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """退出上下文并关闭文件."""
        self.close()

    def _require_group(self, group: str) -> h5py.Group:
        """获取组（不存在则按需创建，仅读写模式下）."""
        try:
            return self._h5.require_group(group)
        except TypeError as exc:
            raise ProjectFileError(f"组名与已有数据集冲突: {group!r}") from exc

    def _require_dataset(self, group: str, name: str) -> h5py.Dataset:
        """获取数据集，不存在抛 :class:`ProjectFileError`."""
        if group not in self._h5 or name not in self._h5[group]:
            raise ProjectFileError(f"数据集不存在: {group}/{name}")
        node = self._h5[group][name]
        if not isinstance(node, h5py.Dataset):
            raise ProjectFileError(f"目标不是数据集: {group}/{name}")
        return node
