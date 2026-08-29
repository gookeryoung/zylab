"""zylab 插件注册表.

两类来源：
- 手工注册：内置单元/材料/求解器随包注册；
- entry points：第三方包在 ``pyproject.toml`` 声明 ``[project.entry-points."zylab.<kind>"]`` 自动发现。

``factory`` 统一为 ``"module:attr"`` 全限定字符串，懒解析（首次 :meth:`PluginRegistry.resolve` 时导入），
避免插件导入拖慢启动，也便于进程隔离场景按需加载。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import threading
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Iterator

from .errors import PluginError, PluginNotFoundError

__all__ = ["ENTRY_POINT_PREFIX", "PluginKind", "PluginRegistry", "PluginSpec"]

logger = logging.getLogger(__name__)

ENTRY_POINT_PREFIX = "zylab"


@unique
class PluginKind(Enum):
    """插件类别（entry point group 为 ``zylab.<value>``）."""

    ELEMENT = "element"
    MATERIAL = "material"
    SOLVER = "solver"
    POSTPROCESS = "postprocess"
    FILE_FORMAT = "file_format"
    TEMPLATE = "template"  # 分析模板（studio.Template 或其字典表示）


@dataclass(frozen=True)
class PluginSpec:
    """插件描述.

    :param name: 唯一名称（同一注册表内不可重复）。
    :param kind: 插件类别。
    :param factory: 工厂全限定名 ``"module:attr"``，惰性解析。
    :param version: 版本字符串，外部插件取自发行包元数据。
    """

    name: str
    kind: PluginKind
    factory: str
    version: str = ""


class PluginRegistry:
    """线程安全的插件注册表."""

    def __init__(self) -> None:
        """初始化空注册表."""
        self._specs: dict[str, PluginSpec] = {}
        self._resolved: dict[str, Any] = {}
        self._lock = threading.RLock()

    def register(self, spec: PluginSpec, *, replace: bool = False) -> None:
        """注册插件；同名冲突默认抛 :class:`PluginError`，``replace=True`` 时覆盖."""
        with self._lock:
            if spec.name in self._specs and not replace:
                raise PluginError(f"插件重名: {spec.name!r}（已注册 {self._specs[spec.name].factory!r}）")
            self._specs[spec.name] = spec
            self._resolved.pop(spec.name, None)
        logger.debug("注册插件: %s kind=%s factory=%s", spec.name, spec.kind.value, spec.factory)

    def unregister(self, name: str) -> None:
        """注销插件；未注册抛 :class:`PluginNotFoundError`."""
        with self._lock:
            try:
                del self._specs[name]
            except KeyError:
                raise PluginNotFoundError(f"插件未注册: {name!r}") from None
            self._resolved.pop(name, None)

    def get(self, name: str) -> PluginSpec:
        """按名称取插件描述；未注册抛 :class:`PluginNotFoundError`."""
        with self._lock:
            try:
                return self._specs[name]
            except KeyError:
                raise PluginNotFoundError(f"插件未注册: {name!r}") from None

    def list(self, kind: PluginKind | None = None) -> list[PluginSpec]:
        """列出插件（按名称排序）；``kind`` 非空时按类别过滤."""
        with self._lock:
            specs = list(self._specs.values())
        if kind is not None:
            specs = [s for s in specs if s.kind is kind]
        return sorted(specs, key=lambda s: s.name)  # type: ignore[implicit-any-lambda]

    def resolve(self, name: str) -> Any:
        """解析插件工厂（``"module:attr"`` → 对象），结果缓存；失败抛 :class:`PluginError`."""
        with self._lock:
            if name in self._resolved:
                return self._resolved[name]
            spec = self.get(name)
        try:
            module_name, _, attr = spec.factory.partition(":")
            if not module_name or not attr:
                raise ValueError(f"工厂格式应为 'module:attr'，得到 {spec.factory!r}")
            obj = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError, ValueError) as exc:
            raise PluginError(f"插件工厂解析失败: {name!r} -> {spec.factory!r}: {exc}") from exc
        with self._lock:
            self._resolved[name] = obj
        return obj

    def load_entry_points(self, *, replace: bool = False) -> int:
        """发现并注册所有已安装包声明的 ``zylab.<kind>`` entry points，返回注册数量.

        单个 entry point 加载失败仅记录日志并跳过，不影响其余插件。
        """
        count = 0
        for kind in PluginKind:
            for ep in _iter_group_entry_points(f"{ENTRY_POINT_PREFIX}.{kind.value}"):
                dist = getattr(ep, "dist", None)
                version = getattr(dist, "version", "") if dist is not None else ""
                spec = PluginSpec(name=ep.name, kind=kind, factory=ep.value, version=version)
                try:
                    self.register(spec, replace=replace)
                except PluginError:
                    logger.warning("跳过冲突的外部插件: %s（group=%s）", ep.name, kind.value)
                    continue
                count += 1
        return count


def _iter_group_entry_points(group: str) -> Iterator[importlib.metadata.EntryPoint]:
    """按 group 迭代 entry points（兼容 3.8/3.9 的 dict 接口与 3.10+ 的 select 接口）."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        yield from eps.select(group=group)
    else:  # pragma: no cover（3.10+ 环境不覆盖旧接口分支）
        yield from eps.get(group, ())  # type: ignore[union-attr]
