"""模板注册表：内置模板随包注册，用户模板从 ``data_dir/templates/*.json`` 目录加载.

第三方包可通过 entry point group ``zylab.template`` 注册插件模板（工厂解析为
Template 实例或模板字典），经 :meth:`TemplateRegistry.load_entry_points` 发现。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from zylab.core.registry import PluginKind, PluginRegistry

from .builtin import BUILTIN_TEMPLATES
from .errors import TemplateError, TemplateNotFoundError
from .template import Template, load_template

__all__ = ["TemplateRegistry"]

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """模板注册表（按 id 索引，学科分组检索）."""

    def __init__(self, templates: tuple[Template, ...] = ()) -> None:
        """初始化注册表并登记给定模板."""
        self._templates: dict[str, Template] = {}
        for template in templates:
            self.register(template)

    @classmethod
    def with_builtin(cls) -> TemplateRegistry:
        """构造含全部内置模板的注册表."""
        return cls(BUILTIN_TEMPLATES)

    def register(self, template: Template, *, replace: bool = False) -> None:
        """注册模板；同 id 冲突默认抛 :class:`TemplateError`，``replace=True`` 时覆盖."""
        if template.id in self._templates and not replace:
            raise TemplateError(f"模板 id 重复注册: {template.id!r}")
        self._templates[template.id] = template
        logger.debug("注册模板: %s (%s)", template.id, template.name)

    def unregister(self, template_id: str) -> None:
        """注销模板；未注册抛 :class:`TemplateNotFoundError`."""
        try:
            del self._templates[template_id]
        except KeyError:
            raise TemplateNotFoundError(f"模板未注册: {template_id!r}") from None

    def get(self, template_id: str) -> Template:
        """按 id 取模板；未注册抛 :class:`TemplateNotFoundError`."""
        try:
            return self._templates[template_id]
        except KeyError:
            raise TemplateNotFoundError(f"模板未注册: {template_id!r}") from None

    def list(self, discipline: str | None = None) -> list[Template]:
        """列出模板（按名称排序）；``discipline`` 非空时按学科过滤."""
        templates = list(self._templates.values())
        if discipline is not None:
            templates = [t for t in templates if t.discipline == discipline]
        return sorted(templates, key=lambda t: t.name)  # type: ignore[implicit-any-lambda]

    def disciplines(self) -> tuple[str, ...]:
        """全部已注册模板的学科标识（出现序去重）."""
        seen: dict[str, None] = {}
        for template in self._templates.values():
            seen.setdefault(template.discipline)
        return tuple(seen)

    def load_dir(self, directory: Path) -> int:
        """加载目录下全部模板（JSON 经典格式 + YAML DSL 格式），返回成功数量.

        顶层为历史平铺布局，子目录为按学科归类的现行布局；二者并存时全部加载。
        ``*.yaml``/``*.yml`` 走 DSL 解析（归一化为 :class:`DslTemplate`，
        与经典模板同池注册）；单个文件非法仅记录告警并跳过。
        """
        directory = Path(directory)
        if not directory.is_dir():
            logger.warning("模板目录不存在，跳过加载: %s", directory)
            return 0
        from .dsl import load_dsl

        count = 0
        paths = sorted({*directory.glob("*.json"), *directory.glob("*/*.json")})
        paths += sorted(
            {
                *directory.glob("*.yaml"),
                *directory.glob("*.yml"),
                *directory.glob("*/*.yaml"),
                *directory.glob("*/*.yml"),
            }
        )
        for path in paths:
            try:
                template = load_dsl(path) if path.suffix.lower() in (".yaml", ".yml") else load_template(path)
                self.register(template)
            except TemplateError as exc:
                logger.warning("跳过非法模板文件 %s: %s", path.name, exc)
                continue
            count += 1
        return count

    def load_entry_points(self, plugins: PluginRegistry | None = None) -> int:
        """发现并注册 entry point group ``zylab.template`` 声明的插件模板，返回注册数量.

        工厂解析结果可为 :class:`Template` 实例或模板字典（经 from_dict 校验）；
        单个插件失败仅记录告警并跳过。

        :param plugins: 插件注册表（缺省时新建并完成 entry point 发现）。
        """
        if plugins is None:
            plugins = PluginRegistry()
            plugins.load_entry_points()
        count = 0
        for spec in plugins.list(kind=PluginKind.TEMPLATE):
            try:
                resolved = plugins.resolve(spec.name)
                template = (
                    resolved
                    if isinstance(resolved, Template)
                    else Template.from_dict(_expect_mapping(resolved, spec.name))
                )
                self.register(template)
            except Exception as exc:  # 第三方插件故障不阻断主流程
                logger.warning("跳过非法插件模板 %s: %s", spec.name, exc)
                continue
            count += 1
        return count


def _expect_mapping(obj: object, name: str) -> Mapping[str, Any]:
    """要求插件解析结果为字典."""
    if not isinstance(obj, Mapping):
        raise TemplateError(f"插件模板 {name!r} 工厂应解析为 Template 或字典，得到 {type(obj).__name__}")
    return obj
