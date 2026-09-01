"""studio.registry 模板注册表测试：注册/查询/目录加载/插件 entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zylab.core.registry import PluginKind, PluginRegistry, PluginSpec
from zylab.studio import BUILTIN_TEMPLATES, Template, TemplateError, TemplateNotFoundError, TemplateRegistry

__all__ = []


class TestRegisterQuery:
    """注册与查询."""

    def test_with_builtin(self) -> None:
        """内置注册表含全部内置模板."""
        registry = TemplateRegistry.with_builtin()
        assert len(registry.list()) == len(BUILTIN_TEMPLATES)
        assert registry.get("structural.cantilever_static").name == "悬臂梁静力分析"

    def test_disciplines(self) -> None:
        """学科分组标识（math 为 DSL 数学模板学科，structural/thermal 为既有学科）."""
        registry = TemplateRegistry.with_builtin()
        assert registry.disciplines() == ("math", "structural", "thermal")

    def test_list_filter_by_discipline(self) -> None:
        """按学科过滤."""
        registry = TemplateRegistry.with_builtin()
        assert registry.list(discipline="structural")
        assert registry.list(discipline="thermal")
        assert registry.list(discipline="fluid") == []

    def test_register_duplicate_rejected(self) -> None:
        """同 id 重复注册拒绝；replace=True 覆盖."""
        registry = TemplateRegistry()
        template = BUILTIN_TEMPLATES[0]
        registry.register(template)
        with pytest.raises(TemplateError, match="重复注册"):
            registry.register(template)
        renamed = Template(id=template.id, name="改名", nodes=template.nodes)
        registry.register(renamed, replace=True)
        assert registry.get(template.id).name == "改名"

    def test_unregister(self) -> None:
        """注销与未注册查询."""
        registry = TemplateRegistry.with_builtin()
        registry.unregister("structural.cantilever_static")
        with pytest.raises(TemplateNotFoundError, match="未注册"):
            registry.get("structural.cantilever_static")
        with pytest.raises(TemplateNotFoundError, match="未注册"):
            registry.unregister("structural.cantilever_static")


class TestLoadDir:
    """用户模板目录加载."""

    def _write(self, directory: Path, name: str, data: dict) -> None:
        """写模板 JSON 文件."""
        (directory / name).write_text(json.dumps(data), encoding="utf-8")

    def test_load_dir_counts_valid_only(self, tmp_path: Path) -> None:
        """目录加载：合法文件注册，非法文件跳过（数量只计成功）."""
        valid = {
            "id": "user.custom",
            "name": "用户模板",
            "nodes": [
                {"id": "model", "type": "example.truss2_two_bar"},
                {"id": "solve", "type": "analysis.static", "inputs": {"model": "model.model"}},
            ],
        }
        self._write(tmp_path, "good.json", valid)
        self._write(tmp_path, "bad.json", {"id": "x"})
        (tmp_path / "notes.txt").write_text("非 json 忽略", encoding="utf-8")
        registry = TemplateRegistry()
        assert registry.load_dir(tmp_path) == 1
        assert registry.get("user.custom").name == "用户模板"

    def test_load_dir_recursive_discipline_subdirs(self, tmp_path: Path) -> None:
        """按学科子目录归类与顶层平铺布局并存时全部加载."""
        flat = {
            "id": "user.legacy",
            "name": "平铺模板",
            "nodes": [{"id": "model", "type": "example.truss2_two_bar"}],
        }
        nested = {
            "id": "user.nested",
            "name": "归类模板",
            "discipline": "thermal",
            "nodes": [{"id": "model", "type": "example.joule_plate_2d"}],
        }
        self._write(tmp_path, "legacy.json", flat)
        (tmp_path / "thermal").mkdir()
        self._write(tmp_path / "thermal", "nested.json", nested)
        registry = TemplateRegistry()
        assert registry.load_dir(tmp_path) == 2
        assert registry.get("user.legacy").name == "平铺模板"
        assert registry.get("user.nested").discipline == "thermal"

    def test_builtin_templates_loaded_from_assets(self) -> None:
        """内置模板从 assets/templates/<学科>/*.json|*.yaml 加载且与包内资产一致."""
        from zylab.studio.builtin import builtin_templates_dir
        from zylab.studio.dsl import DslTemplate

        assets = builtin_templates_dir()
        files = sorted(assets.glob("*/*.json")) + sorted(assets.glob("*/*.yaml"))
        assert len(files) == len(BUILTIN_TEMPLATES)
        by_id = {t.id: t for t in BUILTIN_TEMPLATES}
        for path in files:
            template = by_id[path.stem]  # 文件名即模板 id
            assert template.discipline == path.parent.name
            assert path.parent == assets / template.discipline
        # DSL 模板与经典模板同池注册
        assert isinstance(by_id["dsl.cantilever_sweep"], DslTemplate)
        assert isinstance(by_id["dsl.math_compare"], DslTemplate)

    def test_load_dir_missing_directory(self, tmp_path: Path) -> None:
        """目录不存在返回 0 且不抛异常."""
        registry = TemplateRegistry()
        assert registry.load_dir(tmp_path / "nope") == 0


class TestLoadEntryPoints:
    """插件模板 entry point 发现."""

    def _plugins_with(self, *specs: PluginSpec) -> PluginRegistry:
        """构造含给定插件的注册表."""
        plugins = PluginRegistry()
        for spec in specs:
            plugins.register(spec)
        return plugins

    def test_dict_factory_registered(self) -> None:
        """字典工厂解析为模板并注册."""
        plugins = self._plugins_with(
            PluginSpec(name="sample", kind=PluginKind.TEMPLATE, factory="tests._targets:SAMPLE_TEMPLATE")
        )
        registry = TemplateRegistry()
        assert registry.load_entry_points(plugins) == 1
        assert registry.get("plugin.sample").name == "插件示例模板"

    def test_template_instance_factory(self) -> None:
        """Template 实例工厂直接注册."""
        template = BUILTIN_TEMPLATES[0]
        registry = TemplateRegistry()

        class _FakeResolver:
            """替身：resolve 返回 Template 实例."""

            def list(self, kind=None):
                """返回单个模板插件描述."""
                return [PluginSpec(name="t", kind=PluginKind.TEMPLATE, factory="x:y")]

            def resolve(self, name):
                """返回内置模板实例."""
                return template

        assert registry.load_entry_points(_FakeResolver()) == 1
        assert registry.get(template.id).name == template.name

    def test_bad_factory_skipped(self) -> None:
        """非模板结果（元组）跳过不阻断."""
        plugins = self._plugins_with(
            PluginSpec(name="bad", kind=PluginKind.TEMPLATE, factory="zylab.studio:BUILTIN_TEMPLATES"),
            PluginSpec(name="good", kind=PluginKind.TEMPLATE, factory="tests._targets:SAMPLE_TEMPLATE"),
        )
        registry = TemplateRegistry()
        assert registry.load_entry_points(plugins) == 1
        assert registry.get("plugin.sample").name == "插件示例模板"
