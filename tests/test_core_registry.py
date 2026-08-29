"""core.registry 插件注册表测试."""

from __future__ import annotations

import pytest

from zylab.core.errors import PluginError, PluginNotFoundError
from zylab.core.registry import PluginKind, PluginRegistry, PluginSpec


def _make_spec(name: str, kind: PluginKind = PluginKind.SOLVER, factory: str = "tests._targets:add") -> PluginSpec:
    return PluginSpec(name=name, kind=kind, factory=factory)


def test_register_and_get() -> None:
    """注册后应能通过名称获取描述."""
    reg = PluginRegistry()
    spec = _make_spec("my_solver")
    reg.register(spec)
    assert reg.get("my_solver") == spec


def test_register_duplicate_rejected() -> None:
    """同名不覆盖时默认拒绝."""
    reg = PluginRegistry()
    reg.register(_make_spec("dup"))
    with pytest.raises(PluginError, match="插件重名"):
        reg.register(_make_spec("dup"))


def test_register_replace() -> None:
    """replace=True 时应允许覆盖."""
    reg = PluginRegistry()
    reg.register(_make_spec("replace_me"))
    new = _make_spec("replace_me", factory="tests._targets:echo_report")
    reg.register(new, replace=True)
    assert reg.get("replace_me").factory == "tests._targets:echo_report"


def test_unregister() -> None:
    """注销后应无法获取."""
    reg = PluginRegistry()
    reg.register(_make_spec("gone"))
    reg.unregister("gone")
    with pytest.raises(PluginNotFoundError):
        reg.get("gone")


def test_unregister_not_found() -> None:
    """注销未注册插件应抛 PluginNotFoundError."""
    reg = PluginRegistry()
    with pytest.raises(PluginNotFoundError):
        reg.unregister("nonexistent")


def test_list_all() -> None:
    """list 应返回按名称排序的全部插件."""
    reg = PluginRegistry()
    reg.register(_make_spec("b_solver"))
    reg.register(_make_spec("a_element", kind=PluginKind.ELEMENT))
    all_plugins = reg.list()
    assert [p.name for p in all_plugins] == ["a_element", "b_solver"]


def test_list_by_kind() -> None:
    """按 kind 过滤应仅返回目标类别."""
    reg = PluginRegistry()
    reg.register(_make_spec("s", kind=PluginKind.SOLVER))
    reg.register(_make_spec("e", kind=PluginKind.ELEMENT))
    solvers = reg.list(kind=PluginKind.SOLVER)
    assert len(solvers) == 1 and solvers[0].name == "s"


def test_resolve_builtin() -> None:
    """解析已注册工厂应返回可调用对象."""
    reg = PluginRegistry()
    reg.register(_make_spec("adder", factory="tests._targets:add"))
    func = reg.resolve("adder")
    assert callable(func)
    assert func(2, 3) == 5


def test_resolve_caches() -> None:
    """多次解析同一工厂应返回缓存实例."""
    reg = PluginRegistry()
    reg.register(_make_spec("adder", factory="tests._targets:add"))
    first = reg.resolve("adder")
    second = reg.resolve("adder")
    assert first is second


def test_resolve_not_found() -> None:
    """解析未注册插件应抛 PluginNotFoundError."""
    reg = PluginRegistry()
    with pytest.raises(PluginNotFoundError):
        reg.resolve("ghost")


def test_resolve_bad_factory() -> None:
    """工厂格式非法或模块/属性不存在应抛 PluginError."""
    reg = PluginRegistry()
    reg.register(PluginSpec(name="bad", kind=PluginKind.SOLVER, factory="nosuchmod:func"))
    with pytest.raises(PluginError, match="工厂解析失败"):
        reg.resolve("bad")


def test_resolve_bad_factory_format() -> None:
    """工厂缺少冒号分隔应抛 PluginError."""
    reg = PluginRegistry()
    reg.register(PluginSpec(name="no_colon", kind=PluginKind.SOLVER, factory="noattr"))
    with pytest.raises(PluginError, match="工厂解析失败"):
        reg.resolve("no_colon")


def test_load_entry_points(monkeypatch) -> None:
    """entry points 发现应注册到对应类别."""
    from importlib.metadata import EntryPoint

    from zylab.core import registry as registry_mod

    fake_eps = [EntryPoint("ext_add", "tests._targets:add", "zylab.solver")]

    def fake_iter(group: str):
        return iter(fake_eps) if group == "zylab.solver" else iter(())

    monkeypatch.setattr(registry_mod, "_iter_group_entry_points", fake_iter)
    reg = PluginRegistry()
    count = reg.load_entry_points()
    assert count == 1
    spec = reg.get("ext_add")
    assert spec.kind is PluginKind.SOLVER
    assert spec.factory == "tests._targets:add"


def test_load_entry_points_conflict_skipped(monkeypatch) -> None:
    """与已注册插件同名的外部插件应跳过且不重复计数."""
    from importlib.metadata import EntryPoint

    from zylab.core import registry as registry_mod

    fake_eps = [EntryPoint("ext_add", "tests._targets:add", "zylab.solver")]
    monkeypatch.setattr(
        registry_mod, "_iter_group_entry_points", lambda group: iter(fake_eps) if group == "zylab.solver" else iter(())
    )
    reg = PluginRegistry()
    reg.register(_make_spec("ext_add"))  # 预注册同名
    count = reg.load_entry_points()
    assert count == 0


def test_iter_group_entry_points_real() -> None:
    """真实 entry_points 查询（现代 select 接口）；未安装的 group 返回空."""
    from zylab.core.registry import _iter_group_entry_points

    assert list(_iter_group_entry_points("zylab.no_such_kind")) == []
