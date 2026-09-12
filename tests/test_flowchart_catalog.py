"""模块分类目录 catalog.py 测试."""

from __future__ import annotations

from zylab.flowchart.catalog import (
    CatalogNode,
    catalog_path_of,
    catalog_tree,
    list_by_path,
    subcategories_of,
)
from zylab.flowchart.module import ModuleCategory, ModuleSpec, PortSpec, PortType


def _spec(type_id: str, category: ModuleCategory = ModuleCategory.SOURCE, catalog: tuple[str, ...] = ()) -> ModuleSpec:
    """快速构造 ModuleSpec."""
    return ModuleSpec(
        type_id=type_id,
        name=type_id,
        category=category,
        target="zylab.flowchart.split_nodes:material_linear_elastic",
        inputs=(PortSpec("model", PortType.MODEL),) if category != ModuleCategory.SOURCE else (),
        outputs=(PortSpec("model", PortType.MODEL),) if category != ModuleCategory.POST else (),
        catalog=catalog if catalog else (),
    )


class TestCatalogPathOf:
    """catalog_path_of 路径推断."""

    def test_explicit_catalog_override(self) -> None:
        """显式 catalog 优先于前缀推断."""
        spec = _spec("material.linear_elastic", catalog=("自定义", "特殊子类"))
        assert catalog_path_of(spec) == ("自定义", "特殊子类")

    def test_material_prefix(self) -> None:
        """material.* → 材料参数/弹性材料."""
        spec = _spec("material.linear_elastic")
        assert catalog_path_of(spec) == ("材料参数", "弹性材料")

    def test_conduction_prefix(self) -> None:
        """conduction.* → 材料参数/导热/导电材料."""
        spec = _spec("conduction.isotropic")
        assert catalog_path_of(spec) == ("材料参数", "导热/导电材料")

    def test_geom_cantilever(self) -> None:
        """geom.cantilever.* → 参数化建模/梁系几何."""
        spec = _spec("geom.cantilever_2d")
        assert catalog_path_of(spec) == ("参数化建模", "梁系几何")

    def test_geom_truss(self) -> None:
        """geom.truss.* → 参数化建模/梁系几何."""
        spec = _spec("geom.truss_3d")
        assert catalog_path_of(spec) == ("参数化建模", "梁系几何")

    def test_geom_joule(self) -> None:
        """geom.joule.* → 参数化建模/热-电几何."""
        spec = _spec("geom.joule_hole")
        assert catalog_path_of(spec) == ("参数化建模", "热-电几何")

    def test_geom_fallback(self) -> None:
        """其他 geom.* → 参数化建模/其他几何."""
        spec = _spec("geom.weird_shape")
        assert catalog_path_of(spec) == ("参数化建模", "其他几何")

    def test_mesh_prefix(self) -> None:
        """mesh.* → 有限元网格划分/通用网格划分."""
        spec = _spec("mesh.generate_structural")
        assert catalog_path_of(spec) == ("有限元网格划分", "通用网格划分")

    def test_analysis_static(self) -> None:
        """analysis.static* → 求解器/结构静力."""
        spec = _spec("analysis.static")
        assert catalog_path_of(spec) == ("求解器", "结构静力")

    def test_analysis_modal(self) -> None:
        """analysis.modal* → 求解器/结构模态."""
        spec = _spec("analysis.modal")
        assert catalog_path_of(spec) == ("求解器", "结构模态")

    def test_analysis_buckling(self) -> None:
        """analysis.buckling → 求解器/结构屈曲."""
        spec = _spec("analysis.buckling")
        assert catalog_path_of(spec) == ("求解器", "结构屈曲")

    def test_analysis_electrothermal(self) -> None:
        """analysis.electrothermal → 求解器/稳态电-热耦合."""
        spec = _spec("analysis.electrothermal")
        assert catalog_path_of(spec) == ("求解器", "稳态电-热耦合")

    def test_post_prefix(self) -> None:
        """post.* → 后处理/结果提取."""
        spec = _spec("post.extract_max_stress", category=ModuleCategory.POST)
        assert catalog_path_of(spec) == ("后处理", "结果提取")

    def test_compute_expr(self) -> None:
        """compute.expr → 后处理/计算工具."""
        spec = _spec("compute.expr", category=ModuleCategory.POST)
        assert catalog_path_of(spec) == ("后处理", "计算工具")

    def test_compute_lsc(self) -> None:
        """compute.lsc → 后处理/曲线优化."""
        spec = _spec("compute.lsc", category=ModuleCategory.POST)
        assert catalog_path_of(spec) == ("后处理", "曲线优化")

    def test_reliability_prefix(self) -> None:
        """reliability.* → 后处理/可靠性/感度试验."""
        spec = _spec("reliability.form", category=ModuleCategory.POST)
        assert catalog_path_of(spec) == ("后处理", "可靠性/感度试验")

    def test_example_prefix(self) -> None:
        """example.* → 遗留一体化源."""
        spec = _spec("example.simple_beam")
        assert catalog_path_of(spec) == ("遗留一体化", "几何+网格+材料+载荷一体化源")

    def test_unknown_analysis_category(self) -> None:
        """未在规则表的 analysis.* → 求解器/动态求解器."""
        spec = _spec("analysis.weird", category=ModuleCategory.ANALYSIS)
        assert catalog_path_of(spec) == ("求解器", "动态求解器")

    def test_unknown_post_category(self) -> None:
        """未在规则表的 POST 模块 → 后处理/后处理工具."""
        spec = _spec("custom.something", category=ModuleCategory.POST)
        assert catalog_path_of(spec) == ("后处理", "后处理工具")

    def test_unknown_fallback_to_legacy(self) -> None:
        """完全未匹配的 → 遗留一体化/未分类."""
        spec = _spec("mystery.module")
        assert catalog_path_of(spec) == ("遗留一体化", "未分类")


class TestCatalogNode:
    """CatalogNode 数据类."""

    def test_is_leaf_no_children(self) -> None:
        node = CatalogNode(label="测试")
        assert node.is_leaf()

    def test_is_leaf_with_children(self) -> None:
        node = CatalogNode(label="测试", children=[CatalogNode(label="子")])
        assert not node.is_leaf()


class TestCatalogTree:
    """catalog_tree 构建分类树."""

    def test_empty_specs(self) -> None:
        """空 specs 列表 → 全空树."""
        tree = catalog_tree([])
        assert tree == []

    def test_basic_grouping(self) -> None:
        """三个不同前缀的模块正确归入不同大类."""
        specs = [
            _spec("material.x"),
            _spec("geom.y"),
            _spec("analysis.z", category=ModuleCategory.ANALYSIS),
        ]
        tree = catalog_tree(specs)
        top_labels = [n.label for n in tree]
        assert "材料参数" in top_labels
        assert "参数化建模" in top_labels
        assert "求解器" in top_labels

    def test_filter_empty_categories(self) -> None:
        """没有模块的大类不出现在结果中."""
        specs = [_spec("material.x")]
        tree = catalog_tree(specs)
        top_labels = [n.label for n in tree]
        assert "有限元网格划分" not in top_labels

    def test_subcategories_created(self) -> None:
        """同一大类下的模块按子类分组."""
        specs = [
            _spec("material.linear_elastic"),
            _spec("material.plasticity"),
            _spec("conduction.isotropic"),
        ]
        tree = catalog_tree(specs)
        material_node = next(n for n in tree if n.label == "材料参数")
        sub_labels = [c.label for c in material_node.children]
        assert "弹性材料" in sub_labels
        assert "导热/导电材料" in sub_labels


class TestListByPath:
    """list_by_path 按路径列出模块."""

    def test_existing_path(self) -> None:
        specs = [_spec("material.a"), _spec("material.b"), _spec("geom.c")]
        result = list_by_path(("材料参数", "弹性材料"), specs)
        assert len(result) == 2
        assert all(s.type_id.startswith("material.") for s in result)

    def test_nonexistent_path(self) -> None:
        specs = [_spec("material.a")]
        result = list_by_path(("不存在的大类", "子类"), specs)
        assert result == []


class TestSubcategoriesOf:
    """subcategories_of 取某大类下的子类名."""

    def test_existing_category(self) -> None:
        specs = [_spec("material.a"), _spec("conduction.b")]
        subs = subcategories_of("材料参数", specs)
        assert "弹性材料" in subs
        assert "导热/导电材料" in subs

    def test_nonexistent_category(self) -> None:
        subs = subcategories_of("不存在", [])
        assert subs == []
