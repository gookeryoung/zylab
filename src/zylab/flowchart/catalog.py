"""模块分类目录：五大类分类树 + type_id → 路径自动推断 + GUI 工具箱数据源.

ANSYS Workbench 风格的一级分类 + 二级子类树，顶层节点：

- 材料参数 (MATERIAL)
- 参数化建模 (GEOMETRY)
- 有限元网格划分 (MESH)
- 求解器 (SOLVER)
- 后处理 (POST)
- 遗留一体化 (LEGACY)  — 旧 ``example.*`` 一体化源节点

ModuleSpec 新增可选 ``catalog`` 字段（空时按 type_id 前缀自动推断），
GUI 工具箱直接消费 :func:`catalog_tree` 返回的树。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .module import ModuleCategory, ModuleSpec, all_modules

__all__ = [
    "CATEGORY_LABELS",
    "CatalogNode",
    "catalog_path_of",
    "catalog_tree",
    "list_by_path",
    "subcategories_of",
]


#: 顶层大类中文标签与英文 key 对应（保持稳定字符串 key 便于持久化）
CATEGORY_LABELS: dict[str, str] = {
    "material": "材料参数",
    "geometry": "参数化建模",
    "mesh": "有限元网格划分",
    "solver": "求解器",
    "post": "后处理",
    "legacy": "遗留一体化",
}


#: type_id 前缀 → (顶层大类 key, 子类名) 的推断规则表（越具体的前缀越前）
_PREFIX_RULES: list[tuple[str, tuple[str, str]]] = [
    # ---- 材料 ----
    ("material.", ("material", "弹性材料")),
    ("conduction.", ("material", "导热/导电材料")),
    # ---- 几何 ----
    ("geom.cantilever", ("geometry", "梁系几何")),
    ("geom.column", ("geometry", "梁系几何")),
    ("geom.truss", ("geometry", "梁系几何")),
    ("geom.plate", ("geometry", "连续体几何")),
    ("geom.cylinder", ("geometry", "连续体几何")),
    ("geom.joule", ("geometry", "热-电几何")),
    ("geom.vfilm", ("geometry", "热-电几何")),
    ("geom.", ("geometry", "其他几何")),
    # ---- 网格 ----
    ("mesh.", ("mesh", "通用网格划分")),
    # ---- 求解器 ----
    ("analysis.static", ("solver", "结构静力")),
    ("analysis.modal", ("solver", "结构模态")),
    ("analysis.harmonic", ("solver", "结构谐响应")),
    ("analysis.transient", ("solver", "结构瞬态")),
    ("analysis.buckling", ("solver", "结构屈曲")),
    ("analysis.nonlinear", ("solver", "结构非线性")),
    ("analysis.electrothermal_transient", ("solver", "瞬态电-热耦合")),
    ("analysis.electrothermal", ("solver", "稳态电-热耦合")),
    # ---- 后处理 ----
    ("post.", ("post", "结果提取")),
    ("compute.expr", ("post", "计算工具")),
    ("compute.sweep", ("post", "计算工具")),
    ("compute.lsc", ("post", "曲线优化")),
    ("reliability.", ("post", "可靠性/感度试验")),
    # ---- 遗留一体化 ----
    ("example.", ("legacy", "几何+网格+材料+载荷一体化源")),
]


def catalog_path_of(spec: ModuleSpec) -> tuple[str, ...]:
    """返回模块在分类树中的完整路径 ``(大类标签, 子类名)``.

    优先取 ``spec.catalog`` 显式声明；空时按 type_id 前缀匹配推断表；
    均不命中则归入 ``("遗留一体化", "未分类")``。
    """
    if getattr(spec, "catalog", ()):
        return tuple(spec.catalog)
    for prefix, path in _PREFIX_RULES:
        if spec.type_id.startswith(prefix):
            return (CATEGORY_LABELS[path[0]], path[1])
    # analysis.xxx 这类可能是 solver_adapter 动态生成的、未在规则表中的
    if spec.category is ModuleCategory.ANALYSIS:
        return (CATEGORY_LABELS["solver"], "动态求解器")
    if spec.category is ModuleCategory.POST:
        return (CATEGORY_LABELS["post"], "后处理工具")
    return (CATEGORY_LABELS["legacy"], "未分类")


# ------------------------------------------------------------------ 分类树结构


@dataclass
class CatalogNode:
    """分类树节点（GUI 工具箱渲染用）.

    :param label: 中文显示名。
    :param children: 子节点列表（子类或模块）。
    :param modules: 直接挂在本节点下的模块规格（叶子节点）。
    """

    label: str
    children: list[CatalogNode] = field(default_factory=list)
    modules: list[ModuleSpec] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """是否叶子节点（仅含模块，无子类）."""
        return not self.children


def catalog_tree(specs: Iterable[ModuleSpec] | None = None) -> list[CatalogNode]:
    """按五大类构建完整分类树.

    :param specs: 要纳入树的模块规格；None 时取 :func:`all_modules`。
    :returns: 顶层大类节点列表（有序：material → geometry → mesh → solver → post → legacy）。
    """
    if specs is None:
        specs = all_modules()

    # 初始化顶层节点（保持稳定排序）
    top_order = ["material", "geometry", "mesh", "solver", "post", "legacy"]
    top_labels = [CATEGORY_LABELS[k] for k in top_order]
    tree: dict[str, CatalogNode] = {label: CatalogNode(label=label) for label in top_labels}

    for spec in specs:
        path = catalog_path_of(spec)
        if len(path) == 1:
            top = tree[path[0]]
            top.modules.append(spec)
            continue
        top_label, sub_label = path[0], path[1]
        top = tree[top_label]
        # 查找或创建子节点
        sub = next((c for c in top.children if c.label == sub_label), None)
        if sub is None:
            sub = CatalogNode(label=sub_label)
            top.children.append(sub)
        sub.modules.append(spec)

    # 过滤空的大类节点
    result = [tree[label] for label in top_labels if tree[label].modules or tree[label].children]
    return result


def list_by_path(path: tuple[str, ...], specs: Iterable[ModuleSpec] | None = None) -> list[ModuleSpec]:
    """按完整路径列出该节点下的全部模块（含嵌套子类）."""
    tree = catalog_tree(specs)
    node = _find_node(tree, list(path))
    if node is None:
        return []
    result: list[ModuleSpec] = []
    _collect(node, result)
    return result


def _find_node(nodes: list[CatalogNode], path: list[str]) -> CatalogNode | None:
    if not path:
        return None
    for node in nodes:
        if node.label == path[0]:
            if len(path) == 1:
                return node
            return _find_node(node.children, path[1:])
    return None


def _collect(node: CatalogNode, out: list[ModuleSpec]) -> None:
    out.extend(node.modules)
    for child in node.children:
        _collect(child, out)


def subcategories_of(top_label: str, specs: Iterable[ModuleSpec] | None = None) -> list[str]:
    """取某大类下的全部子类名."""
    tree = catalog_tree(specs)
    for node in tree:
        if node.label == top_label:
            return [c.label for c in node.children]
    return []
