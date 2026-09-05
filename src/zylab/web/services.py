"""zylab Web 服务层：画布布局 + 工程 CRUD + 模板列表.

本模块隔离 Django 视图层与 studio 核心层，纯数据结构操作、无 Django 依赖，
便于单元测试。复用 :mod:`zylab.studio` 的 project_io / builtin / graph / template。
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

from zylab.studio import BUILTIN_TEMPLATES
from zylab.studio.graph import WorkflowGraph
from zylab.studio.project_io import load_workflow, save_workflow
from zylab.studio.template import Template

__all__ = [
    "BuiltinTemplateEntry",
    "CanvasEdge",
    "CanvasLayout",
    "CanvasNode",
    "ProjectCRUDError",
    "ProjectService",
    "builtin_template_entries",
    "compute_canvas_layout",
]


# --------------------------------------------------------------------------- 数据类


@dataclass(frozen=True)
class CanvasNode:
    """画布节点（布局后坐标 + 模块元信息）."""

    node_id: str
    label: str
    type_label: str
    state: str  # NodeState.value
    col: int  # 分层（从左到右，0 = 源节点列）
    row: int  # 同列序（模板定义序）
    inputs: tuple[tuple[str, str], ...]  # (端口名, 上游引用)
    elapsed: float
    error: str


@dataclass(frozen=True)
class CanvasEdge:
    """画布连线（SVG 路径坐标）."""

    src_node: str
    src_port: str
    dst_node: str
    dst_port: str
    # 节点行号差 + 端口序 → 贝塞尔控制点（服务端生成，避免前端做布局）
    row_diff: int


@dataclass(frozen=True)
class CanvasLayout:
    """画布完整布局：节点网格 + 连线."""

    template: Template
    nodes: tuple[CanvasNode, ...]
    edges: tuple[CanvasEdge, ...]
    cols: int  # 分层数
    rows: int  # 最大列节点数


# --------------------------------------------------------------------------- 布局算法


def _max_depth(graph: WorkflowGraph, node_id: str, memo: dict[str, int]) -> int:
    """节点分层深度（最长上游链路长度）：source 节点深度为 0."""
    if node_id in memo:
        return memo[node_id]
    upstream = graph.upstream_ids(node_id)
    depth = 0 if not upstream else 1 + max(_max_depth(graph, uid, memo) for uid in upstream)
    memo[node_id] = depth
    return depth


def compute_canvas_layout(template: Template) -> CanvasLayout:
    """按竖向流式（GitHub Actions 风格）布局模板画布.

    布局规则：
    - 按拓扑深度分层（source 列 = 0，analysis 列 = 1+，post 列 = 最深列）；
    - 同列内按模板定义序排列；
    - 连线记录源/目标节点 id 与端口名，前端据此绘制 SVG。

    :param template: 待布局的分析模板。
    """
    graph = WorkflowGraph(template)
    order = graph.execution_order()
    memo: dict[str, int] = {}
    depth_of = {nid: _max_depth(graph, nid, memo) for nid in order}

    # 同列按模板定义序（保证稳定）
    col_rows: dict[int, list[str]] = {}
    for nid in order:
        col = depth_of[nid]
        col_rows.setdefault(col, []).append(nid)

    nodes: list[CanvasNode] = []
    for col, ids in col_rows.items():
        for row, nid in enumerate(ids):
            ni = graph.node(nid)
            nodes.append(
                CanvasNode(
                    node_id=nid,
                    label=ni.id,
                    type_label=ni.spec.name,
                    state=ni.state.value,
                    col=col,
                    row=row,
                    inputs=tuple(ni.inputs.items()),
                    elapsed=ni.elapsed,
                    error=ni.error,
                )
            )

    # 连线：node.inputs → (src_node, src_port)
    edges: list[CanvasEdge] = []
    for cn in nodes:
        ni = graph.node(cn.node_id)
        for port_name, src_ref in ni.inputs.items():
            src_node, _, src_port = src_ref.partition(".")
            src_row = next(n.row for n in nodes if n.node_id == src_node)
            edges.append(
                CanvasEdge(
                    src_node=src_node,
                    src_port=src_port,
                    dst_node=cn.node_id,
                    dst_port=port_name,
                    row_diff=src_row - cn.row,
                )
            )

    cols = len(col_rows)
    rows = max((len(ids) for ids in col_rows.values()), default=0)
    return CanvasLayout(
        template=template,
        nodes=tuple(nodes),
        edges=tuple(edges),
        cols=cols,
        rows=rows,
    )


# --------------------------------------------------------------------------- 工程 CRUD


class ProjectCRUDError(Exception):
    """工程读写失败（包装 studio ProjectIOError）."""


class ProjectService:
    """工程目录服务层（复用 studio.project_io，屏蔽 Path/异常细节）.

    :param projects_dir: 工程根目录（由 Django settings.PROJECTS_DIR 注入）。
    """

    def __init__(self, projects_dir: Path) -> None:
        self._dir = Path(projects_dir)

    @property
    def dir(self) -> Path:
        return self._dir

    def ensure_dir(self) -> Path:
        """目录不存在则创建（首次新建工程前调用）."""
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def list_projects(self) -> tuple[Path, ...]:
        """目录下全部 ``.zprj`` 工程文件（按文件名序）."""
        if not self._dir.is_dir():
            return ()
        return tuple(sorted(self._dir.glob("*.zprj")))

    def open_project(self, name: str) -> tuple[Path, Template]:
        """打开工程文件（校验后缀、读取模板）."""
        path = self._resolve(name)
        try:
            template = load_workflow(path)
        except Exception as exc:
            raise ProjectCRUDError(f"工程文件打开失败: {path.name}: {exc}") from exc
        return path, template

    def create_from_template(self, name: str, template: Template) -> Path:
        """用内置模板创建新工程（另存为 .zprj；同名已存在则报错）."""
        path = self._resolve(name)
        if path.exists():
            raise ProjectCRUDError(f"工程名已存在: {name}")
        self.ensure_dir()
        try:
            return save_workflow(path, template)
        except Exception as exc:
            raise ProjectCRUDError(f"工程创建失败: {path.name}: {exc}") from exc

    def save_project(self, name: str, template: Template) -> Path:
        """保存工程（覆盖同名 .zprj）."""
        path = self._resolve(name)
        self.ensure_dir()
        try:
            return save_workflow(path, template)
        except Exception as exc:
            raise ProjectCRUDError(f"工程保存失败: {path.name}: {exc}") from exc

    def delete_project(self, name: str) -> None:
        """删除工程文件（不存在静默返回）."""
        path = self._resolve(name)
        path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ 内部

    def _resolve(self, name: str) -> Path:
        name = name.strip()
        if not name:
            raise ProjectCRUDError("工程名不能为空")
        # 禁止路径穿越
        if "/" in name or "\\" in name or ".." in name:
            raise ProjectCRUDError(f"非法工程名: {name!r}")
        if name.lower().endswith(".zprj"):
            name = name[:-5]
        return self._dir / f"{name}.zprj"


# --------------------------------------------------------------------------- 内置模板


@dataclass(frozen=True)
class BuiltinTemplateEntry:
    """内置模板条目（供模板选择器渲染）."""

    template_id: str
    name: str
    discipline: str
    description: str
    node_count: int


def builtin_template_entries() -> tuple[BuiltinTemplateEntry, ...]:
    """全部内置模板条目（按学科、模板 id 序）."""
    rows = [
        BuiltinTemplateEntry(
            template_id=t.id,
            name=t.name,
            discipline=t.discipline,
            description=t.description,
            node_count=len(t.nodes),
        )
        for t in BUILTIN_TEMPLATES
    ]
    rows.sort(key=attrgetter("discipline", "template_id"))
    return tuple(rows)
