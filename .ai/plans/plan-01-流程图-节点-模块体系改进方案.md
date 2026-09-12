# 流程图-节点-模块体系改进方案（参考 ANSYS Workbench）

## 现状与缺口

现有 `src/zylab/flowchart/` 已具备 Workbench 式骨架：ModuleSpec/PortSpec/ParamSpec 类型系统、DAG + 6 态节点状态机、拓扑执行、端口同类型校验、级联脏传播、内容哈希缓存、模板/DSL/工程持久化、进程隔离执行、DOE/优化/可靠性层。主要缺口：

1. **画布只读**：`gui/widgets/node_canvas.py` 仅展示/选中/运行，无拖拽建节点、拉线连接、删除、撤销重做；`WorkflowGraph.add_node/add_link` 后端 API 已就绪但无 GUI 入口，toolbox 与画布未联动。
2. **模块未按五大类组织**：模块散落在 BUILTIN_MODULES + SOLVER_MODULES，无"大类→子类→模块"的层级目录。
3. **建模硬编码、网格内嵌**：源节点为固定几何（build_cantilever 等），网格在建模节点内生成，无独立网格划分节点。
4. **PortType 封闭枚举**：新增端口类型须改枚举；ANY/DATA 端口无内部 schema 校验；无类型兼容矩阵（不匹配只能靠同类型强校验拒绝）。

## 分层目标架构

```
流程图 (WorkflowGraph/DAG)
  └─ 节点 (NodeInstance = ModuleSpec 实例 + 参数 + 位置)
       └─ 模块 (ModuleSpec，五大类目录树)
            └─ 节点函数 node_fn(inputs, params) → 输出（可 pickle 跨进程）
数据流：端口 (PortSpec, PortType) + 兼容矩阵 → 连接即数据流契约
```

## Phase 1 — 模块体系重构（五大类目录）

### 1.1 模块分类目录（`flowchart/catalog.py` 新建）

定义 `ModuleCategory` 树，统一组织 BUILTIN_MODULES 与 SOLVER_MODULES，GUI 工具箱按此树渲染：

```
材料参数 (MATERIAL)
 ├─ 弹性材料（LinearElastic：E/ν/ρ/状态）
 ├─ 导热/导电材料（ConductionMaterial）
 ├─ 材料选择器（从材料库/表读取）
 └─ 材料-截面属性（Section，合并到建模输入或独立）
参数化建模 (GEOMETRY)
 ├─ 梁系几何：悬臂梁 / 简支梁 / 桁架 / 框架（现有 build_* 泛化为参数驱动）
 ├─ 连续体几何：矩形板 / 圆柱 / 薄膜（vfilm）
 ├─ 焦耳热板等热-电几何
 └─ 输出 GEOMETRY（几何+材料引用+截面，不含网格）
有限元网格划分 (MESH)
 ├─ 通用网格划分（单元类型 ElementType + 全局尺寸 + 分块）
 ├─ 网格细化/重划分（局部尺寸控制）
 └─ 输出 MODEL（ModelBundle，含 Mesh + 材料 + 截面）
求解器 (SOLVER)
 ├─ 结构静力 / 模态 / 谐响应 / 瞬态 / 屈曲 / 非线性（solver_adapter 自动生成）
 ├─ 热分析 / 电分析 / 电-热耦合（electrothermal）
 └─ 输入 MODEL + 载荷 DATA，输出 XxxSolution
后处理 (POST)
 ├─ 变形/位移云图（viewdata.deformed_coords）
 ├─ 应力/应变场（nodal_stress_field）
 ├─ 路径/ probes 结果提取（compute.expr、post.static）
 ├─ 曲线/图表（DATA 输出）
 └─ 报告生成（report.py HTML/Markdown）
```

### 1.2 建模与网格拆分

- 新增 `PortType.GEOMETRY`、`PortType.MATERIAL`；新增节点：
  - `geom.*`（各几何源，输出 GEOMETRY payload = 几何描述 + 材料输入口 + Section 参数）
  - `material.linear` / `material.conduction`（输出 MATERIAL payload = LinearElastic/ConductionMaterial 实例）
  - `mesh.generate`（输入 GEOMETRY + MATERIAL，参数：element_type/target_size，输出 MODEL）
- 兼容旧图：保留现有 build_* 一体化 SOURCE 节点为 legacy 类别，模板加载时按 `project_io` 版本号自动迁移（一体化节点 → geom+mesh 两节点自动连线）；`.zprj`/模板 JSON 增加 schema version 字段。

### 1.3 关键文件

- 改 `flowchart/module.py`（PortType 扩展、BUILTIN_MODULES 重组为按类别注册）
- 新建 `flowchart/catalog.py`（类别树 + 模块注册 API：`register_module(spec, category_path)`）
- 改 `flowchart/nodes.py`（拆出 geom.*/mesh.*/material.* 节点函数）
- 改 `flowchart/graph.py`（无需大改，类型校验走新机制见 Phase 2）

## Phase 2 — 端口类型系统改进

### 2.1 开放式端口类型 + 兼容矩阵

- `PortType` 保留为内置枚举，新增 `PortTypeRegistry`（`flowchart/ports.py` 新建）：
  - `register(name, *, color, label, schema=None, aliases=())` 支持插件经 entry point `zylab.port_type` 注册
  - 兼容矩阵 `compat[dest][src] -> bool`（声明式谓词），替代 `graph.py:212` 的严格同类型判断；默认规则：同名兼容 + ANY 收发 + 显式声明（如 MODAL 结果 → 谐响应输入可声明兼容）
- DATA 端口 payload schema：PortSpec 增加可选 `payload_spec`（字段名→类型/形状的轻量声明），`resolve_input` 后校验，失败抛带节点/端口定位的 `PortError`

### 2.2 连接校验 API

`WorkflowGraph.can_connect(src_node, src_port, dst_node, dst_port) -> (bool, reason)` 供画布实时反馈（拖线时预判高亮可连端口、标红不可连）；`add_link` 复用同一谓词，保证 GUI 与后端一致。

## Phase 3 — 可编辑画布（Qt）

### 3.1 模块工具箱联动（`gui/widgets/toolbox.py`）

- 按 Phase 1 类别树渲染：大类 → 子类 → 模块（图标+名称+端口摘要 tooltip）
- 支持拖拽（QDrag mime 携带 type_id）与双击两种方式向画布添加节点

### 3.2 画布编辑交互（`gui/widgets/node_canvas.py`）

- **拖放新建**：dropEvent → `graph.add_node(type_id, position)`，自动命名（如 "静力分析 2"）
- **拉线连接**：端口锚点（QGraphicsItem 于节点边缘，按 PortSpec 渲染，颜色=端口类型色）拖出橡皮线；实时用 `can_connect` 高亮合法目标端口、非法标红；释放即 `add_link`
- **编辑操作**：选中删除（Delete 键，`remove_node`/断链）、参数双击打开 `param_form`（已有）、节点拖动改 position、右键菜单（运行到此/从此运行、插入节点）
- **撤销重做**：`QUndoStack` + 命令对象（AddNode/RemoveNode/AddLink/RemoveLink/MoveNode/SetParams），命令直接操作 WorkflowGraph，天然复用脏传播与缓存失效
- **运行保护**：runner 运行期间锁结构编辑（与 runner.py:46 约束一致），工具栏显示运行态

### 3.3 增量运行

- 连线/参数变更后，受影响节点回退 UNFULFILLED（已有级联失效），画布节点按 NodeState 着色；"运行"只重算脏节点（缓存命中跳过，runner.py:127 已支持）

## 实施顺序与验证

1. **Phase 1**（模块目录 + 建模/网格拆分 + 模板迁移）：新增 tests/test_catalog.py、geom/mesh 节点测试、旧模板迁移测试
2. **Phase 2**（端口注册表 + 兼容矩阵 + payload 校验）：tests/test_ports.py、can_connect 单测（含 ANY/别名/兼容声明/环检测）
3. **Phase 3**（画布编辑 + 工具箱 + 撤销重做）：GUI 手动验收 + 已有 tests 中 graph 操作的回归
4. 全程保持 95% 覆盖率门禁；所有公共 API（catalog 注册、port 注册、QUndo 命令）配套单测

## 不做的事

- 不引入 Web 端（iter-26 的 Django 规划另立项）
- 不做多图/子图嵌套（compute.sweep 机制维持现状）
- 不做通用 CAD 内核，参数化几何仍为现有内置几何族 + 参数化