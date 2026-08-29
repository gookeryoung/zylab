"""zylab.fea - 有限元分析内核（Qt-free，v1 线弹性静力）.

模块划分：
- :mod:`zylab.fea.mesh`：网格数据结构（节点坐标 + 单元块）；
- :mod:`zylab.fea.material`：线弹性材料与截面属性；
- :mod:`zylab.fea.elements`：单元库（杆/梁/T3/Q4/Tet4/Hex8）；
- :mod:`zylab.fea.boundary`：约束与载荷工况（节点力/边压力/体力）；
- :mod:`zylab.fea.assemble`：CSR 稀疏装配；
- :mod:`zylab.fea.solve`：约束消元 + 稀疏 LU 直接求解；
- :mod:`zylab.fea.static`：静力分析编排与结果。
"""

from __future__ import annotations

from .boundary import BodyForce, Constraint, EdgePressure, NodalLoad, StaticCase
from .elements import element_measure, element_stiffness, element_stress, element_stress_at
from .errors import ElementError, MeshError, SolverError
from .material import LinearElastic, Section, StressState
from .mesh import ElementBlock, ElementType, Mesh, element_dofs_per_node
from .static import ElementResult, StaticSolution, solve_static

__all__ = [
    "BodyForce",
    "Constraint",
    "EdgePressure",
    "ElementBlock",
    "ElementError",
    "ElementResult",
    "ElementType",
    "LinearElastic",
    "Mesh",
    "MeshError",
    "NodalLoad",
    "Section",
    "SolverError",
    "StaticCase",
    "StaticSolution",
    "StressState",
    "element_dofs_per_node",
    "element_measure",
    "element_stiffness",
    "element_stress",
    "element_stress_at",
    "solve_static",
]
