"""zylab.fea - 有限元分析内核（Qt-free，v1 线弹性静力）.

模块划分：
- :mod:`zylab.fea.mesh`：网格数据结构（节点坐标 + 单元块）；
- :mod:`zylab.fea.material`：线弹性材料与截面属性；
- :mod:`zylab.fea.elements`：单元库（杆/T3/Q4/Tet4/Hex8）；
- :mod:`zylab.fea.boundary`：约束与载荷工况；
- :mod:`zylab.fea.assemble`：CSR 稀疏装配；
- :mod:`zylab.fea.solve`：约束消元 + 稀疏 LU 直接求解；
- :mod:`zylab.fea.static`：静力分析编排与结果。
"""

from __future__ import annotations

from .boundary import Constraint, NodalLoad, StaticCase
from .elements import element_stiffness, element_stress
from .errors import ElementError, MeshError, SolverError
from .material import LinearElastic, Section, StressState
from .mesh import ElementBlock, ElementType, Mesh
from .static import ElementResult, StaticSolution, solve_static

__all__ = [
    "Constraint",
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
    "element_stiffness",
    "element_stress",
    "solve_static",
]
