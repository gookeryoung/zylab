"""zylab.fea - 有限元分析内核（Qt-free，线弹性静力/模态/谐响应/屈曲）.

模块划分：
- :mod:`zylab.fea.mesh`：网格数据结构（节点坐标 + 单元块）；
- :mod:`zylab.fea.material`：线弹性材料（含质量密度）与截面属性；
- :mod:`zylab.fea.elements`：单元库（杆/梁/T3/Q4/Tet4/Hex8，刚度/质量/几何刚度）；
- :mod:`zylab.fea.boundary`：约束与载荷工况（节点力/边压力/体力）；
- :mod:`zylab.fea.assemble`：CSR 稀疏装配（刚度/质量/几何刚度/载荷）；
- :mod:`zylab.fea.solve`：约束消元 + 稀疏 LU 直接求解；
- :mod:`zylab.fea.static`：静力分析编排与结果；
- :mod:`zylab.fea.modal`：模态分析编排与结果；
- :mod:`zylab.fea.harmonic`：谐响应分析编排与结果（Rayleigh 阻尼）；
- :mod:`zylab.fea.buckling`：线性屈曲分析编排与结果（杆/梁几何刚度）。
"""

from __future__ import annotations

from .assemble import assemble_geometric, assemble_loads, assemble_mass, assemble_stiffness
from .boundary import BodyForce, Constraint, EdgePressure, NodalLoad, StaticCase
from .buckling import BucklingSolution, solve_buckling
from .elements import (
    element_geometric_stiffness,
    element_mass,
    element_measure,
    element_stiffness,
    element_stress,
    element_stress_at,
)
from .errors import ElementError, MeshError, SolverError
from .harmonic import HarmonicResponse, solve_harmonic
from .material import LinearElastic, Section, StressState
from .mesh import ElementBlock, ElementType, Mesh, element_dofs_per_node
from .modal import ModalSolution, solve_modal
from .static import ElementResult, StaticSolution, solve_static

__all__ = [
    "BodyForce",
    "BucklingSolution",
    "Constraint",
    "EdgePressure",
    "ElementBlock",
    "ElementError",
    "ElementResult",
    "ElementType",
    "HarmonicResponse",
    "LinearElastic",
    "Mesh",
    "MeshError",
    "ModalSolution",
    "NodalLoad",
    "Section",
    "SolverError",
    "StaticCase",
    "StaticSolution",
    "StressState",
    "assemble_geometric",
    "assemble_loads",
    "assemble_mass",
    "assemble_stiffness",
    "element_dofs_per_node",
    "element_geometric_stiffness",
    "element_mass",
    "element_measure",
    "element_stiffness",
    "element_stress",
    "element_stress_at",
    "solve_buckling",
    "solve_harmonic",
    "solve_modal",
    "solve_static",
]
