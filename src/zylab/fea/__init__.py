"""zylab.fea - 有限元分析内核（Qt-free，结构/热/电多学科）.

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
- :mod:`zylab.fea.transient`：瞬态动力分析（Newmark 直接时间积分）；
- :mod:`zylab.fea.buckling`：线性屈曲分析编排与结果（杆/梁几何刚度）；
- :mod:`zylab.fea.nonlinear`：几何非线性静力（TRUSS2 大位移 Newton-Raphson）；
- :mod:`zylab.fea.conduction`：标量场传导内核（电/热共用，每节点 1 DOF）；
- :mod:`zylab.fea.electric`：稳态电传导分析（电压场/电场/耗散功率）；
- :mod:`zylab.fea.thermal`：稳态热传导分析（温度场/热流/对流边界）；
- :mod:`zylab.fea.electrothermal`：电-热耦合（Joule 热顺序耦合，稳态/瞬态）；
- :mod:`zylab.fea.thermal_transient`：瞬态热传导（backward Euler + 一致热容）；
- :mod:`zylab.fea.export`：结果 CSV 导出（六类解，GUI 与 CLI 共用）。
"""

from __future__ import annotations

from .assemble import assemble_geometric, assemble_loads, assemble_mass, assemble_stiffness
from .boundary import BodyForce, Constraint, EdgePressure, NodalLoad, StaticCase
from .buckling import BucklingSolution, solve_buckling
from .conduction import (
    ConductionMaterial,
    NodalSource,
    NodalValue,
    assemble_capacity,
    assemble_conduction,
    element_conductance,
    element_field_load,
    element_scalar_gradient,
)
from .electric import ElectricCase, ElectricSolution, solve_electric
from .electrothermal import (
    ElectroThermalSolution,
    ElectroThermalTransientSolution,
    solve_electrothermal,
    solve_electrothermal_transient,
)
from .elements import (
    element_geometric_stiffness,
    element_mass,
    element_measure,
    element_stiffness,
    element_stress,
    element_stress_at,
    truss2_internal_force,
    truss2_tangent_stiffness,
)
from .errors import ElementError, MeshError, SolverError
from .export import export_csv
from .harmonic import HarmonicResponse, solve_harmonic
from .material import LinearElastic, Section, StressState
from .mesh import ElementBlock, ElementType, Mesh, element_dofs_per_node
from .modal import ModalSolution, solve_modal
from .nonlinear import NonlinearSolution, solve_nonlinear_static
from .static import ElementResult, StaticSolution, solve_static
from .thermal import Convection, ThermalCase, ThermalSolution, solve_thermal
from .thermal_transient import ThermalTransientSolution, solve_thermal_transient
from .transient import TransientSolution, solve_transient

__all__ = [
    "BodyForce",
    "BucklingSolution",
    "ConductionMaterial",
    "Constraint",
    "Convection",
    "EdgePressure",
    "ElectricCase",
    "ElectricSolution",
    "ElectroThermalSolution",
    "ElectroThermalTransientSolution",
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
    "NodalSource",
    "NodalValue",
    "NonlinearSolution",
    "Section",
    "SolverError",
    "StaticCase",
    "StaticSolution",
    "StressState",
    "ThermalCase",
    "ThermalSolution",
    "ThermalTransientSolution",
    "TransientSolution",
    "assemble_capacity",
    "assemble_conduction",
    "assemble_geometric",
    "assemble_loads",
    "assemble_mass",
    "assemble_stiffness",
    "element_conductance",
    "element_dofs_per_node",
    "element_field_load",
    "element_geometric_stiffness",
    "element_mass",
    "element_measure",
    "element_scalar_gradient",
    "element_stiffness",
    "element_stress",
    "element_stress_at",
    "export_csv",
    "solve_buckling",
    "solve_electric",
    "solve_electrothermal",
    "solve_electrothermal_transient",
    "solve_harmonic",
    "solve_modal",
    "solve_nonlinear_static",
    "solve_static",
    "solve_thermal",
    "solve_thermal_transient",
    "solve_transient",
    "truss2_internal_force",
    "truss2_tangent_stiffness",
]
