"""拆分式建模节点：几何节点 → 材料节点 → 网格节点三解耦.

与 ``nodes.py`` 中一体化 ``build_*`` 节点并行共存。拆分流程::

    material.linear_elastic ──┐
    geom.cantilever_2d ───────► mesh.generate_structural ──► MODEL ──► analysis.*

节点函数签名与 ``nodes.py`` 保持一致，target 格式为
``"zylab.flowchart.split_nodes:fn_name"``，可被 runner 按全限定名导入。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from zylab.fea import (
    Constraint,
    ElementBlock,
    ElementType,
    LinearElastic,
    Mesh,
    NodalLoad,
    Section,
    StaticCase,
)

from .bundle import GeometryBundle, MaterialPayload, ModelBundle

__all__ = [
    "geom_cantilever_2d",
    "material_linear_elastic",
    "mesh_generate_structural",
]

#: 节点输入表（源节点为空 Mapping）
NodeInputs = Mapping[str, Any]
#: 节点参数表
NodeParams = Mapping[str, Any]


def material_linear_elastic(inputs: NodeInputs, params: NodeParams, report: Any | None = None) -> MaterialPayload:
    """线弹性材料节点：输出 LinearElastic 载荷.

    参数：e_modulus（弹性模量 MPa）、density（密度 t/mm³）、poisson（泊松比，默认 0.3）。
    """
    del inputs  # 源节点无输入
    del report
    return MaterialPayload(
        material=LinearElastic(
            e_modulus=float(params.get("e_modulus", 2.1e5)),
            poisson=float(params.get("poisson", 0.3)),
            density=float(params.get("density", 7.85)),
        ),
    )


def geom_cantilever_2d(inputs: NodeInputs, params: NodeParams, report: Any | None = None) -> GeometryBundle:
    """悬臂梁 Q4 平面应力几何节点：输出纯几何载荷（mesh + sections + case）.

    参数：length/height（mm）、nx/ny（单元数）、tip_load（端部载荷 N，负向）、thickness（mm）。
    """
    del inputs  # 源节点无输入
    del report
    length, height = float(params["length"]), float(params["height"])
    nx, ny = int(params["nx"]), int(params["ny"])

    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    conn = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            conn.append((n00, n00 + 1, n00 + nx + 2, n00 + nx + 1))
    block = ElementBlock(etype=ElementType.QUAD4, conn=np.asarray(conn), name="梁")
    mesh = Mesh(coords=coords, blocks=(block,))

    n_nodes = mesh.n_nodes
    fixed = tuple(Constraint(node=n, dofs=(0, 1)) for n in range(ny + 1))
    tip = tuple(NodalLoad(node=n, forces=(0.0, float(params["tip_load"]))) for n in range(n_nodes - (ny + 1), n_nodes))
    case = StaticCase(constraints=fixed, loads=tip)

    return GeometryBundle(
        mesh=mesh,
        sections=(Section(thickness=float(params["thickness"])),),
        case=case,
    )


def mesh_generate_structural(inputs: NodeInputs, params: NodeParams, report: Any | None = None) -> ModelBundle:
    """结构网格装配节点：GEOMETRY + MATERIAL → MODEL.

    将几何节点输出的 mesh/sections/case 与材料节点输出的 LinearElastic 合并
    为完整 ModelBundle，供求解器消费。参数保留（当前为空，预留 element_type 等）。
    """
    del params, report  # 当前无参数
    geo = inputs["geometry"]
    mat = inputs["material"]
    if not isinstance(geo, GeometryBundle):
        raise TypeError(f"geometry 端口应为 GeometryBundle，得到 {type(geo).__name__}")
    if not isinstance(mat, MaterialPayload):
        raise TypeError(f"material 端口应为 MaterialPayload，得到 {type(mat).__name__}")
    elastic = mat.material
    if not isinstance(elastic, LinearElastic):
        raise TypeError(f"material 载荷应为 LinearElastic，得到 {type(elastic).__name__}")
    return ModelBundle(
        mesh=geo.mesh,
        materials=(elastic,),
        sections=geo.sections,
        case=geo.case,
    )
