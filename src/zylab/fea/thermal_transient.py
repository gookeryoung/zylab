"""瞬态热传导分析：backward Euler 时间积分 + 一致热容矩阵.

半离散方程 ``C Ṫ + K T = F``（C 为一致热容 ``∫ ρc NᵀN dΩ``，K 含对流
Robin 项）。常物性、常载荷假设下采用均匀步长 backward Euler：

    (C/dt + K) T_{n+1} = (C/dt) T_n + F

有效矩阵全程仅分解一次（稀疏 LU 复用），给定温度（Dirichlet）采用
自由/约束划块法消元。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse.linalg import splu

from .conduction import ConductionMaterial, _batch_gradients, assemble_capacity, assemble_conduction
from .errors import MeshError, SolverError
from .material import Section
from .mesh import Mesh
from .thermal import ThermalCase, _apply_convections, _convection_total

__all__ = ["ThermalTransientSolution", "solve_thermal_transient"]


@dataclass(frozen=True)
class ThermalTransientSolution:
    """瞬态热传导结果.

    Attributes:
        mesh: 参与求解的网格。
        times: 帧时刻 ``(n_frames,)``，``n_frames = n_steps + 1``，含 t=0 初始帧。
        temperatures: 各帧节点温度 ``(n_frames, n_nodes)``。
        element_gradients: 末帧单元温度梯度 ``(n_elements, dim)``。
        element_heat_flux: 末帧单元热流密度模长 ``k|∇T| (n_elements,)``（W/mm²）。
        t_min: 全程最低温度。
        t_max: 全程最高温度。
        convection_heat: 各帧对流换热量 ``(n_frames,)``（W，正值 = 向环境散热）。
        total_time: 总时长（与末帧时刻一致）。
    """

    mesh: Mesh
    times: np.ndarray
    temperatures: np.ndarray
    element_gradients: np.ndarray
    element_heat_flux: np.ndarray
    t_min: float
    t_max: float
    convection_heat: np.ndarray
    total_time: float


def solve_thermal_transient(  # noqa: PLR0912, PLR0913  模型四要素 + 工况/初值/时长/步数/附加热载荷/回调，参数与校验分支语义不可合并
    mesh: Mesh,
    materials: Sequence[ConductionMaterial],
    sections: Sequence[Section],
    case: ThermalCase,
    *,
    initial: np.ndarray,
    total_time: float,
    n_steps: int,
    extra_heat: np.ndarray | None = None,
    report: Callable[[float, str], None] | None = None,
) -> ThermalTransientSolution:
    """瞬态热传导分析主流程（backward Euler，常物性常载荷）.

    Args:
        mesh: 网格（2D 连续体 TRIA3/QUAD4，或 3D 六面体 HEX8）。
        materials: 传导材料表，瞬态须提供 ``volumetric_heat_capacity``。
        sections: 截面表（平面单元取厚度，HEX8 忽略）。
        case: 热学工况（给定温度 + 热源 + 对流，时间不变）。
        initial: 初始温度 ``(n_nodes,)``（给定温度节点自动取工况值）。
        total_time: 总时长（> 0，单位与材料热容口径一致，通常 s）。
        n_steps: 时间步数（≥ 1，均匀步长 ``dt = total_time / n_steps``）。
        extra_heat: 附加节点热载荷（如 Joule 热一致节点载荷，时间不变）。
        report: 进度回调 ``(progress, message)``（进程执行器自动注入）。

    Returns:
        :class:`ThermalTransientSolution`，含各帧温度、末帧热流与温度范围。

    Raises:
        MeshError: 工况/初值/时长参数非法或材料缺少体积热容时抛出。
        SolverError: 有效矩阵奇异（约束不足）时抛出。
    """
    progress = report if report is not None else _no_report
    progress(0.05, "校验瞬态热工况")
    case.validate(mesh)
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (mesh.n_nodes,):
        raise MeshError(f"初始温度维度 {initial.shape} 与节点数 {mesh.n_nodes} 不符")
    if total_time <= 0.0:
        raise MeshError(f"瞬态总时长须为正，实际 {total_time}")
    if n_steps < 1:
        raise MeshError(f"时间步数须 ≥ 1，实际 {n_steps}")
    if extra_heat is not None and extra_heat.shape != (mesh.n_nodes,):
        raise MeshError(f"附加热载荷维度 {extra_heat.shape} 与节点数 {mesh.n_nodes} 不符")

    progress(0.15, "装配导热与热容矩阵")
    k_thermal = assemble_conduction(mesh, materials, sections, "thermal")
    capacity = assemble_capacity(mesh, materials, sections)
    if float(np.max(np.abs(capacity.diagonal()))) <= 0.0:
        raise MeshError("瞬态热分析缺少热容：材料表未提供体积热容 volumetric_heat_capacity")

    force = np.zeros(mesh.n_nodes)
    for source in case.heat_sources:
        force[source.node] += source.value
    if extra_heat is not None:
        force = force + extra_heat
    k_total, force = _apply_convections(mesh, case, k_thermal, force)

    seen: dict[int, float] = {}
    for prescribed in case.temperatures:
        seen.setdefault(prescribed.node, prescribed.value)
    fixed = np.fromiter(seen.keys(), dtype=np.intp)
    values = np.fromiter(seen.values(), dtype=float)

    dt = total_time / n_steps
    # 有效矩阵 C/dt + K：均匀步长全程分解一次
    k_eff = (capacity.multiply(1.0 / dt) + k_total).tocsr()
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    mask[fixed] = True
    free = np.flatnonzero(~mask)
    u = initial.copy()
    u[fixed] = values

    frames: list[np.ndarray] = [u.copy()]
    conv_heat: list[float] = []
    progress(0.25, "分解有效矩阵（稀疏 LU）")
    # scipy 稀疏矩阵的花式切片在运行时可用，pyrefly 的 scipy 存根未覆盖
    k_ff = k_eff[free][:, free].tocsc()  # type: ignore[bad-index]
    k_fc = k_eff[free][:, fixed]  # type: ignore[bad-index]
    try:
        lu = splu(k_ff)
    except RuntimeError as exc:
        raise SolverError("瞬态有效矩阵奇异：约束不足无法求解") from exc
    inv_dt = 1.0 / dt
    for step in range(n_steps):
        rhs = capacity @ (u * inv_dt) + force
        rhs_free = rhs[free]
        if fixed.size:
            rhs_free = rhs_free - k_fc @ values
        u = u.copy()
        u[free] = lu.solve(rhs_free)
        frames.append(u.copy())
        progress(0.25 + 0.7 * (step + 1) / n_steps, f"时间积分 {step + 1}/{n_steps}")

    progress(0.95, "恢复末帧热流")
    for frame in frames:
        conv_heat.append(_convection_total(mesh, case, frame))
    gradients: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    final = frames[-1]
    for block in mesh.blocks:
        material = materials[block.material]
        grads = _batch_gradients(block.etype, mesh.coords[block.conn], final[block.conn])
        gradients.append(grads)
        fluxes.append(material.thermal_k * np.linalg.norm(grads, axis=1))
    all_temperatures = np.stack(frames)
    progress(1.0, "瞬态温度场求解完成")
    return ThermalTransientSolution(
        mesh=mesh,
        times=np.linspace(0.0, total_time, n_steps + 1),
        temperatures=all_temperatures,
        element_gradients=np.concatenate(gradients) if gradients else np.empty((0, mesh.dim)),
        element_heat_flux=np.concatenate(fluxes) if fluxes else np.empty(0),
        t_min=float(all_temperatures.min()),
        t_max=float(all_temperatures.max()),
        convection_heat=np.asarray(conv_heat),
        total_time=total_time,
    )


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调."""
