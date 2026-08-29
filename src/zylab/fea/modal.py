"""模态分析编排：装配 K/M -> 约束划块 -> 广义特征值求解.

对外主入口 :func:`solve_modal`，返回 :class:`ModalSolution`
（固有频率/圆频率、质量归一化振型）。求解采用自由自由度划块 +
shift-invert Lanczos（``eigsh(sigma=0)``），振型按 ``φ^T M φ = 1`` 归一。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse.linalg import eigsh

from .assemble import assemble_mass, assemble_stiffness
from .boundary import Constraint
from .errors import SolverError
from .material import LinearElastic, Section
from .mesh import Mesh

__all__ = ["ModalSolution", "solve_modal"]


def _no_report(_progress: float, _message: str) -> None:
    """空进度回调（未注入 report 时使用）."""


@dataclass(frozen=True)
class ModalSolution:
    """模态分析求解结果.

    Attributes:
        mesh: 参与求解的网格。
        frequencies: 固有圆频率 ``(n_modes,)``（rad/s，升序）。
        mode_shapes: 质量归一化振型 ``(n_dofs, n_modes)``，约束自由度分量为 0，
            满足 ``Φ^T M Φ = I``。
    """

    mesh: Mesh
    frequencies: np.ndarray
    mode_shapes: np.ndarray

    @property
    def n_modes(self) -> int:
        """模态阶数."""
        return int(self.frequencies.size)

    @property
    def frequencies_hz(self) -> np.ndarray:
        """固有频率（Hz）."""
        return self.frequencies / (2.0 * np.pi)

    def mode_shape(self, index: int) -> np.ndarray:
        """取指定阶振型并整形为 ``(n_nodes, dofs_per_node)``."""
        return self.mode_shapes[:, index].reshape(self.mesh.n_nodes, self.mesh.dofs_per_node).copy()


def solve_modal(  # noqa: PLR0913  静力/模态共用四要素 + 可选阶数与回调，语义不可合并
    mesh: Mesh,
    materials: Sequence[LinearElastic],
    sections: Sequence[Section],
    constraints: Sequence[Constraint],
    *,
    n_modes: int = 10,
    report: Callable[[float, str], None] | None = None,
) -> ModalSolution:
    """无阻尼自由振动模态分析（线弹性小位移）.

    求解广义特征值问题 ``K φ = ω² M φ``，约束自由度划块消元
    （约束值须为 0，非零约束在模态分析中无意义）。

    Args:
        mesh: 网格。
        materials: 材料表（须配置正的质量密度）。
        sections: 截面表（ElementBlock.section 索引引用）。
        constraints: 位移约束（消除刚体模态；约束值须为 0）。
        n_modes: 提取模态阶数（须小于自由自由度数）。
        report: 进度回调 ``(progress, message)``；进程执行器会自动注入
            （见 :mod:`zylab.core.executor` 的任务协议）。

    Returns:
        :class:`ModalSolution`，含升序固有圆频率与质量归一化振型。

    Raises:
        SolverError: 约束缺失/非零、模态数越界或特征值求解失败时抛出。
    """
    progress = _no_report if report is None else report

    progress(0.1, "校验约束")
    fixed_dofs = _validate_constraints(mesh, constraints)
    progress(0.35, "装配刚度与质量矩阵")
    k_global = assemble_stiffness(mesh, materials, sections)
    m_global = assemble_mass(mesh, materials, sections)

    mask = np.zeros(mesh.n_dofs, dtype=bool)
    mask[fixed_dofs] = True
    free = np.flatnonzero(~mask)
    if n_modes >= free.size:
        raise SolverError(f"模态阶数 {n_modes} 须小于自由自由度数 {free.size}")
    if n_modes < 1:
        raise SolverError(f"模态阶数须至少为 1，实际 {n_modes}")

    # scipy 稀疏矩阵的花式切片在运行时可用，pyrefly 的 scipy 存根未覆盖
    k_ff = k_global[free][:, free]  # type: ignore[bad-index]
    m_ff = m_global[free][:, free]  # type: ignore[bad-index]
    progress(0.6, "求解广义特征值问题")
    try:
        eigenvalues, eigenvectors = eigsh(
            k_ff.tocsc(),  # type: ignore[missing-attribute]  scipy 存根对切片返回类型标注不全
            k=n_modes,
            M=m_ff.tocsc(),  # type: ignore[missing-attribute]
            sigma=0.0,
            which="LM",
        )
    except (RuntimeError, ValueError) as exc:
        raise SolverError(f"特征值求解失败：约束不足或存在机构（{exc}）") from exc

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # sigma=0 的 shift-invert 可能返回极小的负特征值（数值噪声），截断为 0
    omega = np.sqrt(np.clip(eigenvalues, 0.0, None))

    shapes = np.zeros((mesh.n_dofs, n_modes))
    shapes[free, :] = eigenvectors
    progress(1.0, "模态求解完成")
    return ModalSolution(mesh=mesh, frequencies=omega, mode_shapes=shapes)


def _validate_constraints(mesh: Mesh, constraints: Sequence[Constraint]) -> np.ndarray:
    """展开约束为全局自由度索引，并校验约束值全为 0."""
    width = mesh.dofs_per_node
    seen: dict[int, float] = {}
    for constraint in constraints:
        if not 0 <= constraint.node < mesh.n_nodes:
            raise SolverError(f"约束引用节点 {constraint.node} 越界（共 {mesh.n_nodes} 节点）")
        for dof in constraint.dofs:
            if not 0 <= dof < width:
                raise SolverError(f"约束节点 {constraint.node} 的自由度 {dof} 超出 [0, {width})")
            seen.setdefault(constraint.node * width + dof, constraint.value)
    if not seen:
        raise SolverError("模态分析缺少位移约束，刚体模态未消除")
    nonzero = {dof: value for dof, value in seen.items() if value != 0.0}
    if nonzero:
        raise SolverError("模态分析约束值须为 0（非零约束仅适用于静力分析）")
    return np.fromiter(seen.keys(), dtype=np.intp)
