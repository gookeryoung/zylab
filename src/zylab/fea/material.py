"""线弹性材料与截面属性.

- :class:`LinearElastic` 携带应力状态（平面应力/平面应变/空间），按需生成弹性矩阵 D；
- :class:`Section` 描述截面属性：桁架杆取 ``area``，平面连续体单元取 ``thickness``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .errors import ElementError

__all__ = ["LinearElastic", "Section", "StressState"]


class StressState(Enum):
    """应力状态（决定弹性矩阵 D 的形式）."""

    PLANE_STRESS = "plane_stress"
    PLANE_STRAIN = "plane_strain"
    SOLID = "solid"


@dataclass(frozen=True)
class LinearElastic:
    """线弹性各向同性材料.

    Attributes:
        e_modulus: 弹性模量 E（> 0）。
        poisson: 泊松比 nu（平面应变/空间单元要求 < 0.5）。
        state: 应力状态，平面单元默认平面应力。
    """

    e_modulus: float
    poisson: float = 0.0
    state: StressState = StressState.PLANE_STRESS

    def __post_init__(self) -> None:
        """校验材料参数范围."""
        if self.e_modulus <= 0.0:
            raise ElementError(f"弹性模量须为正，实际 E={self.e_modulus}")
        if self.poisson < -1.0 or self.poisson >= 0.5:
            raise ElementError(f"泊松比须在 [-1, 0.5) 内，实际 nu={self.poisson}")

    @property
    def shear_modulus(self) -> float:
        """剪切模量 G = E / (2(1+nu))."""
        return self.e_modulus / (2.0 * (1.0 + self.poisson))

    def d_matrix(self) -> np.ndarray:
        """生成弹性矩阵 D.

        平面应力/平面应变返回 (3, 3)（εxx, εyy, γxy 序），空间返回 (6, 6)
        （εxx, εyy, εzz, γxy, γyz, γxz 序）。
        """
        e, nu = self.e_modulus, self.poisson
        if self.state is StressState.PLANE_STRESS:
            factor = e / (1.0 - nu * nu)
            return factor * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - nu) / 2.0],
                ]
            )
        if self.state is StressState.PLANE_STRAIN:
            factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return factor * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu, 0.0],
                    [0.0, 0.0, 0.5 - nu],
                ]
            )
        factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return factor * np.array(
            [
                [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5 - nu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5 - nu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 - nu],
            ]
        )


@dataclass(frozen=True)
class Section:
    """截面属性：桁架杆取 ``area``（面积），平面连续体取 ``thickness``（厚度），梁取 ``inertia``（惯性矩）."""

    area: float = 1.0
    thickness: float = 1.0
    inertia: float = 1.0

    def __post_init__(self) -> None:
        """校验截面参数为正."""
        if self.area <= 0.0:
            raise ElementError(f"截面面积须为正，实际 A={self.area}")
        if self.thickness <= 0.0:
            raise ElementError(f"截面厚度须为正，实际 t={self.thickness}")
        if self.inertia <= 0.0:
            raise ElementError(f"截面惯性矩须为正，实际 I={self.inertia}")
