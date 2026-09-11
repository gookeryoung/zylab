"""LSC 曲线优化：带几何约束的最小二乘三断点四曲线拟合.

从 lscopt 项目迁移而来，去除 bitool/attrs 依赖，改为标准库 logging +
普通类（cached_property 保留计算缓存语义）。核心算法不变：以
scipy.optimize.lsq_linear 求解带等式约束的 16 维多项式系数。

输入 10 个参数，输出解向量 x(16)、残差 cost、四段曲线采样数据与
特定点角度，供 studio 节点与 DSL 参数化计算直接消费。
"""

from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
from scipy.optimize import OptimizeResult, minimize

__all__ = ["LSCCurve", "solve_lsc_curve"]

logger = logging.getLogger(__name__)


class LSCCurve:
    """LSC 曲线计算器，带几何约束的最小二乘三断点四曲线拟合.

    以三个断点 ``m < m1 < 0`` 将 x 轴划分为内部段 ``[m, m1]`` 和外部段
    ``[m1, 0]``，每段上下各一条三次多项式曲线（共四条），约束条件包括：
    断点处函数值、一阶/二阶导数连续、特定点 ``m2`` 的高度匹配、
    端点角度 ``J/J1`` 匹配。

    参数
    ----
    m : float
        内部断点（须为负），默认 -1.3
    m1 : float
        外部断点（须为负且 < m），默认 -2.4
    s : float
        内部斜率（内部段 x=0 处的目标斜率），默认 1.2183
    s1 : float
        外部斜率（外部段 x=0 处的目标斜率），默认 8.1
    H : float
        切削高度（特定点 m2 处两条曲线的目标间距），默认 0.5
    m2 : float
        特定点（正负皆可），默认 0.5
    H1 : float
        内部保留高度（内部段特定点处的目标高度），默认 0.2
    H2 : float
        外部保留高度（外部段特定点处的目标高度），默认 0.65
    J : float
        总角度（度），默认 80
    J1 : float
        内部角度（度），默认 40
    """

    # 基础参数（类默认值，实例 __init__ 后由 cached_property 派生）
    m: float = -1.3
    m1: float = -2.4
    s: float = 1.2183
    s1: float = 8.1
    H: float = 0.5
    m2: float = 0.5
    H1: float = 0.2
    H2: float = 0.65
    J: float = 80.0
    J1: float = 40.0

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        m: float = -1.3,
        m1: float = -2.4,
        s: float = 1.2183,
        s1: float = 8.1,
        H: float = 0.5,
        m2: float = 0.5,
        H1: float = 0.2,
        H2: float = 0.65,
        J: float = 80.0,
        J1: float = 40.0,
    ) -> None:
        self.m = float(m)
        self.m1 = float(m1)
        self.s = float(s)
        self.s1 = float(s1)
        self.H = float(H)
        self.m2 = float(m2)
        self.H1 = float(H1)
        self.H2 = float(H2)
        self.J = float(J)
        self.J1 = float(J1)
        self._validate_parameters()

    # ------------------------------------------------------------------ 校验

    def _validate_parameters(self) -> None:
        """验证所有输入参数的正确性和可行性."""
        self._validate_angles()
        self._validate_breakpoints()
        self._validate_heights()
        self._validate_slopes()
        logger.debug("LSCCurve 初始化参数: m=%s, m1=%s, H=%s", self.m, self.m1, self.H)

    def _validate_angles(self) -> None:
        """验证角度参数 J 和 J1 均在 [0, 180] 度范围内."""
        if not 0 <= self.J <= 180:
            msg = f"总角度 J 必须在 0~180 度之间，实际为 {self.J}"
            raise ValueError(msg)
        if not 0 <= self.J1 <= 180:
            msg = f"内部角度 J1 必须在 0~180 度之间，实际为 {self.J1}"
            raise ValueError(msg)

    def _validate_breakpoints(self) -> None:
        """验证断点 m < m1 < 0."""
        if self.m >= 0:
            msg = f"内部断点 m 必须为负数，实际为 {self.m}"
            raise ValueError(msg)
        if self.m1 >= 0:
            msg = f"外部断点 m1 必须为负数，实际为 {self.m1}"
            raise ValueError(msg)
        if self.m1 >= self.m:
            msg = f"外部断点 m1({self.m1}) 必须小于内部断点 m({self.m})"
            raise ValueError(msg)

    def _validate_heights(self) -> None:
        """高度参数为负时记录警告."""
        if self.H < 0:
            logger.warning("负的切削高度 H=%s 可能导致意外结果", self.H)
        if self.H1 < 0:
            logger.warning("负的内部保留高度 H1=%s 可能导致意外结果", self.H1)
        if self.H2 < 0:
            logger.warning("负的外部保留高度 H2=%s 可能导致意外结果", self.H2)

    def _validate_slopes(self) -> None:
        """斜率参数为负时记录警告."""
        if self.s < 0:
            logger.warning("负的内部斜率 s=%s 可能导致意外结果", self.s)
        if self.s1 < 0:
            logger.warning("负的外部斜率 s1=%s 可能导致意外结果", self.s1)

    # ------------------------------------------------------------------ 辅助

    @staticmethod
    def _cot(x: float) -> float:
        """余切函数（弧度输入）."""
        return 1.0 / np.tan(x)

    @cached_property
    def n(self) -> float:
        """cot(J)，J 为总角度."""
        return self._cot(np.radians(self.J))

    @cached_property
    def t(self) -> float:
        """cot(J1)，J1 为内部角度."""
        return self._cot(np.radians(self.J1))

    @cached_property
    def ms(self) -> float:
        """m 的平方."""
        return self.m**2

    @cached_property
    def mc(self) -> float:
        """m 的立方."""
        return self.m**3

    @cached_property
    def ms4(self) -> float:
        """m 的四次方."""
        return self.m**4

    @cached_property
    def m1s(self) -> float:
        """m1 的平方."""
        return self.m1**2

    @cached_property
    def m1c(self) -> float:
        """m1 的立方."""
        return self.m1**3

    @cached_property
    def m1s4(self) -> float:
        """m1 的四次方."""
        return self.m1**4

    @cached_property
    def m2s(self) -> float:
        """m2 的平方."""
        return self.m2**2

    @cached_property
    def m2c(self) -> float:
        """m2 的立方."""
        return self.m2**3

    @cached_property
    def m2s4(self) -> float:
        """m2 的四次方."""
        return self.m2**4

    # ------------------------------------------------------------------ 约束矩阵

    @cached_property
    def C(self) -> np.ndarray:
        """等式约束矩阵 C (8x16)."""
        return np.array(
            [
                [1, self.m, self.ms, self.mc, -1, -self.m, -self.ms, -self.mc, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, self.m, self.ms, self.mc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, self.m, self.ms, self.mc, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    self.m,
                    self.ms / 2,
                    self.mc / 3,
                    self.ms4 / 4,
                    -self.m,
                    -self.ms / 2,
                    -self.mc / 3,
                    -self.ms4 / 4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, self.m1, self.m1s, self.m1c, -1, -self.m1, -self.m1s, -self.m1c],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, self.m1, self.m1s, self.m1c, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.m1, self.m1s, self.m1c],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    self.m1,
                    self.m1s / 2,
                    self.m1c / 3,
                    self.m1s4 / 4,
                    -self.m1,
                    -self.m1s / 2,
                    -self.m1c / 3,
                    -self.m1s4 / 4,
                ],
            ],
        )

    @cached_property
    def i(self) -> np.ndarray:
        """内部段采样向量：linspace(m1, 0, 100)."""
        return np.linspace(self.m1, 0, 100)

    @cached_property
    def j(self) -> np.ndarray:
        """外部段采样向量：linspace(m1, 0, 100)."""
        return np.linspace(self.m1, 0, 100)

    @cached_property
    def d(self) -> np.ndarray:
        """目标向量 d."""
        return np.array(
            [0, self.n * self.m, self.t * self.m, self.s / 2, 0, self.n * self.m1, self.t * self.m1, self.s1 / 2]
        )

    @cached_property
    def A_ineq(self) -> np.ndarray:
        """不等式约束矩阵 A_ineq."""
        return np.array(
            [
                [0, 1, 2 * self.m, 3 * self.ms, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, -2 * self.m, -3 * self.ms, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * self.m1, 3 * self.m1s, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2 * self.m1, -3 * self.m1s],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

    @cached_property
    def b_ineq(self) -> np.ndarray:
        """不等式约束向量 b_ineq."""
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -self.H])

    @cached_property
    def A_eq(self) -> np.ndarray:
        """等式约束矩阵 A_eq."""
        return np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [1, self.m2, self.m2s, self.m2c, 0, 0, 0, 0, -1, -self.m2, -self.m2s, -self.m2c, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, -self.m2, -self.m2s, -self.m2c, 0, 0, 0, 0, 1, self.m2, self.m2s, self.m2c],
                [1, self.m, self.ms, self.mc, -1, -self.m, -self.ms, -self.mc, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, self.m1, self.m1s, self.m1c, -1, -self.m1, -self.m1s, -self.m1c],
                [
                    self.m,
                    self.ms / 2,
                    self.mc / 3,
                    self.ms4 / 4,
                    -self.m,
                    -self.ms / 2,
                    -self.mc / 3,
                    -self.ms4 / 4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ],
        )

    @cached_property
    def b_eq(self) -> np.ndarray:
        """等式约束向量 b_eq."""
        return np.array([0, 0, 0, 0, self.H1, self.H1, self.H2, self.H2, 0, 0, self.s / 2])

    # ------------------------------------------------------------------ 求解

    @cached_property
    def R(self) -> OptimizeResult:
        """用 SLSQP 求解带等式+不等式约束的最小二乘."""

        def _obj(x: np.ndarray) -> float:
            r = self.C @ x - self.d
            return float(np.dot(r, r))

        def _jac(x: np.ndarray) -> np.ndarray:
            r = self.C @ x - self.d
            return 2.0 * self.C.T @ r

        constraints: list[dict] = []
        for i in range(self.A_eq.shape[0]):
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, i=i: self.A_eq[i] @ x - self.b_eq[i],
                    "jac": lambda _x, i=i: self.A_eq[i].copy(),
                }
            )
        for i in range(self.A_ineq.shape[0]):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, i=i: self.b_ineq[i] - self.A_ineq[i] @ x,
                    "jac": lambda _x, i=i: -self.A_ineq[i].copy(),
                }
            )

        from scipy.optimize import lsq_linear as _lsq

        x0 = _lsq(self.C, self.d, bounds=(-np.inf, np.inf), lsmr_tol="auto", verbose=0).x

        result = minimize(
            _obj,
            x0,
            method="SLSQP",
            jac=_jac,
            constraints=constraints,
            options={"maxiter": 10000, "ftol": 1e-15},
        )
        return OptimizeResult(
            x=result.x,
            fun=result.fun,
            cost=result.fun,
            success=result.success,
            message=result.message,
        )

    @cached_property
    def x(self) -> np.ndarray:
        """16 维多项式系数向量."""
        return self.R.x

    # ------------------------------------------------------------------ 后处理

    def curve_segments(self) -> dict[str, dict[str, np.ndarray]]:
        """返回四段曲线采样数据（内部上/下、外部上/下）.

        键名：``inner_upper`` / ``inner_lower`` / ``outer_upper`` / ``outer_lower``；
        每段含 ``x``（采样点）和 ``y``（三次多项式求值）。
        """
        x = self.x
        i, j = self.i, self.j
        return {
            "inner_upper": {"x": i, "y": x[0] + x[1] * i + x[2] * i**2 + x[3] * i**3},
            "inner_lower": {"x": i, "y": x[4] + x[5] * i + x[6] * i**2 + x[7] * i**3},
            "outer_upper": {"x": j, "y": x[8] + x[9] * j + x[10] * j**2 + x[11] * j**3},
            "outer_lower": {"x": j, "y": x[12] + x[13] * j + x[14] * j**2 + x[15] * j**3},
        }

    def breakpoint_points(self) -> dict[str, dict[str, float]]:
        """返回断点处曲线坐标（用于在图上标注）."""
        x = self.x
        return {
            "inner": {
                "x": float(self.m),
                "y_upper": float(x[0] + x[1] * self.m + x[2] * self.ms + x[3] * self.mc),
                "y_lower": float(x[4] + x[5] * self.m + x[6] * self.ms + x[7] * self.mc),
            },
            "outer": {
                "x": float(self.m1),
                "y_upper": float(x[8] + x[9] * self.m1 + x[10] * self.m1s + x[11] * self.m1c),
                "y_lower": float(x[12] + x[13] * self.m1 + x[14] * self.m1s + x[15] * self.m1c),
            },
        }

    def calculate_angles(self) -> tuple[float, float, float, float]:
        """计算断点处曲线切线角度（度）.

        返回 (m 上, m 下, m1 上, m1 下)。
        """
        x = self.x
        y3 = x[0] + x[1] * self.m + x[2] * self.ms + x[3] * self.mc
        g3 = x[8] + x[9] * self.m1 + x[10] * self.m1s + x[11] * self.m1c

        angle_m_upper = float(np.degrees(np.arctan2(y3 - x[0], self.m)))
        angle_m_lower = float(np.degrees(np.arctan2(y3 - x[4], self.m)))
        angle_m1_upper = float(np.degrees(np.arctan2(g3 - x[8], self.m1)))
        angle_m1_lower = float(np.degrees(np.arctan2(g3 - x[12], self.m1)))

        logger.debug(
            "计算角度: m_upper=%.2f°, m_lower=%.2f°, m1_upper=%.2f°, m1_lower=%.2f°",
            angle_m_upper,
            angle_m_lower,
            angle_m1_upper,
            angle_m1_lower,
        )
        return angle_m_upper, angle_m_lower, angle_m1_upper, angle_m1_lower


def solve_lsc_curve(  # noqa: PLR0913, PLR0917
    m: float = -1.3,
    m1: float = -2.4,
    s: float = 1.2183,
    s1: float = 8.1,
    H: float = 0.5,
    m2: float = 0.5,
    H1: float = 0.2,
    H2: float = 0.65,
    J: float = 80.0,
    J1: float = 40.0,
) -> dict[str, object]:
    """LSC 曲线优化求解便捷函数（供节点调用，返回可 pickle 结构）.

    返回
    ----
    dict 包含：
    - x: 16 维系数数组（list 形式，pickle 安全）
    - cost: 残差
    - curves: 四段曲线采样数据
    - breakpoints: 断点坐标
    - angles: 四角度
    """
    curve = LSCCurve(m=m, m1=m1, s=s, s1=s1, H=H, m2=m2, H1=H1, H2=H2, J=J, J1=J1)
    segments = curve.curve_segments()
    # 将 numpy 数组转为 list 便于 pickle 和 JSON 序列化
    curves: dict[str, dict[str, list[float]]] = {
        name: {"x": seg["x"].tolist(), "y": seg["y"].tolist()} for name, seg in segments.items()
    }
    angles = curve.calculate_angles()
    return {
        "x": curve.x.tolist(),
        "cost": float(curve.R.cost),
        "curves": curves,
        "breakpoints": curve.breakpoint_points(),
        "angles": {
            "m_upper": angles[0],
            "m_lower": angles[1],
            "m1_upper": angles[2],
            "m1_lower": angles[3],
        },
    }
