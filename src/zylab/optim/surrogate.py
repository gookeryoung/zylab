"""代理模型：从 DOE 样本拟合目标函数的低成本替身.

两种实现：

- :class:`GprSurrogate` — 高斯过程回归（GPR），输出带预测方差，
  供主动学习/探索利用（PI/EI 采集函数）。
- :class:`RbfSurrogate` — 径向基函数插值（scipy RBFInterpolator），
  纯确定性，训练更快，适合样本量中等时快速建代理。

统一接口 ``fit(X, y)`` / ``predict(X)``，与
:mod:`~zylab.doe` 的 :class:`~zylab.doe.design_space.DesignSpace`
通过变量边界显式对齐——训练/预测都在 **实际值空间**（变量的
lower/upper 区间内）进行，而非 [0,1] 单位超立方体，用户无需
手动 denormalize。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from typing_extensions import override

from .errors import SurrogateError

__all__ = ["GprSurrogate", "RbfSurrogate", "Surrogate"]


class Surrogate(ABC):
    """代理模型抽象基类."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        """用 (n_samples, d) 输入矩阵 + (n_samples,) 目标向量拟合."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 (m, d) 输入的目标值，返回 shape (m,) 的向量."""

    def predict_std(self, X: np.ndarray) -> np.ndarray | None:
        """预测标准差——GPR 返回值，RBF 返回 None."""
        del X  # 基类默认实现不使用 X
        return None

    # ------------------------------------------------------------------ 内部校验

    @staticmethod
    def _check_shapes(X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise SurrogateError(f"X 须为 2D 矩阵 (n_samples, d)，得到 {X.shape}")
        if y.ndim != 1:
            raise SurrogateError(f"y 须为 1D 向量 (n_samples,)，得到 {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise SurrogateError(f"X 与 y 样本数不一致: X={X.shape[0]}, y={y.shape[0]}")


@dataclass
class GprSurrogate(Surrogate):
    """高斯过程回归代理模型（sklearn GaussianProcessRegressor）.

    :param length_scale: RBF 核初始长度尺度（自动优化；None 由 sklearn 默认）。
    :param normalize_y: 是否对训练 y 做标准化（GPR 推荐开启）。
    :param alpha: 观测噪声方差（GPR nugget），默认 1e-10 处理奇异矩阵。
    """

    length_scale: float | None = None
    normalize_y: bool = True
    alpha: float = 1e-10

    _gp: GaussianProcessRegressor | None = None

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> GprSurrogate:
        self._check_shapes(X, y)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=self.length_scale or 1.0, length_scale_bounds=(1e-2, 1e2)
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=self.normalize_y,
            alpha=self.alpha,
            n_restarts_optimizer=10,
            random_state=42,
        )
        self._gp.fit(X, y)
        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._gp is None:
            raise SurrogateError("GprSurrogate 尚未 fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise SurrogateError(f"X 须为 2D 矩阵 (m, d)，得到 {X.shape}")
        return np.asarray(self._gp.predict(X), dtype=float)

    @override
    def predict_std(self, X: np.ndarray) -> np.ndarray:
        if self._gp is None:
            raise SurrogateError("GprSurrogate 尚未 fit")
        result = self._gp.predict(np.asarray(X, dtype=float), return_std=True)
        std = result[1]  # sklearn predict 在 return_std=True 时返回 (y, std) 元组
        return np.asarray(std, dtype=float)

    @property
    def kernel_(self):
        """拟合后的核（调试/可视化用）."""
        if self._gp is None:
            raise SurrogateError("GprSurrogate 尚未 fit")
        return self._gp.kernel_


@dataclass
class RbfSurrogate(Surrogate):
    """RBF 径向基函数插值代理（scipy.interpolate.RBFInterpolator）.

    纯确定性——不返回方差——适合样本量中等（几十个）、追求快训
    练速度的场景。大样本或噪声较大时 GPR 更合适。

    :param kernel: RBF 核类型（scipy 默认 thin_plate_spline）。
    :param smoothing: 正则化系数（0.0 = 精确插值；> 0 容错噪）。
    """

    kernel: str = "thin_plate_spline"
    smoothing: float = 0.0

    _rbf: Callable[[np.ndarray], np.ndarray] | None = None

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> RbfSurrogate:
        self._check_shapes(X, y)
        from scipy.interpolate import RBFInterpolator

        self._rbf = RBFInterpolator(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
            kernel=self.kernel,
            smoothing=self.smoothing,
        )
        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._rbf is None:
            raise SurrogateError("RbfSurrogate 尚未 fit")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise SurrogateError(f"X 须为 2D 矩阵 (m, d)，得到 {X.shape}")
        return np.asarray(self._rbf(X), dtype=float)
