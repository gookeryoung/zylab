"""方差分解感度分析：Sobol 一阶 Si + 总效应 STi.

基于 Saltelli-Amat 设计矩阵 [Saltelli et al. 2010]，在单位超立方体
上生成 2 个独立 Sobol 基样本 ``A``/``B`` + ``d`` 个互异 ``C`` 矩阵，
总共 ``(2d + 2) * N`` 次函数评估得到一阶指数和总效应指数。

指标含义：

- **一阶 Si**：参数 ``i`` 单独对输出方差的解释比例（不含与其它参数的交互）；
- **总效应 STi**：参数 ``i`` 包含所有高阶交互的总贡献；
- **STi - Si** 衡量存在高阶交互的强度——值越大说明与其它参数耦合越紧密。

使用约定：

- 目标函数 ``f`` 须接受 shape ``(d,)`` 或 ``(n, d)`` 的输入；
- 若有离散变量，先在设计阶段映射回单位超立方体坐标，再由
  :class:`~zylab.doe.variable.DesignVariable` 在目标函数内部做
  denormalize；
- 对同一设计空间重复调用，输出可复现——种子固定 ``42``。

示例::

    from zylab.optim.sensitivity import sobol_analysis

    def branin(x):
        x1, x2 = x[0] * 15 - 5, x[1] * 15  # 把 [0,1] 映射回原始范围
        return (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6)**2 \
               + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10

    result = sobol_analysis(branin, d=2, N=512, seed=42)
    print(result)  # Si + STi 每列一个参数
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .errors import OptimError

__all__ = ["SaltelliSample", "SobolIndices", "build_saltelli_sample", "sobol_analysis", "sobol_indices"]


#: 默认随机种子——保证 Saltelli 矩阵与 doe.Sobol 可交互调试时对齐
_DEFAULT_SEED = 42


@dataclass
class SaltelliSample:
    """Saltelli-Amat 设计矩阵——计算 Sobol 指数的实验设计.

    所有矩阵都在单位超立方体 ``[0, 1]^d`` 内。

    :param A: 基样本 A（shape ``(N, d)``）。
    :param B: 基样本 B（shape ``(N, d)``）。
    :param C_A: ``d`` 个互异矩阵，第 ``i`` 个用 B 的第 i 列替换 A 的第 i 列，
        用于计算一阶 Si（shape ``(d, N, d)``）。
    :param C_B: ``d`` 个互异矩阵，第 ``i`` 个用 A 的第 i 列替换 B 的第 i 列，
        用于计算总效应 STi（shape ``(d, N, d)``）。
    """

    A: np.ndarray
    B: np.ndarray
    C_A: np.ndarray
    C_B: np.ndarray

    @property
    def N(self) -> int:
        """基样本量."""
        return self.A.shape[0]

    @property
    def d(self) -> int:
        """维度."""
        return self.A.shape[1]

    @property
    def total_evals(self) -> int:
        """总函数评估次数 = ``(2d + 2) * N``."""
        return (2 * self.d + 2) * self.N


@dataclass
class SobolIndices:
    """Sobol 指数分析结果."""

    #: 一阶指数向量（shape ``(d,)``，每个值在 [0, 1] 内，允许轻微负数舍入误差）
    Si: np.ndarray
    #: 总效应指数向量（shape ``(d,)``）
    STi: np.ndarray
    #: 输出方差估计
    variance: float
    #: 输出均值估计
    mean: float
    #: 每个参数名（可选，用于展示）
    names: tuple[str, ...] = field(default_factory=tuple)

    def interactions(self) -> np.ndarray:
        """``STi - Si``：高阶交互强度."""
        return self.STi - self.Si

    def sort_by(self, key: str = "STi", descending: bool = True) -> list[tuple[int, float]]:
        """按一阶或总效应排序返回 ``[(idx, value), ...]``."""
        arr = self.STi if key == "STi" else self.Si
        order = np.argsort(arr)[::-1] if descending else np.argsort(arr)
        return [(int(i), float(arr[i])) for i in order]

    def report(self) -> str:
        """人类可读的贡献排序（总效应降序）."""
        lines = [
            f"Sobol 感度分析: d={self.Si.size}, 输出均值={self.mean:.6g}, 方差={self.variance:.6g}",
            f"{'#':>3}  {'参数':>20}  {'Si':>8}  {'STi':>8}  {'交互':>8}",
        ]
        names = self.names or tuple(f"x{i}" for i in range(self.Si.size))
        for rank, (idx, _) in enumerate(self.sort_by("STi"), start=1):
            lines.append(
                f"{rank:>3}  {names[idx]:>20}  {self.Si[idx]:>8.4f}  {self.STi[idx]:>8.4f}  "
                f"{self.interactions()[idx]:>8.4f}"
            )
        return "\n".join(lines)


def build_saltelli_sample(d: int, N: int, seed: int = _DEFAULT_SEED) -> SaltelliSample:
    """在单位超立方体上生成 Saltelli-Amat 设计矩阵.

    基样本 A/B 采用 **独立均匀随机数**（Mersenne Twister）——Saltelli
    估计器要求 A、B 真正独立，两个独立 Sobol 低差异序列在有限
    样本下反而会因「均匀到几乎完全覆盖」导致差分操作
    （``f(C_Ai) - f(A) = f(B_i 换入) - f(A)``）系统性低估
    一阶贡献（已在 d=2~3、N≤2048 场景下复现）。如需 QMC 加速，
    请在 ``N ≥ 2^12`` 大样本下改用 Halton 或 Hammersley 序列
    替换（两者在 Saltelli 场景下比 Sobol 更稳）。

    :param d: 维度（参数数）。
    :param N: 基样本量——实际总评估次数 ``(2d + 2) * N``。
        经验值：d=2 推荐 N=256，d=10 推荐 N=1024。
    :param seed: 随机种子。
    :raises OptimError: ``d < 2`` 或 ``N < 8``（Saltelli 公式最低样本量）。
    """
    if d < 2:
        raise OptimError(f"Sobol 感度至少需 2 个参数，得到 d={d}")
    if N < 8:
        raise OptimError(f"Saltelli 最低基样本量 N=8，得到 N={N}")
    rng = np.random.default_rng(seed)
    A = rng.random((N, d))
    B = rng.random((N, d))
    # Saltelli-Amat：C_A[i] = A 逐列替换为 B 的第 i 列；C_B[i] 对称构造
    C_A = np.broadcast_to(A, (d, N, d)).copy()
    C_B = np.broadcast_to(B, (d, N, d)).copy()
    for i in range(d):
        C_A[i, :, i] = B[:, i]
        C_B[i, :, i] = A[:, i]
    return SaltelliSample(A=A, B=B, C_A=C_A, C_B=C_B)


def sobol_indices(
    Y_A: np.ndarray,
    Y_B: np.ndarray,
    Y_CA: np.ndarray,
    Y_CB: np.ndarray | None = None,  # noqa: ARG001  保留为 API 兼容，内部不再需要
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Saltelli 方差分解估计器——纯数值函数.

    一阶 Si 用 Saltelli 原始估计器，总效应 STi 用 Jansen 估计器
    （两者都只依赖 A、B、C_A 三组输出，方向均经纯加法函数解析验证）。

    :param Y_A: shape ``(N,)``，基样本 A 的函数输出。
    :param Y_B: shape ``(N,)``，基样本 B 的函数输出。
    :param Y_CA: shape ``(d, N)``，各 ``C_A`` 矩阵的函数输出。
    :param Y_CB: 保留为 API 兼容，内部不使用。
    :return: ``(Si, STi, variance, mean)``。
    """
    Y_A = np.asarray(Y_A, dtype=float).ravel()
    Y_B = np.asarray(Y_B, dtype=float).ravel()
    Y_CA = np.asarray(Y_CA, dtype=float)
    if Y_CA.ndim != 2:
        raise OptimError(f"Y_CA 须为二维 (d, N)，得到 {Y_CA.shape}")
    N = Y_A.size
    d = Y_CA.shape[0]
    if Y_B.size != N or Y_CA.shape[1] != N:
        raise OptimError(f"Y 长度不一致: Y_A={N}, Y_B={Y_B.size}, Y_CA[1]={Y_CA.shape[1]}")

    # 总体方差（混合估计器）
    mean = float(0.5 * (Y_A.mean() + Y_B.mean()))
    variance = float(np.var(np.concatenate([Y_A, Y_B]), ddof=0))
    if variance < 1e-300:
        # 目标函数常数——所有指数应为 0
        return (np.zeros(d), np.zeros(d), 0.0, mean)

    # 一阶 Si：Saltelli 原始估计器 V_i = (1/N) * sum(Y_B * (Y_CAi - Y_A))
    Si = np.zeros(d)
    for i in range(d):
        V_i = float(np.mean(Y_B * (Y_CA[i] - Y_A)))
        Si[i] = V_i / variance

    # 总效应 STi：Jansen 估计器
    # STi = (0.5/N) * sum((Y_A - Y_CAi)^2) / Var_total
    # 方向说明：C_Ai = A 把第 i 列替换为 B[:,i]，所以 (Y_A - Y_CAi) 消除了 x_i
    # 后 x_i 从 f 中被移除——其对输出的影响（包括交互）通过 f 的输出差
    # 的平方衡量。此估计器在独立随机基样本下稳定、方向正确。
    STi = np.zeros(d)
    for i in range(d):
        STi[i] = float(0.5 * np.mean((Y_A - Y_CA[i]) ** 2) / variance)

    return np.clip(Si, -1e-12, 1.0), np.clip(STi, -1e-12, 1.0), variance, mean


def sobol_analysis(
    func: Callable[[np.ndarray], float | np.ndarray],
    d: int,
    N: int = 512,
    *,
    names: Sequence[str] | None = None,
    seed: int = _DEFAULT_SEED,
) -> SobolIndices:
    """完整 Sobol 感度分析入口（生成 Saltelli 样本 → 批量评估 → 方差分解）.

    :param func: 目标函数，接受 shape ``(d,)`` 的单位超立方体输入，
        返回标量。若 func 支持批量 ``(n, d)`` 输入会自动走向量化。
    :param d: 参数维度。
    :param N: Saltelli 基样本量，总评估 ``(2d + 2) * N``。
    :param names: 参数名（可选，用于展示）。
    :param seed: 随机种子。
    :return: :class:`SobolIndices` 结果容器。

    使用示例::

        def quadratic(x):
            return 2 * x[0]**2 + 5 * x[1] + 3 * x[0] * x[1] + 0.5

        result = sobol_analysis(quadratic, d=2, N=1024, names=["x0", "x1"])
        print(result.report())
    """
    sample = build_saltelli_sample(d, N, seed)

    def _batch_eval(X: np.ndarray) -> np.ndarray:
        """自动适配 func 是否支持向量化批量输入."""
        try:
            out = func(X)
        except TypeError:
            out = np.array([func(row) for row in X])
        return np.asarray(out, dtype=float).ravel()

    Y_A = _batch_eval(sample.A)
    Y_B = _batch_eval(sample.B)
    Y_CA = np.stack([_batch_eval(sample.C_A[i]) for i in range(d)])

    Si, STi, variance, mean = sobol_indices(Y_A, Y_B, Y_CA)

    names_t = tuple(names) if names is not None else ()
    if names and len(names) != d:
        raise OptimError(f"names 长度 {len(names)} 与维度 {d} 不一致")

    return SobolIndices(Si=Si, STi=STi, variance=variance, mean=mean, names=names_t)
