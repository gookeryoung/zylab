"""结构可靠性分析：FORM / SORM 极限状态求解.

核心思想：在标准化正态空间（U 空间）中搜索距离原点最近且落在极限状态面
``g(X)=0`` 上的设计点 ``u*``。该最短距离即为 **可靠性指数 β**，近似下
失效概率 ``P_f ≈ Φ(-β)``。

FORM（First-Order Reliability Method）在设计点处线性化极限状态面，用
梯度做 HLRF 迭代收敛到 β。当极限状态面曲率较大时，FORM 会低估失效概率；
SORM（Second-Order Reliability Method）在 FORM 基础上捕获设计点处的曲率
信息，通过 Breitung / Hohenbichler 公式修正 ``P_f``。

本文件只做**单极限状态函数、独立随机变量、HLRF 迭代**——不耦合
相关变量（Nataf 变换）、不做多极限状态系统可靠性、不直接跑 FE
（极限状态函数是一个纯 callable，FE 结果可作为其参数）。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from .errors import ReliabilityError

__all__ = [
    "Distribution",
    "FORMResult",
    "MCResult",
    "RandomVariable",
    "SORMResult",
    "form_analysis",
    "mc_analysis",
    "sorm_analysis",
]

#: 极限状态函数签名：``g(X) -> float``，G<0 表示失效面内侧.
LimitStateFn = Callable[[np.ndarray], float]
#: 梯度函数签名：``grad_g(X) -> np.ndarray``，长度 = 随机变量数.
GradientFn = Callable[[np.ndarray], np.ndarray]


class Distribution(str, Enum):
    """支持的边际分布类型."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GUMBEL = "gumbel"
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    # 允许用户直接提供 ``(u -> x)`` / ``(x -> u)`` 变换 callable，绕过 scipy
    CUSTOM = "custom"


@dataclass(frozen=True)
class RandomVariable:
    """描述一个随机输入变量（独立，边际分布已知）.

    :param name: 变量名，用于报告.
    :param dist: :class:`Distribution` 枚举值.
    :param params: 分布参数 dict，按 dist 类型不同：

        - ``NORMAL``: ``{"loc": μ, "scale": σ}``
        - ``LOGNORMAL``: ``{"loc": μ_X, "scale": σ_X}``（物理空间均值/标准差）
        - ``GUMBEL``: ``{"loc": μ, "scale": σ}``
        - ``WEIBULL``: ``{"shape": k, "scale": λ}``
        - ``EXPONENTIAL``: ``{"loc": a, "scale": b}``
        - ``UNIFORM``: ``{"lo": a, "hi": b}``
        - ``CUSTOM``: ``{"cdf": Callable[[float], float], "ppf": Callable[[float], float]}``
          两个 callable 分别是物理空间的 CDF 和它的逆（分位数函数）.
    """

    name: str
    dist: Distribution
    params: Mapping[str, Any]


# ---------- 分布实现（Rosenblatt 变换 / 逆 Rosenblatt） ----------


def _standard_to_physical(u: float, rv: RandomVariable) -> float:  # noqa: PLR0911
    """U ~ N(0,1) → X（物理空间）."""
    Phi = stats.norm.cdf(u)

    if rv.dist == Distribution.NORMAL:
        return float(stats.norm.ppf(Phi, loc=rv.params["loc"], scale=rv.params["scale"]))

    if rv.dist == Distribution.LOGNORMAL:
        # 先转 LN 的 (mu_ln, sigma_ln)
        mu_x = rv.params["loc"]
        sigma_x = rv.params["scale"]
        sigma_ln = math.sqrt(math.log(1 + sigma_x**2 / mu_x**2))
        mu_ln = math.log(mu_x) - sigma_ln**2 / 2
        return float(stats.lognorm.ppf(Phi, s=sigma_ln, scale=math.exp(mu_ln)))

    if rv.dist == Distribution.GUMBEL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.gumbel_r.ppf(Phi, loc=loc, scale=scale))

    if rv.dist == Distribution.WEIBULL:
        k = rv.params["shape"]
        lam = rv.params["scale"]
        return float(stats.weibull_min.ppf(Phi, c=k, scale=lam))

    if rv.dist == Distribution.EXPONENTIAL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.expon.ppf(Phi, loc=loc, scale=scale))

    if rv.dist == Distribution.UNIFORM:
        lo, hi = rv.params["lo"], rv.params["hi"]
        return float(stats.uniform.ppf(Phi, loc=lo, scale=hi - lo))

    if rv.dist == Distribution.CUSTOM:
        return float(rv.params["ppf"](Phi))

    raise ReliabilityError(f"不支持的分布: {rv.dist}")


def _physical_to_standard(x: float, rv: RandomVariable) -> float:  # noqa: PLR0911
    """X（物理空间）→ U ~ N(0,1)."""
    if rv.dist == Distribution.NORMAL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.norm.ppf(stats.norm.cdf(x, loc=loc, scale=scale)))

    if rv.dist == Distribution.LOGNORMAL:
        mu_x = rv.params["loc"]
        sigma_x = rv.params["scale"]
        sigma_ln = math.sqrt(math.log(1 + sigma_x**2 / mu_x**2))
        mu_ln = math.log(mu_x) - sigma_ln**2 / 2
        return float(stats.norm.ppf(stats.lognorm.cdf(x, s=sigma_ln, scale=math.exp(mu_ln))))

    if rv.dist == Distribution.GUMBEL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.norm.ppf(stats.gumbel_r.cdf(x, loc=loc, scale=scale)))

    if rv.dist == Distribution.WEIBULL:
        k = rv.params["shape"]
        lam = rv.params["scale"]
        return float(stats.norm.ppf(stats.weibull_min.cdf(x, c=k, scale=lam)))

    if rv.dist == Distribution.EXPONENTIAL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.norm.ppf(stats.expon.cdf(x, loc=loc, scale=scale)))

    if rv.dist == Distribution.UNIFORM:
        lo, hi = rv.params["lo"], rv.params["hi"]
        return float(stats.norm.ppf(stats.uniform.cdf(x, loc=lo, scale=hi - lo)))

    if rv.dist == Distribution.CUSTOM:
        return float(stats.norm.ppf(rv.params["cdf"](x)))

    raise ReliabilityError(f"不支持的分布: {rv.dist}")


def _from_standard(u_std: np.ndarray, variables: Sequence[RandomVariable]) -> np.ndarray:
    """标准化正态空间向量 → 物理空间向量."""
    return np.array([_standard_to_physical(float(u), rv) for u, rv in zip(u_std, variables, strict=False)])


# ---------- 梯度辅助 ----------


def _finite_diff(
    limit_state: LimitStateFn,
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    """中心差分估计 limit_state 在 x 处的梯度（物理空间）."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        dx = h * max(1.0, abs(x[i]))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += dx
        x_minus[i] -= dx
        grad[i] = (limit_state(x_plus) - limit_state(x_minus)) / (2 * dx)
    return grad


def _grad_x_to_u(
    grad_x: np.ndarray,
    x: np.ndarray,
    variables: Sequence[RandomVariable],
) -> np.ndarray:
    """物理空间梯度 → 标准化空间梯度.

    链式法则：∂G/∂u_i = ∂G/∂x_i · ∂x_i/∂u_i
    其中 ∂x_i/∂u_i = φ(u_i) / f_{X_i}(x_i)（Rosenblatt 变换的 Jacobian）.
    """
    grad_u = np.zeros_like(grad_x)
    for i, (xi, rv) in enumerate(zip(x, variables, strict=False)):
        u = _physical_to_standard(float(xi), rv)
        phi_u = stats.norm.pdf(u)
        f_x = _pdf_physical(float(xi), rv)
        if f_x < 1e-15:
            grad_u[i] = 0.0
        else:
            grad_u[i] = grad_x[i] * phi_u / f_x
    return grad_u


def _pdf_physical(x: float, rv: RandomVariable) -> float:  # noqa: PLR0911
    """变量在物理空间的 PDF."""
    if rv.dist == Distribution.NORMAL:
        return float(stats.norm.pdf(x, loc=rv.params["loc"], scale=rv.params["scale"]))
    if rv.dist == Distribution.LOGNORMAL:
        mu_x = rv.params["loc"]
        sigma_x = rv.params["scale"]
        sigma_ln = math.sqrt(math.log(1 + sigma_x**2 / mu_x**2))
        mu_ln = math.log(mu_x) - sigma_ln**2 / 2
        return float(stats.lognorm.pdf(x, s=sigma_ln, scale=math.exp(mu_ln)))
    if rv.dist == Distribution.GUMBEL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.gumbel_r.pdf(x, loc=loc, scale=scale))
    if rv.dist == Distribution.WEIBULL:
        return float(stats.weibull_min.pdf(x, c=rv.params["shape"], scale=rv.params["scale"]))
    if rv.dist == Distribution.EXPONENTIAL:
        loc, scale = rv.params["loc"], rv.params["scale"]
        return float(stats.expon.pdf(x, loc=loc, scale=scale))
    if rv.dist == Distribution.UNIFORM:
        lo, hi = rv.params["lo"], rv.params["hi"]
        return float(stats.uniform.pdf(x, loc=lo, scale=hi - lo))
    if rv.dist == Distribution.CUSTOM:
        pdf = rv.params.get("pdf")
        if pdf is not None:
            return float(pdf(x))
        return float(stats.norm.pdf(x))
    return 0.0


# ---------- FORM 结果 ----------


@dataclass(frozen=True)
class FORMResult:
    """HLRF FORM 分析结果.

    :param beta: 可靠性指数（标准化空间设计点到原点的距离）.
    :param pf_form: FORM 近似失效概率 ``Φ(-β)``.
    :param u_star: 设计点（标准化正态空间向量）.
    :param x_star: 设计点（物理空间向量）.
    :param g_star: 设计点处极限状态函数值（应 ≈ 0）.
    :param converged: 是否收敛.
    :param n_iter: HLRF 迭代次数.
    :param variables: 输入随机变量列表（保留引用方便下游画图）.
    """

    beta: float
    pf_form: float
    u_star: np.ndarray
    x_star: np.ndarray
    g_star: float
    converged: bool
    n_iter: int
    variables: Sequence[RandomVariable]


@dataclass(frozen=True)
class SORMResult:
    """SORM 修正结果.

    :param form: 底层 FORM 结果.
    :param pf_breitung: Breitung 公式修正后的失效概率.
    :param pf_hohenbichler: Hohenbichler 公式修正后的失效概率（曲率信息更完整时用）.
    :param kappa: 设计点处主曲率数组（长度 = N-1，N = 变量数）.
    """

    form: FORMResult
    pf_breitung: float
    pf_hohenbichler: float
    kappa: np.ndarray


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo 可靠性分析结果.

    :param pf: 失效概率估计值 ``P̂_f = N_fail / N``.
    :param n_samples: 总样本数.
    :param n_fail: 失效样本数.
    :param beta: 可靠性指数 ``β = -Φ^{-1}(P̂_f)``（P̂_f 过小时设为 inf）.
    :param variables: 随机变量列表.
    :param method: 抽样方法——``crude``（独立正态）/ ``lhc``（拉丁超立方）/
        ``sobol``（Sobol 低差异序列）.
    :param cov: ``P̂_f`` 的变异系数 ``√(P̂_f·(1-P̂_f)/N) / P̂_f``——反映 MC 估计的相对精度.
    :param pf_ci_95: ``P̂_f`` 的 95% 置信区间 (下, 上).
    """

    pf: float
    n_samples: int
    n_fail: int
    beta: float
    variables: Sequence[RandomVariable]
    method: str
    cov: float
    pf_ci_95: tuple[float, float]


# ---------- FORM 主算法 ----------


def form_analysis(  # noqa: PLR0913
    limit_state: LimitStateFn,
    variables: Sequence[RandomVariable],
    *,
    grad: GradientFn | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    start_std: np.ndarray | None = None,
) -> FORMResult:
    """HLRF FORM 迭代求可靠性指数 β.

    :param limit_state: 极限状态函数 ``g(X)``，返回值 >0 表示安全域.
    :param variables: 随机变量列表.
    :param grad: 解析梯度 ``∂g/∂X``（物理空间），不传则用中心差分估计.
    :param max_iter: HLRF 最大迭代次数.
    :param tol: 设计点收敛判据（``||u_new - u_old||``）.
    :param start_std: 迭代起点（标准化空间），默认全 0（各变量均值处）.
    :return: :class:`FORMResult`.

    :raises ReliabilityError: 迭代发散 / 梯度为零向量 / 极限状态函数返回 NaN.

    经典 HLRF 迭代公式（Hasofer & Lind, 1974）：

    .. math::

        u_{k+1} = \\frac{G(u_k) - \
abla G_{u_k}^T u_k}{\\|\\nabla G_{u_k}\\|} \\cdot \
abla G_{u_k} / \\|\\nabla G_{u_k}\\|

    其中 :math:`\\nabla G_u` 是极限状态函数在标准化空间的梯度.
    """
    n = len(variables)
    if n == 0:
        raise ReliabilityError("variables 不能为空")

    if start_std is None:
        u = np.zeros(n)
    else:
        u = np.asarray(start_std, dtype=float).copy()

    prev_beta = float("inf")

    for it in range(max_iter):
        x = _from_standard(u, variables)
        G = float(limit_state(x))

        if not np.isfinite(G):
            raise ReliabilityError(f"极限状态函数在设计点返回非有限值: G={G}")

        # 物理空间梯度
        if grad is not None:
            grad_x = np.asarray(grad(x), dtype=float)
        else:
            grad_x = _finite_diff(limit_state, x)

        # 物理空间 → 标准化空间梯度
        grad_u = _grad_x_to_u(grad_x, x, variables)

        norm_grad = np.linalg.norm(grad_u)
        if norm_grad < 1e-14:
            raise ReliabilityError(
                f"设计点处梯度为零向量（第 {it + 1} 次迭代），HLRF 无法继续；检查极限状态函数是否平坦或起点是否合适"
            )

        alpha = grad_u / norm_grad  # 单位方向，指向 G 增大方向
        beta_new = (G - float(np.dot(grad_u, u))) / norm_grad
        u_new = -beta_new * alpha

        # 收敛判据：设计点位置 + beta 变化都小
        du = np.linalg.norm(u_new - u)
        db = abs(beta_new - prev_beta)
        u = u_new
        prev_beta = beta_new

        if du < tol * max(1.0, float(abs(beta_new))) and db < tol * max(1.0, float(abs(beta_new))):
            break

    # 最终设计点处精确评估
    x_star = _from_standard(u, variables)
    g_star = float(limit_state(x_star))
    beta = float(np.linalg.norm(u))
    pf = float(stats.norm.cdf(-beta))

    converged = bool(du < tol * max(1.0, float(abs(beta))) or it < max_iter - 1)

    return FORMResult(
        beta=beta,
        pf_form=pf,
        u_star=u,
        x_star=x_star,
        g_star=g_star,
        converged=converged,
        n_iter=it + 1,
        variables=list(variables),
    )


# ---------- SORM ----------


def sorm_analysis(
    limit_state: LimitStateFn,
    variables: Sequence[RandomVariable],
    *,
    grad: GradientFn | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> SORMResult:
    """FORM + Breitung 曲率修正的二阶可靠性分析.

    流程：

    1. 跑 :func:`form_analysis` 拿 FORM 设计点 ``u*``.
    2. 在 ``u*`` 处构造标准化空间的 Hessian（用二阶有限差分）.
    3. Hessian 投影到与梯度正交的超平面，得到 :math:`N-1` 维约化 Hessian.
    4. 约化 Hessian 的特征值就是主曲率 κ_i.
    5. Breitung 公式：:math:`P_{f,\\text{SORM}} ≈ Φ(-β) · Π (1 - β κ_i)^{-1/2}`.

    Hohenbichler 公式在每个曲率绝对值都 << 1 时与 Breitung 非常接近，
    这里一并输出作为 sanity check.
    """
    form_result = form_analysis(limit_state, variables, grad=grad, max_iter=max_iter, tol=tol)

    u_star = form_result.u_star
    x_star = form_result.x_star
    n = len(u_star)

    # 物理空间 Hessian（有限差分）
    # 物理 → 标准化空间 Hessian：H_u = J^T H_x J + 对角修正项
    #   为简洁，这里直接在标准化空间做二阶有限差分（避免 Jacobian 二阶导数）
    H_u = _hessian_finite_diff(lambda u_vec: limit_state(_from_standard(u_vec, variables)), u_star)

    # 梯度方向（用于构造投影矩阵 P = I - α α^T）
    if grad is not None:
        grad_x = np.asarray(grad(x_star), dtype=float)
    else:
        grad_x = _finite_diff(limit_state, x_star)
    grad_u = _grad_x_to_u(grad_x, x_star, variables)
    norm_g = np.linalg.norm(grad_u)
    if norm_g < 1e-14:
        kappa = np.array([])
        pf_breitung = form_result.pf_form
        pf_hohenbichler = form_result.pf_form
    else:
        alpha = grad_u / norm_g

        # 投影矩阵（从 R^N 到与 grad 正交的子空间 R^{N-1}）
        if n == 1:
            kappa = np.array([])
            pf_breitung = form_result.pf_form
            pf_hohenbichler = form_result.pf_form
        else:
            P = np.eye(n) - np.outer(alpha, alpha)
            # 约化 Hessian: H_red = P H_u P 在子空间上的表示
            # 取特征向量 Q 张成子空间: H_red = Q^T H_u Q
            eigvals, eigvecs = np.linalg.eigh(P)
            subspace_mask = eigvals > 0.5  # 正交子空间
            Q = eigvecs[:, subspace_mask]  # (N, N-1)
            H_red = Q.T @ H_u @ Q  # (N-1, N-1)
            evals = np.linalg.eigvalsh(H_red)
            # 主曲率 κ = 特征值 / ||∇G_u||（曲率的定义）
            kappa = evals / norm_g

            # Breitung 公式
            prod = 1.0
            for ki in kappa:
                denom = 1.0 - form_result.beta * ki
                if denom <= 0:
                    # 曲率太大，Breitung 公式不适用（通常是极限状态面在设计点凹陷）
                    prod = float("nan")
                    break
                prod *= 1.0 / math.sqrt(denom)
            pf_breitung = form_result.pf_form * prod if math.isfinite(prod) else form_result.pf_form

            # Hohenbichler 近似（曲率小时与 Breitung 一致）
            # pf = Φ(-β) · exp(Σ (β^2 κ_i^2)/(2(1 - β κ_i)))
            # 为避免极端情况，只在 |β κ_i| < 0.8 时启用
            safe = True
            for ki in kappa:
                if abs(form_result.beta * ki) >= 0.8:
                    safe = False
                    break
            if safe and len(kappa) > 0:
                term = sum(form_result.beta**2 * ki**2 / (2.0 * (1.0 - form_result.beta * ki)) for ki in kappa)
                pf_hohenbichler = form_result.pf_form * math.exp(term)
            else:
                pf_hohenbichler = pf_breitung

    return SORMResult(
        form=form_result,
        pf_breitung=float(pf_breitung),
        pf_hohenbichler=float(pf_hohenbichler),
        kappa=kappa,
    )


# ---------- Hessian 辅助 ----------


def _hessian_finite_diff(fn: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """二阶中心差分估计 fn 在 x 处的 Hessian 矩阵."""
    n = len(x)
    H = np.zeros((n, n))
    h = 1e-5
    for i in range(n):
        for j in range(i, n):
            dx_i = h * max(1.0, abs(x[i]))
            dx_j = h * max(1.0, abs(x[j]))
            xpp = x.copy()
            xpp[i] += dx_i
            xpp[j] += dx_j
            xpm = x.copy()
            xpm[i] += dx_i
            xpm[j] -= dx_j
            xmp = x.copy()
            xmp[i] -= dx_i
            xmp[j] += dx_j
            xmm = x.copy()
            xmm[i] -= dx_i
            xmm[j] -= dx_j
            val = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * dx_i * dx_j)
            H[i, j] = val
            H[j, i] = val
    return H


# ---------- Monte Carlo 可靠性分析 ----------


def mc_analysis(
    limit_state: LimitStateFn,
    variables: Sequence[RandomVariable],
    *,
    n_samples: int = 100_000,
    method: str = "crude",
    seed: int = 0,
) -> MCResult:
    """Monte Carlo reliability analysis - estimate Pf and beta.

    Generate standard-normal (U-space) samples, transform to physical space via
    Rosenblatt, evaluate limit state, count g<0 as failure.

    :param limit_state: g(X) -> float, G<0 = failure.
    :param variables: independent random variables.
    :param n_samples: total samples (default 100k). Use >= 1e6 for Pf around 1e-3
        to keep relative error below 10 percent.
    :param method: sampling - 'crude' (independent normal), 'lhc' (Latin Hypercube),
        'sobol' (low-discrepancy, n_samples auto-rounded up to 2^k).
    :param seed: RNG seed.
    :return: MCResult with Pf, beta, 95 percent CI, COV.
    :raises ValueError: bad method / n_samples <= 0 / empty variables.
    """
    if n_samples <= 0:
        raise ValueError("mc_analysis: n_samples must be positive")
    if not variables:
        raise ValueError("mc_analysis: variables must not be empty")
    if method not in ("crude", "lhc", "sobol"):
        raise ValueError(f"mc_analysis: unsupported method '{method}'")

    D = len(variables)
    rng = np.random.default_rng(seed)

    # ---- U-space sampling ----
    if method == "crude":
        U = rng.standard_normal((n_samples, D))
    elif method == "lhc":
        from scipy.stats import qmc as _qmc

        lhs = _qmc.LatinHypercube(D, seed=seed)
        U = stats.norm.ppf(lhs.random(n_samples))
    else:  # sobol
        k = math.ceil(math.log2(max(n_samples, 1)))
        n2 = 1 << k  # 2^k
        from scipy.stats import qmc as _qmc

        sampler = _qmc.Sobol(D, scramble=True, seed=seed)
        U = stats.norm.ppf(sampler.random(n2))[:n_samples]

    # ---- U -> X (physical space) ----
    X = np.array([_from_standard(U[j], variables) for j in range(n_samples)])

    # ---- Evaluate limit state ----
    g_vals = np.array([limit_state(X[j]) for j in range(n_samples)])
    n_fail = int(np.sum(g_vals < 0))
    pf = n_fail / n_samples

    # ---- Derived stats ----
    if pf <= 0:
        beta = math.inf
    elif pf >= 1.0:
        beta = -math.inf
    else:
        beta = float(-stats.norm.ppf(pf))

    pf_var = pf * (1.0 - pf) / n_samples
    cov = float(np.sqrt(pf_var) / pf) if pf > 0 else float("inf")

    # 95 percent Wilson score CI
    if n_fail == 0:
        ci_lo = 0.0
        ci_hi = 1.0 - (0.05 / n_samples)
    elif n_fail == n_samples:
        ci_lo = 0.05 / n_samples
        ci_hi = 1.0
    else:
        z = stats.norm.ppf(0.975)
        denom = 1.0 + z**2 / n_samples
        center = pf + z**2 / (2 * n_samples)
        spread = z * np.sqrt(pf * (1.0 - pf) / n_samples + z**2 / (4 * n_samples**2))
        ci_lo = float((center - spread) / denom)
        ci_hi = float((center + spread) / denom)
        ci_lo = max(0.0, min(1.0, ci_lo))
        ci_hi = max(0.0, min(1.0, ci_hi))

    return MCResult(
        pf=pf,
        n_samples=n_samples,
        n_fail=n_fail,
        beta=beta,
        variables=tuple(variables),
        method=method,
        cov=cov,
        pf_ci_95=(ci_lo, ci_hi),
    )
