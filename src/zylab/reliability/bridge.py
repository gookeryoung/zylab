"""Reliability-Workflow Bridge.

让 FORM / SORM / MC 可靠性分析直接接受 :class:`~zylab.flowchart.Template`
而无需用户手写极限状态函数。桥接层自动做三件事：

1. 把 :class:`RandomVariable` 采样的物理向量按 ``rv_mapping`` 顺序注入到
   :func:`run_workflow` 的 overrides。
2. 从 FE 节点结果里提取常用标量（最大主应力、应变能、最大位移等）。
3. 用 ``limit_state_expr``（字符串 ``eval``）组装 ``g(x)`` 并喂给
   :func:`form_analysis` / :func:`sorm_analysis` / :func:`mc_analysis`。

典型极限状态表达式示例：

* ``"fe.max_stress - rv.sigma_y"`` — 最大主应力低于屈服强度
* ``"rv.R - fe.strain_energy * 0.01"`` — 抗力减功能函数
* ``"fe.max_displacement - rv.L / 250"`` — 挠跨比限值

``rv_mapping`` 的 key 决定可靠性采样向量的顺序：

* ``"node_id.param_key"`` 形式 → FE 参数随机变量，直接进 overrides
* 纯名称（不含 ``"."``） → 计算用随机变量（屈服、抗力、荷载系数等）
"""

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .form import (
    FORMResult,
    LimitStateFn,
    MCResult,
    RandomVariable,
    SORMResult,
    form_analysis,
    mc_analysis,
    sorm_analysis,
)

if TYPE_CHECKING:  # pragma: no cover
    from zylab.flowchart import Template

# pytest-xdist worker 内检测到则禁用内部 ProcessPoolExecutor，避免 Windows
# spawn 模式下嵌套进程池极低频死锁。正常用户调用不受影响。
_IS_XDIST_WORKER = bool(os.environ.get("PYTEST_XDIST_WORKER"))


# ---------------------------------------------------------------------------
# 默认 FE 输出提取器
# ---------------------------------------------------------------------------


def _extract_static_solution(node_result: Any) -> dict[str, float]:
    """从 StaticSolution 风格的节点结果提取常用标量.

    提取项：

    * ``strain_energy`` — 总应变能
    * ``max_stress`` / ``min_stress`` — 最大/最小主应力的 S1 分量
    * ``max_displacement`` — 位移模长最大值（若 displacements 存在）
    * ``n_elements`` — 单元数
    """
    out: dict[str, float] = {}
    # 直接标量字段
    for attr in ("strain_energy", "total_load", "reaction_sum"):
        v = getattr(node_result, attr, None)
        if isinstance(v, (int, float, np.floating)) and not math.isnan(float(v)):
            out[attr] = float(v)

    # 单元应力
    er = getattr(node_result, "element_results", None)
    if er is not None and len(er) > 0:
        # stress: (3,) → 取 S1 分量（第一主应力）
        stresses = np.array([er[i].stress[0] for i in range(len(er))], dtype=float)
        if stresses.size > 0:
            out["max_stress"] = float(stresses.max())
            out["min_stress"] = float(stresses.min())
            out["n_elements"] = float(len(er))

    # 位移
    disp = getattr(node_result, "displacements", None)
    if disp is not None:
        arr = np.asarray(disp, dtype=float)
        if arr.ndim >= 2 and arr.size > 0:
            mag = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=1)
            out["max_displacement"] = float(mag.max())

    # 支持数组字段的 max/min 约定（max_stress 已从 element_results 来）
    return out


def _extract_generic(node_result: Any) -> dict[str, float]:
    """通用兜底提取器：从 result.__dict__ 里收集标量."""
    out: dict[str, float] = {}
    d = getattr(node_result, "__dict__", None)
    if not d:
        return out
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if (
            (isinstance(v, (int, float)) and not isinstance(v, bool))
            or isinstance(v, np.floating)
            or (isinstance(v, np.ndarray) and v.ndim == 0)
        ):
            out[k] = float(v)
    return out


def _collect_fe_outputs(result: Any) -> dict[str, float]:
    """遍历 RunOutcome.outcomes，合并各节点的 FE 输出."""
    out: dict[str, float] = {}
    if not result or not hasattr(result, "outcomes"):
        return out
    for node_out in result.outcomes:
        r = getattr(node_out, "result", None)
        if r is None:
            continue
        # 优先 StaticSolution 风格
        if hasattr(r, "element_results") or hasattr(r, "strain_energy"):
            extracted = _extract_static_solution(r)
        else:
            extracted = _extract_generic(r)
        for k, v in extracted.items():
            # 同名键不覆盖（不同节点极少冲突）
            out.setdefault(k, v)
    return out


# ---------------------------------------------------------------------------
# 表达式解析
# ---------------------------------------------------------------------------


def _parse_namespace_refs(expr: str) -> tuple[set[str], set[str]]:
    """从 ``expr`` 中提取 ``fe.xxx`` 和 ``rv.xxx`` 引用."""
    fe_keys: set[str] = set()
    rv_keys: set[str] = set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return fe_keys, rv_keys

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "fe":
                fe_keys.add(node.attr)
            elif node.value.id == "rv":
                rv_keys.add(node.attr)
    return fe_keys, rv_keys


# ---------------------------------------------------------------------------
# 桥接主类
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityBridge:
    """把可靠性分析与 :class:`~zylab.flowchart.Template` 工作流绑定.

    :param template: FE 工作流模板
    :param rv_mapping: 有序字典——key 决定可靠性采样向量的顺序.
        ``"node_id.param_key"`` 形式的 key 会自动注入到 overrides；
        纯名称 key 作为计算用随机变量（如 ``sigma_y``、``R``），
        在极限状态表达式中通过 ``rv.xxx`` 访问。
    :param limit_state_expr: Python ``eval`` 表达式，可用
        ``fe``（FE 输出字典）和 ``rv``（所有随机变量当前值）两个名字空间.
        例如 ``"fe.max_stress - rv.sigma_y"``。
    :param n_workers: 预留，暂未使用；FORM/SORM 串行（内部用闭包），
        MC 可直接用批量跑 ``run_batch``（由调用方自己包装）。
    """

    template: Template
    rv_mapping: dict[str, RandomVariable]
    limit_state_expr: str
    n_workers: int | None = None
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- 内部辅助 --------------------------------------------------------

    def _fe_param_keys(self) -> list[str]:
        return [k for k in self.rv_mapping if "." in k]

    def _compute_keys(self) -> list[str]:
        return [k for k in self.rv_mapping if "." not in k]

    def _build_overrides(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """把可靠性采样向量 x 按 rv_mapping 顺序转成 overrides dict."""
        rv_keys = list(self.rv_mapping.keys())
        if x.shape[0] != len(rv_keys):
            raise ValueError(f"x 维度 {x.shape[0]} 与 rv_mapping 数量 {len(rv_keys)} 不匹配")
        overrides: dict[str, dict[str, float]] = {}
        rv_values: dict[str, float] = {}
        fe_keys = self._fe_param_keys()
        for i, k in enumerate(rv_keys):
            v = float(x[i])
            rv_values[k] = v
            if k in fe_keys:
                node_id, param_key = k.split(".", 1)
                overrides.setdefault(node_id, {})[param_key] = v
        self._cache["_last_rv_values"] = rv_values
        return overrides

    # ---- LimitStateFn 工厂 ----------------------------------------------

    def make_limit_state(self) -> LimitStateFn:
        """返回喂给 ``form_analysis`` / ``mc_analysis`` 的闭包 ``g(x)``.

        内部每次调用都会：

        1. 把 x 按 rv_mapping 拆成 FE overrides 和计算 RV 值。
        2. 调 :func:`run_workflow`。
        3. 从 FE 结果提取常用标量（最大主应力、应变能等）。
        4. 用 ``eval`` 求值 ``limit_state_expr``。

        FE 失败（``RunOutcome.succeeded == False``）时返回 ``1e10``，
        让优化器自动远离不可行区域。
        """
        from zylab.flowchart import run_workflow

        tpl = self.template
        expr = self.limit_state_expr
        fe_keys = self._fe_param_keys()
        rv_keys = list(self.rv_mapping.keys())

        def g(x: np.ndarray) -> float:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim != 1:
                raise ValueError(f"x 必须是一维数组，实际 {x_arr.shape}")
            if x_arr.shape[0] != len(rv_keys):
                raise ValueError(f"x 维度 {x_arr.shape[0]} 与 rv_mapping 数量 {len(rv_keys)} 不匹配")

            overrides: dict[str, dict[str, float]] = {}
            rv_values: dict[str, float] = {}
            for i, k in enumerate(rv_keys):
                v = float(x_arr[i])
                rv_values[k] = v
                if k in fe_keys:
                    nid, pkey = k.split(".", 1)
                    overrides.setdefault(nid, {})[pkey] = v

            try:
                outcome = run_workflow(tpl, overrides=overrides)
            except Exception:
                # ParamError / 运行时异常 → 视为不可行
                return 1e10

            if not outcome.succeeded:
                return 1e10  # FE 失败视为不可行

            fe = _collect_fe_outputs(outcome)
            # 用 SimpleNamespace 包装，让 eval 支持 fe.max_stress / rv.sigma_y 点号访问
            from types import SimpleNamespace as _NS

            ns = {"fe": _NS(**fe), "rv": _NS(**rv_values)}
            # 空 builtins — 只允许算术/比较运算
            return float(eval(expr, {"__builtins__": {}}, ns))

        return g

    # ---- 分析入口 --------------------------------------------------------

    def _variables(self) -> list[RandomVariable]:
        return list(self.rv_mapping.values())

    def run_form(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1e-8,
        start_std: np.ndarray | None = None,
    ) -> FORMResult:
        """运行 FORM（HLRF）可靠性分析."""
        return form_analysis(
            self.make_limit_state(),
            self._variables(),
            max_iter=max_iter,
            tol=tol,
            start_std=start_std,
        )

    def run_sorm(self, *, max_iter: int = 50, tol: float = 1e-8) -> SORMResult:
        """运行 SORM（Breitung 曲率修正）."""
        return sorm_analysis(
            self.make_limit_state(),
            self._variables(),
            max_iter=max_iter,
            tol=tol,
        )

    def run_mc(
        self,
        *,
        n_samples: int = 100_000,
        method: str = "crude",
        seed: int = 0,
        n_workers: int | None = None,
    ) -> MCResult:
        """Run Monte Carlo reliability analysis.

        :param n_workers: >=2 enables multiprocessing. Bridge itself is picklable,
            each worker constructs the closure locally.
        """
        _w = n_workers if (n_workers is not None and n_workers >= 2 and not _IS_XDIST_WORKER) else 1
        if _w == 1:
            return mc_analysis(
                self.make_limit_state(),
                self._variables(),
                n_samples=n_samples,
                method=method,
                seed=seed,
            )
        return _run_mc_parallel(
            bridge=self,
            n_samples=n_samples,
            method=method,
            seed=seed,
            n_workers=_w,
        )


# ---------------------------------------------------------------------------
# Monte Carlo parallel (bridge-only)
# ---------------------------------------------------------------------------


def _mc_chunk_worker(args: tuple[bytes, np.ndarray]) -> np.ndarray:
    """Process pool worker - unpickle bridge, evaluate g(x) over chunk."""
    import pickle as _pickle

    bridge_bytes, X_chunk = args
    bridge: ReliabilityBridge = _pickle.loads(bridge_bytes)
    g = bridge.make_limit_state()
    return np.array([g(X_chunk[j]) for j in range(X_chunk.shape[0])])


def _run_mc_parallel(
    *,
    bridge: ReliabilityBridge,
    n_samples: int,
    method: str,
    seed: int,
    n_workers: int,
) -> MCResult:
    """Parallel MC - U sampling + U->X on main proc, g-eval split to workers."""
    import math as _math
    from concurrent.futures import ProcessPoolExecutor

    from scipy import stats as _stats

    from .form import _from_standard

    variables = bridge._variables()
    D = len(variables)
    rng = np.random.default_rng(seed)

    if method == "crude":
        U = rng.standard_normal((n_samples, D))
    elif method == "lhc":
        from scipy.stats import qmc as _qmc

        lhs = _qmc.LatinHypercube(D, seed=seed)
        U = _stats.norm.ppf(lhs.random(n_samples))
    else:  # sobol
        k = _math.ceil(_math.log2(max(n_samples, 1)))
        n2 = 1 << k
        from scipy.stats import qmc as _qmc

        sampler = _qmc.Sobol(D, scramble=True, seed=seed)
        U = _stats.norm.ppf(sampler.random(n2))[:n_samples]

    X = np.array([_from_standard(U[j], variables) for j in range(n_samples)])

    import pickle as _pickle

    bridge_bytes = _pickle.dumps(bridge)
    chunk_size = max(1, _math.ceil(n_samples / n_workers))
    chunks = [X[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    args_list = [(bridge_bytes, c) for c in chunks]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        g_chunks = list(ex.map(_mc_chunk_worker, args_list))

    g_vals = np.concatenate(g_chunks)

    n_fail = int(np.sum(g_vals < 0))
    pf = n_fail / n_samples

    if pf <= 0:
        beta = _math.inf
    elif pf >= 1.0:
        beta = -_math.inf
    else:
        beta = float(-_stats.norm.ppf(pf))

    pf_var = pf * (1.0 - pf) / n_samples
    cov = float(np.sqrt(pf_var) / pf) if pf > 0 else float("inf")

    if n_fail == 0:
        ci_lo = 0.0
        ci_hi = 1.0 - (0.05 / n_samples)
    elif n_fail == n_samples:
        ci_lo = 0.05 / n_samples
        ci_hi = 1.0
    else:
        z = _stats.norm.ppf(0.975)
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
