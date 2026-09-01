"""zylab.reliability.model 响应模型测试：CDF 求值与似然正确性."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gumbel_l, norm, weibull_min

from zylab.reliability.errors import ReliabilityError
from zylab.reliability.model import (
    _info_matrix,
    inverse_prob,
    neg_log_likelihood,
    response_prob,
)


def test_response_prob_logistic() -> None:
    """Logistic 模型：中点 50%、对称性与饱和性."""
    assert response_prob("logistic", 0.0, 0.0, 1.0) == pytest.approx(0.5)
    assert response_prob("logistic", 1.0, 0.0, 1.0) == pytest.approx(1.0 - response_prob("logistic", -1.0, 0.0, 1.0))
    assert response_prob("logistic", 40.0, 0.0, 1.0) == pytest.approx(1.0, abs=1.0e-12)
    assert response_prob("logistic", -40.0, 0.0, 1.0) == pytest.approx(0.0, abs=1.0e-12)


def test_response_prob_normal_matches_scipy() -> None:
    """正态模型与 scipy.stats.norm.cdf 一致."""
    x = np.linspace(-3.0, 3.0, 11)
    expected = norm.cdf((x - 10.0) / 2.0)
    np.testing.assert_allclose(response_prob("normal", x, 10.0, 2.0), expected)


def test_response_prob_rejects_bad_arguments() -> None:
    """非法模型名与非正 σ 报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="不受支持"):
        response_prob("cauchy", 0.0, 0.0, 1.0)
    with pytest.raises(ReliabilityError, match="须为正"):
        response_prob("logistic", 0.0, 0.0, 0.0)


def test_response_prob_gumbel_matches_scipy() -> None:
    """Gumbel（最小极值）模型与 scipy.stats.gumbel_l.cdf 一致."""
    x = np.linspace(-3.0, 3.0, 11)
    expected = gumbel_l.cdf(x, loc=10.0, scale=2.0)
    np.testing.assert_allclose(response_prob("gumbel", x, 10.0, 2.0), expected)


def test_response_prob_weibull_matches_scipy() -> None:
    """Weibull 模型（μ=尺度 η、σ=形状 k）与 scipy.stats.weibull_min.cdf 一致."""
    x = np.linspace(0.5, 20.0, 11)
    expected = weibull_min.cdf(x, 1.5, scale=10.0)
    np.testing.assert_allclose(response_prob("weibull", x, 10.0, 1.5), expected)


def test_response_prob_weibull_rejects_nonpositive() -> None:
    """Weibull 模型：非正刺激量或非正尺度 η 报 ReliabilityError."""
    with pytest.raises(ReliabilityError, match="刺激量须全为正"):
        response_prob("weibull", np.array([1.0, 0.0]), 10.0, 1.5)
    with pytest.raises(ReliabilityError, match="尺度参数"):
        response_prob("weibull", 5.0, -1.0, 1.5)


def test_neg_log_likelihood_symmetry() -> None:
    """似然对称：关于 μ 镜像刺激量并翻转响应标签，NLL 不变.

    p(2μ−x; μ,σ) = F((μ−x)/σ) = 1 − p(x; μ,σ)，故 (x, y=1) 项与
    (2μ−x, y=0) 项互为镜像，两组数据的负对数似然相等。
    """
    mu, log_sigma = 10.5, 0.0
    x = np.array([9.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 1, 1])
    a = neg_log_likelihood(np.array([mu, log_sigma]), x, y, "logistic")
    b = neg_log_likelihood(np.array([mu, log_sigma]), 2.0 * mu - x, 1 - y, "logistic")
    assert a == pytest.approx(b)


def test_neg_log_likelihood_minimum_at_truth() -> None:
    """真值附近似然优于偏移参数（单峰性抽查）."""
    rng = np.random.default_rng(3)
    x = rng.uniform(8.0, 12.0, 60)
    p = np.asarray(response_prob("logistic", x, 10.0, 1.0))
    y = (rng.random(60) < p).astype(int)
    at_truth = neg_log_likelihood(np.array([10.0, 0.0]), x, y, "logistic")
    offset = neg_log_likelihood(np.array([10.8, 0.0]), x, y, "logistic")
    assert at_truth < offset


def test_inverse_prob_round_trip() -> None:
    """分位数往返：F⁻¹(F(x)) = x（四模型，Weibull 刺激量为正）."""
    cases = [
        ("logistic", 10.0, 2.0, 12.34),
        ("normal", 10.0, 2.0, 12.34),
        ("gumbel", 10.0, 2.0, 12.34),
        ("weibull", 10.0, 1.5, 12.34),
    ]
    for model, mu, sigma, x in cases:
        p = float(response_prob(model, x, mu, sigma))
        assert inverse_prob(model, p, mu, sigma) == pytest.approx(x)


def test_neg_log_likelihood_matches_probabilities() -> None:
    """四模型 NLL 与按响应概率直算一致（Weibull 参数化 (ln η, ln k)）.

    刺激量取内点区间避免参考式 ``log1p(-p)`` 在 p 双精度饱和为 1 时下溢。
    """
    rng = np.random.default_rng(5)
    x = rng.uniform(8.0, 12.0, 40)
    y = (rng.random(40) < 0.5).astype(int)
    cases = [
        ("logistic", np.array([10.0, 0.0])),
        ("normal", np.array([10.0, 0.0])),
        ("gumbel", np.array([10.0, 0.0])),
        ("weibull", np.array([np.log(10.0), np.log(1.5)])),
    ]
    for model, theta in cases:
        mu, sigma = (
            (float(np.exp(theta[0])), float(np.exp(theta[1])))
            if model == "weibull"
            else (float(theta[0]), float(np.exp(theta[1])))
        )
        p = np.asarray(response_prob(model, x, mu, sigma), dtype=float)
        expected = -float(np.sum(np.where(y > 0, np.log(p), np.log1p(-p))))
        assert neg_log_likelihood(theta, x, y, model) == pytest.approx(expected)


def test_info_matrix_positive_definite() -> None:
    """信息矩阵（数值微分通用实现）对角元为正且行列式为正."""
    for model, mu, sigma in [("normal", 10.0, 1.0), ("weibull", 10.0, 1.5)]:
        info = _info_matrix(np.array([8.0, 10.0, 12.0]), model, mu, sigma)
        assert np.all(np.diag(info) > 0.0)
        assert float(np.linalg.det(info)) > 0.0
