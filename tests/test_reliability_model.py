"""zylab.reliability.model 响应模型测试：CDF 求值与似然正确性."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from zylab.reliability.errors import ReliabilityError
from zylab.reliability.model import (
    MODEL_NAMES,
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
    """分位数往返：F⁻¹(F(x)) = x（两模型）."""
    for model in MODEL_NAMES:
        x = 12.34
        p = float(response_prob(model, x, 10.0, 2.0))
        assert inverse_prob(model, p, 10.0, 2.0) == pytest.approx(x)
