import numpy as np
import pytest
import scipy.special as ss

try:
    import src._vol_decomposition_c as _c  # noqa: F401
    C_EXT_AVAILABLE = True
except ImportError:
    C_EXT_AVAILABLE = False
    pytest.skip("C extension not compiled", allow_module_level=True)

from src.vol_decomposition import (
    bipower_variance,
    realised_variance,
    tripower_quarticity,
    z_stats,
)

RTOL = 1e-10


@pytest.fixture
def simple_returns():
    return np.array([0.01, -0.02, 0.015, -0.01, 0.005], dtype=np.float64)


@pytest.fixture
def simple_day_indices():
    return np.array([0, 0, 0, 0, 0], dtype=np.int64)


@pytest.fixture
def multi_day_returns():
    # 2 days, 5 observations each
    return np.array([
        0.01, -0.02, 0.015, -0.01, 0.005,  # Day 0
        0.02, -0.01, 0.008, -0.015, 0.012   # Day 1
    ], dtype=np.float64)


@pytest.fixture
def multi_day_indices():
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)


def test_ealised_variance_simple(simple_returns, simple_day_indices):
    actual = realised_variance(simple_returns, simple_day_indices, 1)
    expected = np.sum(simple_returns ** 2)

    np.testing.assert_allclose(actual[0], expected, rtol=RTOL)


def test_ealised_variance_multi_day(
    multi_day_returns, multi_day_indices
):
    actual = realised_variance(multi_day_returns, multi_day_indices, 2)

    # Day 0
    expected_day0 = np.sum(multi_day_returns[:5] ** 2)
    np.testing.assert_allclose(actual[0], expected_day0, rtol=1e-10)

    # Day 1
    expected_day1 = np.sum(multi_day_returns[5:] ** 2)
    np.testing.assert_allclose(actual[1], expected_day1, rtol=1e-10)


def test_realised_variance_non_negative(
    multi_day_returns, multi_day_indices
):
    actual = realised_variance(multi_day_returns, multi_day_indices, 2)
    assert np.all(actual >= 0)


def test_realised_variance_zero_returns():
    returns = np.zeros(10, dtype=np.float64)
    day_indices = np.zeros(10, dtype=np.int64)

    actual = realised_variance(returns, day_indices, 1)
    assert actual[0] == 0.0



def mu_func_testing(p):
    return 2**(p/2) * ss.gamma((p+1) / 2) / ss.gamma(1/2)


def test_bipower_variance_simple(simple_returns, simple_day_indices):
    actual = bipower_variance(simple_returns, simple_day_indices, 1)

    # manual calculation
    mu_1 = mu_func_testing(1.0)
    mu_1_inv_sq = mu_1 ** (-2)
    abs_returns = np.abs(simple_returns)
    expected = np.sum(abs_returns[:-1] * abs_returns[1:]) * mu_1_inv_sq

    np.testing.assert_allclose(actual[0], expected, rtol=RTOL)


def test_bipower_variance_multi_day(multi_day_returns, multi_day_indices):
    actual = bipower_variance(multi_day_returns, multi_day_indices, 2)

    # Manual calculation
    mu_1 = mu_func_testing(1.0)
    mu_1_inv_sq = mu_1 ** (-2)

    # Day 0
    abs_ret_day0 = np.abs(multi_day_returns[:5])
    expected_day0 = np.sum(abs_ret_day0[:-1] * abs_ret_day0[1:]) * mu_1_inv_sq
    np.testing.assert_allclose(actual[0], expected_day0, rtol=1e-10)

    # Day 1
    abs_ret_day1 = np.abs(multi_day_returns[5:])
    expected_day1 = np.sum(abs_ret_day1[:-1] * abs_ret_day1[1:]) * mu_1_inv_sq
    np.testing.assert_allclose(actual[1], expected_day1, rtol=1e-10)


def test_bipower_variance_non_negative(multi_day_returns, multi_day_indices):
    actual = bipower_variance(multi_day_returns, multi_day_indices, 2)
    assert np.all(actual >= 0)


def test_bipower_variance_no_cross_contamination():
    returns = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    day_indices = np.array([0, 0, 1, 1], dtype=np.int64)

    actual = bipower_variance(returns, day_indices, 2)

    mu_1 = mu_func_testing(1.0)
    mu_1_inv_sq = mu_1**(-2)

    # Day 0: only |0.01| * |0.02|
    expected_day0 = 0.01 * 0.02 * mu_1_inv_sq
    np.testing.assert_allclose(actual[0], expected_day0, rtol=1e-10)

    # Day 1: only |0.03| * |0.04|
    expected_day1 = 0.03 * 0.04 * mu_1_inv_sq
    np.testing.assert_allclose(actual[1], expected_day1, rtol=1e-10)


def test_tripower_quarticity_simple(simple_returns, simple_day_indices):
    mu_43 = mu_func_testing(4.0/3.0)
    delta = 1.0 / 288.0

    actual = tripower_quarticity(simple_returns, simple_day_indices, 1, delta)

    # Manual calculation
    abs_returns = np.abs(simple_returns)
    expected = 0.0
    for i in range(2, len(abs_returns)):
        expected += (
            abs_returns[i-2]**(4/3)
            * abs_returns[i-1]**(4/3)
            * abs_returns[i]**(4/3)
        )
    expected /= (delta * mu_43**3)

    np.testing.assert_allclose(actual[0], expected, rtol=1e-10)


def test_tripower_quarticity_multi_day(multi_day_returns, multi_day_indices):
    delta = 1.0 / 288.0

    actual = tripower_quarticity(
        multi_day_returns, multi_day_indices, 2, delta
    )

    assert actual.shape == (2,)
    assert np.all(actual >= 0)


def test_tripower_quarticity_no_cross_contamination():
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    day_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    mu_43 = mu_func_testing(4.0/3.0)
    delta = 1.0 / 288.0

    actual = tripower_quarticity(returns, day_indices, 2, delta)

    # Day 0: has 3 observations, so 1 triplet (0.01, 0.02, 0.03)
    expected_day0 = (0.01**(4/3) * 0.02**(4/3) * 0.03**(4/3)) / (delta * mu_43**3)  # noqa: E501
    np.testing.assert_allclose(actual[0], expected_day0, rtol=1e-10)

    # Day 1: only 2 observations, no triplets
    assert actual[1] == 0.0


def test_z_stats():
    rv = np.array([0.001, 0.002], dtype=np.float64)
    bpvar = np.array([0.0008, 0.0015], dtype=np.float64)
    tpq = np.array([0.0001, 0.0002], dtype=np.float64)
    delta = 1.0 / 288.0

    stats = z_stats(rv, bpvar, tpq, delta)

    assert stats.shape == (2,)
    assert stats.dtype == np.float64
    # Z-stats should be finite
    assert np.all(np.isfinite(stats))


def test_z_stats_zero_bipower_variance():
    rv = np.array([0.001], dtype=np.float64)
    bpvar = np.array([0.0], dtype=np.float64)
    tpq = np.array([0.0001], dtype=np.float64)
    delta = 1.0 / 288.0

    stats = z_stats(rv, bpvar, tpq, delta)

    # Should handle gracefully without division by zero
    assert np.isfinite(stats[0])


def test_z_stats_manual_comparison():
    rv = np.array([0.01], dtype=np.float64)
    bpvar = np.array([0.008], dtype=np.float64)
    tpq = np.array([0.0001], dtype=np.float64)
    delta = 1.0 / 288.0

    stats = z_stats(rv, bpvar, tpq, delta)

    # Manual calculation NPY_PI
    const_term = (np.pi**2) / 4 + np.pi - 5
    max_func = max(1.0, tpq[0] / bpvar[0]**2)
    expected = (rv[0] - bpvar[0]) / (rv[0] * np.sqrt(const_term * max_func))
    expected = expected / np.sqrt(delta)

    np.testing.assert_allclose(stats[0], expected, rtol=1e-10)


def test_z_stats_high_jump():
    # Simulate a clear jump: RV much larger than BPV
    rv = np.array([0.01], dtype=np.float64)
    bpvar = np.array([0.001], dtype=np.float64)
    tpq = np.array([0.00001], dtype=np.float64)
    delta = 1.0 / 288.0

    stats = z_stats(rv, bpvar, tpq, delta)

    # Should produce a large positive Z-statistic
    assert stats[0] > 1.0
