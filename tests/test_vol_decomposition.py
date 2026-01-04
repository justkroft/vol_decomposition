import numpy as np
import pytest

try:
    import src._vol_decomposition_c as _c  # noqa: F401
    C_EXT_AVAILABLE = True
except ImportError:
    C_EXT_AVAILABLE = False
    pytest.skip("C extension not compiled", allow_module_level=True)

from src.vol_decomposition import compute_realised_variance

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


def test_compute_realised_variance_simple(simple_returns, simple_day_indices):
    actual = compute_realised_variance(simple_returns, simple_day_indices, 1)
    expected = np.sum(simple_returns ** 2)

    np.testing.assert_allclose(actual[0], expected, rtol=RTOL)


def test_compute_realised_variance_multi_day(
    multi_day_returns, multi_day_indices
):
    actual = compute_realised_variance(multi_day_returns, multi_day_indices, 2)

    # Day 0
    expected_day0 = np.sum(multi_day_returns[:5] ** 2)
    np.testing.assert_allclose(actual[0], expected_day0, rtol=1e-10)

    # Day 1
    expected_day1 = np.sum(multi_day_returns[5:] ** 2)
    np.testing.assert_allclose(actual[1], expected_day1, rtol=1e-10)


def test_compute_realised_variance_non_negative(
    multi_day_returns, multi_day_indices
):
    actual = compute_realised_variance(multi_day_returns, multi_day_indices, 2)
    assert np.all(actual >= 0)


def test_compute_realised_variance_zero_returns():
    returns = np.zeros(10, dtype=np.float64)
    day_indices = np.zeros(10, dtype=np.int64)

    actual = compute_realised_variance(returns, day_indices, 1)
    assert actual[0] == 0.0
