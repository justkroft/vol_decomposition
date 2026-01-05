import numpy as np

import src._vol_decomposition_c as _c


def compute_realised_variance(
    returns: np.ndarray,
    day_indices: np.ndarray,
    n_days: int
) -> np.ndarray:
    """
    Compute realized variance for each day from intraday returns.

    This function calculates the realized variance by summing the squared
    returns for each trading day. The realized variance is a non-parametric
    estimator of return variance that uses high-frequency intraday data.

    Parameters
    ----------
    returns : np.ndarray
        1D array of intraday returns (log returns or simple returns).
        Must be of dtype float64.
    day_indices : np.ndarray
        1D array of integer day identifiers corresponding to each return.
        Each element indicates which day (0 to n_days-1) the corresponding
        return belongs to. Must be of dtype int64.
    n_days : int
        Total number of unique trading days in the dataset.

    Returns
    -------
    np.ndarray
        1D array of shape (n_days,) containing the realized variance for
        each day.

    Examples
    --------
    >>> import numpy as np
    >>> from src.vol_decomposition import compute_realised_variance

    >>> # Two days with 3 and 2 returns respectively
    >>> returns = np.array([0.01, -0.02, 0.015, 0.01, -0.005])
    >>> day_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    >>> n_days = 2

    >>> rv = compute_realised_variance(returns, day_indices, n_days)
    >>> print(rv)
    [0.000725 0.000125]

    >>> # Day 0: 0.01² + (-0.02)² + 0.015² = 0.000725
    >>> # Day 1: 0.01² + (-0.005)² = 0.000125
    """
    return _c.compute_realised_variance(returns, day_indices, n_days)


def compute_bipower_variance(
    returns: np.ndarray,
    day_indices: np.ndarray,
    n_days: int
) -> np.ndarray:
    """
    Compute the bipower variance for each day from intraday returns.

    Parameters
    ----------
    returns : np.ndarray
        1D array of intraday returns (log returns or simple returns).
        Must be of dtype float64.
    day_indices : np.ndarray
        1D array of integer day identifiers corresponding to each return.
        Each element indicates which day (0 to n_days-1) the corresponding
        return belongs to. Must be of dtype int64.
    n_days : int
        Total number of unique trading days in the dataset.

    Returns
    -------
    np.ndarray
        1D array of shape (n_days,) containing the bi-power variance for each
        day.
    """
    return _c.compute_bipower_variance(returns, day_indices, n_days)
