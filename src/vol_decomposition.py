import numpy as np

import src._vol_decomposition_c as _c


def compute_realised_variance(
    returns: np.ndarray,
    day_indices: np.ndarray,
    n_days: int
) -> np.ndarray:
    return _c.compute_realised_variance(returns, day_indices, n_days)
