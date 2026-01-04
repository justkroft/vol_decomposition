import numpy as np

from src.vol_decomposition import compute_realised_variance

returns = np.array([
    0.01, -0.02, 0.015, -0.01, 0.005,  # Day 0
    0.02, -0.01, 0.008, -0.015, 0.012   # Day 1
], dtype=np.float64)

day_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)

rv = compute_realised_variance(returns, day_indices, n_days=2)
print(rv)
print(np.sum(returns[:5] ** 2))
