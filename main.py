import numpy as np
import scipy.special as ss

# from src._vol_decomposition_c import compute_bipower_variance, compute_tripower_quarticity
from src.vol_decomposition import realised_variance, bipower_variance, z_stats, tripower_quarticity


def mu_func(p):
    """Compute mu function using gamma functions."""
    return 2**(p/2) * ss.gamma((p+1) / 2) / ss.gamma(1/2)



returns = np.array([
    0.01, -0.02, 0.015, -0.01, 0.005,  # Day 0
    0.02, -0.01, 0.008, -0.015, 0.012   # Day 1
], dtype=np.float64)

day_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)

bv = realised_variance(returns, day_indices, 2)
print(bv)

# print(type(bv))

# mu_1 = mu_func(1.0)
# mu_1_inv_sq = mu_1 ** (-2)

# # Day 0 - manual calculation
# abs_ret_day0 = np.abs(returns[:5])
# expected_day0 = np.sum(abs_ret_day0[:-1] * abs_ret_day0[1:]) * mu_1_inv_sq

# # Day 1 - manual calculation
# abs_ret_day1 = np.abs(returns[5:])
# expected_day1 = np.sum(abs_ret_day1[:-1] * abs_ret_day1[1:]) * mu_1_inv_sq

# print([expected_day0, expected_day1])