from datetime import timedelta

import numpy as np
import pandas as pd


def generate_jump_diffision_process(
    X0: float = np.log(100.),
    start_date: str = "2010-01-02",
    end_date: str = "2025-12-31",
    mu: float = 0.08,
    sigma: float = 0.15,
    lambda_J: float = 50.0,
    mu_J: float = 0.0,
    sigma_J: float = 0.02,
    intraday_freq: int = 5
) -> pd.DataFrame:
    """
    Generate intraday log-price process with jump diffusion dynamics:

    dX(t) = μ dt + σ dW^P(t) + J dX_P(t)

    where:
    - X(t) = log(S(t)) is the log price
    - μ is the drift term
    - σ is the stochastic volatility process
    - W^P(t) is a standard Geometric Brownian Motion under measure P
    - X_P(t) is a Poisson process with intensity λ
    - J is the jump magnitude (typically J ~ N(μ_J, σ_J²))

    Parameters:
    -----------
    X0 : float
        Initial log price (e.g., log(100)), default: np.log(100.)
    start_date : str
        Start date of the time series, default: '2010-01-02'
    end_date : str
        End date of the time series, default: '2025-12-31'
    mu : float
        Annual drift parameter, default: 0.08
    sigma : float
        Annual volatility parameter, default: 0.15
    lambda_J : float
        Annual Poisson intensity (λ) - expected number of jumps per year,
        default: 50.0
    mu_J : float
        Mean of jump magnitude distribution (μ_J), default: 0.0
    sigma_J : float
        Std dev of jump magnitude distribution (σ_J), default: 0.02
    intraday_freq : int
        The frequency of the intraday prices in minutes, default: 5

    Returns:
    --------
    pd.DataFrame : DataFrame with timestamp, log_price, price, returns, etc.
    """
    np.random.seed(42)

    market_open_hour: int = 9,
    market_open_minute: int = 30,
    market_close_hour: int = 16,
    market_close_minute: int = 0
    intervals_per_day = (6 * 60 + 30) / intraday_freq
    intervals_per_year = 252 * intervals_per_day

    timestamps = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    current_date = start
    while current_date <= end:
        if current_date.weekday() < 5:
            current_time = current_date.replace(
                hour=market_open_hour,
                minute=market_open_minute,
                second=0,
                microsecond=0
            )
            end_time = current_date.replace(
                hour=market_close_hour,
                minute=market_close_minute,
                second=0,
                microsecond=0
            )

            while current_time <= end_time:
                timestamps.append(current_time)
                current_time += timedelta(minutes=intraday_freq)

        current_date += timedelta(days=1)

    n = len(timestamps)
    dt = 1 / intervals_per_year

    X = np.zeros(n)
    X[0] = X0

    # heston like stochastic volatility
    v = np.zeros(n)
    v[0] = sigma**2
    kappa = 2.0  # mean reversion speed
    theta = sigma**2  # long run variance
    xi = 0.3  # vol-of-vol
    rho = -0.7

    for i in range(1, n):
        current_sigma = np.sqrt(v[i-1])

        # drift term
        drift = mu * dt

        # diffusion term
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = current_sigma * dW

        # jump term
        prob_J = lambda_J * dt
        flag_J = np.random.binomial(1, min(prob_J, 1.0))

        if flag_J:
            J = np.random.normal(mu_J, sigma_J)
            jump_component = J
        else:
            jump_component = 0.0

        # Update log price: dX(t) = μ dt + σ dW^P(t) + J dX_P(t)
        dX = drift + diffusion + jump_component
        X[i] = X[i-1] + dX

        # correlated brownian motion
        dW_v = np.random.normal(0, np.sqrt(dt))
        dW_v_corr = rho * dW + np.sqrt(1 - rho**2) * dW_v

        # Heston dynamics: dv = κ(θ - v)dt + ξ√v dW_v
        dv = kappa * (theta - v[i-1]) * dt + xi * np.sqrt(max(v[i-1], 0)) * dW_v_corr # noqa: E501
        v[i] = max(v[i-1] + dv, 1e-6)  # ensure positivity

    df = pd.DataFrame({
        'timestamp': timestamps,
        'log_price': X
    })
    df['log_return'] = df['log_price'].diff()

    return df[["timestamp", "log_return"]]
