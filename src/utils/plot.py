from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'


def plot_overview(
    return_df: pd.DataFrame | None,
    variance_df: pd.DataFrame,
    **kwargs: Any
) -> plt.Figure:
    """
    Plot overview of returns, realized variance, jumps, and z-statistics.

    Parameters
    ----------
    return_df : pd.DataFrame | None
        DataFrame with logarithmic returns. Expected to have a DatetimeIndex.
    variance_df : pd.DataFrame
        DataFrame with variance components (realised_variance, jump, z_stats).
        Expected to have a DatetimeIndex.
    **kwargs : Any
        Additional keyword arguments:
        - figsize : tuple, optional
            Figure size (width, height) in inches. Default is (14, 10).
        - alpha : float, optional
            Significance level for z-statistic threshold. Default is 0.99.

    Returns
    -------
    plt.Figure
        Matplotlib figure object containing four subplots.
    """
    clr = "black"
    ls = "-"
    fig, axes = plt.subplots(4, 1, figsize=kwargs.get("figsize", (14, 10)))

    xlim = None
    if return_df is not None and isinstance(return_df.index, pd.DatetimeIndex):
        xlim = (return_df.index.min(), return_df.index.max())
    elif isinstance(variance_df.index, pd.DatetimeIndex):
        xlim = (variance_df.index.min(), variance_df.index.max())

    # Panel A: Logarithmic Returns
    if return_df is not None:
        axes[0].plot(return_df, color=clr, linestyle=ls)
    if xlim:
        axes[0].set_xlim(xlim)
    axes[0].set_title(r'Panel A.  Logarithmic Return ($r_{t+j,\delta} = X_{t+j,\delta} - X_{t+(j-1),\delta}$)')  # noqa: E501

    # Panel B: Realized Variance
    axes[1].plot(variance_df["realised_variance"], color=clr, linestyle=ls)
    if xlim:
        axes[1].set_xlim(xlim)
    axes[1].set_title(r'Panel B.  Realised Variance $(RV_{t+1})$')

    # Panel C: Jump Component
    axes[2].plot(variance_df["jump"], color=clr, linestyle=ls)
    if xlim:
        axes[2].set_xlim(xlim)
    axes[2].set_title(r'Panel C.  Jump Component $(J_{t+1})$')

    # Panel D: Z-Statistic
    axes[3].plot(variance_df["z_stats"], color=clr, linestyle=ls)
    axes[3].axhline(
        y=st.norm.ppf(kwargs.get("alpha", 0.99)),
        color='#95a3a6'
    )
    if xlim:
        axes[3].set_xlim(xlim)
    axes[3].set_title(r'Panel D.  Z-Statistic $(Z_{t+1}(\delta))$')
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    return fig
