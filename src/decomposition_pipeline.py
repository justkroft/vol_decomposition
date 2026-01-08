import dataclasses

import numpy as np
import pandas as pd
import scipy.stats as st

from src.vol_decomposition import (
    apply_jump_filter,
    bipower_variance,
    realised_variance,
    tripower_quarticity,
    z_stats,
)


@dataclasses.dataclass(frozen=True)
class _TransformOutput:
    """Internal data structure for transformed inputs."""
    returns: np.ndarray
    day_indices: np.ndarray
    n_days: int
    unique_dates: pd.Index


class VarianceDecomposition:
    """
    Decompose realised variance into continuous and jump components.

    This class implements the methodology for separating the continuous and
    jump components of realised variance using high-frequency intraday returns.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing intraday (log or simple )returns and timestamps.
    return_column : str
        Name of the column containing returns (log or simple returns).
    date_column : str
        Name of the column containing dates/timestamps.
    alpha : float, optional
        Significance level for jump detection (default: 0.975).
        Higher values mean more conservative jump detection.
    delta : float, optional
        Intraday sampling frequency (default: 1/288, representing
        5-minute intervals in a 24-hour trading day).

    Attributes
    ----------
    threshold : float
        Z-score threshold computed from alpha using the inverse CDF of the
        normal distribution.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        return_column: str,
        date_column: str,
        alpha: float = 0.975,
        delta: float = 1.0 / 288.0
    ) -> None:

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )

        if data.empty:
            raise ValueError("data must not be empty")

        if return_column not in data.columns:
            raise KeyError(
                f"return_column '{return_column}' not found in data"
            )

        if date_column not in data.columns:
            raise KeyError(f"date_column '{date_column}' not found in data")

        self.data = data
        self.return_column = return_column
        self.date_column = date_column
        self.threshold = st.norm.ppf(alpha)
        self.delta = delta
        self._results: pd.DataFrame | None = None

    def decompose(self) -> pd.DataFrame:
        """
        Decompose realized variance into continuous and jump components.

        Returns
        -------
        pd.DataFrame
            DataFrame with date index and the following columns:

            - realised_variance : Total realized variance (sum of squared returns)
            - bipower_variance : Bi-power variation (robust to jumps)
            - tripower_quarticity : Tri-power quarticity (for test statistic)
            - z_stats : Jump test statistics (higher absolute values indicate jumps)
            - continuous : Continuous component of variance
            - jump : Jump component of variance (0 if no jump detected)

        Raises
        ------
        ValueError
            If data contains insufficient observations, missing values,
            or invalid numeric values (raised by C extension).
        RuntimeError
            If computation fails in the C extension.

        Notes
        -----
        The decomposition satisfies:

            realised_variance = continuous + jump

        Jump detection is based on comparing z_stats against the threshold
        derived from alpha. Days with |z_stats| > threshold are identified
        as having statistically significant jumps.

        Examples
        --------
        >>> decomp = VarianceDecomposition(data, 'returns', 'date')
        >>> results = decomp.decompose()
        >>>
        >>> # Identify jump days
        >>> jump_days = decomp.n_jump_days
        >>> print(f"Found {jump_days} days with jumps")
        """  # noqa: E501
        inputs = self._transform_data()
        returns = inputs.returns
        day_indices = inputs.day_indices
        n_days = inputs.n_days

        # C extensions handle all numeric validation
        rv = realised_variance(
            returns=returns,
            day_indices=day_indices,
            n_days=n_days
        )

        bpv = bipower_variance(
            returns=returns,
            day_indices=day_indices,
            n_days=n_days
        )

        tpq = tripower_quarticity(
            returns=returns,
            day_indices=day_indices,
            n_days=n_days,
            delta=self.delta
        )

        stats = z_stats(
            realised_variance=rv,
            bipower_variance=bpv,
            tripower_quarticity=tpq,
            delta=self.delta
        )

        cont, jump = apply_jump_filter(
            realised_variance=rv,
            bipower_variance=bpv,
            z_stats=stats,
            sig_threshold=self.threshold,
            truncate_zero=1
        )

        results = pd.DataFrame(
            {
                "realised_variance": rv,
                "bipower_variance": bpv,
                "tripower_quarticity": tpq,
                "z_stats": stats,
                "continuous": cont,
                "jump": jump
            },
            index=inputs.unique_dates
        )
        # cache for property access
        self._results = results

        return results

    def _transform_data(self) -> _TransformOutput:
        """
        Transform input DataFrame to arrays suitable for C extension.

        Returns
        -------
        _TransformOutput
            Transformed data ready for computation.

        Raises
        ------
        ValueError
            If date parsing fails or data type conversion is not possible.
        """
        try:
            returns = self.data[self.return_column].to_numpy().astype(np.float64)  # noqa: E501
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert '{self.return_column} to float64: {e}"
            ) from e

        try:
            dates = pd.to_datetime(self.data[self.date_column])
        except Exception as e:  # noqa: BLE001; ignore blind exception once
            raise ValueError(
                f"Cannot parse '{self.date_column}' as dates: {e}"
            ) from e

        sort_idx = dates.argsort()
        returns = returns[sort_idx]
        dates = dates.iloc[sort_idx].dt.date

        day_indices, unique_dates = pd.factorize(dates, sort=True)
        day_indices = day_indices.astype(np.int64)
        n_days = len(unique_dates)

        return _TransformOutput(
            returns=returns,
            day_indices=day_indices,
            n_days=n_days,
            unique_dates=pd.Index(unique_dates)
        )

    @property
    def jump_days(self) -> pd.DataFrame:
        """Get dates where jumps were detected."""
        if self._results is None:
            raise RuntimeError(
                "Must call decompose() before accessing results"
            )
        return self._results[self._results["jump"] > 0]

    @property
    def jump_ratio(self) -> float:
        """Calculate the proportion of total variance attributable to jumps."""
        if self._results is None:
            raise RuntimeError(
                "Must call decompose() before accessing results"
            )

        total_rv = self._results["realised_variance"].sum()
        total_jump = self._results["jump"].sum()
        return total_jump / total_rv if total_rv > 0 else 0.0

    @property
    def n_jump_days(self) -> int:
        """Get the number of days with detected jumps."""
        if self._results is None:
            raise RuntimeError(
                "Must call decompose() before accessing results"
            )
        return int(np.sum(self._results["jump"] > 0))
