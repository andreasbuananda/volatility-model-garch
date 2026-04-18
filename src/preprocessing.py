"""
Preprocessing module for financial time series.

Provides functions for computing log returns, testing stationarity,
handling outliers, and preparing data for GARCH modeling.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series.

    Log returns are defined as:
        r_t = log(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.Series
        Time series of asset prices (must be positive).

    Returns
    -------
    pd.Series
        Log return series (first observation dropped due to differencing).

    Raises
    ------
    ValueError
        If ``prices`` contains non-positive values.

    Examples
    --------
    >>> import pandas as pd
    >>> p = pd.Series([100.0, 110.0, 121.0])
    >>> r = compute_log_returns(p)
    >>> len(r)
    2
    """
    if (prices <= 0).any():
        raise ValueError("Price series must contain strictly positive values.")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns


def handle_missing_values(
    df: pd.DataFrame,
    price_col: str = "Close",
    ffill_limit: int = 2,
) -> pd.DataFrame:
    """Handle missing values in OHLCV data.

    Forward-fills short gaps (up to ``ffill_limit`` consecutive NaNs)
    and drops any remaining rows with missing values in ``price_col``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    price_col : str, optional
        Column name to check for missing values. Default is ``"Close"``.
    ffill_limit : int, optional
        Maximum number of consecutive NaNs to forward-fill. Default is 2.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame without missing values in ``price_col``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2020-01-01", periods=3)
    >>> df = pd.DataFrame({"Close": [100.0, np.nan, 102.0]}, index=dates)
    >>> result = handle_missing_values(df)
    >>> result["Close"].isna().any()
    False
    """
    df = df.copy()
    df = df.ffill(limit=ffill_limit)
    n_dropped = df[price_col].isna().sum()
    if n_dropped > 0:
        logger.warning(
            "Dropping %d rows with missing '%s' values after forward-fill.",
            n_dropped,
            price_col,
        )
    df = df.dropna(subset=[price_col])
    return df


def adf_test(
    series: pd.Series,
    significance: float = 0.05,
    verbose: bool = True,
) -> Tuple[float, float, bool]:
    """Perform the Augmented Dickey-Fuller (ADF) unit root test.

    Null hypothesis (H0): The series has a unit root (non-stationary).
    Alternative hypothesis (H1): The series is stationary.

    Parameters
    ----------
    series : pd.Series
        Time series to test. Should have no NaN values.
    significance : float, optional
        Significance level for the test. Default is 0.05.
    verbose : bool, optional
        If ``True``, print test results. Default is ``True``.

    Returns
    -------
    tuple of (float, float, bool)
        - ADF test statistic
        - p-value
        - ``True`` if the series is stationary (reject H0)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> stationary = pd.Series(rng.standard_normal(500))
    >>> stat, pval, is_stationary = adf_test(stationary, verbose=False)
    >>> is_stationary
    True
    """
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    adf_stat: float = result[0]
    p_value: float = result[1]
    is_stationary: bool = bool(p_value < significance)

    if verbose:
        print(f"ADF Test Results for '{series.name}':")
        print(f"  ADF Statistic : {adf_stat:.6f}")
        print(f"  p-value       : {p_value:.6f}")
        print(f"  Critical values: {result[4]}")
        conclusion = "STATIONARY" if is_stationary else "NON-STATIONARY"
        print(f"  Conclusion    : {conclusion} (α={significance})")

    return adf_stat, p_value, is_stationary


def kpss_test(
    series: pd.Series,
    significance: float = 0.05,
    regression: str = "c",
    verbose: bool = True,
) -> Tuple[float, float, bool]:
    """Perform the KPSS stationarity test.

    Null hypothesis (H0): The series is stationary.
    Alternative hypothesis (H1): The series has a unit root (non-stationary).

    Note: KPSS complements ADF — use both for robust conclusions.

    Parameters
    ----------
    series : pd.Series
        Time series to test. Should have no NaN values.
    significance : float, optional
        Significance level. Default is 0.05.
    regression : str, optional
        ``"c"`` for constant (level stationarity) or ``"ct"`` for constant
        and trend. Default is ``"c"``.
    verbose : bool, optional
        If ``True``, print test results. Default is ``True``.

    Returns
    -------
    tuple of (float, float, bool)
        - KPSS test statistic
        - p-value (interpolated from table)
        - ``True`` if the series is stationary (fail to reject H0)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> stationary = pd.Series(rng.standard_normal(500))
    >>> stat, pval, is_stationary = kpss_test(stationary, verbose=False)
    >>> is_stationary
    True
    """
    series = series.dropna()
    kpss_stat, p_value, _, crit_values = kpss(series, regression=regression, nlags="auto")
    is_stationary: bool = bool(p_value > significance)  # Fail to reject H0 → stationary

    if verbose:
        print(f"KPSS Test Results for '{series.name}':")
        print(f"  KPSS Statistic: {kpss_stat:.6f}")
        print(f"  p-value       : {p_value:.6f}")
        print(f"  Critical values: {crit_values}")
        conclusion = "STATIONARY" if is_stationary else "NON-STATIONARY"
        print(f"  Conclusion    : {conclusion} (α={significance})")

    return kpss_stat, p_value, is_stationary


def remove_outliers_zscore(
    series: pd.Series,
    threshold: float = 5.0,
    replace_with: str = "nan",
) -> pd.Series:
    """Detect and handle outliers in a return series using the z-score method.

    For Bitcoin returns, a conservative threshold (e.g., 5.0) is recommended
    because extreme moves are common and may represent genuine market events.

    Parameters
    ----------
    series : pd.Series
        Return series to clean.
    threshold : float, optional
        Z-score threshold above which values are treated as outliers.
        Default is 5.0 (very conservative for crypto).
    replace_with : str, optional
        How to handle outliers: ``"nan"`` to replace with NaN (then
        forward-fill), or ``"clip"`` to winsorize at the threshold.
        Default is ``"nan"``.

    Returns
    -------
    pd.Series
        Cleaned return series with outliers handled.

    Raises
    ------
    ValueError
        If ``replace_with`` is not ``"nan"`` or ``"clip"``.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([0.01, 0.02, -0.01, 100.0, 0.005])
    >>> cleaned = remove_outliers_zscore(s, threshold=3.0)
    >>> cleaned.isna().sum() <= 1
    True
    """
    if replace_with not in ("nan", "clip"):
        raise ValueError("replace_with must be 'nan' or 'clip'.")

    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    outlier_mask = z_scores.abs() > threshold
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        logger.info(
            "Found %d outliers (|z| > %.1f) in '%s'.",
            n_outliers,
            threshold,
            series.name,
        )

    result = series.copy()
    if replace_with == "nan":
        result[outlier_mask] = np.nan
    else:
        # Clip at ±threshold standard deviations
        upper = mean + threshold * std
        lower = mean - threshold * std
        result = result.clip(lower=lower, upper=upper)

    return result


def prepare_returns_for_garch(
    returns: pd.Series,
    scale: float = 100.0,
) -> pd.Series:
    """Scale log returns for GARCH model estimation.

    GARCH models are typically estimated on percentage returns (×100) to
    improve numerical stability of the optimization. This is a common
    convention in the GARCH literature.

    Parameters
    ----------
    returns : pd.Series
        Log returns series (raw, not scaled).
    scale : float, optional
        Scaling factor. Default is 100.0 (percentage returns).

    Returns
    -------
    pd.Series
        Scaled return series.

    Examples
    --------
    >>> import pandas as pd
    >>> r = pd.Series([0.01, -0.02, 0.005])
    >>> scaled = prepare_returns_for_garch(r)
    >>> float(scaled.iloc[0])
    1.0
    """
    scaled = returns * scale
    scaled.name = f"{returns.name}_pct" if returns.name else "log_return_pct"
    return scaled
