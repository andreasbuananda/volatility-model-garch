"""
Diagnostics module for GARCH model residual analysis.

Provides functions for testing model adequacy via Ljung-Box, ARCH-LM tests,
and ACF/PACF plots of standardized residuals and squared residuals.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)


def ljung_box_test(
    residuals: pd.Series,
    lags: int = 10,
    significance: float = 0.05,
    verbose: bool = True,
) -> Tuple[pd.Series, pd.Series, bool]:
    """Perform the Ljung-Box test for autocorrelation in residuals.

    Tests H0: No autocorrelation up to lag ``lags``.
    A significant result (p < α) suggests remaining autocorrelation,
    indicating the mean model may be mis-specified.

    Parameters
    ----------
    residuals : pd.Series
        Standardized residuals from a fitted GARCH model.
    lags : int, optional
        Number of lags to test. Default is 10.
    significance : float, optional
        Significance level. Default is 0.05.
    verbose : bool, optional
        If ``True``, print test results. Default is ``True``.

    Returns
    -------
    tuple of (pd.Series, pd.Series, bool)
        - lb_stat: Ljung-Box test statistics for each lag
        - lb_pval: p-values for each lag
        - passed: ``True`` if no significant autocorrelation detected at any lag

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> resid = pd.Series(rng.standard_normal(500))
    >>> lb_stat, lb_pval, passed = ljung_box_test(resid, lags=5, verbose=False)
    >>> isinstance(passed, bool)
    True
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_result = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    lb_stat = lb_result["lb_stat"]
    lb_pval = lb_result["lb_pvalue"]
    passed = bool((lb_pval > significance).all())

    if verbose:
        print(f"Ljung-Box Test (up to lag {lags}):")
        print(lb_result.to_string())
        status = "PASS (no autocorrelation)" if passed else "FAIL (autocorrelation detected)"
        print(f"  Conclusion: {status} (α={significance})")

    return lb_stat, lb_pval, passed


def arch_lm_test(
    residuals: pd.Series,
    lags: int = 10,
    significance: float = 0.05,
    verbose: bool = True,
) -> Tuple[float, float, bool]:
    """Perform the ARCH-LM test for remaining ARCH effects in residuals.

    Tests H0: No ARCH effects (homoskedastic residuals).
    A significant result (p < α) before fitting indicates ARCH effects are
    present; after fitting a GARCH model, it should be insignificant.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from the mean model (or standardized GARCH residuals).
    lags : int, optional
        Number of lags to include in the test. Default is 10.
    significance : float, optional
        Significance level. Default is 0.05.
    verbose : bool, optional
        If ``True``, print test results. Default is ``True``.

    Returns
    -------
    tuple of (float, float, bool)
        - lm_stat: LM test statistic
        - lm_pval: p-value
        - passed: ``True`` if no significant ARCH effects remain

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> resid = pd.Series(rng.standard_normal(500))
    >>> lm_stat, lm_pval, passed = arch_lm_test(resid, lags=5, verbose=False)
    >>> isinstance(passed, bool)
    True
    """
    clean_resid = residuals.dropna().values
    lm_stat, lm_pval, f_stat, f_pval = het_arch(clean_resid, nlags=lags)
    passed = bool(lm_pval > significance)

    if verbose:
        print(f"ARCH-LM Test (lags={lags}):")
        print(f"  LM Statistic : {lm_stat:.6f}")
        print(f"  LM p-value   : {lm_pval:.6f}")
        print(f"  F Statistic  : {f_stat:.6f}")
        print(f"  F p-value    : {f_pval:.6f}")
        status = "PASS (no ARCH effects)" if passed else "FAIL (ARCH effects remain)"
        print(f"  Conclusion   : {status} (α={significance})")

    return lm_stat, lm_pval, passed


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 30,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ACF and PACF side by side for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series to plot (e.g., returns, squared returns, residuals).
    lags : int, optional
        Number of lags to display. Default is 30.
    title_prefix : str, optional
        Prefix for plot titles (e.g., ``"Standardized Residuals"``).
    figsize : tuple of (int, int), optional
        Figure size. Default is (14, 5).
    save_path : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    prefix = f"{title_prefix} — " if title_prefix else ""

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f"{prefix}ACF")

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title(f"{prefix}PACF")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ACF/PACF plot saved to %s", save_path)

    return fig


def plot_garch_diagnostics(
    fit_result: ARCHModelResult,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate a 4-panel diagnostic plot for a fitted GARCH model.

    Panels:
    1. Standardized residuals over time
    2. Histogram of standardized residuals (with normal overlay)
    3. ACF of standardized residuals
    4. ACF of squared standardized residuals

    Parameters
    ----------
    fit_result : ARCHModelResult
        Fitted GARCH model result from the ``arch`` library.
    figsize : tuple of (int, int), optional
        Figure size. Default is (14, 10).
    save_path : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated diagnostic figure.
    """
    std_resid = fit_result.std_resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Standardized residuals over time
    axes[0, 0].plot(std_resid.index, std_resid.values, linewidth=0.8, color="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Standardized Residuals")
    axes[0, 0].set_xlabel("Date")

    # Panel 2: Histogram with normal overlay
    x = np.linspace(std_resid.min(), std_resid.max(), 200)
    normal_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    axes[0, 1].hist(std_resid, bins=50, density=True, color="steelblue", alpha=0.6)
    axes[0, 1].plot(x, normal_pdf, "r-", linewidth=1.5, label="N(0,1)")
    axes[0, 1].set_title("Distribution of Standardized Residuals")
    axes[0, 1].legend()

    # Panel 3: ACF of standardized residuals
    plot_acf(std_resid, lags=20, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("ACF of Standardized Residuals")

    # Panel 4: ACF of squared standardized residuals
    plot_acf(std_resid**2, lags=20, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("ACF of Squared Standardized Residuals")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("GARCH diagnostics plot saved to %s", save_path)

    return fig


def durbin_watson_test(residuals: pd.Series) -> float:
    """Compute the Durbin-Watson statistic for first-order autocorrelation.

    DW ≈ 2 indicates no autocorrelation; DW < 2 indicates positive
    autocorrelation; DW > 2 indicates negative autocorrelation.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from a fitted model.

    Returns
    -------
    float
        Durbin-Watson statistic (range: 0 to 4).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> r = pd.Series(rng.standard_normal(300))
    >>> dw = durbin_watson_test(r)
    >>> 1.5 < dw < 2.5
    True
    """
    dw_stat = durbin_watson(residuals.dropna().values)
    logger.info("Durbin-Watson statistic: %.4f", dw_stat)
    return float(dw_stat)
