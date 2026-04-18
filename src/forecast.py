"""
Volatility forecasting module for GARCH-family models.

Provides rolling and static forecast functions, along with evaluation
metrics: RMSE, MAE, and the QLIKE loss function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def static_forecast(
    fit_result: ARCHModelResult,
    horizon: int = 10,
    method: str = "analytic",
) -> pd.DataFrame:
    """Generate a static volatility forecast from a fitted GARCH model.

    The static forecast uses the full in-sample data and projects ``horizon``
    steps ahead. No re-estimation is performed.

    Parameters
    ----------
    fit_result : ARCHModelResult
        Fitted GARCH model result from the ``arch`` library.
    horizon : int, optional
        Number of steps ahead to forecast. Default is 10.
    method : str, optional
        Forecast method: ``"analytic"`` (closed-form) or ``"simulation"``.
        Default is ``"analytic"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``h.01`` … ``h.{horizon}``: forecasted conditional variance
        - Rows correspond to the forecast origins.

    Examples
    --------
    >>> # Assumes fit_result is a valid ARCHModelResult
    >>> # forecasts = static_forecast(fit_result, horizon=5)
    >>> # "h.01" in forecasts.columns
    >>> # True
    """
    forecasts = fit_result.forecast(horizon=horizon, method=method)
    variance_df = forecasts.variance
    logger.info("Static forecast: horizon=%d, method=%s", horizon, method)
    return variance_df


def rolling_forecast(
    returns: pd.Series,
    model_spec: "ModelSpec",  # type: ignore[name-defined]
    train_size: float = 0.8,
    refit_frequency: int = 1,
) -> pd.DataFrame:
    """Perform a rolling (out-of-sample) volatility forecast.

    Uses an expanding or rolling window to re-estimate the model at each
    step and produce one-step-ahead conditional variance forecasts.

    Parameters
    ----------
    returns : pd.Series
        Scaled return series (full sample).
    model_spec : ModelSpec
        GARCH model specification.
    train_size : float, optional
        Fraction of data used for initial training. Default is 0.8.
    refit_frequency : int, optional
        How often (in observations) to re-estimate the model.
        ``1`` = refit at every step (fully rolling). Default is 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``forecast_variance``: one-step-ahead conditional variance
        - ``actual_sq_return``: squared actual return (proxy for true variance)
        Indexed by date.

    Notes
    -----
    Rolling forecasting is computationally expensive for large datasets.
    Consider setting ``refit_frequency > 1`` to speed up the calculation.
    """
    from .model_builder import GARCHModelFactory

    n = len(returns)
    split = int(n * train_size)
    dates = returns.index[split:]

    forecast_variances: List[float] = []
    fit_result: Optional[ARCHModelResult] = None

    for i, t in enumerate(range(split, n)):
        train = returns.iloc[:t]

        # Refit at the specified frequency (or on the first iteration)
        if i % refit_frequency == 0 or fit_result is None:
            factory = GARCHModelFactory(train)
            if model_spec.model_type == "GJR-GARCH":
                fr = factory.fit_gjr_garch(
                    p=model_spec.p,
                    q=model_spec.q,
                    distribution=model_spec.distribution,
                )
            else:
                fr = factory.fit(model_spec)
            fit_result = fr.result

        # One-step-ahead forecast
        fc = fit_result.forecast(horizon=1, reindex=False)
        var_forecast = float(fc.variance.iloc[-1, 0])
        forecast_variances.append(var_forecast)

    actual_sq = (returns.iloc[split:].values ** 2)

    result_df = pd.DataFrame(
        {
            "forecast_variance": forecast_variances,
            "actual_sq_return": actual_sq,
        },
        index=dates,
    )

    logger.info(
        "Rolling forecast complete: %d out-of-sample observations.", len(result_df)
    )
    return result_df


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    RMSE = sqrt(mean((actual - predicted)²))

    Parameters
    ----------
    actual : np.ndarray
        Observed values (proxy for true variance, e.g., squared returns).
    predicted : np.ndarray
        Forecasted values (conditional variance).

    Returns
    -------
    float
        RMSE value. Lower is better.

    Examples
    --------
    >>> import numpy as np
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([1.1, 1.9, 3.2])
    >>> round(compute_rmse(actual, pred), 4)
    0.1528
    """
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE).

    MAE = mean(|actual - predicted|)

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    predicted : np.ndarray
        Forecasted values.

    Returns
    -------
    float
        MAE value. Lower is better.

    Examples
    --------
    >>> import numpy as np
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([1.1, 1.9, 3.2])
    >>> round(compute_mae(actual, pred), 4)
    0.1333
    """
    return float(mean_absolute_error(actual, predicted))


def compute_qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the QLIKE loss function for volatility forecast evaluation.

    QLIKE is the standard loss function for volatility forecasting:
        QLIKE = mean(log(predicted) + actual / predicted)

    It is robust to scale and is theoretically motivated by the Gaussian
    log-likelihood. Lower is better.

    Parameters
    ----------
    actual : np.ndarray
        Observed squared returns (proxy for realized variance).
    predicted : np.ndarray
        Forecasted conditional variance (must be positive).

    Returns
    -------
    float
        QLIKE loss. Lower is better.

    Raises
    ------
    ValueError
        If any predicted value is non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> actual = np.array([1.0, 4.0, 9.0])
    >>> predicted = np.array([1.1, 3.8, 9.5])
    >>> qlike = compute_qlike(actual, predicted)
    >>> isinstance(qlike, float)
    True
    """
    if np.any(predicted <= 0):
        raise ValueError("All predicted values must be strictly positive for QLIKE.")
    qlike = np.mean(np.log(predicted) + actual / predicted)
    return float(qlike)


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    variance_col: str = "forecast_variance",
    actual_col: str = "actual_sq_return",
) -> Dict[str, float]:
    """Compute all forecast evaluation metrics for a rolling forecast result.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with forecasted variance and actual squared returns.
        Typically the output of :func:`rolling_forecast`.
    variance_col : str, optional
        Column name for forecasted variance. Default is ``"forecast_variance"``.
    actual_col : str, optional
        Column name for actual squared returns. Default is ``"actual_sq_return"``.

    Returns
    -------
    dict of {str: float}
        Dictionary with keys ``"RMSE"``, ``"MAE"``, ``"QLIKE"``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     "forecast_variance": [1.0, 2.0, 3.0],
    ...     "actual_sq_return": [1.1, 1.9, 3.2],
    ... })
    >>> metrics = evaluate_forecast(df)
    >>> set(metrics.keys()) == {"RMSE", "MAE", "QLIKE"}
    True
    """
    actual = forecast_df[actual_col].values
    predicted = forecast_df[variance_col].values

    metrics = {
        "RMSE": compute_rmse(actual, predicted),
        "MAE": compute_mae(actual, predicted),
        "QLIKE": compute_qlike(actual, predicted),
    }

    logger.info(
        "Forecast evaluation — RMSE: %.6f, MAE: %.6f, QLIKE: %.6f",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["QLIKE"],
    )

    return metrics


def save_forecast_results(
    forecast_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: Path,
    model_name: str = "GARCH",
) -> None:
    """Save rolling forecast results and evaluation metrics to CSV.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Rolling forecast DataFrame (output of :func:`rolling_forecast`).
    metrics : dict
        Evaluation metrics (output of :func:`evaluate_forecast`).
    output_dir : pathlib.Path
        Directory to save results. Will be created if it doesn't exist.
    model_name : str, optional
        Model name used as a prefix for output file names. Default is ``"GARCH"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save forecast DataFrame
    forecast_path = output_dir / f"{model_name}_rolling_forecast.csv"
    forecast_df.to_csv(forecast_path)
    logger.info("Rolling forecast saved to %s", forecast_path)

    # Save metrics
    metrics_path = output_dir / f"{model_name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info("Metrics saved to %s", metrics_path)
