"""
GARCH model factory for Bitcoin volatility modeling.

Provides a factory class and helper functions to build, fit, and compare
GARCH-family models (GARCH, EGARCH, GJR-GARCH) using the ``arch`` library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from arch import arch_model
from arch.univariate import ARX, EGARCH, GARCH, ConstantMean, ZeroMean
from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)

# Supported model types and distributions
SUPPORTED_MODELS = ("GARCH", "EGARCH", "GJR-GARCH")
SUPPORTED_DISTRIBUTIONS = ("normal", "t", "skewt")


@dataclass
class ModelSpec:
    """Specification for a GARCH-family model.

    Attributes
    ----------
    model_type : str
        One of ``"GARCH"``, ``"EGARCH"``, or ``"GJR-GARCH"``.
    p : int
        ARCH order (lag order of squared residuals).
    q : int
        GARCH order (lag order of conditional variance).
    distribution : str
        Error distribution: ``"normal"``, ``"t"``, or ``"skewt"``.
    mean : str
        Mean model: ``"zero"`` or ``"constant"``.
    """

    model_type: str = "GARCH"
    p: int = 1
    q: int = 1
    distribution: str = "normal"
    mean: str = "constant"

    def __post_init__(self) -> None:
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {SUPPORTED_MODELS}, got '{self.model_type}'."
            )
        if self.distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {SUPPORTED_DISTRIBUTIONS}, "
                f"got '{self.distribution}'."
            )
        if self.p < 1 or self.q < 1:
            raise ValueError("Both p and q must be >= 1.")


@dataclass
class FitResult:
    """Container for a fitted GARCH model result.

    Attributes
    ----------
    spec : ModelSpec
        The model specification used for fitting.
    result : ARCHModelResult
        The fitted model result object from the ``arch`` library.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    log_likelihood : float
        Log-likelihood of the fitted model.
    """

    spec: ModelSpec
    result: ARCHModelResult
    aic: float = field(init=False)
    bic: float = field(init=False)
    log_likelihood: float = field(init=False)

    def __post_init__(self) -> None:
        self.aic = self.result.aic
        self.bic = self.result.bic
        self.log_likelihood = self.result.loglikelihood

    def summary(self) -> str:
        """Return a formatted summary string of the fit result."""
        return (
            f"{self.spec.model_type}({self.spec.p},{self.spec.q}) "
            f"[{self.spec.distribution}] — "
            f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}, "
            f"LogL: {self.log_likelihood:.4f}"
        )


class GARCHModelFactory:
    """Factory for building and fitting GARCH-family volatility models.

    This factory supports GARCH(p,q), EGARCH(p,q), and GJR-GARCH(p,q)
    models with Normal, Student-t, and Skewed-t error distributions.

    The canonical GARCH(1,1) equation is:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    For stationarity: α + β < 1.

    Parameters
    ----------
    returns : pd.Series
        Return series for model estimation. Should be scaled (×100)
        for numerical stability.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> r = pd.Series(rng.standard_normal(500))
    >>> factory = GARCHModelFactory(r)
    >>> result = factory.fit(ModelSpec("GARCH", 1, 1, "normal"))
    >>> result.aic < 0 or result.aic > 0  # AIC is a finite number
    True
    """

    def __init__(self, returns: pd.Series) -> None:
        self.returns = returns

    def fit(
        self,
        spec: ModelSpec,
        update_freq: int = 0,
        disp: str = "off",
    ) -> FitResult:
        """Build and fit a GARCH model according to the given specification.

        Parameters
        ----------
        spec : ModelSpec
            Model specification (type, p, q, distribution, mean).
        update_freq : int, optional
            Frequency for optimizer output. ``0`` suppresses output.
        disp : str, optional
            Display optimizer output. ``"off"`` suppresses output.

        Returns
        -------
        FitResult
            Container with the fitted model, AIC, BIC, and log-likelihood.

        Raises
        ------
        ValueError
            If an unsupported model type or distribution is specified.
        RuntimeError
            If the optimizer fails to converge.
        """
        vol_model = self._build_vol_model(spec)
        dist_str = self._map_distribution(spec.distribution)

        am = arch_model(
            self.returns,
            mean=spec.mean,
            vol=vol_model,
            p=spec.p,
            q=spec.q,
            dist=dist_str,
        )

        try:
            fitted = am.fit(update_freq=update_freq, disp=disp)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit {spec.model_type}({spec.p},{spec.q}) "
                f"with '{spec.distribution}' distribution: {exc}"
            ) from exc

        logger.info(
            "Fitted %s(%d,%d) [%s]: AIC=%.4f, BIC=%.4f",
            spec.model_type,
            spec.p,
            spec.q,
            spec.distribution,
            fitted.aic,
            fitted.bic,
        )

        return FitResult(spec=spec, result=fitted)

    def _build_vol_model(self, spec: ModelSpec) -> str:
        """Map ModelSpec.model_type to arch library vol string.

        Parameters
        ----------
        spec : ModelSpec
            Model specification.

        Returns
        -------
        str
            The ``vol`` parameter string for ``arch_model``.
        """
        mapping: Dict[str, str] = {
            "GARCH": "GARCH",
            "EGARCH": "EGARCH",
            "GJR-GARCH": "GARCH",  # GJR-GARCH uses o=1 in arch_model
        }
        return mapping[spec.model_type]

    def _map_distribution(self, distribution: str) -> str:
        """Map distribution name to arch library string.

        Parameters
        ----------
        distribution : str
            One of ``"normal"``, ``"t"``, or ``"skewt"``.

        Returns
        -------
        str
            The ``dist`` parameter string for ``arch_model``.
        """
        mapping: Dict[str, str] = {
            "normal": "normal",
            "t": "t",
            "skewt": "skewt",
        }
        return mapping[distribution]

    def fit_gjr_garch(
        self,
        p: int = 1,
        q: int = 1,
        distribution: str = "normal",
        update_freq: int = 0,
        disp: str = "off",
    ) -> FitResult:
        """Fit a GJR-GARCH model (asymmetric GARCH with leverage effect).

        The GJR-GARCH(1,1) model is:
            σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
        where I_{t-1} = 1 if ε_{t-1} < 0 (bad news), 0 otherwise.
        γ > 0 implies a leverage effect.

        Parameters
        ----------
        p : int, optional
            ARCH order. Default is 1.
        q : int, optional
            GARCH order. Default is 1.
        distribution : str, optional
            Error distribution. Default is ``"normal"``.
        update_freq : int, optional
            Optimizer update frequency. Default is 0 (silent).
        disp : str, optional
            Display optimizer output. Default is ``"off"``.

        Returns
        -------
        FitResult
            Fitted GJR-GARCH model result.
        """
        dist_str = self._map_distribution(distribution)
        am = arch_model(
            self.returns,
            mean="constant",
            vol="GARCH",
            p=p,
            o=1,  # asymmetric (leverage) term
            q=q,
            dist=dist_str,
        )
        try:
            fitted = am.fit(update_freq=update_freq, disp=disp)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit GJR-GARCH({p},{q}) with '{distribution}' distribution: {exc}"
            ) from exc

        spec = ModelSpec("GJR-GARCH", p=p, q=q, distribution=distribution)
        logger.info(
            "Fitted GJR-GARCH(%d,%d) [%s]: AIC=%.4f, BIC=%.4f",
            p,
            q,
            distribution,
            fitted.aic,
            fitted.bic,
        )
        return FitResult(spec=spec, result=fitted)


def select_best_model(
    fit_results: List[FitResult],
    criterion: str = "aic",
) -> FitResult:
    """Select the best model from a list of fitted models by AIC or BIC.

    Lower AIC/BIC indicates a better model (balances fit and complexity).

    Parameters
    ----------
    fit_results : list of FitResult
        List of fitted model results to compare.
    criterion : str, optional
        Selection criterion: ``"aic"`` or ``"bic"``. Default is ``"aic"``.

    Returns
    -------
    FitResult
        The best-fitting model result.

    Raises
    ------
    ValueError
        If ``criterion`` is not ``"aic"`` or ``"bic"``, or if the list
        is empty.

    Examples
    --------
    >>> # Assumes fit_results is a list of FitResult objects
    >>> # best = select_best_model(fit_results, criterion="bic")
    """
    if not fit_results:
        raise ValueError("fit_results must not be empty.")
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'.")

    key = criterion
    best = min(fit_results, key=lambda r: getattr(r, key))
    logger.info(
        "Best model by %s: %s (%.4f)",
        criterion.upper(),
        best.spec.model_type,
        getattr(best, key),
    )
    return best


def model_comparison_table(fit_results: List[FitResult]) -> pd.DataFrame:
    """Create a comparison table for multiple fitted GARCH models.

    Parameters
    ----------
    fit_results : list of FitResult
        Fitted model results to compare.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Model, p, q, Distribution, LogL, AIC, BIC.
        Sorted by AIC (ascending).

    Examples
    --------
    >>> # table = model_comparison_table(fit_results)
    >>> # table.columns.tolist()
    >>> # ['Model', 'p', 'q', 'Distribution', 'LogL', 'AIC', 'BIC']
    """
    rows = []
    for fr in fit_results:
        rows.append(
            {
                "Model": fr.spec.model_type,
                "p": fr.spec.p,
                "q": fr.spec.q,
                "Distribution": fr.spec.distribution,
                "LogL": round(fr.log_likelihood, 4),
                "AIC": round(fr.aic, 4),
                "BIC": round(fr.bic, 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("AIC", ascending=True).reset_index(drop=True)
    return df


def grid_search_garch(
    returns: pd.Series,
    model_types: Optional[List[str]] = None,
    p_range: Tuple[int, int] = (1, 2),
    q_range: Tuple[int, int] = (1, 2),
    distributions: Optional[List[str]] = None,
) -> Tuple[List[FitResult], pd.DataFrame]:
    """Perform a grid search over GARCH model specifications.

    Fits all combinations of model_types × p × q × distributions and
    returns a ranked comparison table.

    Parameters
    ----------
    returns : pd.Series
        Scaled return series for model fitting.
    model_types : list of str, optional
        Model types to try. Default is ``["GARCH", "EGARCH", "GJR-GARCH"]``.
    p_range : tuple of (int, int), optional
        (min_p, max_p) range for ARCH order. Default is (1, 2).
    q_range : tuple of (int, int), optional
        (min_q, max_q) range for GARCH order. Default is (1, 2).
    distributions : list of str, optional
        Distributions to try. Default is ``["normal", "t"]``.

    Returns
    -------
    tuple of (list of FitResult, pd.DataFrame)
        - All fitted models
        - Comparison table sorted by AIC
    """
    if model_types is None:
        model_types = ["GARCH", "EGARCH", "GJR-GARCH"]
    if distributions is None:
        distributions = ["normal", "t"]

    factory = GARCHModelFactory(returns)
    results: List[FitResult] = []

    for model_type in model_types:
        for p in range(p_range[0], p_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                for dist in distributions:
                    spec = ModelSpec(
                        model_type=model_type, p=p, q=q, distribution=dist
                    )
                    try:
                        if model_type == "GJR-GARCH":
                            fr = factory.fit_gjr_garch(p=p, q=q, distribution=dist)
                        else:
                            fr = factory.fit(spec)
                        results.append(fr)
                    except RuntimeError as exc:
                        logger.warning("Skipping %s: %s", spec, exc)

    table = model_comparison_table(results)
    return results, table
