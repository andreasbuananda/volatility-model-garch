"""
Tests for the preprocessing module.

Tests cover log return computation, missing value handling,
stationarity tests, outlier detection, and return scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    adf_test,
    compute_log_returns,
    handle_missing_values,
    kpss_test,
    prepare_returns_for_garch,
    remove_outliers_zscore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def price_series() -> pd.Series:
    """Simple upward-trending price series."""
    prices = [100.0, 105.0, 110.25, 104.74, 109.98, 115.48, 109.71, 115.19]
    return pd.Series(prices, name="Close")


@pytest.fixture()
def btc_like_returns() -> pd.Series:
    """Simulated stationary return series (white noise, approx. BTC-like)."""
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(500) * 0.03  # 3% daily std
    return pd.Series(returns, name="log_return")


@pytest.fixture()
def ohlcv_df_with_nan() -> pd.DataFrame:
    """OHLCV DataFrame with some NaN Close values."""
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    closes = [30000.0, np.nan, 31000.0, 31500.0, np.nan, np.nan, 32000.0, None, 33000.0, 34000.0]
    return pd.DataFrame({"Close": closes}, index=dates)


# ---------------------------------------------------------------------------
# Tests: compute_log_returns
# ---------------------------------------------------------------------------


class TestComputeLogReturns:
    def test_output_length(self, price_series: pd.Series) -> None:
        """Log returns should have one fewer observation than prices."""
        returns = compute_log_returns(price_series)
        assert len(returns) == len(price_series) - 1

    def test_no_nan_in_output(self, price_series: pd.Series) -> None:
        """Output should not contain NaN values."""
        returns = compute_log_returns(price_series)
        assert not returns.isna().any()

    def test_correct_first_return(self, price_series: pd.Series) -> None:
        """First log return should equal log(105/100)."""
        returns = compute_log_returns(price_series)
        expected = np.log(105.0 / 100.0)
        assert abs(returns.iloc[0] - expected) < 1e-10

    def test_name_set(self, price_series: pd.Series) -> None:
        """Output Series should be named 'log_return'."""
        returns = compute_log_returns(price_series)
        assert returns.name == "log_return"

    def test_raises_on_non_positive_prices(self) -> None:
        """Should raise ValueError for non-positive prices."""
        prices = pd.Series([100.0, 0.0, 105.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_log_returns(prices)

    def test_raises_on_negative_prices(self) -> None:
        """Should raise ValueError for negative prices."""
        prices = pd.Series([100.0, -50.0, 105.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_log_returns(prices)

    def test_constant_prices_yield_zero_returns(self) -> None:
        """Constant prices should yield zero returns."""
        prices = pd.Series([100.0] * 5)
        returns = compute_log_returns(prices)
        assert (returns == 0.0).all()

    def test_single_price_raises_or_empty(self) -> None:
        """A single price should yield an empty return series."""
        prices = pd.Series([100.0])
        returns = compute_log_returns(prices)
        assert len(returns) == 0


# ---------------------------------------------------------------------------
# Tests: handle_missing_values
# ---------------------------------------------------------------------------


class TestHandleMissingValues:
    def test_no_nan_after_cleaning(self, ohlcv_df_with_nan: pd.DataFrame) -> None:
        """Cleaned DataFrame should have no NaN in Close column."""
        result = handle_missing_values(ohlcv_df_with_nan)
        assert not result["Close"].isna().any()

    def test_forward_fill_short_gaps(self) -> None:
        """Single NaN between valid values should be forward-filled."""
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"Close": [100.0, np.nan, 102.0]}, index=dates)
        result = handle_missing_values(df)
        assert len(result) == 3
        assert result["Close"].iloc[1] == 100.0  # forward-filled

    def test_long_gaps_are_dropped(self) -> None:
        """Gaps longer than ffill_limit=2 should be dropped."""
        dates = pd.date_range("2020-01-01", periods=6)
        df = pd.DataFrame(
            {"Close": [100.0, np.nan, np.nan, np.nan, 105.0, 106.0]}, index=dates
        )
        result = handle_missing_values(df, ffill_limit=2)
        # The 3rd NaN (index 3) cannot be filled; row is dropped
        assert len(result) == 5
        assert not result["Close"].isna().any()

    def test_no_rows_dropped_when_clean(self) -> None:
        """No rows should be dropped when there are no missing values."""
        dates = pd.date_range("2020-01-01", periods=5)
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
        result = handle_missing_values(df)
        assert len(result) == len(df)

    def test_does_not_modify_original(self, ohlcv_df_with_nan: pd.DataFrame) -> None:
        """Function should not modify the original DataFrame in-place."""
        original_nan_count = ohlcv_df_with_nan["Close"].isna().sum()
        handle_missing_values(ohlcv_df_with_nan)
        assert ohlcv_df_with_nan["Close"].isna().sum() == original_nan_count


# ---------------------------------------------------------------------------
# Tests: adf_test
# ---------------------------------------------------------------------------


class TestADFTest:
    def test_stationary_series_detected(self, btc_like_returns: pd.Series) -> None:
        """White noise should be detected as stationary."""
        _, _, is_stationary = adf_test(btc_like_returns, verbose=False)
        assert is_stationary is True

    def test_nonstationary_series_detected(self) -> None:
        """Random walk should be detected as non-stationary."""
        rng = np.random.default_rng(0)
        random_walk = pd.Series(rng.standard_normal(500).cumsum(), name="price")
        _, _, is_stationary = adf_test(random_walk, verbose=False)
        assert is_stationary is False

    def test_returns_tuple_of_three(self, btc_like_returns: pd.Series) -> None:
        """Should return a 3-tuple: (stat, pval, bool)."""
        result = adf_test(btc_like_returns, verbose=False)
        assert len(result) == 3
        assert isinstance(result[2], bool)

    def test_pvalue_is_float(self, btc_like_returns: pd.Series) -> None:
        """p-value should be a float in [0, 1]."""
        _, pval, _ = adf_test(btc_like_returns, verbose=False)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Tests: kpss_test
# ---------------------------------------------------------------------------


class TestKPSSTest:
    def test_stationary_series_passes(self, btc_like_returns: pd.Series) -> None:
        """White noise should fail to reject H0 (stationary)."""
        _, _, is_stationary = kpss_test(btc_like_returns, verbose=False)
        assert is_stationary is True

    def test_returns_tuple_of_three(self, btc_like_returns: pd.Series) -> None:
        """Should return a 3-tuple."""
        result = kpss_test(btc_like_returns, verbose=False)
        assert len(result) == 3
        assert isinstance(result[2], bool)


# ---------------------------------------------------------------------------
# Tests: remove_outliers_zscore
# ---------------------------------------------------------------------------


class TestRemoveOutliersZscore:
    def test_extreme_outlier_replaced_with_nan(self) -> None:
        """Extreme outlier should be replaced with NaN."""
        rng = np.random.default_rng(1)
        s = pd.Series(rng.standard_normal(100))
        s.iloc[50] = 1000.0  # Extreme outlier
        result = remove_outliers_zscore(s, threshold=3.0, replace_with="nan")
        assert result.isna().sum() >= 1

    def test_clip_mode_no_nan(self) -> None:
        """Clip mode should not introduce NaN values."""
        rng = np.random.default_rng(1)
        s = pd.Series(rng.standard_normal(100))
        s.iloc[50] = 1000.0
        result = remove_outliers_zscore(s, threshold=3.0, replace_with="clip")
        assert not result.isna().any()

    def test_clip_reduces_extreme_values(self) -> None:
        """Clipped values should be smaller in absolute value than the outlier."""
        rng = np.random.default_rng(99)
        # Large sample so z-score of outlier clearly exceeds threshold
        s = pd.Series(rng.standard_normal(200))
        s.iloc[100] = 1000.0  # Extreme outlier; z-score >> 2.0
        result = remove_outliers_zscore(s, threshold=2.0, replace_with="clip")
        assert result.abs().max() < 1000.0

    def test_invalid_replace_with_raises(self) -> None:
        """Invalid replace_with value should raise ValueError."""
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="replace_with must be"):
            remove_outliers_zscore(s, replace_with="drop")

    def test_no_outliers_unchanged(self) -> None:
        """Series without outliers should be unchanged."""
        rng = np.random.default_rng(42)
        s = pd.Series(rng.standard_normal(200))  # All z-scores << 5
        result = remove_outliers_zscore(s, threshold=5.0, replace_with="nan")
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests: prepare_returns_for_garch
# ---------------------------------------------------------------------------


class TestPrepareReturnsForGARCH:
    def test_default_scale_is_100(self) -> None:
        """Default scaling should multiply returns by 100."""
        r = pd.Series([0.01, -0.02, 0.005])
        result = prepare_returns_for_garch(r)
        assert abs(result.iloc[0] - 1.0) < 1e-10
        assert abs(result.iloc[1] - (-2.0)) < 1e-10

    def test_custom_scale(self) -> None:
        """Custom scale should be applied correctly."""
        r = pd.Series([0.01, 0.02])
        result = prepare_returns_for_garch(r, scale=10.0)
        assert abs(result.iloc[0] - 0.1) < 1e-10

    def test_output_has_name(self) -> None:
        """Output should have a descriptive name."""
        r = pd.Series([0.01, 0.02], name="log_return")
        result = prepare_returns_for_garch(r)
        assert result.name is not None
        assert "pct" in result.name
