"""
Bitcoin Volatility GARCH Modeling Package.

This package provides tools for downloading, preprocessing, modeling,
and forecasting Bitcoin/USD financial volatility using GARCH-family models.

Submodules
----------
data_loader
    Download and save BTC-USD OHLCV data via yfinance.
preprocessing
    Log returns, stationarity tests, outlier handling.
model_builder
    GARCH/EGARCH/GJR-GARCH model factory.
diagnostics
    Ljung-Box, ARCH-LM tests and ACF/PACF plots.
forecast
    Rolling and static volatility forecasting with evaluation metrics.
"""

__version__ = "0.1.0"
__author__ = "Andreas Buananda"
