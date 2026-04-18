"""
Data loader module for Bitcoin/USD OHLCV data.

Downloads BTC-USD daily price data from Yahoo Finance via yfinance,
computes log returns, and saves raw and processed data to CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .preprocessing import compute_log_returns, handle_missing_values

logger = logging.getLogger(__name__)

# Repository root is two levels above this file (src/data_loader.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = _REPO_ROOT / "data" / "processed"


def download_btc_data(
    ticker: str = "BTC-USD",
    start_date: str = "2017-01-01",
    end_date: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Download daily BTC-USD OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str, optional
        Yahoo Finance ticker symbol. Default is ``"BTC-USD"``.
    start_date : str, optional
        Start date in ``"YYYY-MM-DD"`` format. Default is ``"2017-01-01"``.
    end_date : str or None, optional
        End date in ``"YYYY-MM-DD"`` format. If ``None``, uses today's date.
    save : bool, optional
        If ``True``, save raw CSV to ``data/raw/``. Default is ``True``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[Open, High, Low, Close, Volume]``
        indexed by date (DatetimeIndex, UTC-normalized).

    Raises
    ------
    ValueError
        If the downloaded DataFrame is empty (e.g., invalid ticker or dates).

    Examples
    --------
    >>> df = download_btc_data(start_date="2020-01-01", save=False)
    >>> "Close" in df.columns
    True
    """
    logger.info("Downloading %s data from %s to %s", ticker, start_date, end_date)

    raw_df: pd.DataFrame = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw_df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}."
        )

    # Flatten multi-level columns produced by yfinance when auto_adjust=True
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    # Ensure DatetimeIndex is timezone-naive for downstream compatibility
    if raw_df.index.tz is not None:
        raw_df.index = raw_df.index.tz_localize(None)

    raw_df.index.name = "Date"

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DATA_DIR / f"{ticker.replace('-', '_')}_daily.csv"
        raw_df.to_csv(raw_path)
        logger.info("Raw data saved to %s", raw_path)

    return raw_df


def load_processed_data(ticker: str = "BTC-USD") -> pd.DataFrame:
    """Load the processed BTC data (with log returns) from disk.

    Parameters
    ----------
    ticker : str, optional
        Ticker symbol used when the file was saved. Default is ``"BTC-USD"``.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns including ``log_return``.

    Raises
    ------
    FileNotFoundError
        If the processed CSV file does not exist. Run
        :func:`prepare_dataset` first.
    """
    processed_path = (
        PROCESSED_DATA_DIR / f"{ticker.replace('-', '_')}_processed.csv"
    )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Run prepare_dataset() first."
        )
    df = pd.read_csv(processed_path, index_col="Date", parse_dates=True)
    logger.info("Loaded processed data from %s (%d rows)", processed_path, len(df))
    return df


def prepare_dataset(
    ticker: str = "BTC-USD",
    start_date: str = "2017-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Full pipeline: download → clean → compute log returns → save.

    Parameters
    ----------
    ticker : str, optional
        Yahoo Finance ticker symbol. Default is ``"BTC-USD"``.
    start_date : str, optional
        Start date in ``"YYYY-MM-DD"`` format. Default is ``"2017-01-01"``.
    end_date : str or None, optional
        End date in ``"YYYY-MM-DD"`` format. If ``None``, uses today.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing OHLCV columns plus ``log_return``.

    Examples
    --------
    >>> # df = prepare_dataset(start_date="2022-01-01")
    >>> # "log_return" in df.columns
    >>> # True
    """
    # 1. Download raw data
    raw_df = download_btc_data(ticker=ticker, start_date=start_date, end_date=end_date)

    # 2. Handle missing values
    clean_df = handle_missing_values(raw_df)

    # 3. Compute log returns
    clean_df = clean_df.copy()
    clean_df["log_return"] = compute_log_returns(clean_df["Close"])

    # Drop the first row (NaN log return)
    clean_df = clean_df.dropna(subset=["log_return"])

    # 4. Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = (
        PROCESSED_DATA_DIR / f"{ticker.replace('-', '_')}_processed.csv"
    )
    clean_df.to_csv(processed_path)
    logger.info(
        "Processed data (%d rows) saved to %s", len(clean_df), processed_path
    )

    return clean_df
