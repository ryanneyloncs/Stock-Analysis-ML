"""
Data fetching module for retrieving stock data from Yahoo Finance.

This module handles all data retrieval operations including retry logic
for handling transient network failures.
"""

import time

import pandas as pd
import yfinance as yf


def fetch_data_with_retry(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> pd.DataFrame | None:
    """
    Fetch stock data from Yahoo Finance with automatic retry on failure.

    Uses linear backoff strategy where wait time increases with each retry attempt.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay between retries in seconds.

    Returns:
        DataFrame with OHLCV columns (Open, High, Low, Close, Volume),
        or None if all retry attempts failed.
    """
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}: Fetching data for {symbol}...")

        if attempt > 0:
            wait_time = retry_delay * attempt
            print(f"Waiting {wait_time} seconds...")
            time.sleep(wait_time)

        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )

            if not data.empty:
                print(f"Successfully fetched {len(data)} days of data")
                return data

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")

    return None


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw stock data for analysis.

    Handles multi-level column indices that may come from yfinance
    and removes rows with missing Close prices.

    Args:
        data: Raw stock data DataFrame from yfinance.

    Returns:
        Cleaned DataFrame with flattened columns and no missing Close values.
    """
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Remove rows with missing Close prices (essential for analysis)
    data = data.dropna(subset=["Close"])

    return data
